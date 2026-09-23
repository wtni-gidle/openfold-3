# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Portable data/inference hand-off utilities for EnsembleFold inference.

The prepared bundle deliberately stores model-independent resources next to a
single-query JSON file.  Runtime OpenFold objects and caches are rebuilt from this
bundle by inference-only jobs and are never part of the persistent interface.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from biotite.structure.io import pdbx
from pydantic import BaseModel

from openfold3.core.config.msa_pipeline_configs import (
    MsaSampleProcessorInputInference,
)
from openfold3.core.data.af3_input import load_af3_query_set as load_af3_query_set
from openfold3.core.data.af3_input import write_af3_query_sets as write_af3_query_sets
from openfold3.core.data.io.compression import read_text_auto, write_zstd_text
from openfold3.core.data.io.sequence.msa import (
    MsaSampleParserInference,
)
from openfold3.core.data.pipelines.sample_processing.msa import (
    create_paired,
    create_paired_from_precomputed,
)
from openfold3.core.data.primitives.sequence.msa import MsaArray, MsaArrayCollection
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.projects.of3_all_atom.config.dataset_config_components import (
    MSASettings,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
    PreparedTemplate,
)

if TYPE_CHECKING:
    from openfold3.core.data.pipelines.preprocessing.template import (
        TemplatePreprocessorSettings,
    )

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_RESOURCE_FIELDS = {
    "main_msa_file_paths",
    "paired_msa_file_paths",
    "template_alignment_file_path",
    "template_cif_paths",
    "prepared_template_file_path",
    "sdf_file_path",
}


def sanitise_job_name(name: str) -> str:
    """Return a safe, single-component output name."""
    value = _SAFE_NAME_RE.sub("_", name.strip()).strip("._")
    if not value or value in {".", ".."}:
        raise ValueError(f"Unsafe or empty query name: {name!r}")
    return value


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically publish UTF-8 text in the destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _relative_resource(value: Any, owner_directory: Path) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return [_relative_resource(item, owner_directory) for item in value]
    return os.path.relpath(Path(value), owner_directory)


def query_set_to_portable_dict(
    query_set: InferenceQuerySet, owner_directory: Path
) -> dict[str, Any]:
    """Serialize a query set with resource paths relative to its JSON directory."""
    data = query_set.model_dump(mode="json", exclude_none=True)
    for query in data["queries"].values():
        # query_name is an internal convenience field, not input schema.
        query.pop("query_name", None)
        for chain in query["chains"]:
            for field in _RESOURCE_FIELDS & chain.keys():
                chain[field] = _relative_resource(chain[field], owner_directory)
            for template in chain.get("templates") or []:
                template["mmcif_path"] = _relative_resource(
                    template["mmcif_path"], owner_directory
                )
    return data


def validate_prepared_query_names(query_set: InferenceQuerySet) -> dict[str, str]:
    """Map safe names to original names, rejecting ambiguous output locations."""
    safe_names: dict[str, str] = {}
    for name in query_set.queries:
        safe = sanitise_job_name(name)
        previous = safe_names.get(safe)
        if previous is not None and previous != name:
            raise ValueError(
                f"Query names {previous!r} and {name!r} both map to {safe!r}"
            )
        safe_names[safe] = name
    return safe_names


def write_prepared_query_sets(
    query_set: InferenceQuerySet, output_root: Path
) -> dict[str, Path]:
    """Write one portable ``<name>_data.json`` for every query.

    The function expects resource materialization to have happened already.  It
    splits multi-query inputs so GPU jobs and seeds can be scheduled independently.
    """
    output_root = Path(output_root)
    safe_names = validate_prepared_query_names(query_set)

    outputs = {}
    for safe, original in safe_names.items():
        job_directory = output_root / safe
        prepared = InferenceQuerySet(
            seeds=list(query_set.seeds),
            queries={original: query_set.queries[original].model_copy(deep=True)},
        )
        data = query_set_to_portable_dict(prepared, job_directory)
        output_path = job_directory / f"{safe}_data.json"
        atomic_write_text(output_path, json.dumps(data, indent=2) + "\n")
        outputs[original] = output_path
    return outputs


def msa_array_to_a3m(msa: MsaArray, *, query_first: bool = True) -> str:
    """Serialize the MSA information consumed by OpenFold back to A3M.

    OpenFold features use insertion counts, not the identities of insertion
    residues.  A lowercase ``x`` therefore provides a lossless round trip for the
    alignment matrix and deletion matrix while keeping the hand-off human readable.
    """
    lines = []
    for row_index, (sequence, deletion_row) in enumerate(
        zip(msa.msa, msa.deletion_matrix, strict=True)
    ):
        header = "query" if query_first and row_index == 0 else f"row_{row_index}"
        encoded = "".join(
            f"{'x' * int(n_insertions)}{residue}"
            for residue, n_insertions in zip(sequence, deletion_row, strict=True)
        )
        lines.extend((f">{header}", encoded))
    return "\n".join(lines) + "\n"


def _deduplicated_main_msa(
    collection: MsaArrayCollection, rep_id: str, config: MSASettings
) -> MsaArray:
    """Build the native pre-subsampling main pool for one representative."""
    source_map = collection.rep_id_to_main_msa[rep_id]
    arrays = [source_map[key] for key in config.aln_order if key in source_map]
    if not arrays:
        query = collection.rep_id_to_query_seq[rep_id]
        return MsaArray(
            msa=query.copy(),
            deletion_matrix=np.zeros(query.shape, dtype=int),
            metadata=pd.DataFrame(),
        )

    msa = np.concatenate([array.msa for array in arrays], axis=0)
    deletion = np.concatenate([array.deletion_matrix for array in arrays], axis=0)
    view = msa.view(np.dtype((np.void, msa.dtype.itemsize * msa.shape[1])))
    _, unique_indices = np.unique(view, return_index=True)
    unique_indices.sort()
    return MsaArray(
        msa=msa[unique_indices],
        deletion_matrix=deletion[unique_indices],
        metadata=pd.DataFrame(),
    )


def _create_final_paired_msa(
    collection: MsaArrayCollection, config: MSASettings
) -> dict[str, MsaArray]:
    if collection.rep_id_to_paired_msa:
        return create_paired_from_precomputed(
            msa_array_collection=collection,
            max_rows_paired=config.max_rows_paired,
            paired_msa_order=config.paired_msa_order,
        )
    return create_paired(
        msa_array_collection=collection,
        max_rows_paired=config.max_rows_paired,
        min_chains_paired_partial=config.min_chains_paired_partial,
        pairing_mask_keys=config.pairing_mask_keys,
        max_seq_per_species=config.max_seq_per_species,
        msas_to_pair=config.msas_to_pair,
    )


def configure_af3_msa_sources(config: MSASettings) -> None:
    """Make explicit public channels readable even with a native source whitelist.

    Preserve all native source quotas/order and downstream row budgets. Canonical
    main input is the finalized profile pool, not another raw search database.
    """
    defaults = MSASettings()
    for key in ("colabfold_main", "colabfold_paired"):
        config.max_seq_counts.setdefault(key, defaults.max_seq_counts[key])
    if "colabfold_main" not in config.aln_order:
        config.aln_order.append("colabfold_main")
    if "colabfold_paired" not in config.paired_msa_order:
        config.paired_msa_order.append("colabfold_paired")


def materialise_msas(
    query_set: InferenceQuerySet,
    output_root: Path,
    config: MSASettings,
    *,
    compress: bool = False,
) -> None:
    """Replace native MSA sources with canonical paired/unpaired bundle files."""
    for query_name, query in query_set.queries.items():
        if not query.use_msas:
            # A use flag disables features, not the user's declared conditions.
            # Public AF3 resources have already been staged privately by the reader.
            continue
        processor_input = (
            MsaSampleProcessorInputInference.create_from_inference_query_entry(query)
        )
        collection = MsaSampleParserInference(config=config)(input=processor_input)
        paired_by_chain = (
            _create_final_paired_msa(collection, config)
            if query.use_paired_msas
            else {}
        )
        job_directory = Path(output_root) / sanitise_job_name(query_name)
        msa_directory = job_directory / "msas"
        main_by_rep = {
            rep_id: _deduplicated_main_msa(collection, rep_id, config)
            for rep_id in collection.rep_id_to_main_msa
        }
        materialized_by_rep = {}

        for chain in query.chains:
            if chain.molecule_type not in config.moltypes:
                continue
            if chain.main_msa_file_paths == [] and not chain.paired_msa_file_paths:
                # Explicitly absent conditions have no native MSA representative.
                # Do not turn them into query-only features just by publishing.
                continue
            rep_id = collection.chain_id_to_rep_id.get(chain.chain_ids[0])
            if rep_id in materialized_by_rep:
                # Distinct entities may explicitly share both source channels.
                # Preserve that identity when replacing sources with bundle paths.
                representative = materialized_by_rep[rep_id]
                chain.main_msa_file_paths = representative.main_msa_file_paths
                if query.use_paired_msas or chain.molecule_type != MoleculeType.PROTEIN:
                    chain.paired_msa_file_paths = representative.paired_msa_file_paths
                continue
            if rep_id is not None:
                materialized_by_rep[rep_id] = chain
            main_msa = main_by_rep.get(rep_id)
            if main_msa is None:
                query_row = np.asarray([list(chain.sequence)], dtype="<U1")
                main_msa = MsaArray(
                    msa=query_row,
                    deletion_matrix=np.zeros(query_row.shape, dtype=int),
                    metadata=pd.DataFrame(),
                )
            entity_id = sanitise_job_name(str(chain.chain_ids[0]))
            main_suffix = ".a3m.zst" if compress else ".a3m"
            main_path = msa_directory / (
                f"{sanitise_job_name(query_name)}__{entity_id}_unpairedmsa{main_suffix}"
            )
            main_text = msa_array_to_a3m(main_msa)
            if compress:
                write_zstd_text(main_path, main_text)
            else:
                atomic_write_text(main_path, main_text)
            chain.main_msa_file_paths = [main_path]

            if chain.molecule_type != MoleculeType.PROTEIN:
                # RNA gap rows only align internal matrix depth; the native
                # consumer reconstructs them from the protein paired channels.
                chain.paired_msa_file_paths = []
                continue
            if not query.use_paired_msas:
                # Do not erase a declared paired condition when only its use is off.
                continue
            representative_chain = collection.rep_id_to_chain_id.get(rep_id)
            paired = paired_by_chain.get(representative_chain)
            if paired is None or len(paired) == 0:
                chain.paired_msa_file_paths = []
                continue
            paired_path = msa_directory / (
                f"{sanitise_job_name(query_name)}__{entity_id}_pairedmsa{main_suffix}"
            )
            # Preserve native paired rows, including an ordinary query row if
            # present. A local first hit is not necessarily the query.
            paired_text = msa_array_to_a3m(paired, query_first=False)
            if compress:
                write_zstd_text(paired_path, paired_text)
            else:
                atomic_write_text(paired_path, paired_text)
            chain.paired_msa_file_paths = [paired_path]


def _normalise_release_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def extract_single_chain_mmcif(source_path: Path, chain_id: str) -> str:
    """Keep one chain and its entity metadata, without renumbering SEQRES."""
    cif_file = pdbx.CIFFile.read(StringIO(read_text_auto(source_path)))
    cif_file = copy.deepcopy(cif_file)
    atom_site = cif_file.block["atom_site"]
    chain_mask = atom_site["label_asym_id"].as_array() == chain_id
    if not np.any(chain_mask):
        available = sorted(set(atom_site["label_asym_id"].as_array().tolist()))
        raise ValueError(
            f"Template chain {chain_id!r} is absent from {source_path}; "
            f"available label_asym_id values: {available}"
        )

    cif_file.block["atom_site"] = pdbx.CIFCategory(
        {name: column.as_array()[chain_mask] for name, column in atom_site.items()}
    )
    block = cif_file.block

    def subset(category_name, column_name, values):
        if category_name not in block or column_name not in block[category_name]:
            return
        category = block[category_name]
        mask = np.isin(category[column_name].as_array(), list(values))
        if np.any(mask):
            block[category_name] = pdbx.CIFCategory(
                {name: column.as_array()[mask] for name, column in category.items()}
            )
        else:
            del block[category_name]

    if "struct_asym" in block:
        asym = block["struct_asym"]
        entities = set(asym["entity_id"].as_array()[asym["id"].as_array() == chain_id])
        subset("struct_asym", "id", {chain_id})
        subset("entity", "id", entities)
        for category in ("entity_poly", "entity_poly_seq", "pdbx_entity_nonpoly"):
            subset(category, "entity_id", entities)
    for category in (
        "pdbx_poly_seq_scheme",
        "pdbx_nonpoly_scheme",
        "pdbx_branch_scheme",
    ):
        subset(category, "asym_id", {chain_id})
    if "entity_poly" in block and "pdbx_strand_id" in block["entity_poly"]:
        # This column uses author chain IDs, not label_asym_id.
        author_ids = sorted(set(atom_site["auth_asym_id"].as_array()[chain_mask]))
        block["entity_poly"]["pdbx_strand_id"] = pdbx.CIFColumn(
            [",".join(author_ids)] * block["entity_poly"].row_count
        )
    output = StringIO()
    cif_file.write(output)
    return output.getvalue()


def materialise_templates(
    query_set: InferenceQuerySet,
    output_root: Path,
    config: TemplatePreprocessorSettings,
    *,
    compress: bool = False,
) -> None:
    """Embed finalized templates and export one single-chain mmCIF per template."""
    for query_name, query in query_set.queries.items():
        job_directory = Path(output_root) / sanitise_job_name(query_name)
        template_directory = job_directory / "msas"
        for chain in query.chains:
            # Explicit prepared templates do not need native search/realignment.
            # In particular, repeated data must not mistake them for no hits.
            if chain.templates is not None:
                for index, template in enumerate(chain.templates):
                    text = read_text_auto(template.mmcif_path)
                    block = pdbx.CIFFile.read(StringIO(text)).block
                    chains = np.unique(block["atom_site"]["label_asym_id"].as_array())
                    if len(chains) != 1:
                        raise ValueError(
                            "Explicit prepared templates must be single-chain"
                        )
                    text = extract_single_chain_mmcif(
                        template.mmcif_path, str(chains[0])
                    )
                    suffix = ".cif.zst" if compress else ".cif"
                    path = template_directory / (
                        f"{sanitise_job_name(query_name)}__"
                        f"{sanitise_job_name(chain.chain_ids[0])}_template_{index}{suffix}"
                    )
                    if compress:
                        write_zstd_text(path, text)
                    else:
                        atomic_write_text(path, text)
                    template.mmcif_path = path
                continue
            cache_path = chain.template_alignment_file_path
            template_ids = chain.template_entry_chain_ids or []
            if cache_path is None or not template_ids:
                chain.template_alignment_file_path = None
                chain.template_entry_chain_ids = []
                chain.template_cif_paths = None
                chain.template_cif_chain_ids = None
                chain.templates = []
                chain.prepared_template_file_path = None
                continue

            with np.load(cache_path, allow_pickle=True) as cache_npz:
                cache = {key: value.item() for key, value in cache_npz.items()}

            entity_id = sanitise_job_name(str(chain.chain_ids[0]))
            prepared_templates = []
            cif_paths_by_template: dict[tuple[str, str], Path] = {}
            for template_index, template_id in enumerate(template_ids):
                if template_id not in cache:
                    raise ValueError(
                        f"Template {template_id!r} is absent from cache {cache_path}"
                    )
                entry_id, chain_id = template_id.rsplit("_", 1)
                entry = cache[template_id]
                source_path = entry.get("cif_path")
                if source_path is None:
                    if config.structure_file_format != "cif":
                        raise ValueError(
                            "Portable template finalization requires the original "
                            f"mmCIF for {template_id}; the configured template store "
                            f"contains only {config.structure_file_format!r} data."
                        )
                    source_path = (
                        Path(config.structure_directory)
                        / f"{entry_id}.{config.structure_file_format}"
                    )
                source_path = Path(source_path)
                if not source_path.exists():
                    raise FileNotFoundError(
                        f"Structure for template {template_id} not found: {source_path}"
                    )

                template_key = (entry_id, chain_id)
                cif_path = cif_paths_by_template.get(template_key)
                if cif_path is None:
                    suffix = ".cif.zst" if compress else ".cif"
                    cif_path = template_directory / (
                        f"{sanitise_job_name(query_name)}__{entity_id}_"
                        f"template_{template_index}{suffix}"
                    )
                    cif_text = extract_single_chain_mmcif(source_path, chain_id)
                    if compress:
                        write_zstd_text(cif_path, cif_text)
                    else:
                        atomic_write_text(cif_path, cif_text)
                    cif_paths_by_template[template_key] = cif_path

                idx_map = np.asarray(entry["idx_map"], dtype=int)
                prepared_templates.append(
                    PreparedTemplate(
                        entry_id=entry_id,
                        mmcif_path=cif_path,
                        query_indices=idx_map[:, 0].tolist(),
                        template_indices=idx_map[:, 1].tolist(),
                        release_date=_normalise_release_date(entry.get("release_date")),
                        source_index=int(entry["index"]),
                    )
                )

            chain.templates = prepared_templates
            chain.prepared_template_file_path = None
            chain.template_alignment_file_path = None
            chain.template_entry_chain_ids = []
            chain.template_cif_paths = None
            chain.template_cif_chain_ids = None


def clear_template_inputs(query_set: InferenceQuerySet) -> None:
    """Remove every raw or prepared template reference from a query set."""
    for query in query_set.queries.values():
        for chain in query.chains:
            chain.template_alignment_file_path = None
            chain.template_entry_chain_ids = []
            chain.template_cif_paths = None
            chain.template_cif_chain_ids = None
            chain.templates = None
            chain.prepared_template_file_path = None


def _native_safe_template_chain(cif_text: str, chain_id: str) -> tuple[str, str]:
    """Normalize only private label-asym identities unsafe in native id splitting.

    Author identifiers, residue numbers, coordinates and the public CIF stay
    unchanged. Update label-asym references together for native unresolved-residue
    reconstruction, not only the coordinate table.
    """
    if "_" not in chain_id:
        return cif_text, chain_id
    cif = pdbx.CIFFile.read(StringIO(cif_text))
    for category_name, category in cif.block.items():
        for column_name, column in list(category.items()):
            if (
                (category_name == "struct_asym" and column_name == "id")
                or column_name in {"asym_id", "label_asym_id"}
                or column_name.endswith("_label_asym_id")
            ):
                values = column.as_array().astype(object)
                values[values == chain_id] = "A"
                category[column_name] = pdbx.CIFColumn(values.tolist())
            elif column_name == "asym_id_list":
                category[column_name] = pdbx.CIFColumn(
                    [
                        ",".join(
                            "A" if part == chain_id else part
                            for part in value.split(",")
                        )
                        for value in column.as_array()
                    ]
                )
    stream = StringIO()
    cif.write(stream)
    return stream.getvalue(), "A"


def restore_prepared_templates(
    query_set: InferenceQuerySet,
    runtime_directory: Path,
    config: TemplatePreprocessorSettings,
) -> None:
    """Rebuild native cache NPZ and plain CIF resources in a private directory."""
    runtime_directory = Path(runtime_directory)
    structure_directory = runtime_directory / "template_structures"
    cache_directory = runtime_directory / "template_cache"
    structure_directory.mkdir(parents=True, exist_ok=True)
    cache_directory.mkdir(parents=True, exist_ok=True)

    for query_name, query in query_set.queries.items():
        for chain_index, chain in enumerate(query.chains):
            templates = chain.templates
            if templates is None and chain.prepared_template_file_path is not None:
                templates = PreparedTemplateSet.from_json(
                    chain.prepared_template_file_path
                ).templates
            if not templates:
                continue
            cache = {}
            template_ids = []
            identities = {}
            for template_index, template in enumerate(templates):
                cif_text = read_text_auto(template.mmcif_path)
                template_chain_id = template.chain_id
                if template_chain_id is None:
                    cif_file = pdbx.CIFFile.read(StringIO(cif_text))
                    chain_ids = np.unique(
                        cif_file.block["atom_site"]["label_asym_id"].as_array()
                    ).tolist()
                    if len(chain_ids) != 1:
                        raise ValueError(
                            f"Prepared template {template.mmcif_path} must contain "
                            f"exactly one label_asym_id; found {chain_ids}"
                        )
                    template_chain_id = str(chain_ids[0])
                signature = (
                    template.entry_id,
                    template_chain_id,
                    cif_text,
                    tuple(template.query_indices),
                    tuple(template.template_indices),
                    template.source_index,
                    template.release_date,
                )
                if signature in identities:
                    # Native repeated identical hits share one cache entry and
                    # collapse in sample_templates' dict, while retaining list order.
                    template_ids.append(identities[signature])
                    continue
                cif_text, template_chain_id = _native_safe_template_chain(
                    cif_text, template_chain_id
                )
                native_entry = template.entry_id
                if "_" in native_entry:
                    native_entry = f"ef{template_index}"
                template_id = f"{native_entry}_{template_chain_id}"
                while template_id in cache:
                    native_entry = "ef" + native_entry
                    template_id = f"{native_entry}_{template_chain_id}"
                identities[signature] = template_id
                structure_path = structure_directory / (
                    f"{sanitise_job_name(query_name)}_{chain_index}_{template_index}.cif"
                )
                atomic_write_text(structure_path, cif_text)

                cache[template_id] = {
                    "index": template.source_index,
                    "release_date": (
                        template.release_date.isoformat()
                        if template.release_date is not None
                        else "1900-01-01"
                    ),
                    "idx_map": np.column_stack(
                        [template.query_indices, template.template_indices]
                    ).astype(int),
                    "cif_path": str(structure_path),
                }
                template_ids.append(template_id)

            cache_path = cache_directory / (
                f"{sanitise_job_name(query_name)}_{chain_index}.npz"
            )
            np.savez_compressed(cache_path, **cache)
            chain.template_alignment_file_path = cache_path
            chain.template_entry_chain_ids = template_ids
            chain.templates = None
            chain.prepared_template_file_path = None

    config.output_directory = runtime_directory
    config.structure_directory = structure_directory
    config.structure_file_format = "cif"
    config.cache_directory = cache_directory
    config.structure_array_directory = None


def validate_inference_only_templates(query_set: InferenceQuerySet) -> None:
    """Reject raw template inputs that require the disabled data pipeline."""
    for query_name, query in query_set.queries.items():
        for chain in query.chains:
            if (
                chain.templates is not None
                or chain.prepared_template_file_path is not None
            ):
                continue
            if (
                chain.template_alignment_file_path is not None
                or chain.template_cif_paths is not None
            ):
                raise ValueError(
                    f"Query {query_name!r}, chain {chain.chain_ids} contains raw "
                    "template inputs. Run once with --run-data-pipeline=true to "
                    "create portable inline templates."
                )


class PreparedTemplateSet(BaseModel):
    version: int = 1
    templates: list[PreparedTemplate]

    @classmethod
    def from_json(cls, path: Path) -> PreparedTemplateSet:
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        for template in data.get("templates", []):
            mmcif_path = Path(template["mmcif_path"]).expanduser()
            if not mmcif_path.is_absolute():
                mmcif_path = path.parent / mmcif_path
            template["mmcif_path"] = str(mmcif_path.resolve())
        return cls.model_validate(data)

    def write_json(self, path: Path) -> None:
        path = Path(path)
        data = self.model_dump(mode="json", exclude_none=True)
        for template in data["templates"]:
            template["mmcif_path"] = os.path.relpath(
                template["mmcif_path"], path.parent
            )
        atomic_write_text(path, json.dumps(data, indent=2) + "\n")
