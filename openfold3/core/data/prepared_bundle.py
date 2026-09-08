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

import json
import os
import re
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, FilePath, field_validator, model_validator

from openfold3.core.config.msa_pipeline_configs import (
    MsaSampleProcessorInputInference,
)
from openfold3.core.data.io.compression import read_text_auto, write_zstd_text
from openfold3.core.data.io.sequence.msa import MsaSampleParserInference
from openfold3.core.data.pipelines.sample_processing.msa import (
    create_paired,
    create_paired_from_precomputed,
)
from openfold3.core.data.primitives.sequence.hash import get_sequence_hash
from openfold3.core.data.primitives.sequence.msa import MsaArray, MsaArrayCollection
from openfold3.projects.of3_all_atom.config.dataset_config_components import (
    MSASettings,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
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
    return data


def write_prepared_query_sets(
    query_set: InferenceQuerySet, output_root: Path
) -> dict[str, Path]:
    """Write one portable ``<name>_data.json`` for every query.

    The function expects resource materialization to have happened already.  It
    splits multi-query inputs so GPU jobs and seeds can be scheduled independently.
    """
    output_root = Path(output_root)
    safe_names: dict[str, str] = {}
    for name in query_set.queries:
        safe = sanitise_job_name(name)
        previous = safe_names.get(safe)
        if previous is not None and previous != name:
            raise ValueError(
                f"Query names {previous!r} and {name!r} both map to {safe!r}"
            )
        safe_names[safe] = name

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


def msa_array_to_a3m(msa: MsaArray) -> str:
    """Serialize the MSA information consumed by OpenFold back to A3M.

    OpenFold features use insertion counts, not the identities of insertion
    residues.  A lowercase ``x`` therefore provides a lossless round trip for the
    alignment matrix and deletion matrix while keeping the hand-off human readable.
    """
    lines = []
    for row_index, (sequence, deletion_row) in enumerate(
        zip(msa.msa, msa.deletion_matrix, strict=True)
    ):
        header = "query" if row_index == 0 else f"row_{row_index}"
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


def materialise_msas(
    query_set: InferenceQuerySet,
    output_root: Path,
    config: MSASettings,
    *,
    compress: bool = True,
) -> None:
    """Replace native MSA sources with canonical paired/unpaired bundle files."""
    for query_name, query in query_set.queries.items():
        if not query.use_msas:
            for chain in query.chains:
                chain.main_msa_file_paths = []
                chain.paired_msa_file_paths = []
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

        for chain in query.chains:
            if chain.molecule_type not in config.moltypes:
                continue
            rep_id = get_sequence_hash(chain.sequence)
            if rep_id not in main_by_rep:
                query_row = np.asarray([list(chain.sequence)], dtype="<U1")
                main_by_rep[rep_id] = MsaArray(
                    msa=query_row,
                    deletion_matrix=np.zeros(query_row.shape, dtype=int),
                    metadata=pd.DataFrame(),
                )
            entity_id = sanitise_job_name(str(chain.chain_ids[0]))
            main_suffix = ".a3m.zst" if compress else ".a3m"
            main_path = msa_directory / (
                f"{sanitise_job_name(query_name)}__{entity_id}_unpairedmsa{main_suffix}"
            )
            main_text = msa_array_to_a3m(main_by_rep[rep_id])
            if compress:
                write_zstd_text(main_path, main_text)
            else:
                atomic_write_text(main_path, main_text)
            chain.main_msa_file_paths = [main_path]

            representative_chain = collection.rep_id_to_chain_id.get(rep_id)
            paired = paired_by_chain.get(representative_chain)
            if paired is None or len(paired) == 0:
                chain.paired_msa_file_paths = []
                continue
            paired_path = msa_directory / (
                f"{sanitise_job_name(query_name)}__{entity_id}_pairedmsa{main_suffix}"
            )
            paired_text = msa_array_to_a3m(paired)
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


def materialise_templates(
    query_set: InferenceQuerySet,
    output_root: Path,
    config: TemplatePreprocessorSettings,
    *,
    compress: bool = True,
) -> None:
    """Export native template caches as editable sidecars plus full mmCIF files."""
    for query_name, query in query_set.queries.items():
        job_directory = Path(output_root) / sanitise_job_name(query_name)
        msa_directory = job_directory / "msas"
        for chain in query.chains:
            cache_path = chain.template_alignment_file_path
            template_ids = chain.template_entry_chain_ids or []
            if cache_path is None or not template_ids:
                chain.template_alignment_file_path = None
                chain.template_entry_chain_ids = []
                chain.template_cif_paths = None
                chain.template_cif_chain_ids = None
                chain.prepared_template_file_path = None
                continue

            with np.load(cache_path, allow_pickle=True) as cache_npz:
                cache = {key: value.item() for key, value in cache_npz.items()}

            entity_id = sanitise_job_name(str(chain.chain_ids[0]))
            prepared_templates = []
            for output_index, template_id in enumerate(template_ids):
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

                suffix = ".cif.zst" if compress else ".cif"
                cif_path = msa_directory / (
                    f"{sanitise_job_name(query_name)}__{entity_id}_template_"
                    f"{output_index}{suffix}"
                )
                cif_text = read_text_auto(source_path)
                if compress:
                    write_zstd_text(cif_path, cif_text)
                else:
                    atomic_write_text(cif_path, cif_text)

                idx_map = np.asarray(entry["idx_map"], dtype=int)
                prepared_templates.append(
                    PreparedTemplate(
                        entry_id=entry_id,
                        chain_id=chain_id,
                        mmcif_path=cif_path,
                        query_indices=idx_map[:, 0].tolist(),
                        template_indices=idx_map[:, 1].tolist(),
                        release_date=_normalise_release_date(entry.get("release_date")),
                        source_index=int(entry["index"]),
                    )
                )

            sidecar_path = msa_directory / (
                f"{sanitise_job_name(query_name)}__{entity_id}_templates.json"
            )
            PreparedTemplateSet(templates=prepared_templates).write_json(sidecar_path)
            chain.prepared_template_file_path = sidecar_path
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
            chain.prepared_template_file_path = None


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

    restored_structures: dict[str, str] = {}
    for query_name, query in query_set.queries.items():
        for chain_index, chain in enumerate(query.chains):
            sidecar_path = chain.prepared_template_file_path
            if sidecar_path is None:
                continue
            template_set = PreparedTemplateSet.from_json(sidecar_path)
            cache = {}
            template_ids = []
            for template in template_set.templates:
                template_id = f"{template.entry_id}_{template.chain_id}"
                cif_text = read_text_auto(template.mmcif_path)
                previous = restored_structures.get(template.entry_id)
                if previous is not None and previous != cif_text:
                    raise ValueError(
                        f"Conflicting mmCIF contents for template entry "
                        f"{template.entry_id!r}"
                    )
                restored_structures[template.entry_id] = cif_text
                structure_path = structure_directory / f"{template.entry_id}.cif"
                if not structure_path.exists():
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
                }
                template_ids.append(template_id)

            cache_path = cache_directory / (
                f"{sanitise_job_name(query_name)}_{chain_index}.npz"
            )
            np.savez_compressed(cache_path, **cache)
            chain.template_alignment_file_path = cache_path
            chain.template_entry_chain_ids = template_ids
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
            if chain.prepared_template_file_path is not None:
                continue
            if (
                chain.template_alignment_file_path is not None
                or chain.template_cif_paths is not None
            ):
                raise ValueError(
                    f"Query {query_name!r}, chain {chain.chain_ids} contains raw "
                    "template inputs. Run once with --run-data-pipeline=true to "
                    "create a portable prepared_template_file_path sidecar."
                )


class PreparedTemplate(BaseModel):
    """One finalized template used to reconstruct an OpenFold template cache."""

    entry_id: str
    chain_id: str
    mmcif_path: FilePath
    query_indices: list[int]
    template_indices: list[int]
    release_date: date | None = None
    source_index: int

    @field_validator("query_indices", "template_indices")
    @classmethod
    def validate_nonnegative_indices(cls, value: list[int]) -> list[int]:
        if any(index < 0 for index in value):
            raise ValueError("Template residue indices must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_mapping(self) -> PreparedTemplate:
        if len(self.query_indices) != len(self.template_indices):
            raise ValueError(
                "query_indices and template_indices must have equal length"
            )
        if not self.query_indices:
            raise ValueError("A prepared template must map at least one residue")
        return self


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
