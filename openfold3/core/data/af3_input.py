"""AF3-style public JSON boundary for the EnsembleFold OpenFold3 wrapper.

The native query and cache models remain internal. In particular template cache
indices are one-based; only this boundary translates to/from public zero-based
indices. No searches, alignment, pairing or sampling are performed here.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from io import StringIO
from pathlib import Path

import numpy as np
import zstandard
from biotite.structure.io import pdbx

from openfold3.core.data.io.compression import read_text_auto
from openfold3.core.data.io.sequence.msa import parse_a3m, reject_legacy_paired_context
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)


def _keys(value, allowed, context):
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"Unsupported {context} fields: {sorted(unknown)}")


def _indices(value, bound, field):
    if (
        not isinstance(value, list)
        or not value
        or any(type(x) is not int or x < 0 or x >= bound for x in value)
        or len(set(value)) != len(value)
        or value != sorted(value)
    ):
        raise ValueError(
            f"{field} must contain increasing, unique zero-based "
            f"integer indices below {bound}"
        )
    return [x + 1 for x in value]


def _text(body, field, base):
    inline, path = body.get(field), body.get(field + "Path")
    if inline is not None and path is not None:
        raise ValueError(f"Cannot supply both {field} and {field}Path")
    if path is not None:
        if not isinstance(path, str) or not path:
            raise ValueError(f"{field}Path must be a nonempty file path")
        source = Path(path).expanduser()
        return read_text_auto(source if source.is_absolute() else base / source)
    if inline is not None and not isinstance(inline, str):
        raise ValueError(f"{field} must be a string")
    return inline


def _put(path, text):
    from openfold3.core.data.prepared_bundle import atomic_write_text

    atomic_write_text(path, text)
    return path


def _template(body, sequence, base, directory, index):
    _keys(
        body,
        {"mmcif", "mmcifPath", "queryIndices", "templateIndices", "openfold3"},
        "template",
    )
    text = _text(body, "mmcif", base)
    if not text:
        raise ValueError("Template requires nonempty mmcif or mmcifPath")
    block = pdbx.CIFFile.read(StringIO(text)).block
    chains = np.unique(block["atom_site"]["label_asym_id"].as_array()).tolist()
    if len(chains) != 1:
        raise ValueError(f"Template mmCIF must be single-chain; found {chains}")
    asym = block["struct_asym"]
    entity_ids = asym["entity_id"].as_array()[asym["id"].as_array() == chains[0]]
    if len(entity_ids) != 1 or "entity_poly_seq" not in block:
        raise ValueError("Template requires full polymer sequence metadata")
    seq = block["entity_poly_seq"]
    numbers = seq["num"].as_array(int)[seq["entity_id"].as_array() == entity_ids[0]]
    if not len(numbers) or set(numbers) != set(range(1, int(max(numbers)) + 1)):
        raise ValueError(
            "Template polymer residue numbering must be contiguous and one-based"
        )
    query_indices = _indices(body.get("queryIndices"), len(sequence), "queryIndices")
    template_indices = _indices(
        body.get("templateIndices"), int(max(numbers)), "templateIndices"
    )
    if len(query_indices) != len(template_indices):
        raise ValueError("queryIndices and templateIndices must have equal length")
    metadata = body.get("openfold3", {})
    _keys(metadata, {"entryId", "releaseDate", "sourceIndex"}, "template.openfold3")
    source_index = metadata.get("sourceIndex", index)
    if type(source_index) is not int or source_index < 0:
        raise ValueError("sourceIndex must be a nonnegative integer")
    entry_id = metadata.get("entryId", f"template{index}")
    if not isinstance(entry_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", entry_id):
        raise ValueError("entryId must be a safe identifier")
    return dict(
        entry_id=entry_id,
        source_index=source_index,
        release_date=metadata.get("releaseDate"),
        mmcif_path=_put(directory / f"template_{index}.cif", text),
        query_indices=query_indices,
        template_indices=template_indices,
    )


def load_af3_query_set(json_path: Path, resource_directory: Path) -> InferenceQuerySet:
    """Read AF3-style JSON, resolve paths against it, and stage native resources.

    A batch may contain a list of AF3 jobs with the same modelSeeds. The caller owns
    resource_directory and must keep it alive until data/inference completes.
    """
    from openfold3.core.data.prepared_bundle import validate_prepared_query_names

    json_path = Path(json_path).resolve()
    data = json.loads(json_path.read_text())
    jobs = data if isinstance(data, list) else [data]
    if not jobs:
        raise ValueError("AF3 input must contain at least one job")
    queries, seeds = {}, None
    for job_index, job in enumerate(jobs):
        if not isinstance(job, dict) or "queries" in job or "sequences" not in job:
            raise ValueError(
                "Expected AF3-style name/modelSeeds/sequences JSON, "
                "not native queries/chains"
            )
        _keys(
            job,
            {"name", "modelSeeds", "sequences", "dialect", "version", "openfold3"},
            "AF3 job",
        )
        if job.get("dialect", "alphafold3") != "alphafold3":
            raise ValueError("dialect must be alphafold3")
        if type(job.get("version", 4)) is not int or job.get("version", 4) not in {
            1,
            2,
            3,
            4,
        }:
            raise ValueError("Unsupported AF3 version")
        name = job.get("name")
        if not isinstance(name, str) or not name.strip() or name in queries:
            raise ValueError("AF3 name must be nonempty and unique")
        job_seeds = job.get("modelSeeds", [42])
        if (
            not isinstance(job_seeds, list)
            or not job_seeds
            or any(type(s) is not int or not 0 <= s < 2**32 for s in job_seeds)
            or len(set(job_seeds)) != len(job_seeds)
        ):
            raise ValueError("modelSeeds must be unique uint32 integers")
        if seeds is not None and job_seeds != seeds:
            raise ValueError(
                "AF3 batch jobs must share modelSeeds; "
                "run different seed lists separately"
            )
        seeds = job_seeds
        sequences = job["sequences"]
        if not isinstance(sequences, list) or not sequences:
            raise ValueError("sequences must be a nonempty list")
        chains, used_ids = [], set()
        paired_states, paired_depths = {}, {}
        for chain_index, entity in enumerate(sequences):
            _keys(entity, {"protein", "rna", "dna", "ligand"}, "sequence entity")
            if len(entity) != 1:
                raise ValueError(
                    "Each sequence entity must have exactly one molecule type"
                )
            kind, body = next(iter(entity.items()))
            allowed = {"id", "description"}
            if kind == "ligand":
                allowed |= {"ccdCodes", "smiles"}
            else:
                allowed |= {"sequence", "modifications", "cyclic"}
                if kind in {"protein", "rna"}:
                    allowed |= {"unpairedMsa", "unpairedMsaPath"}
                if kind == "protein":
                    allowed |= {"pairedMsa", "pairedMsaPath", "templates"}
            _keys(body, allowed, kind)
            ids = body.get("id")
            ids = [ids] if isinstance(ids, str) else ids
            if (
                not isinstance(ids, list)
                or not ids
                or any(
                    not isinstance(i, str) or not re.fullmatch(r"[A-Za-z0-9]+", i)
                    for i in ids
                )
                or len(set(ids)) != len(ids)
                or used_ids.intersection(ids)
            ):
                raise ValueError(
                    "Sequence id values must be nonempty, alphanumeric and unique"
                )
            used_ids.update(ids)
            chain = {"molecule_type": kind, "chain_ids": ids}
            if "description" in body:
                chain["description"] = body["description"]
            directory = Path(resource_directory) / str(job_index) / str(chain_index)
            if kind == "ligand":
                if (body.get("ccdCodes") is None) == (body.get("smiles") is None):
                    raise ValueError(
                        "Ligand requires exactly one of ccdCodes or smiles"
                    )
                if "ccdCodes" in body:
                    codes = body["ccdCodes"]
                    if (
                        not isinstance(codes, list)
                        or not codes
                        or any(not isinstance(c, str) or not c for c in codes)
                    ):
                        raise ValueError("ccdCodes must be a nonempty list")
                    chain["ccd_codes"] = codes
                else:
                    if not isinstance(body["smiles"], str) or not body["smiles"]:
                        raise ValueError("smiles must be nonempty")
                    chain["smiles"] = body["smiles"]
            else:
                sequence = body.get("sequence")
                if (
                    not isinstance(sequence, str)
                    or not sequence
                    or not re.fullmatch(r"[A-Z]+", sequence)
                ):
                    raise ValueError("sequence must contain uppercase residue letters")
                chain["sequence"] = sequence
                if "cyclic" in body:
                    if type(body["cyclic"]) is not bool:
                        raise ValueError("cyclic must be boolean")
                    chain["cyclic"] = body["cyclic"]
                modifications = {}
                for mod in body.get("modifications", []):
                    position, code = (
                        ("ptmPosition", "ptmType")
                        if kind == "protein"
                        else ("basePosition", "modificationType")
                    )
                    _keys(mod, {position, code}, "modification")
                    idx = mod.get(position)
                    if (
                        type(idx) is not int
                        or not 1 <= idx <= len(sequence)
                        or idx in modifications
                    ):
                        raise ValueError(
                            "Modification positions must be unique "
                            "one-based residue indices"
                        )
                    if not isinstance(mod.get(code), str) or not mod[code]:
                        raise ValueError("Modification requires a CCD code")
                    modifications[idx] = mod[code]
                if modifications:
                    chain["non_canonical_residues"] = modifications
                for channel, native in (
                    ("unpairedMsa", "main_msa_file_paths"),
                    ("pairedMsa", "paired_msa_file_paths"),
                ):
                    text = _text(body, channel, json_path.parent)
                    if kind == "protein" and channel == "pairedMsa":
                        paired_states[ids[0]] = (
                            "automatic" if text is None else
                            "empty" if text == "" else "provided"
                        )
                    if text is not None:
                        if text == "":
                            if channel == "pairedMsa":
                                chain[native] = []
                                continue
                            text = f">query\n{sequence}\n"
                        reject_legacy_paired_context(text)
                        alignment = parse_a3m(text)
                        if (
                            alignment.msa.shape[1] != len(sequence)
                            or (channel == "unpairedMsa"
                                and "".join(alignment.msa[0]) != sequence)
                        ):
                            raise ValueError(
                                f"{channel} query/width does not match sequence"
                            )
                        if not np.isin(alignment.msa, list("ABCDEFGHIJKLMNOPQRSTUVWXYZ-")).all():
                            raise ValueError(f"{channel} contains invalid alignment characters")
                        if kind == "protein" and channel == "pairedMsa":
                            paired_depths[ids[0]] = len(alignment.msa)
                        filename = (
                            "input_unpairedmsa.a3m"
                            if channel == "unpairedMsa"
                            else "input_pairedmsa.a3m"
                        )
                        chain[native] = [_put(directory / filename, text)]
                if body.get("templates") is not None:
                    if not isinstance(body["templates"], list):
                        raise ValueError("templates must be a list")
                    chain["templates"] = [
                        _template(t, sequence, json_path.parent, directory, i)
                        for i, t in enumerate(body["templates"])
                    ]
            chains.append(chain)
        extensions = job.get("openfold3", {})
        _keys(
            extensions,
            {
                "use_msas",
                "use_main_msas",
                "use_paired_msas",
                "pocket_constraint",
                "covalent_bonds",
            },
            "openfold3",
        )
        if len(set(paired_states.values())) > 1:
            raise ValueError(
                "Protein paired MSA states must match across the complex: "
                "provide all, leave all empty, or leave all unspecified. "
                f"Got {paired_states}"
            )
        if len(set(paired_depths.values())) > 1:
            raise ValueError(f"Protein paired MSA row depths must match: {paired_depths}")
        if paired_states and set(paired_states.values()) == {"empty"}:
            # Explicit absence must not fall back to native species pairing.
            extensions = {**extensions, "use_paired_msas": False}
        queries[name] = {"chains": chains, **extensions}
    result = InferenceQuerySet(seeds=seeds, queries=queries)
    validate_prepared_query_names(result)
    for name, query in result.queries.items():
        if not query.use_main_msas:
            raise ValueError(
                f"Query {name!r}: use_main_msas=false is unsupported because the "
                "pinned native branch omits required MSA statistics. Leave "
                'use_main_msas true (or omitted) and set unpairedMsa to "" '
                "for query-only main input; supplied paired MSA is retained."
            )
    return result


def _atomic_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_af3_query_sets(query_set, output_root, *, compress=True):
    """Publish self-contained per-job snapshots; read all sources before writing.

    On a caught filesystem failure restore overwritten files. This is not a
    multi-file power-loss transaction. Unrelated inference output is untouched.
    """
    from openfold3.core.data.prepared_bundle import validate_prepared_query_names

    names = validate_prepared_query_names(query_set)
    files, outputs = {}, {}
    for safe, name in names.items():
        query = query_set.queries[name]
        job_dir = Path(output_root) / safe
        sequences = []

        def resource(path, stem, suffix, safe=safe, job_dir=job_dir):
            text = read_text_auto(path)
            relative = f"msas/{safe}__{stem}{suffix}" + (".zst" if compress else "")
            raw = text.encode()
            files[job_dir / relative] = (
                zstandard.ZstdCompressor(level=3).compress(raw) if compress else raw
            )
            return relative

        for chain in query.chains:
            kind = chain.molecule_type.name.lower()
            body = {
                "id": chain.chain_ids[0]
                if len(chain.chain_ids) == 1
                else chain.chain_ids
            }
            if chain.description is not None:
                body["description"] = chain.description
            if kind == "ligand":
                body.update(
                    {"ccdCodes": chain.ccd_codes}
                    if chain.ccd_codes is not None
                    else {"smiles": chain.smiles}
                )
            else:
                body["sequence"] = chain.sequence
                if chain.cyclic:
                    body["cyclic"] = True
                if chain.non_canonical_residues:
                    position, code = (
                        ("ptmPosition", "ptmType")
                        if kind == "protein"
                        else ("basePosition", "modificationType")
                    )
                    body["modifications"] = [
                        {code: c, position: p}
                        for p, c in chain.non_canonical_residues.items()
                    ]
                for public, native, suffix in (
                    ("unpairedMsa", "main_msa_file_paths", "unpairedmsa"),
                    ("pairedMsa", "paired_msa_file_paths", "pairedmsa"),
                ):
                    if kind not in {"protein", "rna"} or (
                        public == "pairedMsa" and kind != "protein"
                    ):
                        continue
                    paths = getattr(chain, native)
                    if paths is None:
                        continue
                    if not paths:
                        # RNA has only the public unpaired channel.
                        if kind == "protein" or public == "unpairedMsa":
                            body[public] = ""
                    elif len(paths) != 1:
                        raise ValueError(
                            "Finalize native MSA sources before AF3 publication"
                        )
                    else:
                        body[public + "Path"] = resource(
                            paths[0], f"{chain.chain_ids[0]}_{suffix}", ".a3m"
                        )
                if kind == "protein" and chain.templates is not None:
                    body["templates"] = []
                    for i, template in enumerate(chain.templates):
                        if any(
                            x < 1
                            for x in template.query_indices + template.template_indices
                        ):
                            raise ValueError(
                                "Internal template indices must be one-based"
                            )
                        body["templates"].append(
                            {
                                "mmcifPath": resource(
                                    template.mmcif_path,
                                    f"{chain.chain_ids[0]}_template_{i}",
                                    ".cif",
                                ),
                                "queryIndices": [x - 1 for x in template.query_indices],
                                "templateIndices": [
                                    x - 1 for x in template.template_indices
                                ],
                                "openfold3": {
                                    "entryId": template.entry_id,
                                    "sourceIndex": template.source_index,
                                    **(
                                        {"releaseDate": str(template.release_date)}
                                        if template.release_date
                                        else {}
                                    ),
                                },
                            }
                        )
            sequences.append({kind: body})
        payload = {
            "name": name,
            "modelSeeds": list(query_set.seeds),
            "sequences": sequences,
            "dialect": "alphafold3",
            "version": 4,
        }
        extensions = {
            f: getattr(query, f)
            for f in ("use_msas", "use_main_msas", "use_paired_msas")
            if not getattr(query, f)
        }
        for key in ("pocket_constraint", "covalent_bonds"):
            if getattr(query, key) is not None:
                extensions[key] = query.model_dump(mode="json")[key]
        if extensions:
            payload["openfold3"] = extensions
        path = job_dir / f"{safe}_data.json"
        files[path] = (json.dumps(payload, indent=2) + "\n").encode()
        outputs[name] = path
    previous = {path: path.read_bytes() if path.exists() else None for path in files}
    changed = []
    try:
        for path, raw in files.items():
            _atomic_bytes(path, raw)
            changed.append(path)
    except BaseException:
        for path in reversed(changed):
            if previous[path] is None:
                path.unlink(missing_ok=True)
            else:
                _atomic_bytes(path, previous[path])
        raise
    return outputs
