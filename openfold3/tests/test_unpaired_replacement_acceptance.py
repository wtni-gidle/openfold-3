"""Prepared resource replacement reaches real MSA and template consumers.

Catch stale snapshots/caches, mutation of paired/template conditions, and lost
insertion statistics. Only runner model setup/run is substituted; the replacement
run calls actual MSA processing, native template sampling and mmCIF parsing.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from openfold3 import run_openfold
from openfold3.core.config.msa_pipeline_configs import MsaSampleProcessorInputInference
from openfold3.core.data import prepared_bundle as bundle
from openfold3.core.data.io.compression import read_text_auto, write_zstd_text
from openfold3.core.data.pipelines.sample_processing.msa import MsaSampleProcessorInference
from openfold3.core.data.primitives.structure.template import (
    parse_template_structure,
    sample_templates,
)
from openfold3.entry_points.experiment_runner import InferenceExperimentRunner


@pytest.mark.parametrize("data", [False, True])
@pytest.mark.parametrize("write", [False, True])
@pytest.mark.parametrize("new_path", [False, True])
def test_prepared_unpaired_replacement_preserves_native_template_and_pairing(
    tmp_path, monkeypatch, biotite_ccd_wrapper, data, write, new_path
):
    fixture = Path(__file__).parent / "test_data/mmcifs/1a8q.cif"
    cif = bundle.extract_single_chain_mmcif(fixture, "A")
    source = tmp_path / "request.json"
    source.write_text(json.dumps({
        "name": "job", "modelSeeds": [7], "sequences": [
            {"protein": {"id": "A", "sequence": "ACDE",
                         "unpairedMsa": ">q\nACDE\n>old\nA-DE\n",
                         "pairedMsa": ">q\nACDE\n>pair\nAC-E\n",
                         "templates": [{"mmcif": cif, "queryIndices": [0, 1],
                                        "templateIndices": [1, 2]}]}},
            {"protein": {"id": "B", "sequence": "FGHI",
                         "unpairedMsa": ">q\nFGHI\n>single\nFGH-\n",
                         "pairedMsa": ">q\nFGHI\n>pair\nFG-I\n",
                         "templates": []}},
        ],
    }))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("SLURM_TMPDIR", str(scratch))
    monkeypatch.setenv("OPENFOLD_CACHE", str(tmp_path / "empty-cache"))
    checkpoint = tmp_path / "dummy.ckpt"
    checkpoint.write_text("never loaded")
    output = tmp_path / "out"
    observed = []

    def consume(runner, queries):
        query = queries.queries["job"]
        features = MsaSampleProcessorInference(runner.dataset_config_kwargs.msa)(
            MsaSampleProcessorInputInference.create_from_inference_query_entry(query)
        )
        chain = query.chains[0]
        assembly = {"A": {"template_ids": chain.template_entry_chain_ids,
                          "cache_entry_file_path": chain.template_alignment_file_path}}
        selected = sample_templates(
            assembly, chain.template_alignment_file_path.parent, 4, True, "A", None, "cif"
        )
        assert len(selected) == 1
        entry = next(iter(selected.values()))
        np.testing.assert_array_equal(entry.idx_map, [[1, 2], [2, 3]])
        identity = chain.template_entry_chain_ids[0]
        with np.load(chain.template_alignment_file_path, allow_pickle=True) as cache:
            path = Path(cache[identity].item()["cif_path"])
        atoms = parse_template_structure(
            path.parent, None, identity, "cif", biotite_ccd_wrapper, cif_path=path
        )
        assert atoms is not None and len(atoms) > 100
        observed.append((features, atoms.coord.copy(), atoms.res_id.copy()))

    monkeypatch.setattr(InferenceExperimentRunner, "setup", lambda self: None)
    monkeypatch.setattr(InferenceExperimentRunner, "run", consume)

    def invoke(path, *, data, inference, write):
        result = CliRunner().invoke(run_openfold.cli, [
            "predict", "--query-json", str(path), "--output-dir", str(output),
            "--inference-ckpt-path", str(checkpoint), "--use-msa-server", "false",
            "--use-templates", "true", "--use_tf32", "false", "--skip", "false",
            "-D", str(data), "-P", str(inference), "--write-input-json", str(write),
        ])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert not list(scratch.iterdir())

    invoke(source, data=True, inference=False, write=True)
    prepared = output / "job/job_data.json"
    job = prepared.parent
    payload = json.loads(prepared.read_text())
    proteins = [item["protein"] for item in payload["sequences"]]
    fixed_paths = [job / protein["pairedMsaPath"] for protein in proteins]
    fixed_paths += [job / proteins[0]["templates"][0]["mmcifPath"]]
    fixed_bytes = {path: path.read_bytes() for path in fixed_paths}
    mapping = proteins[0]["templates"]
    invoke(prepared, data=False, inference=True, write=False)
    replacement = ">q\nACDE\n>new\nAxxCD-\n"
    changed = job / proteins[0]["unpairedMsaPath"]
    if new_path:
        changed = job / "msas/replacement.a3m.zst"
        proteins[0]["unpairedMsaPath"] = "msas/replacement.a3m.zst"
        prepared.write_text(json.dumps(payload))
    write_zstd_text(changed, replacement)
    before = {path: path.read_bytes() for path in [prepared, *job.glob("msas/*")]}
    before_paths = {path.relative_to(job) for path in job.rglob("*")}
    invoke(prepared, data=data, inference=True, write=write)
    first, second = observed
    rows = lambda msa: ["".join(row) for row in msa.msa]
    assert "A-DE" in rows(first[0].chain_id_to_main_msa["A"])
    assert "ACD-" in rows(second[0].chain_id_to_main_msa["A"])
    assert "A-DE" not in rows(second[0].chain_id_to_main_msa["A"])
    np.testing.assert_allclose(second[0].chain_id_to_deletion_mean["A"], [0, 1, 0, 0])
    assert not np.array_equal(first[0].chain_id_to_profile["A"], second[0].chain_id_to_profile["A"])
    for chain_id, expected in (("A", "AC-E"), ("B", "FG-I")):
        assert expected in rows(second[0].chain_id_to_paired_msa[chain_id])
        np.testing.assert_array_equal(first[0].chain_id_to_paired_msa[chain_id].msa,
                                      second[0].chain_id_to_paired_msa[chain_id].msa)
    np.testing.assert_allclose(first[1], second[1], equal_nan=True)
    np.testing.assert_array_equal(first[2], second[2])
    assert {path: path.read_bytes() for path in fixed_paths} == fixed_bytes
    if not write:
        assert {path: path.read_bytes() for path in [prepared, *job.glob("msas/*")]} == before
        assert {path.relative_to(job) for path in job.rglob("*")} == before_paths
    else:
        current = json.loads(prepared.read_text())["sequences"][0]["protein"]
        assert current["unpairedMsaPath"] == "msas/job__A_unpairedmsa.a3m"
        # Data serialization may normalize A3M headers, not residues/insertions.
        assert "AxxCD-" in read_text_auto(job / current["unpairedMsaPath"])
        assert current["templates"] == mapping
