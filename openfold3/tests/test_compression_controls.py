"""Compression controls preserve the established writer groups and input paths."""
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
from biotite import structure

from openfold3.core.data.af3_input import load_af3_query_set, write_af3_query_sets
from openfold3.core.data.io.compression import read_text_auto
from openfold3.core.runners.writer import OF3OutputWriter
from openfold3.entry_points.validator import OutputWritingSettings


@pytest.mark.parametrize("compressed", [None, False, True])
def test_bundle_reads_compressed_then_publishes_requested_format(tmp_path, compressed):
    source = tmp_path / "input.json"
    source.write_text(json.dumps({"name": "job", "modelSeeds": [7], "sequences": [{"protein": {
        "id": "A", "sequence": "A", "pairedMsa": "", "unpairedMsa": ">q\nA\n>hit\nW\n", "templates": [],
    }}]}))
    initial = load_af3_query_set(source, tmp_path / "initial")
    old = write_af3_query_sets(initial, tmp_path / "old", compress=True)["job"]
    restored = load_af3_query_set(old, tmp_path / "readback")
    options = {} if compressed is None else {"compress": compressed}
    target = write_af3_query_sets(restored, tmp_path / "new", **options)["job"]
    protein = json.loads(target.read_text())["sequences"][0]["protein"]
    assert protein["unpairedMsaPath"] == "msas/job__A_unpairedmsa.a3m" + (".zst" if compressed else "")
    assert read_text_auto(target.parent / protein["unpairedMsaPath"]) == ">q\nA\n>hit\nW\n"
    assert load_af3_query_set(target, tmp_path / "final").queries["job"].chains[0].sequence == "A"


@pytest.mark.parametrize("values,expected", [({}, "json"), ({"compress_full_confidence": False}, "json"), ({"compress_full_confidence": True}, "npz"), ({"full_confidence_output_format": "npz"}, "npz"), ({"full_confidence_output_format": "json"}, "json"), ({"compress_full_confidence": True, "full_confidence_output_format": "npz"}, "npz")])
def test_effective_settings_drive_real_writer(tmp_path, values, expected):
    settings = OutputWritingSettings(**values)
    writer = OF3OutputWriter(tmp_path, **settings.model_dump())
    atoms = structure.array([structure.Atom([0, 0, 0], chain_id="A")])
    confidence = {key: np.array([0.5], dtype=np.float32) for key in ("plddt", "gpde", "iptm", "ptm", "disorder", "has_clash", "sample_ranking_score")}
    confidence.update(pae=np.array([[0.25]], dtype=np.float32), pde=np.array([[0.75]], dtype=np.float32), chain_ptm={"1": np.array([0.5])}, chain_pair_iptm={}, bespoke_iptm={})
    prefix = tmp_path / "full_data/seed-7_sample-0"
    stale = Path(f"{prefix}_full_data." + ("json" if expected == "npz" else "npz"))
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"previous format")
    embedding = tmp_path / "latents/seed-7_latent_outputs.pt"
    embedding.parent.mkdir()
    embedding.write_bytes(b"keep")
    writer.write_confidence_scores(confidence, atoms, tmp_path / "summary/seed-7_sample-0", prefix)
    output = Path(f"{prefix}_full_data.{expected}")
    assert output.exists()
    assert not stale.exists()
    if expected == "npz":
        with ZipFile(output) as archive:
            assert all(item.compress_type == ZIP_DEFLATED for item in archive.infolist())
        with np.load(output) as archive:
            actual = dict(archive)
            assert actual["plddt"].dtype == np.dtype("float16")
    else:
        actual = json.loads(output.read_text())
    assert set(actual) == {"plddt", "pae", "pde"}
    for key, value in {"plddt": [0.5], "pae": [[0.25]], "pde": [[0.75]]}.items():
        np.testing.assert_array_equal(actual[key], value)
    assert embedding.read_bytes() == b"keep"


@pytest.mark.parametrize("compressed,legacy", [(True, "json"), (False, "npz")])
def test_conflicting_explicit_confidence_options_rejected(compressed, legacy):
    with pytest.raises(ValueError, match="conflict"):
        OutputWritingSettings(compress_full_confidence=compressed, full_confidence_output_format=legacy)
    with pytest.raises(ValueError, match="conflict"):
        OF3OutputWriter(Path("unused"), compress_full_confidence=compressed, full_confidence_output_format=legacy)


@pytest.mark.parametrize("write", [None, False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_inference_cli_refreshes_input_even_when_every_seed_skips(tmp_path, monkeypatch, write, legacy):
    from click.testing import CliRunner
    from openfold3 import run_openfold
    from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
    source = tmp_path / "input.json"
    source.write_text(json.dumps({"name": "job", "modelSeeds": [7], "sequences": [{"protein": {"id": "A", "sequence": "A", "unpairedMsa": ">q\nA\n>hit\nW\n", "pairedMsa": "", "templates": []}}]}))
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()
    monkeypatch.setenv("OPENFOLD_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(run_openfold, "_configure_torch_backend", lambda: None)
    monkeypatch.setattr(run_openfold, "_enable_tf32", lambda: None)
    def forbidden(*args, **kwargs):
        pytest.fail("completed inference must skip model execution")
    monkeypatch.setattr(InferenceExperimentRunner, "run", forbidden)
    output = tmp_path / "out"
    fmt = "npz" if legacy else "json"
    for relative in ("models/seed-7_sample-0_model.cif", "summary_confidences/seed-7_sample-0_summary_confidences.json", f"full_data/seed-7_sample-0_full_data.{fmt}"):
        path = output / "job" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not parsed by skip")
    args = ["predict", "--query_json", str(source), "--output_dir", str(output), "--inference_ckpt_path", str(checkpoint), "-D", "false", "--use_templates", "false", "--num_diffusion_samples", "1", "--skip", "true", "--use_tf32", "false"]
    if write is not None:
        args += ["-J", str(write).lower()]
    if legacy:
        yaml = tmp_path / "runner.yml"
        yaml.write_text("output_writer_settings:\n  full_confidence_output_format: npz\n")
        args += ["--runner_yaml", str(yaml)]
    result = CliRunner().invoke(run_openfold.cli, args)
    assert result.exit_code == 0, (result.output, result.exception)
    snapshot = output / "job/job_data.json"
    assert snapshot.exists() is (write is not False)
    if snapshot.exists():
        resource = output / "job/msas/job__A_unpairedmsa.a3m"
        assert resource.read_text().endswith(">hit\nW\n")
        source.write_text(source.read_text().replace("W\\n", "G\\n"))
        result = CliRunner().invoke(run_openfold.cli, args)
        assert result.exit_code == 0, (result.output, result.exception)
        assert resource.read_text().endswith(">hit\nG\n")
