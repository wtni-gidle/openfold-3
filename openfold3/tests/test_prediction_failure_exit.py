"""Failure reporting must reach callers without discarding successful outputs."""

from types import SimpleNamespace

import pytest
import torch
from biotite import structure
from click.testing import CliRunner

from openfold3.core.runners import writer as writer_module
from openfold3.core.runners.writer import OF3OutputWriter


def _trainer(rank=0, world_size=1):
    return SimpleNamespace(
        global_rank=rank, is_global_zero=rank == 0, world_size=world_size
    )


def _prediction(query_id):
    atoms = structure.array(
        [structure.Atom([1, 2, 3], chain_id="A", res_name="ALA", atom_name="CA")]
    )
    batch = {"atom_array": [atoms], "seed": [42], "query_id": [query_id]}
    confidence = {
        key: torch.ones((1, 1))
        for key in (
            "gpde", "iptm", "ptm", "disorder", "has_clash", "sample_ranking_score"
        )
    }
    confidence.update(
        plddt=torch.full((1, 1, 1), 80.0),
        pde=torch.ones((1, 1, 1, 1)),
        pae=torch.ones((1, 1, 1, 1)),
        chain_ptm={"1": torch.ones((1, 1))},
        chain_pair_iptm={},
        bespoke_iptm={},
    )
    outputs = {
        "atom_positions_predicted": torch.tensor(atoms.coord)[None, None],
        "confidence_scores": confidence,
    }
    return batch, outputs


def _successful_batch(writer, query_id="successful"):
    batch, outputs = _prediction(query_id)
    writer.on_predict_batch_end(_trainer(), None, (batch, outputs), batch, 0)
    assert writer.success_count == 1
    model = writer.output_dir / query_id / "models/seed-42_sample-0_model.pdb"
    assert model.stat().st_size > 0
    return model, model.read_bytes()


def _failed_batch(writer, query_id="failed_query"):
    writer.on_predict_batch_end(
        _trainer(), None, None, {"query_id": [query_id]}, 1
    )


def test_mixed_prediction_failure_raises_after_summary_and_preserves_outputs(tmp_path):
    writer = OF3OutputWriter(tmp_path, structure_format="pdb")
    model, original = _successful_batch(writer)
    _failed_batch(writer)

    with pytest.raises(RuntimeError) as error:
        writer.on_predict_end(_trainer(), None)

    assert "failed_query" in str(error.value)
    summary = (tmp_path / "summary.txt").read_text()
    assert "Successful Queries:  1" in summary
    assert "Failed Queries:      1" in summary
    assert "Failed Queries: failed_query" in summary
    assert model.read_bytes() == original


def test_output_write_failure_is_reported_after_other_outputs_succeed(tmp_path):
    writer = OF3OutputWriter(tmp_path, structure_format="pdb")
    # A regular file at the model directory reproduces a real filesystem failure.
    blocked = tmp_path / "unwritable" / "models"
    blocked.parent.mkdir()
    blocked.write_text("existing file")
    batch, outputs = _prediction("unwritable")
    writer.on_predict_batch_end(_trainer(), None, (batch, outputs), batch, 0)
    assert writer.failed_count == 1
    model, original = _successful_batch(writer)

    with pytest.raises(RuntimeError):
        writer.on_predict_end(_trainer(), None)

    assert "Failed Queries: unwritable" in (tmp_path / "summary.txt").read_text()
    assert model.read_bytes() == original
    assert blocked.read_text() == "existing file"


def test_all_failed_predictions_raise_after_summary(tmp_path):
    writer = OF3OutputWriter(tmp_path)
    _failed_batch(writer)
    with pytest.raises(RuntimeError):
        writer.on_predict_end(_trainer(), None)
    assert "Failed Queries:      1" in (tmp_path / "summary.txt").read_text()


@pytest.mark.parametrize("mode", ["success", "empty", "all_skipped"])
def test_success_empty_and_all_skipped_predictions_return_normally(tmp_path, mode):
    writer = OF3OutputWriter(tmp_path, structure_format="pdb")
    if mode == "success":
        _successful_batch(writer)
    elif mode == "all_skipped":
        writer.on_predict_batch_end(
            _trainer(), None, None,
            {"query_id": ["already_complete"], "repeated_sample": True}, 0,
        )
    writer.on_predict_end(_trainer(), None)
    assert "Failed Queries:      0" in (tmp_path / "summary.txt").read_text()
    assert writer.total_count == (1 if mode == "success" else 0)


@pytest.mark.parametrize("rank", [0, 1])
def test_remote_failure_reaches_every_rank_and_only_rank_zero_writes_summary(
    tmp_path, monkeypatch, rank
):
    writer = OF3OutputWriter(tmp_path, structure_format="pdb")
    model, original = _successful_batch(writer)
    monkeypatch.setattr(writer_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(writer_module.dist, "is_initialized", lambda: True)

    def gather(summaries, local_summary):
        summaries[rank] = local_summary
        summaries[1 - rank] = {
            "total": 1, "success": 0, "failed": 1,
            "failed_queries": ["remote_failed"],
        }

    monkeypatch.setattr(writer_module.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError) as error:
        writer.on_predict_end(_trainer(rank, 2), None)
    assert "remote_failed" in str(error.value)
    assert model.read_bytes() == original
    if rank == 0:
        summary = (tmp_path / "summary.txt").read_text()
        assert "Total Queries Processed: 2" in summary
        assert "Failed Queries: remote_failed" in summary
    else:
        assert not (tmp_path / "summary.txt").exists()


@pytest.mark.parametrize("rank", [0, 1])
def test_distributed_timeout_writes_local_diagnostic_and_propagates(
    tmp_path, monkeypatch, rank
):
    writer = OF3OutputWriter(tmp_path, structure_format="pdb")
    model, original = _successful_batch(writer)
    monkeypatch.setattr(writer_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(writer_module.dist, "is_initialized", lambda: True)

    def timeout(summaries, local_summary):
        raise RuntimeError("Distributed collective timed out")

    monkeypatch.setattr(writer_module.dist, "all_gather_object", timeout)
    with pytest.raises(RuntimeError, match="timed out"):
        writer.on_predict_end(_trainer(rank, 2), None)
    fallback = tmp_path / f"fallback_summary_rank_{rank}.txt"
    assert f"INCOMPLETE (Rank {rank})" in fallback.read_text()
    assert "Successful Queries:  1" in fallback.read_text()
    assert not (tmp_path / "summary.txt").exists()
    assert model.read_bytes() == original


def test_summary_write_failure_propagates(tmp_path):
    writer = OF3OutputWriter(tmp_path, structure_format="pdb")
    model, original = _successful_batch(writer)
    (tmp_path / "summary.txt").mkdir()
    with pytest.raises(OSError):
        writer.on_predict_end(_trainer(), None)
    assert model.read_bytes() == original


def test_callback_failure_reaches_predict_cli_exit_code(tmp_path, monkeypatch):
    from openfold3 import run_openfold
    from openfold3.entry_points.experiment_runner import InferenceExperimentRunner

    monkeypatch.setenv("OPENFOLD_CACHE", str(tmp_path / "cache"))
    query = tmp_path / "query.json"
    query.write_text(
        '{"name":"successful","sequences":[{"protein":'
        '{"id":"A","sequence":"A","templates":[],"unpairedMsa":""}}]}'
    )
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.touch()
    output_dir = tmp_path / "outputs"

    def predict_without_model(self, query_set):
        writer = OF3OutputWriter(self.output_dir, structure_format="pdb")
        _successful_batch(writer)
        _failed_batch(writer)
        writer.on_predict_end(_trainer(), None)

    # Keep CLI configuration, query loading, runner construction and cleanup real;
    # replace only checkpoint/model execution and device configuration.
    monkeypatch.setattr(run_openfold, "_configure_torch_backend", lambda: None)
    monkeypatch.setattr(run_openfold, "_enable_tf32", lambda: None)
    monkeypatch.setattr(InferenceExperimentRunner, "setup", lambda self: None)
    monkeypatch.setattr(InferenceExperimentRunner, "run", predict_without_model)
    result = CliRunner().invoke(
        run_openfold.cli,
        [
            "predict", "--query-json", str(query),
            "--inference-ckpt-path", str(checkpoint),
            "--output-dir", str(output_dir), "--run-data-pipeline", "false",
            "--use-templates", "false", "--write-input-json", "false",
        ],
    )
    assert result.exit_code != 0, result.output
    assert isinstance(result.exception, RuntimeError), result.exception
    assert "Failed Queries: failed_query" in (output_dir / "summary.txt").read_text()
    assert (output_dir / "successful/models/seed-42_sample-0_model.pdb").is_file()
