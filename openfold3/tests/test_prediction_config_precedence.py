"""Explicit runner YAML values win over wrapper CLI values, including false."""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from openfold3 import run_openfold
from openfold3.entry_points import experiment_runner


class CapturedRunner(Exception):
    pass


@pytest.fixture
def invocation(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("OPENFOLD_CACHE", str(cache))
    monkeypatch.setattr(run_openfold, "_configure_torch_backend", lambda: None)
    monkeypatch.setattr(run_openfold, "_enable_tf32", lambda: None)
    query = tmp_path / "query.json"
    query.write_text(json.dumps({"name": "job", "modelSeeds": [7], "sequences": [
        {"protein": {"id": "A", "sequence": "A", "templates": [],
                     "pairedMsa": "", "unpairedMsa": ""}}
    ]}))
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()
    captured = {}
    original = experiment_runner.InferenceExperimentRunner

    def capture(config, **kwargs):
        runner = original(config, **kwargs)
        captured["runner"] = runner
        raise CapturedRunner

    monkeypatch.setattr(experiment_runner, "InferenceExperimentRunner", capture)

    def invoke(settings, options=(), *, cached=False, cached_settings=None):
        if cached_settings is not None:
            (cache / "runner.yml").write_text(yaml.safe_dump(cached_settings))
        runner_yaml = (cache if cached else tmp_path) / "runner.yml"
        runner_yaml.write_text(yaml.safe_dump(settings))
        args = ["predict", "--query_json", str(query), "--output_dir", str(tmp_path / "cli-out"),
                "--inference_ckpt_path", str(checkpoint), "--use_tf32", "false"]
        if not cached:
            args += ["--runner_yaml", str(runner_yaml)]
        result = CliRunner().invoke(run_openfold.cli, args + list(options))
        assert isinstance(result.exception, CapturedRunner), (result.output, result.exception)
        return captured["runner"]

    return invoke


@pytest.mark.parametrize("cached", [False, True])
def test_yaml_false_and_skip_override_explicit_cli(invocation, cached):
    runner = invocation({"experiment_settings": {
        "use_msa_server": False, "use_templates": False, "skip_existing": True,
        "run_data_pipeline": False, "run_inference": True, "write_input_json": False,
        "compress_fold_input": False,
    }}, ["--use_msa_server", "true", "--use_templates", "true", "--skip", "false",
         "-D", "true", "-J", "true", "--compress_fold_input", "true"], cached=cached)
    settings = runner.experiment_config.experiment_settings
    assert runner.use_msa_server is False
    assert runner.use_templates is False
    assert settings.skip_existing is True
    assert settings.run_data_pipeline is False
    assert settings.write_input_json is False
    assert settings.compress_fold_input is False


def test_absent_yaml_keys_leave_cli_effective(invocation):
    runner = invocation({}, ["--use_msa_server", "false", "--use_templates", "false",
                             "--skip", "true", "--seeds", "13", "--num_diffusion_samples", "2"])
    assert runner.use_msa_server is False
    assert runner.use_templates is False
    assert runner.experiment_config.experiment_settings.skip_existing is True
    assert runner.seeds == [13]
    assert runner.num_diffusion_samples == 2


def test_yaml_seed_samples_output_and_checkpoint_win(invocation, tmp_path):
    checkpoint = tmp_path / "yaml.ckpt"
    checkpoint.touch()
    runner = invocation({
        "inference_ckpt_path": str(checkpoint),
        "experiment_settings": {"seeds": [19], "output_dir": str(tmp_path / "yaml-out")},
        "model_update": {"custom": {"architecture": {"shared": {"diffusion": {
            "no_full_rollout_samples": 3}}}}},
    }, ["--seeds", "13", "--num_diffusion_samples", "2"])
    assert runner.seeds == [19]
    assert runner.num_diffusion_samples == 3
    assert runner.output_dir == tmp_path / "yaml-out"
    assert runner.experiment_config.inference_ckpt_path == checkpoint


@pytest.mark.parametrize("yaml_setting", [
    {"compress_full_confidence": True}, {"full_confidence_output_format": "npz"},
])
def test_yaml_confidence_wins_over_cli_alias(invocation, yaml_setting):
    runner = invocation({"output_writer_settings": yaml_setting},
                        ["--compress_full_confidence", "false"])
    assert runner.experiment_config.output_writer_settings.full_confidence_output_format == "npz"


def test_yaml_template_cutoff_wins_without_double_conversion(invocation):
    runner = invocation({"template_preprocessor_settings": {"max_release_date": "2020-01-02"}},
                        ["--max_template_date", "2024-01-01"])
    assert runner.experiment_config.template_preprocessor_settings.max_release_date == datetime(2020, 1, 2)


def test_stage_conflict_is_checked_after_yaml_precedence(invocation):
    runner = invocation({"experiment_settings": {"run_data_pipeline": True, "run_inference": False}},
                        ["-D", "false", "-P", "false"])
    assert runner.experiment_config.experiment_settings.run_data_pipeline is True
    assert runner.experiment_config.experiment_settings.run_inference is False


@pytest.mark.parametrize("quoted", [False, True])
def test_yaml_controls_actual_skip_and_prepared_writes(tmp_path, monkeypatch, quoted):
    from openfold3.core.data.framework.data_module import InferenceDataModule

    monkeypatch.setenv("OPENFOLD_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(run_openfold, "_configure_torch_backend", lambda: None)
    monkeypatch.setattr(run_openfold, "_enable_tf32", lambda: None)

    def forbidden(*args, **kwargs):
        pytest.fail("YAML disables data and skips the complete seed")

    monkeypatch.setattr(InferenceDataModule, "prepare_data", forbidden)
    monkeypatch.setattr(experiment_runner.InferenceExperimentRunner, "setup", forbidden)
    query = tmp_path / "query.json"
    query.write_text(json.dumps({"name": "job", "modelSeeds": [7], "sequences": [
        {"protein": {"id": "A", "sequence": "A", "templates": [],
                     "pairedMsa": "", "unpairedMsa": ""}}
    ]}))
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()
    output = tmp_path / "out"
    for relative in ("models/seed-7_sample-0_model.cif",
                     "summary_confidences/seed-7_sample-0_summary_confidences.json",
                     "full_data/seed-7_sample-0_full_data.npz"):
        path = output / "job" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("skip checks existence and size only")
    config = tmp_path / "runner.yml"
    settings = {
        "experiment_settings": {"run_data_pipeline": False, "use_templates": False,
                                "write_input_json": False, "skip_existing": True},
        "output_writer_settings": {"compress_full_confidence": True},
    }
    if quoted:
        settings["experiment_settings"] = {
            key: str(value).lower() for key, value in settings["experiment_settings"].items()
        }
    config.write_text(yaml.safe_dump(settings))
    result = CliRunner().invoke(run_openfold.cli, [
        "predict", "--query_json", str(query), "--output_dir", str(output),
        "--runner_yaml", str(config), "--inference_ckpt_path", str(checkpoint),
        "--num_diffusion_samples", "1", "-D", "true", "-J", "true", "--skip", "false",
        "--use_templates", "true", "--compress_full_confidence", "false",
    ])
    assert result.exit_code == 0, (result.output, result.exception)
    assert not (output / "job/job_data.json").exists()


def test_shell_defers_stage_validation_until_yaml_is_applied(tmp_path):
    query = tmp_path / "input.json"
    query.write_text("{}")
    config = tmp_path / "runner.yml"
    config.write_text("experiment_settings:\n  run_inference: true\n")
    executable = tmp_path / "capture-cli"
    executable.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "print(json.dumps({'args': sys.argv[1:], 'cuda': os.environ.get('CUDA_VISIBLE_DEVICES')}))\n"
    )
    executable.chmod(0o755)
    environment = dict(os.environ, OPENFOLD_BIN=str(executable))
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    result = subprocess.run([
        "bash", str(Path(__file__).resolve().parents[2] / "run_openfold.sh"),
        "-i", str(query), "-o", str(tmp_path / "out"), "-y", str(config),
        "-D", "false", "-P", "false", "-d", "3",
    ], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    invoked = json.loads(result.stdout.splitlines()[-1])
    assert invoked["args"][invoked["args"].index("--runner_yaml") + 1] == str(config)
    assert invoked["cuda"] == "3"


def test_quoted_false_disables_inference_before_backend_setup(invocation, monkeypatch):
    def forbidden():
        pytest.fail("inference disabled in YAML must not configure the Torch backend")

    monkeypatch.setattr(run_openfold, "_configure_torch_backend", forbidden)
    runner = invocation({"experiment_settings": {
        "run_data_pipeline": "true", "run_inference": "false",
        "write_input_json": "false", "compress_fold_input": "false",
    }})
    settings = runner.experiment_config.experiment_settings
    assert settings.run_inference is False
    assert settings.write_input_json is False
    assert settings.compress_fold_input is False


@pytest.mark.parametrize("cached_format,explicit_format,expected", [
    ({"full_confidence_output_format": "npz"}, {"compress_full_confidence": False}, "json"),
    ({"compress_full_confidence": False}, {"full_confidence_output_format": "npz"}, "npz"),
])
def test_explicit_yaml_format_alias_overrides_cached_alternative(
    invocation, cached_format, explicit_format, expected
):
    runner = invocation({"output_writer_settings": explicit_format},
                        cached_settings={"output_writer_settings": cached_format})
    assert runner.experiment_config.output_writer_settings.full_confidence_output_format == expected


def test_explicit_yaml_checkpoint_name_replaces_cached_path(invocation, tmp_path):
    old_checkpoint = tmp_path / "old.ckpt"
    old_checkpoint.touch()
    runner = invocation({
        "inference_ckpt_name": "openbind-2025-06-30-174k",
        "experiment_settings": {"run_inference": False},
    }, cached_settings={"inference_ckpt_path": str(old_checkpoint)})
    assert runner.experiment_config.inference_ckpt_name == "openbind-2025-06-30-174k"
    assert runner.experiment_config.inference_ckpt_path is None
