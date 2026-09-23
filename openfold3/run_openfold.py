# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""

Main run script for OpenFold3. Please see the README for usage details.

"""
# ruff: noqa: F821

import logging
import os
import tempfile
from datetime import timedelta
from pathlib import Path

import click

from openfold3.core.config import config_utils
from openfold3.entry_points.import_utils import (
    _configure_torch_backend,
    _enable_tf32,
)
from openfold3.entry_points.parameters import DEFAULT_CACHE_PATH

logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--runner-yaml",
    "--runner_yaml",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Yaml that specifies model and dataset parameters,"
    " see examples/training_new.yml",
)
@click.option("--seed", type=int, help="Initial seed for all processes")
@click.option(
    "--data-seed",
    "--data_seed",
    type=int,
    help="Initial seed for data pipeline. Defaults to seed if not specified.",
)
@click.option(
    "--use_tf32",
    type=bool,
    default=False,
    help="Use tf32 precision",
)
def train(
    runner_yaml: Path,
    seed: int | None = None,
    data_seed: int | None = None,
    use_tf32: bool = False,
):
    """Perform a training experiment with a preprepared dataset cache."""
    _configure_torch_backend()
    if use_tf32:
        _enable_tf32()

    from openfold3.entry_points.experiment_runner import (
        TrainingExperimentRunner,
    )
    from openfold3.entry_points.validator import (
        TrainingExperimentConfig,
    )

    runner_dict = config_utils.load_yaml(runner_yaml)

    # overwrite seed defaults if provided:
    if seed is not None:
        runner_dict["experiment_settings"]["seed"] = seed

    if data_seed is not None:
        runner_dict["data_module_args"]["data_seed"] = data_seed

    expt_config = TrainingExperimentConfig.model_validate(runner_dict)

    expt_runner = TrainingExperimentRunner(expt_config)
    expt_runner.setup()
    expt_runner.run()


@cli.command()
@click.option(
    "--query-json",
    "--query_json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Json containing the queries for prediction.",
)
@click.option(
    "--inference-ckpt-path",
    "--inference_ckpt_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
    required=False,
    help="Path for model checkpoint to be used for inference. "
    "If not specified, will attempt to find or download parameters to "
    "$OPENFOLD_CACHE [default: ~/.openfold3/]",
)
@click.option(
    "--inference-ckpt-name",
    "--inference_ckpt_name",
    type=str,
    required=False,
    help="Name of the checkpoint to be used for inference."
    " Only used if `inference_ckpt_path` is not specified.",
)
@click.option(
    "--num-diffusion-samples",
    "--num_diffusion_samples",
    type=int,
    default=None,
    required=False,
    help="Number of diffusion samples to generate for each query.",
)
@click.option(
    "--num-model-seeds",
    "--num_model_seeds",
    type=int,
    default=None,
    required=False,
    help="Number of model seeds to use for each query.",
)
@click.option(
    "--seeds",
    "--model-seeds",
    "--model_seeds",
    type=str,
    default=None,
    help="Explicit comma-separated uint32 model seeds, e.g. 40,41,42.",
)
@click.option(
    "--runner-yaml",
    "--runner_yaml",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
    help="Yaml that specifies model and dataset parameters, see examples/runner.yml",
)
@click.option(
    "--use-msa-server",
    "--use_msa_server",
    type=bool,
    default=None,
    help=(
        "Use ColabFold MSA server to perform alignments. If unset, the value from"
        " the runner yaml (or config default) is used."
    ),
)
@click.option(
    "--use-templates",
    "--use_templates",
    type=bool,
    default=None,
    help=(
        "Whether to use templates for prediction. If unset, the value from the"
        " runner yaml (or config default) is used."
    ),
)
@click.option(
    "--output-dir",
    "--output_dir",
    type=click.Path(exists=False, file_okay=True, dir_okay=True, path_type=Path),
    required=False,
    help="Output directory for writing results",
)
@click.option(
    "-D",
    "--run-data-pipeline",
    "--run_data_pipeline",
    type=bool,
    default=True,
    show_default=True,
    help="Run MSA/template preparation and publish prepared bundles.",
)
@click.option(
    "-P",
    "--run-inference",
    "--run_inference",
    type=bool,
    default=True,
    show_default=True,
    help="Run model inference.",
)
@click.option(
    "-J",
    "--write-input-json",
    "--write_input_json",
    type=bool,
    default=None,
    show_default=True,
    help="Write an AF3-style <query>_data.json (default: true).",
)
@click.option(
    "--compress-fold-input",
    "--compress_fold_input",
    type=bool,
    default=False,
    show_default=True,
    help="Write prepared A3M and mmCIF resources with zstd compression.",
)
@click.option(
    "--compress-full-confidence", "--compress_full_confidence",
    type=bool, default=None,
    help="Write detailed confidence as compressed NPZ (effective default: false; honors explicit legacy runner format).",
)
@click.option(
    "--skip",
    type=bool,
    default=None,
    help="Skip seeds whose complete expected outputs already exist.",
)
@click.option(
    "--max-template-date",
    "--max_template_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Inclusive maximum release date for searched templates.",
)
@click.option(
    "--use_tf32",
    type=bool,
    default=True,
    help="Use tf32 precision",
)
def predict(
    query_json: Path,
    inference_ckpt_path: Path | None = None,
    inference_ckpt_name: str | None = None,
    num_diffusion_samples: int | None = None,
    num_model_seeds: int | None = None,
    seeds: str | None = None,
    runner_yaml: Path | None = None,
    use_msa_server: bool | None = None,
    use_templates: bool | None = None,
    output_dir: Path | None = None,
    run_data_pipeline: bool = True,
    run_inference: bool = True,
    write_input_json: bool | None = None,
    compress_fold_input: bool = False,
    compress_full_confidence: bool | None = None,
    skip: bool | None = None,
    max_template_date=None,
    use_tf32: bool = True,
):
    """Perform inference on a set of queries defined in the query_json."""
    if not run_data_pipeline and not run_inference:
        raise click.UsageError(
            "At least one of --run-data-pipeline and --run-inference must be true"
        )
    if seeds is not None and num_model_seeds is not None:
        raise click.UsageError("--seeds and --num-model-seeds are mutually exclusive")
    if num_model_seeds is not None and num_model_seeds < 1:
        raise click.BadParameter(
            "must be a positive integer", param_hint="--num-model-seeds"
        )
    if num_diffusion_samples is not None and num_diffusion_samples < 1:
        raise click.BadParameter(
            "must be a positive integer", param_hint="--num-diffusion-samples"
        )

    explicit_seeds = None
    if seeds is not None:
        try:
            explicit_seeds = [int(value.strip()) for value in seeds.split(",")]
        except ValueError as exc:
            raise click.BadParameter(
                "seeds must be comma-separated integers", param_hint="--seeds"
            ) from exc
        if (
            not explicit_seeds
            or len(explicit_seeds) != len(set(explicit_seeds))
            or any(seed < 0 or seed > 2**32 - 1 for seed in explicit_seeds)
        ):
            raise click.BadParameter(
                "seeds must be unique uint32 values", param_hint="--seeds"
            )

    if run_inference:
        _configure_torch_backend()
        if use_tf32:
            _enable_tf32()

    from openfold3.core.data.framework.data_module import InferenceDataModule
    from openfold3.core.data.prepared_bundle import (
        clear_template_inputs,
        configure_af3_msa_sources,
        load_af3_query_set,
        materialise_msas,
        materialise_templates,
        restore_prepared_templates,
        validate_inference_only_templates,
        validate_prepared_query_names,
        write_af3_query_sets,
    )
    from openfold3.entry_points.experiment_runner import (
        InferenceExperimentRunner,
    )
    from openfold3.entry_points.validator import (
        InferenceExperimentConfig,
    )

    # Reject ambiguous output paths before config/runner construction can resolve
    # assets or create directories, including when snapshots are disabled.
    runtime_parent = os.environ.get("SLURM_TMPDIR")
    if runtime_parent and not (
        Path(runtime_parent).is_dir() and os.access(runtime_parent, os.W_OK)
    ):
        runtime_parent = None
    workspace = tempfile.TemporaryDirectory(
        prefix="openfold3-input-", dir=runtime_parent
    )
    # Click closes resources on success, early skip and exceptions alike.
    click.get_current_context().call_on_close(workspace.cleanup)
    runtime_directory = Path(workspace.name)
    query_set = load_af3_query_set(query_json, runtime_directory / "input")
    validate_prepared_query_names(query_set)
    if write_input_json is None:
        write_input_json = True

    logging.basicConfig(level=logging.INFO)

    default_yml = (
        Path(os.environ.get("OPENFOLD_CACHE") or DEFAULT_CACHE_PATH) / "runner.yml"
    )
    user_default_runner_path = None

    if default_yml.exists():
        runner_args = config_utils.load_yaml(default_yml)
        user_default_runner_path = default_yml.resolve()
    else:
        runner_args = dict()

    if runner_yaml:
        config_utils.deep_update(runner_args, config_utils.load_yaml(runner_yaml))

    if compress_full_confidence is not None:
        runner_args.setdefault("output_writer_settings", {})["compress_full_confidence"] = compress_full_confidence
    experiment_settings = runner_args.setdefault("experiment_settings", {})
    runner_configures_seeds = any(
        key in experiment_settings for key in ("seeds", "num_seeds")
    )
    experiment_settings["run_data_pipeline"] = run_data_pipeline
    experiment_settings["run_inference"] = run_inference
    experiment_settings["write_input_json"] = write_input_json
    experiment_settings["compress_fold_input"] = compress_fold_input
    if skip is not None:
        experiment_settings["skip_existing"] = skip

    # Inject owned defaults BEFORE validation derives download/cache directories.
    # Explicit structure/precache stores are user inputs and remain untouched;
    # per-query computed caches/logs always belong to this invocation.
    template_args = runner_args.setdefault("template_preprocessor_settings", {})
    template_root = runtime_directory / "template_data"
    template_args["output_directory"] = template_root
    template_args["cache_directory"] = template_root / "template_cache"
    template_args["log_directory"] = template_root / "template_logs"
    for key, child in (
        ("structure_directory", "template_structures"),
        ("precache_directory", "template_precache"),
        ("structure_array_directory", "template_structure_arrays"),
    ):
        enabled = (
            key == "structure_directory"
            or (key == "precache_directory" and template_args.get("create_precache"))
            or (
                key == "structure_array_directory"
                and template_args.get("preparse_structures")
            )
        )
        if enabled and template_args.get(key) is None:
            template_args[key] = template_root / child

    expt_config = InferenceExperimentConfig(
        inference_ckpt_path=inference_ckpt_path,
        inference_ckpt_name=inference_ckpt_name,
        user_default_runner_yaml_path=user_default_runner_path,
        **runner_args,
    )
    configure_af3_msa_sources(expt_config.dataset_config_kwargs.msa)
    if run_inference and expt_config.output_writer_settings.structure_format != "cif":
        raise click.UsageError(
            "EnsembleFold wrapper structure_format must be cif; "
            "update output_writer_settings.structure_format in runner YAML."
        )
    msa_compute_settings = expt_config.msa_computation_settings
    if msa_compute_settings.msa_output_directory is None:
        msa_compute_settings.save_openfold_outputs = False
    if msa_compute_settings.colabfold_output_dir is None:
        msa_compute_settings.save_colabfold_outputs = False
    if max_template_date is not None:
        # The native preprocessor uses an exclusive cutoff.  EnsembleFold exposes
        # an inclusive date consistently with its other wrappers.
        expt_config.template_preprocessor_settings.max_release_date = (
            max_template_date + timedelta(days=1)
        )
    expt_runner = InferenceExperimentRunner(
        expt_config,
        num_diffusion_samples=num_diffusion_samples,
        num_model_seeds=num_model_seeds,
        use_msa_server=use_msa_server,
        use_templates=use_templates,
        output_dir=output_dir,
        model_seeds=explicit_seeds,
    )

    if (
        explicit_seeds is not None
        or num_model_seeds is not None
        or runner_configures_seeds
    ):
        query_set.seeds = list(expt_runner.seeds)
    else:
        expt_runner.set_model_seeds(query_set.seeds)

    # Run the requested preparation stage without creating a model or checkpoint.
    try:
        if run_data_pipeline:
            expt_runner.inference_query_set = query_set
            data_module = InferenceDataModule(
                expt_runner.data_module_config,
                use_msa_server=expt_runner.use_msa_server,
                use_templates=expt_runner.use_templates,
                msa_computation_settings=expt_config.msa_computation_settings,
            )
            data_module.prepare_data()
            query_set = data_module.inference_config.query_set
            materialise_msas(
                query_set,
                runtime_directory / "prepared",
                expt_config.dataset_config_kwargs.msa,
                compress=compress_fold_input,
            )
            materialise_templates(
                query_set,
                runtime_directory / "prepared",
                expt_config.template_preprocessor_settings,
                compress=compress_fold_input,
            )

        # Saving conditions is independent of searching. This also handles D=false,
        # J=true; no public resources are created at all when J=false.
        if write_input_json:
            write_af3_query_sets(
                query_set, expt_runner.output_dir, compress=compress_fold_input
            )

        if run_inference:
            # Conditions were freshly read into private files above. Keep the
            # public declaration intact when disabling their use for this run.
            query_set = query_set.model_copy(deep=True)
            inference_uses_templates = expt_runner.use_templates
            if inference_uses_templates:
                if not run_data_pipeline:
                    validate_inference_only_templates(query_set)
            else:
                clear_template_inputs(query_set)
            expt_runner.set_preprocessing_flags(
                use_msa_server=False, use_templates=False
            )
            if not expt_runner.has_pending_queries(query_set):
                logger.info("All requested query/seed outputs are complete; skipping")
                expt_runner.cleanup()
                return
            if inference_uses_templates:
                restore_prepared_templates(
                    query_set,
                    runtime_directory / "inference",
                    expt_config.template_preprocessor_settings,
                )
            expt_runner.setup()
            expt_runner.run(query_set)
    except BaseException:
        expt_runner.cleanup_msa_workspace()
        raise
    expt_runner.cleanup()


@cli.command()
@click.option(
    "--query-json",
    "--query_json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Json containing the queries for prediction.",
)
@click.option(
    "--output-dir",
    "--output_dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Output directory for writing alignments",
)
@click.option(
    "--msa-computation-settings-yaml",
    "--msa_computation_settings_yaml",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
    help="Yaml file to customize Colabfold MSA settings,"
    " see MsaComputationSettings for options.",
)
def align_msa_server(
    query_json: Path,
    output_dir: Path,
    msa_computation_settings_yaml: Path | None = None,
):
    """Generate ColabFold alignments without running model inference.

    Submit all queries for the job in one JSON file. The command writes a new
    ``query_msa.json`` whose alignment paths point to files in ``output_dir``.
    Server settings can be supplied with ``msa_computation_settings_yaml``.
    """
    from openfold3.core.data.tools.colabfold_msa_server import (
        MsaComputationSettings,
        preprocess_colabfold_msas,
    )
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )

    msa_settings = MsaComputationSettings.from_config_with_cli_override(
        output_dir, msa_computation_settings_yaml
    )
    try:
        query_set = InferenceQuerySet.from_json(query_json)
        query_set = preprocess_colabfold_msas(
            inference_query_set=query_set,
            compute_settings=msa_settings,
        )

        with open(output_dir / "query_msa.json", "w") as fp:
            fp.write(query_set.model_dump_json(indent=4))
    finally:
        try:
            msa_settings.cleanup_workspace()
        except OSError:
            logger.warning(
                "Could not remove temporary MSA workspace for run %s",
                msa_settings.run_directory_name,
                exc_info=True,
            )


if __name__ == "__main__":
    cli()
