"""Public AF3 boundary tests; native feature code is deliberately not mocked."""

import json

import numpy as np
import pytest

from openfold3.core.data import prepared_bundle as bundle
from openfold3.core.data.io.compression import read_text_auto


def load(tmp_path, payload):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(payload))
    assert hasattr(bundle, "load_af3_query_set"), "AF3 public reader is missing"
    return bundle.load_af3_query_set(source, tmp_path / "private")


def protein(**kwargs):
    return {
        "name": "target",
        "modelSeeds": [7],
        "sequences": [{"protein": {"id": ["A", "B"], "sequence": "ACDE", **kwargs}}],
    }


def test_af3_inline_msa_reaches_native_parser(tmp_path):
    from openfold3.core.config.msa_pipeline_configs import (
        MsaSampleProcessorInputInference,
    )
    from openfold3.core.data.pipelines.sample_processing.msa import (
        MsaSampleProcessorInference,
    )
    from openfold3.projects.of3_all_atom.config.dataset_config_components import (
        MSASettings,
    )

    queries = load(
        tmp_path, protein(unpairedMsa=">query\nACDE\n>hit\nAxC-E\n", pairedMsa="")
    )
    assert queries.seeds == [7]
    query = queries.queries["target"]
    assert query.chains[0].chain_ids == ["A", "B"]
    features = MsaSampleProcessorInference(MSASettings())(
        MsaSampleProcessorInputInference.create_from_inference_query_entry(query)
    )
    np.testing.assert_allclose(features.chain_id_to_deletion_mean["A"], [0, 0.5, 0, 0])


def test_af3_external_arbitrary_msa_filename_is_read_relative_to_json(tmp_path):
    (tmp_path / "deepmsa.result").write_text(">query\nACDE\n>hit\nAC-E\n")
    queries = load(tmp_path, protein(unpairedMsaPath="deepmsa.result"))
    path = queries.queries["target"].chains[0].main_msa_file_paths[0]
    assert path.name.endswith("_unpairedmsa.a3m")
    assert read_text_auto(path) == ">query\nACDE\n>hit\nAC-E\n"


@pytest.mark.parametrize(
    "payload,match",
    [
        ({"queries": {}}, "AF3"),
        (protein(unpairedMsa="", unpairedMsaPath="missing.a3m"), "both"),
        (protein(templatesPath="old.json"), "templatesPath"),
        ({**protein(), "userCCD": "data_x"}, "userCCD"),
        ({**protein(), "modelSeeds": [True]}, "modelSeeds"),
        (protein(id=["A", "A"]), "id"),
    ],
)
def test_af3_rejects_ambiguous_or_unsupported_input(tmp_path, payload, match):
    with pytest.raises(ValueError, match=match):
        load(tmp_path, payload)


def test_af3_polymer_modifications_and_ligands_round_trip(tmp_path):
    payload = {
        "name": "chemistry",
        "modelSeeds": [11],
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": "AST",
                    "modifications": [{"ptmType": "SEP", "ptmPosition": 2}],
                }
            },
            {"rna": {"id": "R", "sequence": "AC", "unpairedMsa": ""}},
            {"dna": {"id": "D", "sequence": "AT"}},
            {"ligand": {"id": "L", "ccdCodes": ["ATP"]}},
            {"ligand": {"id": "S", "smiles": "CCO"}},
        ],
    }
    queries = load(tmp_path, payload)
    assert queries.queries["chemistry"].chains[0].non_canonical_residues == {2: "SEP"}
    paths = bundle.write_af3_query_sets(queries, tmp_path / "out")
    result = json.loads(paths["chemistry"].read_text())
    actual = result["sequences"]
    rna = actual[1]["rna"]
    assert bundle.read_text_auto(paths["chemistry"].parent / rna.pop("unpairedMsaPath")) == ">query\nAC\n"
    rna["unpairedMsa"] = ""
    assert actual == payload["sequences"]
    assert result["modelSeeds"] == [11]
    assert "queries" not in result


def test_af3_template_indices_convert_once_and_use_full_polymer_sequence(tmp_path):
    cif = """data_test
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
1 1 ALA
1 2 CYS
1 3 ASP
1 4 GLU
loop_
_atom_site.label_asym_id
_atom_site.label_seq_id
A 1
A 3
"""
    queries = load(
        tmp_path,
        protein(
            templates=[
                {"mmcif": cif, "queryIndices": [0, 2], "templateIndices": [1, 3]}
            ]
        ),
    )
    template = queries.queries["target"].chains[0].templates[0]
    assert template.query_indices == [1, 3]
    assert template.template_indices == [2, 4]
    paths = bundle.write_af3_query_sets(queries, tmp_path / "out")
    payload = json.loads(paths["target"].read_text())
    public = payload["sequences"][0]["protein"]["templates"][0]
    assert public["queryIndices"] == [0, 2]
    assert public["templateIndices"] == [1, 3]
    assert public["mmcifPath"].startswith("msas/")
    assert "chainId" not in public
    reread = bundle.load_af3_query_set(paths["target"], tmp_path / "reload")
    assert reread.queries["target"].chains[0].templates[0].template_indices == [2, 4]

    # Exercise the real native residue consumer, not only serializer symmetry.
    from biotite.structure import AtomArray

    from openfold3.core.data.primitives.structure.template import (
        TemplateCacheEntry,
        map_token_pos_to_template_residues,
    )

    query_atoms = AtomArray(4)
    query_atoms.res_id = np.array([1, 2, 3, 4])
    query_atoms.set_annotation("token_id", np.array([0, 1, 2, 3]))
    query_atoms.set_annotation("token_position", np.array([0, 1, 2, 3]))
    template_atoms = AtomArray(4)
    template_atoms.res_id = np.array([1, 2, 3, 4])
    template_atoms.res_name = np.array(["ALA", "CYS", "ASP", "GLU"])
    template_atoms.set_annotation("molecule_type_id", np.zeros(4, dtype=int))
    template_atoms.coord = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])
    entry = TemplateCacheEntry(
        index=0,
        release_date="1900-01-01",
        idx_map=np.column_stack([template.query_indices, template.template_indices]),
    )
    slices = []
    map_token_pos_to_template_residues(slices, entry, query_atoms, template_atoms)
    assert len(slices) == 1
    np.testing.assert_array_equal(slices[0].query_token_positions, [0, 2])
    np.testing.assert_array_equal(slices[0].atom_array.coord, [[2, 0, 0], [4, 0, 0]])

    from types import SimpleNamespace

    bundle.restore_prepared_templates(reread, tmp_path / "runtime", SimpleNamespace())
    chain = reread.queries["target"].chains[0]
    with np.load(chain.template_alignment_file_path, allow_pickle=True) as cache:
        entry = cache[chain.template_entry_chain_ids[0]].item()
    np.testing.assert_array_equal(entry["idx_map"], [[1, 2], [3, 4]])


@pytest.mark.parametrize("indices", [[True], [-1], [4], [0, 0], [2, 0]])
def test_public_template_mapping_rejects_invalid_or_nonmonotonic_indices(
    tmp_path, indices
):
    from pathlib import Path

    fixture = Path(__file__).parent / "test_data/mmcifs/1a8q.cif"
    single = bundle.extract_single_chain_mmcif(fixture, "A")
    with pytest.raises(ValueError, match="queryIndices"):
        load(
            tmp_path,
            protein(
                templates=[
                    {
                        "mmcif": single,
                        "queryIndices": indices,
                        "templateIndices": list(range(len(indices))),
                    }
                ]
            ),
        )


def test_data_prepared_rna_can_be_read_again(tmp_path):
    from types import SimpleNamespace

    from openfold3.projects.of3_all_atom.config.dataset_config_components import (
        MSASettings,
    )

    queries = load(
        tmp_path,
        {
            "name": "rna",
            "sequences": [
                {
                    "rna": {
                        "id": "R",
                        "sequence": "ACGU",
                        "unpairedMsa": ">q\nACGU\n>hit\nAC-U\n",
                    }
                }
            ],
        },
    )
    bundle.materialise_msas(queries, tmp_path / "prepared", MSASettings())
    bundle.materialise_templates(queries, tmp_path / "prepared", SimpleNamespace())
    path = bundle.write_af3_query_sets(queries, tmp_path / "out")["rna"]
    reread = bundle.load_af3_query_set(path, tmp_path / "reload")
    assert "AC-U" in read_text_auto(
        reread.queries["rna"].chains[0].main_msa_file_paths[0]
    )


@pytest.mark.parametrize("flag", ["use_msas", "use_paired_msas"])
def test_materialisation_preserves_declared_msa_when_use_disabled(tmp_path, flag):
    from openfold3.projects.of3_all_atom.config.dataset_config_components import (
        MSASettings,
    )

    queries = load(
        tmp_path,
        {
            **protein(
                unpairedMsa=">q\nACDE\n>hit\nAC-E\n",
                pairedMsa=">q\nACDE\n>pair\nAC-E\n",
            ),
            "openfold3": {flag: False},
        },
    )
    bundle.materialise_msas(queries, tmp_path / "prepared", MSASettings())
    path = bundle.write_af3_query_sets(queries, tmp_path / "out")["target"]
    payload = json.loads(path.read_text())
    assert payload["openfold3"][flag] is False
    saved = payload["sequences"][0]["protein"]
    assert "AC-E" in read_text_auto(path.parent / saved["unpairedMsaPath"])
    assert "AC-E" in read_text_auto(path.parent / saved["pairedMsaPath"])


def test_failed_snapshot_publication_restores_previous_files(tmp_path, monkeypatch):
    from openfold3.core.data import af3_input

    queries = load(tmp_path, protein(unpairedMsa=">q\nACDE\n"))
    path = bundle.write_af3_query_sets(queries, tmp_path / "out")["target"]
    before = {p: p.read_bytes() for p in path.parent.rglob("*") if p.is_file()}
    queries.queries["target"].chains[0].main_msa_file_paths[0].write_text(
        ">q\nACDE\n>new\nAC-E\n"
    )
    replace = af3_input.os.replace
    failed = False

    def fail_json_once(src, dst):
        nonlocal failed
        if dst == path and not failed:
            failed = True
            raise OSError("injected publication failure")
        return replace(src, dst)

    monkeypatch.setattr(af3_input.os, "replace", fail_json_once)
    with pytest.raises(OSError, match="injected"):
        bundle.write_af3_query_sets(queries, tmp_path / "out")
    assert {p: p.read_bytes() for p in path.parent.rglob("*") if p.is_file()} == before


def test_public_msa_channels_are_not_lost_under_native_source_whitelist(tmp_path):
    from openfold3.core.config.msa_pipeline_configs import (
        MsaSampleProcessorInputInference,
    )
    from openfold3.core.data.pipelines.sample_processing.msa import (
        MsaSampleProcessorInference,
    )
    from openfold3.projects.of3_all_atom.config.dataset_config_components import (
        MSASettings,
    )

    config = MSASettings(
        max_seq_counts={"uniref90_hits": 1},
        aln_order=["uniref90_hits"],
        paired_msa_order=[],
    )
    queries = load(tmp_path, protein(unpairedMsa=">q\nACDE\n>hit\nAxC-E\n"))
    assert hasattr(bundle, "configure_af3_msa_sources"), (
        "Public MSA role adapter missing"
    )
    bundle.configure_af3_msa_sources(config)
    features = MsaSampleProcessorInference(config)(
        MsaSampleProcessorInputInference.create_from_inference_query_entry(
            queries.queries["target"]
        )
    )
    np.testing.assert_allclose(features.chain_id_to_deletion_mean["A"], [0, 0.5, 0, 0])
    assert config.max_seq_counts["uniref90_hits"] == 1


def test_af3_snapshot_overwrites_resources_and_preserves_omitted_vs_empty(tmp_path):
    queries = load(
        tmp_path, protein(unpairedMsa=">q\nACDE\n", pairedMsa="", templates=[])
    )
    paths = bundle.write_af3_query_sets(queries, tmp_path / "out")
    saved = json.loads(paths["target"].read_text())["sequences"][0]["protein"]
    assert saved["pairedMsa"] == ""
    assert saved["templates"] == []
    source = queries.queries["target"].chains[0].main_msa_file_paths[0]
    source.write_text(">q\nACDE\n>new\nAC-E\n")
    bundle.write_af3_query_sets(queries, tmp_path / "out")
    assert "new" in read_text_auto(paths["target"].parent / saved["unpairedMsaPath"])
    paths = bundle.write_af3_query_sets(load(tmp_path, protein()), tmp_path / "omitted")
    saved = json.loads(paths["target"].read_text())["sequences"][0]["protein"]
    assert "unpairedMsa" not in saved and "unpairedMsaPath" not in saved
    assert "templates" not in saved


def test_public_predict_rejects_native_queries_before_runtime(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from openfold3 import run_openfold
    from openfold3.entry_points import validator

    source = tmp_path / "native.json"
    source.write_text('{"queries": {"job": {"chains": []}}}')

    def forbidden_config(**kwargs):
        raise RuntimeError("must reject before configuration")

    monkeypatch.setattr(validator, "InferenceExperimentConfig", forbidden_config)
    result = CliRunner().invoke(
        run_openfold.cli,
        [
            "predict",
            "--query-json",
            str(source),
            "-D",
            "true",
            "-P",
            "false",
        ],
    )
    assert isinstance(result.exception, ValueError), repr(result.exception)
    assert "AF3" in str(result.exception)


def test_runner_refreshes_data_config_after_seed_change(tmp_path):
    from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
    from openfold3.entry_points.validator import InferenceExperimentConfig
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )

    checkpoint = tmp_path / "dummy.ckpt"
    checkpoint.write_text("not loaded")
    runner = InferenceExperimentRunner(
        InferenceExperimentConfig(inference_ckpt_path=checkpoint)
    )
    runner.inference_query_set = InferenceQuerySet.model_validate(
        {"queries": {"job": {"chains": []}}}
    )
    before = runner.data_module_config
    runner.set_model_seeds([19])
    after = runner.data_module_config
    assert before is not after
    assert after.datasets[0].config.seeds == [19]


@pytest.mark.parametrize(
    "data,inference,write",
    [
        (True, False, True),
        (True, False, False),
        (True, True, True),
        (True, True, False),
        (False, True, True),
        (False, True, False),
        (False, True, None),
    ],
)
def test_predict_stage_write_contract_and_private_resource_lifetime(
    tmp_path, monkeypatch, data, inference, write
):
    from click.testing import CliRunner

    from openfold3 import run_openfold
    from openfold3.core.config.msa_pipeline_configs import (
        MsaSampleProcessorInputInference,
    )
    from openfold3.core.data.pipelines.sample_processing.msa import (
        MsaSampleProcessorInference,
    )
    from openfold3.entry_points.experiment_runner import InferenceExperimentRunner

    source = tmp_path / "input.json"
    source.write_text(
        json.dumps(
            protein(unpairedMsa=">q\nACDE\n>hit\nAxC-E\n", pairedMsa="", templates=[])
        )
    )
    checkpoint = tmp_path / "dummy.ckpt"
    checkpoint.write_text("not loaded")
    monkeypatch.setenv("OPENFOLD_CACHE", str(tmp_path / "empty-cache"))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("SLURM_TMPDIR", str(scratch))
    # Only the heavy model boundary is replaced; preparation, public IO and
    # native MSA feature construction still execute.
    consumed = []

    def consume(runner, query_set):
        chain = query_set.queries["target"].chains[0]
        consumed.extend(chain.main_msa_file_paths)
        features = MsaSampleProcessorInference(runner.dataset_config_kwargs.msa)(
            MsaSampleProcessorInputInference.create_from_inference_query_entry(
                query_set.queries["target"]
            )
        )
        np.testing.assert_allclose(
            features.chain_id_to_deletion_mean["A"], [0, 0.5, 0, 0]
        )

    monkeypatch.setattr(InferenceExperimentRunner, "setup", lambda self: None)
    monkeypatch.setattr(InferenceExperimentRunner, "run", consume)
    args = [
        "predict",
        "--query-json",
        str(source),
        "--output-dir",
        str(tmp_path / "out"),
        "--inference-ckpt-path",
        str(checkpoint),
        "-D",
        str(data),
        "-P",
        str(inference),
        "--use-msa-server",
        "false",
        "--use-templates",
        "false",
        "--use_tf32",
        "false",
    ]
    if write is not None:
        args += ["--write-input-json", str(write)]
    result = CliRunner().invoke(run_openfold.cli, args)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    job = tmp_path / "out/target"
    assert (job / "target_data.json").exists() == (write is True)
    assert (job / "msas").exists() == (write is True)
    assert not list(scratch.iterdir())
    if inference:
        assert consumed and all(not p.exists() for p in consumed)
