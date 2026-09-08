# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import json
from types import SimpleNamespace

import numpy as np

from openfold3.core.config.msa_pipeline_configs import (
    MsaSampleProcessorInputInference,
)
from openfold3.core.data.io.compression import read_text_auto, write_zstd_text
from openfold3.core.data.io.sequence.msa import parse_a3m, parse_msas_direct
from openfold3.core.data.pipelines.sample_processing.msa import (
    MsaSampleProcessorInference,
)
from openfold3.core.data.prepared_bundle import (
    PreparedTemplate,
    PreparedTemplateSet,
    clear_template_inputs,
    materialise_msas,
    materialise_templates,
    msa_array_to_a3m,
    restore_prepared_templates,
    write_prepared_query_sets,
)
from openfold3.projects.of3_all_atom.config.dataset_config_components import (
    MSASettings,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)


def test_a3m_round_trip_preserves_alignment_and_deletion_matrix(tmp_path):
    original = parse_a3m(">query\nACDE\n>hit\nAxxC-E\n")
    path = tmp_path / "target__A_unpairedmsa.a3m.zst"
    write_zstd_text(path, msa_array_to_a3m(original))

    reparsed = parse_msas_direct([path])["colabfold_main"]

    np.testing.assert_array_equal(reparsed.msa, original.msa)
    np.testing.assert_array_equal(reparsed.deletion_matrix, original.deletion_matrix)
    assert path.read_bytes().startswith(b"\x28\xb5\x2f\xfd")
    assert read_text_auto(path).startswith(">query\n")


def test_prepared_query_paths_are_relative_and_flags_are_preserved(tmp_path):
    msa_path = tmp_path / "target" / "msas" / "target__A_unpairedmsa.a3m"
    msa_path.parent.mkdir(parents=True)
    msa_path.write_text(">query\nACDE\n")
    query_set = InferenceQuerySet.model_validate(
        {
            "queries": {
                "target": {
                    "use_msas": True,
                    "use_main_msas": False,
                    "use_paired_msas": False,
                    "chains": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": ["A"],
                            "sequence": "ACDE",
                            "main_msa_file_paths": [msa_path],
                            "paired_msa_file_paths": [],
                        }
                    ],
                }
            }
        }
    )

    output_path = write_prepared_query_sets(query_set, tmp_path)["target"]
    payload = json.loads(output_path.read_text())
    query = payload["queries"]["target"]

    assert query["use_msas"] is True
    assert query["use_main_msas"] is False
    assert query["use_paired_msas"] is False
    assert query["chains"][0]["main_msa_file_paths"] == [
        "msas/target__A_unpairedmsa.a3m"
    ]

    relocated = InferenceQuerySet.from_json(output_path)
    assert relocated.queries["target"].chains[0].main_msa_file_paths == [
        msa_path.resolve()
    ]


def test_materialised_msa_bundle_preserves_native_processed_features(tmp_path):
    chains = []
    for chain_id, sequence, hit in (("A", "ACDE", "AC-E"), ("B", "FGHI", "FG-I")):
        source_dir = tmp_path / "raw" / chain_id
        source_dir.mkdir(parents=True)
        main_path = source_dir / "colabfold_main.a3m"
        paired_path = source_dir / "colabfold_paired.a3m"
        main_path.write_text(f">query\n{sequence}\n>hit\n{hit}\n")
        paired_path.write_text(f">pair\n{sequence}\n")
        chains.append(
            {
                "molecule_type": "protein",
                "chain_ids": [chain_id],
                "sequence": sequence,
                "main_msa_file_paths": [main_path],
                "paired_msa_file_paths": [paired_path],
            }
        )

    query_set = InferenceQuerySet.model_validate(
        {"queries": {"target": {"chains": chains}}}
    )
    query = query_set.queries["target"]
    config = MSASettings()

    original_input = MsaSampleProcessorInputInference.create_from_inference_query_entry(
        query
    )
    np.random.seed(123)
    original = MsaSampleProcessorInference(config)(original_input)

    materialise_msas(query_set, tmp_path / "out", config, compress=True)
    prepared_input = MsaSampleProcessorInputInference.create_from_inference_query_entry(
        query
    )
    np.random.seed(123)
    prepared = MsaSampleProcessorInference(config)(prepared_input)

    for field in (
        "chain_id_to_query_seq",
        "chain_id_to_paired_msa",
        "chain_id_to_main_msa",
        "chain_id_to_profile",
        "chain_id_to_deletion_mean",
    ):
        expected = getattr(original, field)
        actual = getattr(prepared, field)
        assert expected.keys() == actual.keys()
        for chain_id in expected:
            if hasattr(expected[chain_id], "msa"):
                np.testing.assert_array_equal(
                    expected[chain_id].msa, actual[chain_id].msa
                )
                np.testing.assert_array_equal(
                    expected[chain_id].deletion_matrix,
                    actual[chain_id].deletion_matrix,
                )
            else:
                np.testing.assert_allclose(expected[chain_id], actual[chain_id])


def test_template_sidecar_owns_relative_mmcif_paths(tmp_path):
    cif_path = tmp_path / "template.cif"
    cif_path.write_text("data_template\n")
    sidecar_path = tmp_path / "target__A_templates.json"
    template_set = PreparedTemplateSet(
        templates=[
            PreparedTemplate(
                entry_id="1abc",
                chain_id="A",
                mmcif_path=cif_path,
                query_indices=[0, 1],
                template_indices=[2, 3],
                release_date="2020-01-01",
                source_index=0,
            )
        ]
    )

    template_set.write_json(sidecar_path)
    payload = json.loads(sidecar_path.read_text())

    assert payload["templates"][0]["mmcif_path"] == "template.cif"
    restored = PreparedTemplateSet.from_json(sidecar_path)
    assert restored.templates[0].mmcif_path == cif_path.resolve()


def test_materialised_templates_reuse_mmcif_for_duplicate_occurrences(tmp_path):
    structure_directory = tmp_path / "structures"
    structure_directory.mkdir()
    cif_path = structure_directory / "1abc.cif"
    cif_path.write_text("data_template\n")
    cache_path = tmp_path / "template_cache.npz"
    cache_entry = {
        "index": 4,
        "release_date": "2020-01-01",
        "idx_map": np.asarray([[0, 2], [1, 3]]),
    }
    np.savez_compressed(cache_path, **{"1abc_A": cache_entry})
    query_set = InferenceQuerySet.model_validate(
        {
            "queries": {
                "target": {
                    "chains": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": ["A"],
                            "sequence": "ACDE",
                            "template_alignment_file_path": cache_path,
                            "template_entry_chain_ids": ["1abc_A", "1abc_A"],
                        }
                    ]
                }
            }
        }
    )
    config = SimpleNamespace(
        structure_file_format="cif", structure_directory=structure_directory
    )

    materialise_templates(query_set, tmp_path / "output", config, compress=True)

    chain = query_set.queries["target"].chains[0]
    template_set = PreparedTemplateSet.from_json(chain.prepared_template_file_path)
    assert len(template_set.templates) == 2
    assert template_set.templates[0].mmcif_path == template_set.templates[1].mmcif_path
    assert len(list((tmp_path / "output/target/msas").glob("*.cif.zst"))) == 1


def test_restore_prepared_templates_rebuilds_native_runtime_cache(tmp_path):
    cif_path = tmp_path / "template.cif"
    cif_path.write_text("data_template\n")
    sidecar_path = tmp_path / "target__A_templates.json"
    PreparedTemplateSet(
        templates=[
            PreparedTemplate(
                entry_id="1abc",
                chain_id="A",
                mmcif_path=cif_path,
                query_indices=[0, 1],
                template_indices=[2, 3],
                release_date="2020-01-01",
                source_index=4,
            )
        ]
    ).write_json(sidecar_path)
    query_set = InferenceQuerySet.model_validate(
        {
            "queries": {
                "target": {
                    "chains": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": ["A"],
                            "sequence": "ACDE",
                            "prepared_template_file_path": sidecar_path,
                        }
                    ]
                }
            }
        }
    )
    config = SimpleNamespace()

    restore_prepared_templates(query_set, tmp_path / "runtime", config)

    chain = query_set.queries["target"].chains[0]
    assert chain.prepared_template_file_path is None
    assert chain.template_entry_chain_ids == ["1abc_A"]
    with np.load(chain.template_alignment_file_path, allow_pickle=True) as cache:
        entry = cache["1abc_A"].item()
    assert entry["index"] == 4
    np.testing.assert_array_equal(entry["idx_map"], [[0, 2], [1, 3]])
    assert (config.structure_directory / "1abc.cif").read_text() == "data_template\n"


def test_clear_template_inputs_disables_prepared_and_raw_sources(tmp_path):
    cif_path = tmp_path / "template.cif"
    cif_path.write_text("data_template\n")
    query_set = InferenceQuerySet.model_validate(
        {
            "queries": {
                "prepared": {
                    "chains": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": ["A"],
                            "sequence": "ACDE",
                            "prepared_template_file_path": cif_path,
                        }
                    ]
                },
                "raw": {
                    "chains": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": ["B"],
                            "sequence": "FGHI",
                            "template_cif_paths": [cif_path],
                            "template_cif_chain_ids": ["A"],
                        }
                    ]
                },
            }
        }
    )

    clear_template_inputs(query_set)

    for query in query_set.queries.values():
        chain = query.chains[0]
        assert chain.prepared_template_file_path is None
        assert chain.template_alignment_file_path is None
        assert chain.template_entry_chain_ids == []
        assert chain.template_cif_paths is None
        assert chain.template_cif_chain_ids is None
