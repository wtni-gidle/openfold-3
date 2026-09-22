"""Prepared publication must not change the real native MSA consumers."""

import json

import numpy as np
import pytest
from biotite.structure import AtomArray

from openfold3.core.config.msa_pipeline_configs import MsaSampleProcessorInputInference
from openfold3.core.data import prepared_bundle as bundle
from openfold3.core.data.io.compression import read_text_auto
from openfold3.core.data.pipelines.featurization.msa import (
    MsaFeaturizerOF3,
    MsaFeaturizerOF3Config,
)
from openfold3.core.data.pipelines.sample_processing.msa import MsaSampleProcessorInference
from openfold3.projects.of3_all_atom.config.dataset_config_components import MSASettings
from openfold3.projects.of3_all_atom.config.inference_query_format import InferenceQuerySet


def _conditions(tmp_path, case):
    chains = []
    specs = [("protein", ["A", "C"], "ACDEFG"), ("protein", ["B"], "HIKLMN")]
    if case == "rna":
        specs.append(("rna", ["R"], "ACGUAC"))
    if case == "local_gap":
        specs.append(("protein", ["G"], "PQRSTV"))
    for number, (kind, ids, sequence) in enumerate(specs):
        chain = dict(molecule_type=kind, chain_ids=ids, sequence=sequence,
                     main_msa_file_paths=[], paired_msa_file_paths=[])
        if case == "partial_empty" and number == 1:
            chains.append(chain)
            continue
        directory = tmp_path / "original" / ids[0]
        directory.mkdir(parents=True)
        main = directory / "colabfold_main.a3m"
        main.write_text(f">query\n{sequence}\n>insertion\n{sequence[0]}gg{sequence[1:-1]}-\n")
        chain["main_msa_file_paths"] = [main]
        if case == "empty":
            main.write_text(f">query\n{sequence}\n")
            chains.append(chain)
            continue
        if case.startswith("local") and not (case == "local_gap" and number == 2):
            species = directory / "uniprot_hits.a3m"
            # Include a real hit identical to the query in one chain: it must not
            # be mistaken for a synthetic query and removed on publication.
            hit = sequence if case == "local_identical" and number == 0 else sequence[:-1] + "-"
            species.write_text(
                f">query\n{sequence}\n>tr|P12345|P12345_HUMAN/1-6\n{hit}\n"
                f">tr|P67890|P67890_MOUSE/1-6\n-{sequence[1:]}\n"
            )
            chain["main_msa_file_paths"].append(species)
        elif kind == "protein" and not case.startswith("local"):
            paired = directory / "colabfold_paired.a3m"
            paired.write_text(f">query\n{sequence}\n>hit\n{sequence[:-1]}-\n")
            chain["paired_msa_file_paths"] = [paired]
        chains.append(chain)
    return InferenceQuerySet(queries={"job": {
        "chains": chains, "use_paired_msas": case != "empty",
    }})


def _snapshot(query_set, config):
    query = query_set.queries["job"]
    # Native main-MSA subsampling is stochastic even without band subsampling.
    # Compare the same draw, without changing the native sampling setting.
    state = np.random.get_state()
    np.random.seed(13)
    try:
        collection = MsaSampleProcessorInference(config)(
            MsaSampleProcessorInputInference.create_from_inference_query_entry(query)
        )
    finally:
        np.random.set_state(state)
    ids, residues, entities = [], [], []
    for entity, chain in enumerate(query.chains):
        for chain_id in chain.chain_ids:
            ids.extend([chain_id] * len(chain.sequence))
            residues.extend(range(1, len(chain.sequence) + 1))
            entities.extend([entity] * len(chain.sequence))
    atoms = AtomArray(len(ids))
    atoms.chain_id = np.array(ids)
    atoms.res_id = np.array(residues)
    for key, values in (("token_id", np.arange(len(ids))),
                        ("token_position", np.arange(len(ids))),
                        ("entity_id", np.array(entities))):
        atoms.set_annotation(key, values)
    features = MsaFeaturizerOF3(MsaFeaturizerOF3Config(
        max_rows=config.max_rows, max_rows_paired=config.max_rows_paired,
        subsample_with_bands=False,
    ))(atoms, collection, len(ids))
    result = {f"feature/{key}": value.numpy().copy() for key, value in features.items()}
    for attr in ("chain_id_to_query_seq", "chain_id_to_main_msa", "chain_id_to_paired_msa",
                 "chain_id_to_profile", "chain_id_to_deletion_mean"):
        for chain_id, value in getattr(collection, attr).items():
            if hasattr(value, "msa"):
                result[f"{attr}/{chain_id}/msa"] = value.msa.copy()
                result[f"{attr}/{chain_id}/deletion"] = value.deletion_matrix.copy()
            else:
                result[f"{attr}/{chain_id}"] = value.copy()
    return result


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("disabled", [None, "use_paired_msas"])
@pytest.mark.parametrize("case", ["prepaired", "rna", "empty",
                                  "local_species", "local_identical", "local_gap"])
def test_publication_preserves_native_msa_features(tmp_path, case, compress, disabled):
    queries = _conditions(tmp_path, case)
    if disabled:
        setattr(queries.queries["job"], disabled, False)
    config = MSASettings()
    config.max_rows = 4
    config.max_rows_paired = 2
    config.max_seq_counts["colabfold_paired"] = 2
    before = _snapshot(queries, config)
    if case == "empty":
        assert before["feature/profile"].any()
    if case.startswith("local") and disabled != "use_paired_msas":
        # Real native species pairing has two hit rows, not an added query row.
        assert before["chain_id_to_paired_msa/A/msa"].shape == (2, 6)

    for cycle in range(3):
        bundle.materialise_msas(queries, tmp_path / f"materialised{cycle}", config,
                                compress=compress)
        public = bundle.write_af3_query_sets(queries, tmp_path / f"published{cycle}",
                                             compress=compress)["job"]
        payload = json.loads(public.read_text())
        paired_depths = []
        for entry in payload["sequences"]:
            body = next(iter(entry.values()))
            if "rna" in entry:
                assert "pairedMsaPath" not in body and "pairedMsa" not in body
            if case == "empty":
                assert body["pairedMsa"] == ""
                assert read_text_auto(public.parent / body["unpairedMsaPath"]) == (
                    f">query\n{body['sequence']}\n"
                )
            if "pairedMsaPath" in body:
                text = read_text_auto(public.parent / body["pairedMsaPath"])
                assert "openfold3_context_only" not in text
                paired_depths.append(sum(line.startswith(">") for line in text.splitlines()))
        # Public paired files must also remain row-aligned, including when a
        # genuine first hit matches the query in only one of the chains.
        assert len(set(paired_depths)) <= 1
        moved = tmp_path / f"moved{cycle}"
        public.parent.rename(moved)
        if cycle == 0 and (tmp_path / "original").exists():
            (tmp_path / "original").rename(tmp_path / "retired-original")
        queries = bundle.load_af3_query_set(moved / public.name, tmp_path / f"read{cycle}")
        after = _snapshot(queries, config)
        assert set(after) == set(before)
        for key in before:
            np.testing.assert_array_equal(after[key], before[key], err_msg=f"cycle {cycle}: {key}")


def test_legacy_context_query_requires_regeneration(tmp_path):
    from openfold3.core.data.io.sequence.msa import parse_msas_direct

    path = tmp_path / "job__A_pairedmsa.a3m"
    path.write_text(
        ">query openfold3_context_only\nACDEFG\n>first\nAxxCDEF-\n>second\n-CDEFG\n"
    )
    with pytest.raises(ValueError, match="regenerate"):
        parse_msas_direct([path], {"colabfold_paired": 2})


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("source", ["local_species", "merged_prepaired"])
def test_final_paired_pool_is_not_recropped_by_single_source_quota(tmp_path, source, compress):
    """A source quota must not truncate an already merged/local paired pool."""
    queries = _conditions(tmp_path, "local_species" if source == "local_species" else "prepaired")
    config = MSASettings()
    config.max_rows = 12
    config.max_rows_paired = 8
    config.max_seq_counts["colabfold_paired"] = 1 if source == "local_species" else 2
    expected_depth = 2 if source == "local_species" else 4
    if source == "merged_prepaired":
        config.max_seq_counts["other_paired"] = 2
        config.paired_msa_order.append("other_paired")
        for chain in queries.queries["job"].chains:
            other = chain.paired_msa_file_paths[0].with_name("other_paired.a3m")
            other.write_text(f">query\n{chain.sequence}\n>other\n-{chain.sequence[1:]}\n")
            chain.paired_msa_file_paths.append(other)
    before = _snapshot(queries, config)
    assert before["chain_id_to_paired_msa/A/msa"].shape == (expected_depth, 6)
    for cycle in range(3):
        bundle.materialise_msas(queries, tmp_path / f"materialized{cycle}", config,
                                compress=compress)
        public = bundle.write_af3_query_sets(queries, tmp_path / f"public{cycle}",
                                             compress=compress)["job"]
        moved = tmp_path / f"moved{cycle}"
        public.parent.rename(moved)
        if cycle == 0:
            (tmp_path / "original").rename(tmp_path / "retired-original")
        queries = bundle.load_af3_query_set(moved / public.name, tmp_path / f"read{cycle}")
        after = _snapshot(queries, config)
        assert set(after) == set(before)
        for key in before:
            np.testing.assert_array_equal(after[key], before[key], err_msg=f"cycle {cycle}: {key}")
    # Users can still lower the inference paired budget deliberately.
    config.max_rows_paired = 1
    limited = _snapshot(queries, config)
    assert limited["chain_id_to_paired_msa/A/msa"].shape == (1, 6)
    np.testing.assert_array_equal(limited["feature/num_paired_seqs"], [2])


@pytest.mark.parametrize("canonical", [False, True])
def test_raw_paired_source_keeps_its_quota_but_public_pool_does_not(tmp_path, canonical):
    from openfold3.core.data.io.sequence.msa import parse_msas_direct

    path = tmp_path / ("job__A_pairedmsa.a3m" if canonical else "colabfold_paired.a3m")
    path.write_text(">query\nACDEFG\n>hit\nAxxCDEF-\n>last\n-CDEFG\n")
    parsed = parse_msas_direct([path], {"colabfold_paired": 2})["colabfold_paired"]
    expected = [list("ACDEFG"), list("ACDEF-"), list("-CDEFG")]
    np.testing.assert_array_equal(parsed.msa, expected if canonical else expected[:2])
    assert parsed.deletion_matrix[1, 1] == 2


@pytest.mark.parametrize("channel,text", [
    ("unpairedMsa", ">query openfold3_context_only\nACDEFG\n>hit\nACDEF-\n"),
    ("pairedMsa", ">query openfold3_context_only\nACDEFG\n"),
    ("pairedMsa", ">query openfold3_context_only\nAxxCDEFG\n>hit\nACDEF-\n"),
    ("pairedMsa", ">query openfold3_context_only\nACDEFGxx\n>hit\nACDEF-\n"),
    ("pairedMsa", ">query openfold3_context_only\nACDEFG\n>query openfold3_context_only\nACDEFG\n>hit\nACDEF-\n"),
    ("pairedMsa", ">query\nACDEFG\n>query openfold3_context_only\nACDEF-\n"),
    ("unpairedMsa", ">query\nACDEFG\n>query openfold3_context_only\nACDEF-\n"),
])
def test_invalid_context_query_is_rejected(tmp_path, channel, text):
    source = tmp_path / "invalid.json"
    source.write_text(json.dumps({"name": "job", "sequences": [
        {"protein": {"id": "A", "sequence": "ACDEFG", channel: text}},
    ]}))
    with pytest.raises(ValueError, match="context"):
        bundle.load_af3_query_set(source, tmp_path / "read")
