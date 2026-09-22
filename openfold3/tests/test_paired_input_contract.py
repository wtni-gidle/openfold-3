"""Public MSA states and native paired rows survive the preparation boundary."""

import json

import numpy as np
import pytest

from openfold3.core.data import prepared_bundle as bundle
from openfold3.core.data.io.compression import read_text_auto
from openfold3.projects.of3_all_atom.config.dataset_config_components import MSASettings
from openfold3.tests.test_prepared_msa_parity import _conditions, _snapshot


def _load(tmp_path, paired, unpaired="", name="input"):
    entities = []
    for chain, seq, text in zip(("A", "B"), ("ACDEFG", "HIKLMN"), paired, strict=True):
        body = dict(id=chain, sequence=seq, templates=[], unpairedMsa=unpaired)
        if text is not None:
            body["pairedMsa"] = text.replace("SEQ", seq)
        entities.append({"protein": body})
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(dict(name="job", sequences=entities)))
    return bundle.load_af3_query_set(path, tmp_path / f"{name}-private")


@pytest.mark.parametrize("with_query", [False, True])
@pytest.mark.parametrize("compress", [False, True])
def test_empty_main_and_native_paired_roundtrip(tmp_path, with_query, compress):
    # A local hit-only block must not lose its first hit; a server query must not
    # be stripped. Empty main must supply a real query, never the first hit.
    prefix = ">query\nSEQ\n" if with_query else ""
    paired = [prefix + ">first\nAxxCDEF-\n>second\n-CDEFG\n",
              prefix + ">first\nHIKLM-\n>second\n-IKLMN\n"]
    queries = _load(tmp_path, paired)
    config = MSASettings()
    before = _snapshot(queries, config)
    wanted = ([list("ACDEFG")] if with_query else []) + [list("ACDEF-"), list("-CDEFG")]
    np.testing.assert_array_equal(before["chain_id_to_paired_msa/A/msa"], wanted)
    np.testing.assert_array_equal(before["chain_id_to_query_seq/A/msa"], [list("ACDEFG")])
    np.testing.assert_array_equal(before["chain_id_to_deletion_mean/A"], np.zeros(6))
    assert np.all(before["chain_id_to_profile/A"].sum(axis=1) == 1)
    assert np.count_nonzero(before["chain_id_to_profile/A"]) == 6
    assert before["chain_id_to_paired_msa/A/deletion"][int(with_query), 1] == 2
    for cycle in range(2):
        bundle.materialise_msas(queries, tmp_path / f"prep{cycle}", config, compress=compress)
        public = bundle.write_af3_query_sets(queries, tmp_path / f"out{cycle}", compress=compress)["job"]
        moved = tmp_path / f"moved{cycle}"
        public.parent.rename(moved)
        queries = bundle.load_af3_query_set(moved / public.name, tmp_path / f"read{cycle}")
        after = _snapshot(queries, config)
        assert after.keys() == before.keys()
        for key in before:
            np.testing.assert_array_equal(after[key], before[key], err_msg=key)


@pytest.mark.parametrize("case", ["local_species", "local_identical", "prepaired"])
def test_native_pairing_publication_does_not_add_or_remove_rows(tmp_path, case):
    queries = _conditions(tmp_path, case)
    config = MSASettings()
    before = _snapshot(queries, config)
    bundle.materialise_msas(queries, tmp_path / "prep", config)
    public = bundle.write_af3_query_sets(queries, tmp_path / "out")["job"]
    payload = json.loads(public.read_text())
    text = read_text_auto(public.parent / payload["sequences"][0]["protein"]["pairedMsaPath"])
    assert "openfold3_context_only" not in text
    assert text.count(">") == 2  # two hits locally, query + hit for prepaired
    reread = bundle.load_af3_query_set(public, tmp_path / "read")
    after = _snapshot(reread, config)
    assert after.keys() == before.keys()
    for key in before:
        np.testing.assert_array_equal(after[key], before[key], err_msg=key)


@pytest.mark.parametrize("states", [
    (">query\nSEQ\n>hit\n------\n", ""),
    (">query\nSEQ\n>hit\n------\n", None),
    ("", None),
])
def test_mixed_paired_states_are_rejected(tmp_path, states):
    with pytest.raises(ValueError, match="paired.*state|paired.*together"):
        _load(tmp_path, states)


def test_paired_depth_mismatch_is_rejected_before_cropping(tmp_path):
    with pytest.raises(ValueError, match="paired.*(depth|row)"):
        _load(tmp_path, [">query\nSEQ\n>hit\n------\n", ">query\nSEQ\n"])


def test_explicit_empty_channels_mean_query_only_not_zero_features(tmp_path):
    queries = _load(tmp_path, ["", ""])
    config = MSASettings()
    features = _snapshot(queries, config)
    assert "chain_id_to_query_seq/A/msa" in features
    assert "chain_id_to_paired_msa/A/msa" not in features
    assert np.count_nonzero(features["chain_id_to_profile/A"]) == 6
    bundle.materialise_msas(queries, tmp_path / "prep", config)
    public = bundle.write_af3_query_sets(queries, tmp_path / "out")["job"]
    after = _snapshot(bundle.load_af3_query_set(public, tmp_path / "read"), config)
    assert features.keys() == after.keys()
    for key in features:
        np.testing.assert_array_equal(after[key], features[key], err_msg=key)


def test_old_synthetic_query_marker_requires_regeneration(tmp_path):
    text = ">query openfold3_context_only\nSEQ\n>hit\n------\n"
    with pytest.raises(ValueError, match="context.*(regenerate|regenerat)|regenerat.*context"):
        _load(tmp_path, [text, text], unpaired=None)


@pytest.mark.parametrize("disabled", [False, 0, "false"])
@pytest.mark.parametrize("use_msas", [False, True])
def test_public_input_rejects_disabled_main_with_query_only_guidance(
    tmp_path, disabled, use_msas,
):
    # Reject after native boolean normalization too: coercible values must not
    # bypass the guard and reach the broken native main-disabled branch.
    path = tmp_path / "disabled-main.json"
    path.write_text(json.dumps({
        "name": "job",
        "sequences": [{"protein": {
            "id": "A", "sequence": "ACDEFG", "unpairedMsa": "", "pairedMsa": "",
        }}],
        "openfold3": {"use_main_msas": disabled, "use_msas": use_msas},
    }))
    with pytest.raises(ValueError, match="use_main_msas.*unpairedMsa"):
        bundle.load_af3_query_set(path, tmp_path / "private")
