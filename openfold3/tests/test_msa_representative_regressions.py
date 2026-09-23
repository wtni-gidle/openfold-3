"""Real input/parser/publication regressions for wrapper MSA identities."""

import json

import numpy as np
import pytest

from openfold3.core.config.msa_pipeline_configs import (
    MsaChainDataInference,
    MsaSampleProcessorInputInference,
)
from openfold3.core.data import prepared_bundle as bundle
from openfold3.core.data.io.sequence.msa import MsaSampleParserInference
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.projects.of3_all_atom.config.dataset_config_components import MSASettings
from openfold3.tests.test_prepared_msa_parity import _snapshot


def _load(tmp_path, sequences):
    source = tmp_path / "input.json"
    source.write_text(json.dumps({"name": "job", "sequences": sequences}))
    return bundle.load_af3_query_set(source, tmp_path / "read")


def _protein(ids, main=">query\nACG\n>hit\nA-G\n", paired=""):
    return {"protein": {"id": ids, "sequence": "ACG", "unpairedMsa": main,
                        "pairedMsa": paired, "templates": []}}


def _parse(queries, config):
    return MsaSampleParserInference(config)(
        MsaSampleProcessorInputInference.create_from_inference_query_entry(
            queries.queries["job"]
        )
    )


def test_protein_ccd_ligand_survives_msa_preparation(tmp_path):
    queries = _load(tmp_path, [_protein("A"),
                               {"ligand": {"id": "L", "ccdCodes": ["ATP"]}}])
    processor_input = MsaSampleProcessorInputInference.create_from_inference_query_entry(
        queries.queries["job"]
    )
    assert processor_input.msa_chain_data["L"].sequence is None
    collection = _parse(queries, MSASettings())
    assert set(collection.chain_id_to_rep_id) == {"A"}
    bundle.materialise_msas(queries, tmp_path / "materialized", MSASettings(), compress=True)
    public = bundle.write_af3_query_sets(queries, tmp_path / "public", compress=True)["job"]
    restored = bundle.load_af3_query_set(public, tmp_path / "restored")
    assert restored.queries["job"].chains[1].ccd_codes == ["ATP"]
    assert set(_parse(restored, MSASettings()).chain_id_to_rep_id) == {"A"}


@pytest.mark.parametrize("kind", [MoleculeType.PROTEIN, MoleculeType.RNA, MoleculeType.DNA])
@pytest.mark.parametrize("sequence", [None, ""])
def test_polymer_msa_input_requires_nonempty_sequence(kind, sequence):
    with pytest.raises(ValueError, match="sequence"):
        MsaChainDataInference(molecule_type=kind, sequence=sequence)


@pytest.mark.parametrize("case", ["mixed_types", "different_main", "different_paired", "shared_copies"])
@pytest.mark.parametrize("reverse", [False, True])
def test_representatives_preserve_conditions_through_publication(tmp_path, case, reverse):
    first = _protein(["A", "C"] if case == "shared_copies" else "A")
    second = _protein("B", main=">query\nACG\n>hit\nAC-\n")
    if case == "mixed_types":
        second = {"rna": {"id": "B", "sequence": "ACG",
                          "unpairedMsa": ">query\nACG\n>rna_hit\nAC-\n"}}
    if case == "different_paired":
        first["protein"]["pairedMsa"] = ">hit\nA-G\n>other\n-CG\n"
        second["protein"]["unpairedMsa"] = first["protein"]["unpairedMsa"]
        second["protein"]["pairedMsa"] = ">hit\nAC-\n>other\n-CG\n"
    sequences = [first] if case == "shared_copies" else [first, second]
    queries = _load(tmp_path, list(reversed(sequences)) if reverse else sequences)
    if case == "different_paired":
        # Native queries may share the main source while supplying independent
        # paired rows; both channels must participate in representative identity.
        chains = queries.queries["job"].chains
        chains[1].main_msa_file_paths = chains[0].main_msa_file_paths
    config = MSASettings()
    config.max_rows = 8
    config.max_rows_paired = 4
    before = None
    for cycle in range(3):
        collection = _parse(queries, config)
        reps = collection.chain_id_to_rep_id
        if case == "shared_copies":
            assert reps["A"] == reps["C"]
            assert len(collection.rep_id_to_main_msa) == 1
        else:
            assert reps["A"] != reps["B"]
            assert collection.rep_id_to_mol_type[reps["A"]] == MoleculeType.PROTEIN
            assert collection.rep_id_to_mol_type[reps["B"]] == (
                MoleculeType.RNA if case == "mixed_types" else MoleculeType.PROTEIN
            )
            for chain_id, hit in (("A", "A-G"), ("B", "A-G" if case == "different_paired" else "AC-")):
                msa = collection.rep_id_to_main_msa[reps[chain_id]]["colabfold_main"]
                np.testing.assert_array_equal(msa.msa, [list("ACG"), list(hit)])
            if case == "different_paired":
                for chain_id, hit in (("A", "A-G"), ("B", "AC-")):
                    paired = collection.rep_id_to_paired_msa[reps[chain_id]]["colabfold_paired"]
                    np.testing.assert_array_equal(paired.msa, [list(hit), list("-CG")])
        snapshot = _snapshot(queries, config)
        if before is None:
            before = snapshot
        else:
            assert snapshot.keys() == before.keys()
            for key in before:
                np.testing.assert_array_equal(snapshot[key], before[key], err_msg=key)
        bundle.materialise_msas(queries, tmp_path / f"materialized{cycle}", config, compress=True)
        public = bundle.write_af3_query_sets(queries, tmp_path / f"public{cycle}", compress=True)["job"]
        queries = bundle.load_af3_query_set(public, tmp_path / f"restored{cycle}")


@pytest.mark.parametrize("boundary", ["materialise", "publish", "read", "roundtrip"])
def test_shared_separate_entities_keep_representative_and_features(tmp_path, boundary):
    """Explicit source sharing survives each wrapper hand-off independently."""
    paired = ">hit\nA-G\n>other\n-CG\n"
    queries = _load(tmp_path, [_protein("A", paired=paired),
                               _protein("B", paired=paired)])
    first, second = queries.queries["job"].chains
    # Use real parsed queries with the exact same two native source paths.
    # Separate files with equal bytes deliberately do not imply sharing.
    second.main_msa_file_paths = first.main_msa_file_paths
    second.paired_msa_file_paths = first.paired_msa_file_paths
    config = MSASettings()
    config.max_rows = 8
    config.max_rows_paired = 4
    expected = _snapshot(queries, config)

    def assert_shared(candidate):
        collection = _parse(candidate, config)
        assert collection.chain_id_to_rep_id["A"] == collection.chain_id_to_rep_id["B"]
        assert len(collection.rep_id_to_main_msa) == 1
        assert len(collection.rep_id_to_paired_msa) == 1
        actual = _snapshot(candidate, config)
        assert actual.keys() == expected.keys()
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)

    assert_shared(queries)
    for cycle in range(2 if boundary == "roundtrip" else 1):
        if boundary in {"materialise", "roundtrip"}:
            bundle.materialise_msas(queries, tmp_path / f"materialized{cycle}", config, compress=True)
            assert_shared(queries)
        if boundary != "materialise":
            public = bundle.write_af3_query_sets(queries, tmp_path / f"public{cycle}", compress=True)["job"]
            payload = json.loads(public.read_text())
            a, b = [entry["protein"] for entry in payload["sequences"]]
            for channel in ("unpairedMsaPath", "pairedMsaPath"):
                if boundary == "read":
                    # Isolate the reader from the publisher: shared references
                    # are already expressible with the existing public schema.
                    b[channel] = a[channel]
                else:
                    assert a[channel] == b[channel]
            if boundary == "read":
                public.write_text(json.dumps(payload))
            queries = bundle.load_af3_query_set(public, tmp_path / f"restored{cycle}")
            assert_shared(queries)
