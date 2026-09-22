"""Reproductions for independent review findings, without network or models."""

import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from biotite.structure.io import pdbx
from click.testing import CliRunner

from openfold3.core.data import prepared_bundle as bundle
from openfold3.core.data.primitives.structure.template import (
    parse_template_structure,
    sample_templates,
)


def _source(tmp_path, *, chain_name="A", duplicate=False, entry_id="custom_hit"):
    fixture = Path(__file__).parent / "test_data/mmcifs/1a8q.cif"
    text = bundle.extract_single_chain_mmcif(fixture, "A")
    if chain_name != "A":
        cif = pdbx.CIFFile.read(StringIO(text))
        for category_name, column_name in (
            ("atom_site", "label_asym_id"),
            ("struct_asym", "id"),
            ("pdbx_poly_seq_scheme", "asym_id"),
        ):
            category = cif.block[category_name]
            values = category[column_name].as_array().astype(object)
            values[values == "A"] = chain_name
            category[column_name] = pdbx.CIFColumn(values.tolist())
        stream = StringIO()
        cif.write(stream)
        text = stream.getvalue()
    templates = [
        {
            "mmcif": text,
            "queryIndices": [0, 1],
            "templateIndices": [1, 2],
            "openfold3": {"entryId": entry_id, "sourceIndex": 4},
        }
    ]
    if duplicate:
        templates += [
            {
                "mmcif": text,
                "queryIndices": [1, 2],
                "templateIndices": [3, 4],
                "openfold3": {"entryId": entry_id, "sourceIndex": 9},
            }
        ]
    source = tmp_path / "input.json"
    source.write_text(
        json.dumps(
            {
                "name": "job",
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": "ACDE",
                            "templates": templates,
                        }
                    }
                ],
            }
        )
    )
    return source, fixture


def _restore(tmp_path, source):
    query_set = bundle.load_af3_query_set(source, tmp_path / "input")
    config = SimpleNamespace()
    bundle.restore_prepared_templates(query_set, tmp_path / "runtime", config)
    chain = query_set.queries["job"].chains[0]
    assembly = {
        "A": {
            "template_ids": chain.template_entry_chain_ids,
            "cache_entry_file_path": chain.template_alignment_file_path,
        }
    }
    return chain, assembly, config


@pytest.mark.parametrize("entry_id,chain_name", [("custom_hit", "A"), ("entry", "A_1")])
def test_runtime_template_identity_reaches_native_cif_consumer(
    tmp_path, biotite_ccd_wrapper, entry_id, chain_name
):
    source, fixture = _source(tmp_path, entry_id=entry_id, chain_name=chain_name)
    chain, _, config = _restore(tmp_path, source)
    identity = chain.template_entry_chain_ids[0]
    with np.load(chain.template_alignment_file_path, allow_pickle=True) as cache:
        cif_path = Path(cache[identity].item()["cif_path"])
    actual = parse_template_structure(
        config.structure_directory,
        None,
        identity,
        "cif",
        biotite_ccd_wrapper,
        cif_path=cif_path,
    )
    expected = parse_template_structure(
        fixture.parent, None, "1a8q_A", "cif", biotite_ccd_wrapper, cif_path=fixture
    )
    assert actual is not None and len(actual) > 100
    np.testing.assert_allclose(actual.coord, expected.coord, equal_nan=True)
    np.testing.assert_array_equal(actual.res_id, expected.res_id)


def test_same_provenance_different_mappings_survive_native_sampling(tmp_path):
    source, _ = _source(tmp_path, duplicate=True, entry_id="shared")
    chain, assembly, config = _restore(tmp_path, source)

    def sample(n):
        return sample_templates(
            assembly, config.cache_directory, n, True, "A", None, "cif"
        )

    first = list(sample(1).values())
    assert len(first) == 1 and first[0].index == 4
    np.testing.assert_array_equal(first[0].idx_map, [[1, 2], [2, 3]])
    both = list(sample(2).values())
    assert len(both) == 2
    assert [t.index for t in both] == [4, 9]
    np.testing.assert_array_equal(both[1].idx_map, [[2, 4], [3, 5]])
    assert len(set(chain.template_entry_chain_ids)) == 2


def test_identical_native_duplicate_hits_keep_native_collapse(tmp_path):
    source, _ = _source(tmp_path, entry_id="same")
    payload = json.loads(source.read_text())
    templates = payload["sequences"][0]["protein"]["templates"]
    templates.append(templates[0].copy())
    source.write_text(json.dumps(payload))
    _, assembly, config = _restore(tmp_path, source)
    selected = sample_templates(
        assembly, config.cache_directory, 2, True, "A", None, "cif"
    )
    assert len(selected) == 1


@pytest.mark.parametrize("fail", [False, True])
def test_default_template_artifacts_are_private_and_cleaned(
    tmp_path, monkeypatch, fail
):
    from openfold3 import run_openfold
    from openfold3.core.data.pipelines.preprocessing.template import (
        TemplatePreprocessor,
    )

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("SLURM_TMPDIR", str(scratch))
    monkeypatch.setenv("OPENFOLD_CACHE", str(tmp_path / "cache"))
    source = tmp_path / "input.json"
    source.write_text(
        '{"name":"job","sequences":[{"protein":{"id":"A","sequence":"ACDE","templates":[],"unpairedMsa":""}}]}'
    )
    config = tmp_path / "runner.yml"
    config.write_text(
        "template_preprocessor_settings:\n  create_precache: true\n  preparse_structures: true\n  create_logs: true\n"
    )
    observed = []

    def offline_preprocess(processor):
        # The slow native search/download is simulated; real config construction
        # and path validation run. Never write outside this test's scratch.
        for field in (
            "structure_directory",
            "cache_directory",
            "log_directory",
            "precache_directory",
            "structure_array_directory",
        ):
            path = Path(getattr(processor, field))
            observed.append(path)
            assert path.is_relative_to(scratch), (field, path)
            assert path.is_dir()
            (path / "simulated-artifact").write_text("temporary")
        if fail:
            raise RuntimeError("simulated preparation failure")

    monkeypatch.setattr(TemplatePreprocessor, "__call__", offline_preprocess)
    result = CliRunner().invoke(
        run_openfold.cli,
        [
            "predict",
            "--query-json",
            str(source),
            "--runner-yaml",
            str(config),
            "--output-dir",
            str(tmp_path / "out"),
            "-D",
            "true",
            "-P",
            "false",
            "-J",
            "false",
            "--use-msa-server",
            "false",
        ],
    )
    if fail:
        assert isinstance(result.exception, RuntimeError), repr(result.exception)
    else:
        assert result.exit_code == 0, (result.output, repr(result.exception))
    assert len(observed) == 5
    assert all(not p.exists() for p in observed)
    assert not list(scratch.iterdir())
