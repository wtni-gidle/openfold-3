# EnsembleFold wrapper stage status — 2026-09-24

The public input remains AF3-style JSON; native internals retain their own
scientific processing. Defaults: `write_input_json=true`,
`compress_fold_input=false`, and detailed JSON unless an explicit existing
format setting selects otherwise. Explicit conflicting old/new format options
are rejected. JSON rounding and NPZ dtype behavior remain distinct.

Included fixes: permit ligand sequence=None; distinguish MSA representatives by
molecule type, sequence and source while preserving explicitly shared sources.
Earlier-stage CPU/offline wrapper regression: 235 tests; setup/training/GPU work is excluded.
See `docs/ENSEMBLEFOLD_INPUT.md` for prepared data and paired-row semantics.

The proposed mandatory template declaration was cancelled; omission/null still
allows no-template inference.

Follow-up: partial prediction/write failures and collective timeouts now
propagate nonzero exit. Configured runner YAML values have priority over CLI
values in both shell and Python entry points, with explicit runner YAML above
cached runner YAML. Quoted false values and cross-file format/checkpoint aliases
are covered. Final stat offline CPU regression: 262 passed, 14 deselected,
15 existing dependency warnings (job 92161); no GPU or real OF3 multi-rank run.
Tests and review follow-up are recorded in the project audit.

Final verification (2026-09-24): 262 CPU tests passed again (job 93947,
14 deselected). A real single-GPU smoke in allocation 91413 passed using
the existing of3-ob-2025-06-30-174k checkpoint, one 20-residue protein,
query-only MSA, no templates, seed 7, one sample, one recycle and four
diffusion steps. Canonical CIF, summary JSON and full-data JSON were checked.
This is not default-schedule quality, scientific parity, or real multi-rank
failure validation. The retained all-failed guard overlaps the broader failure
check; final review considered it harmless defensive redundancy.

Remaining limitations: local database search is not connected to the wrapper
(server-off uses supplied MSA or query-only), and individual atomic writes do not
guarantee transactional sample replacement. Those changes were not authorized.
