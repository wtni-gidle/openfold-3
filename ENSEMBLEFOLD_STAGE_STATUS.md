# EnsembleFold wrapper stage status — 2026-09-23

The public input remains AF3-style JSON; native internals retain their own
scientific processing. Defaults: `write_input_json=true`,
`compress_fold_input=false`, and detailed JSON unless an explicit existing
format setting selects otherwise. Explicit conflicting old/new format options
are rejected. JSON rounding and NPZ dtype behavior remain distinct.

Included fixes: permit ligand sequence=None; distinguish MSA representatives by
molecule type, sequence and source while preserving explicitly shared sources.
CPU/offline wrapper regression: 235 tests; setup/training/GPU work is excluded.
See `docs/ENSEMBLEFOLD_INPUT.md` for prepared data and paired-row semantics.

The proposed mandatory template declaration was cancelled; omission/null still
allows no-template inference. Known remaining limitations: shell defaults can
override YAML template/server/skip values, local database search is not connected
to the wrapper (server-off uses supplied MSA or query-only), partial prediction
failures can exit successfully, and individual atomic writes do not guarantee
transactional sample replacement. This stage does not fix those issues.
