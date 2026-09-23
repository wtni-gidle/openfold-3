# EnsembleFold wrapper: AF3-style public input

`run_openfold.sh` and `run_openfold predict` accept AF3-style JSON, not the native
OpenFold3 `queries/chains` JSON. Native standalone tools such as
`align-msa-server` and the underlying Python query classes retain their native
format. Older wrapper bundles must be regenerated; their template index origin
is not guessed or silently migrated.

```json
{
  "name": "example",
  "modelSeeds": [42],
  "sequences": [
    {"protein": {
      "id": "A",
      "sequence": "ACDE",
      "unpairedMsaPath": "inputs/custom_alignment.a3m",
      "pairedMsa": "",
      "templates": []
    }}
  ],
  "dialect": "alphafold3",
  "version": 4
}
```

Supported molecules: `protein`, `rna`, `dna`, and `ligand` (`ccdCodes` or
`smiles`, exclusively). An entity `id` can be a string or a list for copies.
Protein modifications use `ptmType`/`ptmPosition`; nucleic-acid modifications use
`modificationType`/`basePosition`. Modification positions are one-based, as in
AF3. `description` and the method's optional `cyclic` polymer control are retained.
Unknown fields are errors, not ignored. In particular AF3 `bondedAtomPairs` and
`userCCD` are not automatically translated into OpenFold3 atom/CCD definitions.

For batch execution use a list of these job objects with the same `modelSeeds`.
Use separate invocations for differing seed lists. Existing CLI seed overrides
still apply. Output-name collisions are rejected before runtime setup.

## MSA and templates

- Protein: `unpairedMsa` or `unpairedMsaPath`, `pairedMsa` or `pairedMsaPath`.
- RNA: `unpairedMsa` or `unpairedMsaPath`; no invented RNA paired/search service.
- DNA and ligands: no MSA fields.
- An inline string and path cannot both be supplied. Paths are relative to the
  declaring JSON, not the current working directory. Readers detect plain/gzip/
  xz/zstd content. An arbitrary external A3M filename is accepted.
- Missing/null MSA allows the native preparation route to fill it. Explicit
  `unpairedMsa: ""` is staged as a real query-only alignment before either stage;
  it does not request search or mean all-zero sequence statistics. Explicit
  `pairedMsa: ""` means no paired hits; an independent query is still retained.
  All protein entities must have the same paired state: all unspecified, all
  explicitly empty, or all supplied. Mixing these states is rejected.
  Supplied paired files must have equal row counts before any native cropping.
- Protein `templates` entries contain `mmcif` or `mmcifPath`, `queryIndices`, and
  `templateIndices`. CIF must contain one template chain and full polymer sequence
  metadata. Both index arrays are zero-based positions in full sequences, not
  coordinate-row indices; unresolved residues are not removed from numbering.
  Both arrays must be strictly increasing: native OpenFold3's residue consumer
  does not implement arbitrary reordered/circular residue correspondence.
- Explicit templates are not searched, re-aligned, or re-filtered on repeated data.
  Missing/null templates leave the native preparation route available; `[]` means
  no templates. Turning off template use does not erase supplied conditions from
  a written snapshot.
- Native automatic search results are converted to single-chain CIF with mapping.
  Native search tools, databases, pairing, profile computation and inference
  sampling are not replaced. A native route that returns no template hits is not
  supplemented with a new search backend.

RNA prepared output contains only the unpaired channel. Any all-gap RNA paired
block used to align a protein/RNA complex is reconstructed internally, not exported
as an RNA `pairedMsaPath`.

Supplied paired files are already paired: corresponding rows across proteins
define the correspondence; inference does not pair them again by species.
Native local species pairing may return only hit rows, without a query first row.
Server prepaired output may include the query. Publication and reading preserve
both cases without inserting or removing a row, even if a real hit equals the
query. The paired first row need not equal the query; unpaired still must start
with the query. Width and alignment characters are validated for both channels.
Headers may be regenerated and lowercase insertion identities become `x`; the
aligned residues, row order and insertion counts are preserved. This is a
model-input round trip, not byte-for-byte preservation of search output.

Older bundles with `>query openfold3_context_only` are rejected with a regeneration
message. Recreate them from original inputs, rather than deleting that header by
hand (which would turn its synthetic record into an actual paired row).

Use the native `use_msa_server` setting to choose the server preparation route.
Disabling it does not launch local Jackhmmer/HHblits searches: that native database
search is a separate Snakemake workflow. The offline preparation path consumes
existing alignments and supplies an unspecified main with query-only input.
Native local species pairing operates on its UniProt sources before final paired
publication; public paired files are already finalized, not raw UniProt sources.
This change adds no new local-search configuration or database-search entry point.

Public paired/unpaired files represent finalized pools, not a single raw search
source. They are therefore not cropped again by `max_seq_counts` for ColabFold
when loaded. Native raw search files still obey their per-source quotas. Inference
still applies `max_rows_paired` and the total MSA row budget; lowering those limits
deliberately can change the rows used by the model. This also applies to user
supplied public MSA fields, which are staged under the canonical channel names.
Use `use_paired_msas=false` to disable paired features, not a zero per-source quota.

Prepared templates retain native hit rank/date/entry information in an optional
`openfold3` object within each template. This is method metadata, not a sidecar;
it prevents preparation from changing downstream native template selection.
Public mapping is zero-based; the temporary native cache uses one-based indices.
Provenance names are not runtime cache keys: distinct templates sharing an entry
name keep distinct mappings and structures. Unsafe native identifiers (including
underscores in a template chain label) are normalized only in the private runtime
CIF/cache; the published template and its residue indices are not renamed.

Method-specific query controls (`use_msas`, `use_main_msas`, `use_paired_msas`,
`pocket_constraint`, native numeric `covalent_bonds`) may be placed in a top-level
`openfold3` object. They keep their OpenFold3 meanings; this is AF3-style, not a
claim that this extended JSON is accepted unmodified by AF3 itself.

The public wrapper rejects `use_main_msas=false`: this pinned native branch
omits required profile/deletion statistics and total row counts. Leave that
switch omitted or true. To exclude unpaired homologues, set `unpairedMsa: ""`
for the relevant protein/RNA entities (remove `unpairedMsaPath` if present);
the wrapper supplies query-only main
input while keeping supplied protein paired rows. This neither silently changes
the switch nor uses homologues from a disabled file. Explicit false is rejected
even with `use_msas=false`; to request native empty-MSA features, set only the
total `use_msas` switch false. Native Python tools are not changed by this guard.

## Preparation, inference, and writing

```bash
bash run_openfold.sh -i input.json -o results -D true -P false
bash run_openfold.sh -i results/example/example_data.json -o results -D false -P true -w false
```

`-D` controls native preparation/search. `-P` controls prediction. `-w` in the
shell wrapper (`-J` / `--write-input-json` in Python CLI) controls publication.
When omitted, publication defaults to true, including inference-only and fully skipped runs.

With writing enabled, even an existing snapshot is updated:

```text
results/example/example_data.json
results/example/msas/example__A_unpairedmsa.a3m
results/example/msas/example__A_pairedmsa.a3m
results/example/msas/example__A_template_0.cif
```

`--compress_fold_input` / `--compress-fold-input` (shell `-z`) defaults to false;
true writes the external A3M/mmCIF resources as zstd. Reading either format is
independent of this write option.

`--compress_full_confidence` / `--compress-full-confidence` (shell `-f`) selects
compressed NPZ when true and JSON when false. Omitting both it and the legacy
`output_writer_settings.full_confidence_output_format` resolves to false/JSON.
An explicitly configured legacy npz format remains effective if the new bool is
omitted; explicitly conflicting choices fail. Existing dtype conversion and JSON
rounding rules are retained, so the two encodings need not be bitwise equal.
Keys, grouping and dimensions are preserved; features, latents, CIF and summaries
are unaffected. Switching formats removes the matching old confidence file after
successful publication. Skip checks require the effective selected format.

Only supplied/generated resources are written; an empty paired channel has no
paired file. JSON paths use `msas/...`. With D=false/J=true existing conditions
are copied without searching. With J=false no public input bundle is written;
private input/cache resources survive through inference and are cleaned on exit.
Writable `SLURM_TMPDIR` is preferred, otherwise Python's temporary-directory
selection (`TMPDIR`/system) applies. Explicit native raw-search output options in
runner YAML retain their native behavior; omit them to avoid raw output retention.
Default downloaded template structures, precaches and preparsed arrays also live
in the managed workspace. Explicit structure/precache/array store paths in runner
YAML are user-managed stores and are never removed by wrapper cleanup.

Inference builds native caches/features from the current conditions. Completed
seed skipping remains existence/nonempty based and does not compare conditions;
disable skip or use a new output directory when new conditions must be recomputed.
No cross-file crash atomicity is promised. Caught publication errors restore files
already overwritten; unrelated outputs and older unused resources are not deleted.
