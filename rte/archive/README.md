# Research Artifact Archive

This directory preserves historical scratch artifacts without changing their
contents. It is organization, not evidence of numerical or mathematical validity.
Core code, primary reports, and the original results referenced by reports remain
in their existing locations. Neither top-level project README is maintained here.

## Contents

- `smoke/`: unreferenced root RTE scratch reports, tables, results, and histories.
- `smoke_moves_<batch>.tsv`: immutable original-path to archive-path mappings,
  byte counts, and SHA-256 hashes. The batch identifier is derived from the TSV
  content. These files also preserve the original names for historical lookup.
- [Project artifact index](ARTIFACT_INDEX.md): repository-wide links grouped as
  notes, results, traces, core code, and scratch.
- [Machine-readable inventory](artifact_inventory.tsv): the same classification
  with repository-relative paths, byte counts, and SHA-256 hashes.

The index includes existing tracked files and non-ignored untracked files, not
ignored generated caches, logs, checkpoints, or images. Tracked files that match
ignore rules remain included. Only the two generated inventory files exclude
themselves. No timestamps or Git staging-state fields enter the inventory, so
identical paths and contents reproduce it byte for byte.

## Safe Workflow

Run from the repository root with Python 3.9 or newer; no third-party packages
are required:

```sh
python3 scripts/organize_research_artifacts.py
python3 scripts/organize_research_artifacts.py --apply
python3 scripts/organize_research_artifacts.py --verify
python3 scripts/organize_research_artifacts.py --inventory-only --apply
python3 -B -m unittest discover -s scripts -p 'test_organize_research_artifacts.py' -v
```

The default and explicit `--dry-run` are read-only. `--inventory-only` never moves
files and still requires `--apply` to write the generated inventory. Refresh the
inventory after other project edits, then use `--verify` to check all manifest
hashes, source-path absence, and exact inventory reproducibility.

## Selection and Reference Boundaries

Only direct children of `rte/` whose names start with `tmp_` or `TMP_` and end in
`.csv`, `.tsv`, `.md`, `.tex`, or `.txt` are eligible. Code, nested experiments,
non-temporary reports/results, and build auxiliaries are not moved. Ignored
untracked files are not move candidates.

The organizer scans existing tracked and non-ignored untracked files, plus
ignored text configuration/document/source files (including JSON and notebooks),
for case-insensitive filename stems. It also recognizes URL escaping and LaTeX
underscore escaping. Any hit, including a self-reference or another scratch
report's historical command, retains the candidate. Matching a stem deliberately
protects all its same-stem companions. Generated indexes and move manifests are
excluded from the reference scan because they document locations rather than
consume artifacts. Archived content itself remains in the reference scan.

This is a conservative static reference check, not a proof about externally
stored commands, binary checkpoints, compressed document contents, or dynamically
constructed paths. Inspect filesystem-consuming code before applying a new batch.
Do not run experiments that write the selected scratch paths concurrently.
Historical report text is never rewritten to make a candidate eligible.

## Preservation and Recovery

Before applying, the script rechecks the plan and source hashes. Every source and
destination must be a regular file path without symlink components. Existing
destinations are errors, even when their bytes match. The same-filesystem move
uses exclusive hard links, verifies every linked file, writes and flushes the
manifest, then removes the original names only while their identical archive
links exist. The final archive is checked against the manifest. No file contents
are edited, no destination is overwritten, and no last copy is deleted.

An interrupted or failed apply may leave both names, a partial set of archive
links, or a manifest with only some original names removed. The script refuses
collisions and does not auto-clean or roll back such a state. Compare both paths
and their SHA-256 hashes, preserve at least one verified copy, and reconcile the
manifest before retrying. A recreated original scratch path also causes
verification to fail; it must be reviewed as a new experiment, not silently
overwritten or folded into the historical archive.

Only the two explicitly marked generated inventory files may be refreshed in
place. Existing non-generated files at those paths are refused. A move manifest
is never replaced by a later run.

## Initial Batch Validation

The 2026-09-12 organization pass archived **21 files / 29,638 bytes** in
[this mapping manifest](smoke_moves_ef84582494bfba84.tsv). Eleven root scratch
files were retained because of filename/stem references, including the two
uppercase-prefix reports and their original result CSVs. The dry-run was
inspected before applying; no references or historical report text were edited.

All 21 archived files were independently compared byte-for-byte with their
original Git `HEAD` blobs, in addition to checking manifest SHA-256 hashes and
byte counts. All 11 retained scratch files also matched their original blobs.
The only tracked paths removed were the 21 manifest source paths. Fourteen
temporary-repository safety tests passed, including no-write dry-runs,
no-clobber races, reference retention, changed-source rejection, symlink refusal,
interruption before and after manifest creation, and reproducible reruns.

The project inventory is a content snapshot, not a live database. Concurrent
project edits can make `--verify` report it as stale even when every archived
file still verifies. Regenerate with `--inventory-only --apply` after those
edits, without repeating or extending the archive operation.
