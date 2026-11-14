from pathlib import Path
import json,subprocess
D=Path('/data1/wjt/rte_handoff_20261114_slim');H=D/'handoff/rte_20261114';s=json.loads((H/'SLIM_SELECTION.json').read_text())
for rel in s['removed_base_binary']:
 if not rel.startswith('rte/'):
  subprocess.run(['git','restore','--source=HEAD','--',rel],cwd=D,check=True)
p=H/'HANDOFF.md';t=p.read_text().replace('Critical current data/checkpoints and classical controls are included where listed; bulky historical banks and original Arnoldi vectors remain on FWI','No RTE tensor data or model checkpoints are included in this lightweight revision. Current data/checkpoints, classical-control weights, historical banks and original Arnoldi vectors remain on FWI').replace('Current numerical scripts contain','This lightweight handoff requires retrieval of inventoried RTE data/checkpoints before numerical replay. Current numerical scripts contain').replace('RNG/checkpoint state and fixed draws are included where saved','RNG/checkpoint state and fixed draws are inventoried externally, not materialized');p.write_text(t)
(H/'BINARY_DEPENDENCIES.md').write_text('''# Lightweight handoff: external numerical dependencies

This is NOT self-contained for training or numerical replay. At the user request, all RTE tensor data and model checkpoints, images, archives and large raw tables/logs were excluded. No original data were deleted: the full FWI tree and earlier isolated snapshot remain unchanged.

SOURCE_INVENTORY.json contains every original scoped artifact with exact SHA-256, bytes, materialization flag and authoritative FWI retrieval path. SLIM_SELECTION.json records the deterministic type/size selection and omitted-file identities. No new paid storage or LFS service was introduced.

Required before current replay:
- rte/workstation_runs/rte_mcmg_born_20261114_dot_v1/data.pt and physical_2048.pt: exact media and common initialization.
- rte/workstation_runs/rte_krylov_window_20261114_dot_v1/draw_seed*.pt and checkpoints: exact schedules and resumable weights.
- CT4/SCT4 best.pt files in inventoried September controls.
- FIT8 residual tensors/checkpoints, RHS-holdout inputs, fixed-panel inputs, original Arnoldi v/z/Az banks for each relevant diagnostic.
- Direct-C8 also needs historical teacher banks under rte_predictability_20261108_dot_v1 and tail_multiresolution_bank_20260926.

Retrieve exact source paths on FWI and verify SHA-256 before use. Included generators/replay scripts describe creation; regeneration was NOT performed by this handoff and bitwise GPU retraining is not promised. Future outputs after the snapshot require a separate append. No full numerical pass is claimed for this data-free revision.
''')
(H/'VALIDATION.md').write_text('''# Lightweight handoff validation

Earlier isolated 564 MB snapshot: all 2922 copied-source hashes matched, three existing handoff tests passed. The revised lightweight snapshot deliberately removes numerical tensors and checkpoints; these earlier serialization tests are NOT presented as tests of this reduced checkout.

Current checks are source/text selection, original-source hash identity, metadata, and final read-only verifier. Expected malformed historical artifacts are preserved where selected: empty telemetry JSON and window_failed_syntax.py. AppleDouble resource forks are excluded. Originals are never repaired or erased.

Common credential-pattern scan found no private-key headers, GitHub tokens or AWS key IDs in the original text snapshot; final metadata is separately inspected. This is not a general security guarantee.

No training, solver campaign or clean-environment install was run for this handoff. ENVIRONMENT.json and PIP_FREEZE.txt record the actual environment, not a portable lockfile. No Actions workflow found; no CI pass claimed. FWI running worktree and jobs untouched.
''')
