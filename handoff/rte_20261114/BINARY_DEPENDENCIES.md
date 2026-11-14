# Lightweight handoff: external numerical dependencies

This is NOT self-contained for training or numerical replay. At the user request, all RTE tensor data and model checkpoints, images, archives and large raw tables/logs were excluded. No original data were deleted: the full FWI tree and earlier isolated snapshot remain unchanged.

SOURCE_INVENTORY.json contains every original scoped artifact with exact SHA-256, bytes, materialization flag and authoritative FWI retrieval path. SLIM_SELECTION.json records the deterministic type/size selection and omitted-file identities. No new paid storage or LFS service was introduced.

Required before current replay:
- rte/workstation_runs/rte_mcmg_born_20261114_dot_v1/data.pt and physical_2048.pt: exact media and common initialization.
- rte/workstation_runs/rte_krylov_window_20261114_dot_v1/draw_seed*.pt and checkpoints: exact schedules and resumable weights.
- CT4/SCT4 best.pt files in inventoried September controls.
- FIT8 residual tensors/checkpoints, RHS-holdout inputs, fixed-panel inputs, original Arnoldi v/z/Az banks for each relevant diagnostic.
- Direct-C8 also needs historical teacher banks under rte_predictability_20261108_dot_v1 and tail_multiresolution_bank_20260926.

Retrieve exact source paths on FWI and verify SHA-256 before use. Included generators/replay scripts describe creation; regeneration was NOT performed by this handoff and bitwise GPU retraining is not promised. Future outputs after the snapshot require a separate append. No full numerical pass is claimed for this data-free revision.
