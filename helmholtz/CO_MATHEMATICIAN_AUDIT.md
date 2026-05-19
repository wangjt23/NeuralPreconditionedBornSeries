# Co-AI Mathematician Audit: NPBS Helmholtz Repository

Paper anchor: <https://arxiv.org/abs/2603.18527>  
Local co-mathematician project: `helmholtz/.co_math/projects/npbs-repository-audit-20260517-203448-7dc2ae`

## Scope

This audit checks the Helmholtz portion of the repository against the paper's
core mathematical claims:

- CBS/Born coordinates satisfy `I - G_eta V_eta = G_eta A`.
- NPBS feeds learned preconditioners with Born-preconditioned residuals.
- `loss_type: 3` measures residual error in the `G_eta`-induced metric.
- Classical CBS, FGMRES+CBS, learned NPBS, and shifted-Laplacian baselines must
  be labelled by the actual preconditioner used.

## Main Findings

1. `ConvergentBornSeries2d.A_op`, `G`, and the training losses match the paper's
   discrete scaling convention: the Born residual used in code is
   `G((b - A_op(u)) / dx**2)`, equivalently `G(V*u + b/dx**2) - u`.
2. CBS used inside FGMRES previously had only one implicit mode:
   `gamma * G(r / dx**2)`. This is the CBS correction map, not the pure
   constant-reference shifted-Laplacian inverse `G(r / dx**2)`.
3. `ShiftedLaplacianVcycle2d` and its finite-difference variants are classical
   variable-coefficient shifted Helmholtz preconditioners. They are valid
   baselines, but they are not the same operator as the constant-coefficient
   FFT reference `G_eta` used in the paper's Born-equivalence identity.
4. Timing comparisons need explicit methodology notes. Stationary solvers can be
   replayed without residual checks after the converged iteration count is known,
   while FGMRES currently measures true residuals during Arnoldi iterations.
5. Published OpenFWI configs mostly use `pbl`, which is a potential/sponge
   absorbing layer in the CBS path. The repository also contains stretched-grid
   PML operators, but those are not wired into CBS/NPBS.

## Changes Made

- Added `inference.cbs_krylov_preconditioner` with two explicit modes:
  `gamma_green` for the existing CBS correction map, and `green` for the pure
  Born/constant-reference shifted-Laplacian inverse.
- Added `configs/2d/openfwi/CurveFaultB/cbs_gmres_green.yaml` for the pure
  `G_eta` Krylov comparison.
- Added `verify_operator_identities.py`, which checks the CBS Born residual
  identity and a small FGMRES right-preconditioner smoke test.
- Fixed the `ConvergentBornSeries2d` default boundary type from an unrecognized
  `PBL3` string to `pbl`, and normalized boundary type strings to lowercase.
- Added `--checkpoint` support to `inference2d.py`, matching the README command.
- Fixed the `CurveVel_A20` default frequency to match its `ppw = 20` label.
- Added optional seed handling for training/inference and restored the
  MgNO+FGMRES representative config to 50 test samples.
- Started a durable local co-mathematician workspace under `helmholtz/.co_math`.

## Remaining Work

- Add a real test environment with PyTorch in CI and run
  `python helmholtz/verify_operator_identities.py`.
- Add experiment manifests recording seed, checkpoint hash, GPU/CPU versions,
  PyTorch/CUDA versions, `num_test`, batch size, restart length, and timing mode.
- Decide whether paper tables should label FGMRES+CBS as `gamma_green` or
  include both `gamma_green` and `green` rows.
- If true PML is required for CBS/NPBS, wire `helm_op.py`'s PML operator into
  the same operator path used by training, inference, and loss evaluation.
