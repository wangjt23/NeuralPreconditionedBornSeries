# RTE Workstation Handoff, 2026-09-12

**Latest continuation, September 26: global-response neural pilots and larger-grid tests completed.**
See the [bigger-picture review, measured results and next architecture target](RTE_SOURCE_INITIALIZATION_REVIEW_2026_09_26.md).
Two validation-selected neural initializers were tested on separate cohorts
of 24 independent media. The directional variant is tied at N64 and slower
by 1.6% at N128 and 2.0% at N256, with inference included and no failed solves.
There is no significant learned speedup. The campaign has 1,512 measured
held-out solves, not 1,512 independent problems. Exact fine-directional
information helps, but neither simple filtering nor much lower full-field
MSE supplies the needed correction reliably.

The next priority is a **one-shot learned Krylov-tail search direction after
four classical steps**, not another full-flux initializer or dense inverse.
Retain the existing Krylov basis, skip AI on already-converged cases, and
train on projected physical residual reduction. New exact-tail datasets
are ready: 128 training cases (112 eligible) and 16 validation cases (13
eligible). No tail network has been trained and the augmented FGMRES path
still needs implementation and an exact-teacher gate. See the review for
the important happy-breakdown/search-image projection distinction.

Full suite: 127 tests and 66 subtests passed. All budgets and diagnostics
are complete; no RTE experiment is left running. Original September 12
handoff and September 19/September 26 base-report hash manifests verify.
All failed/smoke/partial runs and old checkpoints are preserved. No commit
or push was made. Earlier states below are historical.

**Historical continuation, September 19: recursive response deployment tests completed.**
See the [recursive-response results and next factor hypothesis](RTE_RECURSIVE_RESPONSE_RESULTS_2026_09_19.md).
The actual recursive residual-weighted operator helps at N16 but is slower
than TSA at N32 on the two stiff development media. Its FP32 N32 cache still
uses about 127 MiB. Energy weighting and rank8/32 global repair do not clear
the training gate. Exact fixed-band coarse responses also fail to beat the
classical controls at N64/N128; full odd-reset compositions are particularly
bad and are preserved as negative controls. No neural model was trained.
Next untested hypothesis: a direction-aware nested factor of the
reference-normalized Schur operator, not independent raw inverse blocks.
All 1,080 measured solves completed; these include repeats, not 1,080
independent problems. Full suite: 118 tests and 66 subtests passed. No RTE
experiment is left running. Earlier reports and handoff artifacts are intact.

**Historical update, September 19: structural continuation completed.** See the
[review, parity/response investigations, and stronger classical-depth audit](RTE_PARITY_STRUCTURE_REVIEW_2026_09_19.md).
The old AI timing advantage did not survive expanded classical tuning on the
same 24 N128 cases (later GPU timings are contended). The best new research
lead is residual-weighted hierarchical response compression in stiff media:
a small-grid teacher oracle helps iterations, but is not yet compact or fast
enough for deployment. No new neural model was trained and no RTE experiment
is left running. September 13 and earlier states below are historical.
The full RTE suite passes 105 tests; the original handoff remains intact.

**Historical update, September 13: resumed campaign completed.** See
[high-resolution results and structural audit](RTE_STRUCTURE_AND_RESOLUTION_RESULTS_2026_09_13.md)
and the [next structure-driven plan](RTE_STRUCTURE_DRIVEN_PLAN_2026_09_13.md).
Latest architecture investigation: [learned global/hierarchical response
design and negative teacher-space pilot](RTE_LEARNED_GLOBAL_RESPONSE_DESIGN_2026_09_13.md).
The rank-2/4/8 global-space pilot saved no iterations on 12 held-out source
translations. Next test hierarchical response-block compressibility before
committing to a larger network. The full suite now passes 96 tests.
No RTE job is left running. The two previously suspended timing processes were
replaced by fresh runs; their saved partial outputs remain preserved. The pause
state below is historical, not the current process state.

**Historical pause, September 12 at 21:08.**
See [the pause record](RTE_HIGH_RESOLUTION_PAUSE_2026_09_12.md) for suspended
PIDs, completed N64/N128 checkpoints, partial results and timing-resume cautions.
Do not resume experiments without a new instruction. Older stop states below
are preserved historical records.

New architecture continuation: [NTO design](RTE_NEURAL_TRANSPORT_OPERATOR_DESIGN_2026_09_12.md)
and [matched AI-operator pilots](RTE_NEURAL_TRANSPORT_OPERATOR_RESULTS_2026_09_12.md).
The four-stage neural operator is the next candidate; all four matched pilot
training budgets are complete, with no RTE job left running.

Workstation continuation: see [the diagnostic audit and updated experiment plan](RTE_DIAGNOSTIC_AUDIT_2026_09_12.md).
N16 replications and the N32 training budget are complete. A coefficient-cache
fault invalidated parts of the original GPU test evaluation; corrected results
and measurement limitations are documented there. The handoff below retains its
original stop-state description and historical evidence.

## Stop State

Local training is stopped. The shared collision/transport (CT) run had already
completed its 100-epoch budget: 1,600 optimizer updates, selected epoch 60.
There is no partially running job to recover. Previous research agents were
closed. No training was performed after the stop request; handoff checks only
serialize/load artifacts, check hashes, compile code, and test restart guards.
The research objective of outperforming strong classical solvers generally is
**not complete**. This is a preserved experimental state, not a final claim.

Local changes were based on `389b95d`. The fetched remote update `6fb06bc`
adds CUDA U-Net experiments and a heterogeneous-history CSV writer fix. Both
lines of work are retained. The Git history of this handoff records the merge;
use the final pushed revision, not either parent alone.

## What Is Saved

| Location | Contents |
|---|---|
| `rte/results/collision_transport_ensemble_seed1/last.pt` | Epoch 100 weights, optimizer, scheduler, RNG, best weights, history, data fingerprints |
| `rte/results/collision_transport_ensemble_seed1/best.pt` | Validation-selected epoch 60 weights and configuration |
| `rte/results/collision_transport_ensemble_seed1_paired/` | Same selected weights, reevaluated with cached FFT multipliers and paired randomized method timing |
| `rte/handoff/2026-09-12/{train,validation,test}.pt` | Actual coefficient tensors, geometry codes, compact source parameters, metadata and fingerprints |
| `rte/handoff/2026-09-12/provenance.json` | Checkpoint state, original platform/library versions, parent revisions |
| `rte/handoff/2026-09-12/SHA256SUMS.tsv` | Byte counts and SHA-256 hashes for every saved results/bundle file |
| `rte/results/` | Earlier positive and negative experiments, CSV histories, figures, settings and available local weights |
| `rte/archive/` | Original scratch results moved without changing bytes, plus mapping/hash manifest |

The `.pt` and settings artifacts in this snapshot are explicitly included in
Git, despite the general ignore rules for future training outputs. Historical
remote weights that were never present locally are **not** reconstructed.
Full past stochastic training residuals are not stored: their generation code
and the last epoch-boundary RNG state are saved. Sources can be reconstructed
exactly from their compact parameters; no external benchmark download is needed.

Dataset fingerprints:

```text
train       feaa920e77430f23da7290e9206cc12a93e842c89ce5628511f929e7ccc812f9
validation  9401554ad9419d564ec7a29e8c93ba5159ab61d851d6ef47201c212d4074cd57
test        b955a619f20902289baf113f45606682d3c79695a8af9be798ad64eaa2d365a7
```

## Current Method And Training

The physical problem is the repository's periodic, two-dimensional, discrete-
ordinate RTE, not ATFPS, an upwind sweep, or a physical inflow-boundary model:

```text
A psi = S psi + C psi
S = Omega_l . grad_h,   C = sigma_t I - sigma_s P_h
```

Spatial differentiation is Fourier pseudo-spectral on the unit square with
`dx=1/N`, including the existing even-grid Nyquist convention. Angles are
equally spaced on S1, with normalized quadrature weights `1/L`. `P_h` is the
discrete normalized circular phase operator, not an S2 spherical quadrature.
For the current ensemble, `N=16`, `L=12`, `g=0.65`; coefficients are real32 and
fields complex64. The phase operator is fixed within each run.

The new model is an alternative to U-Net defect learning, not a claim that
the old U-Net was repaired by this architecture. Four residual-correction
stages alternate angular collision FFTs and spatial streaming FFTs:

```text
lambda_m(x) = sigma_t(x) - sigma_s(x) p_m
K_j r = F_x^-1 [ 2 alpha_j / (alpha_j + i Omega.k)
                  F_x F_theta^-1 [ gain_j,m(x) / (alpha_j + lambda_m(x)) F_theta r ] ]
z_0 = 0;  z_(j+1) = z_j + K_j (r - A z_j), j=0,...,3
```

`p_m` is computed from the actual discrete phase matrix. Every angular mode
is retained. With unit gains and fixed positive shifts this is the classical
Hermitian/skew-Hermitian splitting (HSS) correction; the learned variation
does not automatically inherit its convergence theorem. The trained map is
complex-linear in the residual for fixed coefficients, but nonlinear in the
coefficients. Shifts are coefficient-conditioned positive scalars per stage;
bounded positive gain fields depend on local/multiscale material features.
Signed angular modes share gains and the scalar-mode gain is fixed at one.
Width 12 gives 1,340 parameters at 12 angles. No learned angular rotation,
restriction/prolongation, or residual-conditioned router is hidden in this
model. Mode interaction comes from alternating the exact collision/streaming
representations and physical residual updates. Cached multipliers are used
equally for learned CT and classical HSS during no-gradient evaluation.

The training objective is the **pure one-step physical residual loss**:

```text
rho_i = ||r_i - A_i M_theta(A_i) r_i||_2 / ||r_i||_2
loss = mean_i rho_i^2
```

Norms are Euclidean over all angular/spatial complex entries (equivalently
real-stacked Euclidean). There is no Born-metric, teacher, rollout-loss, or
stability penalty here. `sqrt(loss)` is the RMS ratio, not `mean(rho)`.
Training uses 256 independently generated material/source pairs; validation
and test have 32 each, seeds 1001/2001/3001. Both coefficient fields receive
the same independent periodic translation and quarter-turn rotation. Source
centers, orientation, amplitude and angular type are independently sampled.
Each medium still has only one stored source: coefficient diversity and
source diversity have not been separately identified as causes.

Each batch samples a residual from four stages of a fixed TSA trajectory;
with probability 1/3 it instead uses `A` applied to a smooth random complex
error. Fresh probes are drawn every batch. One epoch visits every shuffled
material, including a partial final minibatch. AdamW uses lr 0.003, weight
decay 1e-4, gradient clipping 1, and a cosine schedule to 1% of the initial lr
over the specified total update horizon. Batch size is 16; 100 epochs means
25,600 material presentations. Full-bank train/validation losses use four
fixed residual views per material, independently seeded from training.

Every ten epochs, checkpoint selection minimizes the lexicographic tuple
`(validation failures, mean FGMRES iterations, mean final relative residual)`
over the first eight validation media. Test media do not select checkpoints
or classical settings. All test solves use real-stacked FGMRES, true physical
residual tolerance 1e-5, and at most 30 iterations. The small validation solver
subset is a limitation to enlarge before drawing strong conclusions.

## Latest Evidence, Including Negative Results

Selected epoch 60 bank losses are train **0.0107141**, validation **0.00810655**,
and test **0.00973402**. Initial train loss was 0.131747; epoch 100 train loss
was 0.0104939. A lower last loss did not make it the selected solver checkpoint.

The paired reevaluation averages 32 held-out material/source solves. Times are
CPU, one thread, medians of three warmed solves per medium, then means across
media. Every listed method converged on all 32 test problems.

| Method | Source physical loss | Mean iterations | Warm solve ms | Preparation + warm ms |
|---|---:|---:|---:|---:|
| Learned CT4 | 0.0154248 | 6.28125 | 7.702 | 8.681 |
| Tuned HSS | 0.122424 | 7.75000 | 9.461 | 9.754 |
| Tuned TSA | 0.00696129 | 6.37500 | 13.165 | 15.119 |
| Tuned full reference | 0.0240125 | 9.56250 | 12.007 | 14.031 |
| Tuned DSA | 0.0597034 | 11.53125 | 22.833 | 23.169 |
| Tuned transport-corrected DSA | 0.0535345 | 11.59375 | 22.702 | 23.085 |

`source_loss` uses the initial physical RHS; the bank loss uses a mixture of
trajectory residuals and random probes. They are the same formula on different
inputs, not interchangeable datasets. `first_solve_seconds` in the CSV is
**preparation plus a warmed solve estimate**, not measured cold end-to-end
latency. Operator construction and training are excluded.

Classical depth/shift parameters are selected by validation failures, failure
quality and warmed time. The paired selected settings were HSS depth4/factor1,
TSA depth4/eta0.03, full reference depth4/eta0.03, and both DSA variants
depth4/eta0. The transport-corrected diffusion coefficient uses the actual
discrete first phase moment. These are periodic synthetic-acceleration
prototypes, not the best available production DSA/TSA implementations.

The initial method-major timing gave CT **26.069 ms**, HSS **21.766 ms**, TSA
**20.010 ms**: CT did not win. The subsequent run changed cached multiplier
implementation, method timing order, and selected classical configurations.
It reused the same learned weights and test cases. Thus it is **not** a second
independent training replication, and the time difference cannot be attributed
to caching alone. Keep both result directories in any analysis.

Earlier three-seed, per-medium experiments improved mixed cases at N=16 and
N=32 but failed to show a general advantage on homogeneous anisotropic cases.
Global angular gain learning explains much of the gain; spatial conditioning
adds a smaller increment. A P1 macro augmentation performed badly on the mixed
fixture and is not enabled by default. Shift-only loss reduction did not
reliably improve the selected solver. Original U-Net/replay/subspace negatives
remain relevant and are retained in the audit and results notes.

The stronger classical audit is in `collision_transport_classical_audit`.
Unshifted homogeneous full-reference inversion is now independently tested.
The same-pseudo-spectral sparse ILU/LU implementation has action/inverse tests,
but **its performance benchmark has not run**. No GPU numerical/timing evidence
or multiple training seeds for the shared CT ensemble exists yet.

## Workstation Commands

Install a CUDA-enabled Torch build matching the workstation/driver, plus
NumPy, SciPy and Matplotlib; PyYAML/h5py are used by other repository paths.
The saved laptop environment was Python 3.9.6, Torch 2.8.0, NumPy 2.0.2,
SciPy 1.13.1 on macOS arm64. This is provenance, not a Linux environment lock.
Use Torch 2.8 for the closest initial reproduction; cross-platform FFT results
and timings need not be bitwise identical.

From the repository root, **verify without any training**:

```bash
python scripts/prepare_rte_handoff.py --verify
python -m unittest rte.tests.test_workstation_handoff -v
python scripts/organize_research_artifacts.py --verify
python -m rte.train_collision_transport_ensemble --help
```

To continue learning on CUDA from the selected weights, with the exact saved
material/source tensors and a fresh 100-epoch optimizer schedule:

```bash
python -m rte.train_collision_transport_ensemble --device cuda:0 \
  --train-size 256 --val-size 32 --test-size 32 --size 16 --n-angles 12 \
  --epochs 100 --batch-size 16 --seed 1 \
  --dataset-dir rte/handoff/2026-09-12 \
  --warm-start rte/results/collision_transport_ensemble_seed1/best.pt \
  --output rte/workstation_runs/ct_seed1_gpu_warmstart
```

This reuses previously seen pilot validation/test cases; it is an engineering
continuation, **not** a fresh confirmatory test. `--warm-start` loads only model
weights and records their file hash; optimizer, scheduler and RNG restart.
Changing angle count changes the output head and strict loading rejects it.
Changing spatial size is supported structurally but is an unverified transfer.

`--resume last.pt` instead restores optimizer/scheduler/RNG exactly at an epoch
boundary and requires unchanged settings, including device and epoch horizon.
Use the original full argument set for the checkpoint. `--stop-after E` saves
an epoch checkpoint and exits without final test evaluation when E is below
the total horizon. Resuming the saved **completed** 100/100 checkpoint with
the same horizon runs no additional training; it only reevaluates. Do not
increase `--epochs` and call that an exact resume. CPU-to-CUDA uses warm-start.
Use a new output directory; never overwrite the published laptop results.

## Prioritized Next Experiments

1. **P0, deployment verification.** On the workstation run the CT, transport
   baseline, sparse-action and handoff tests; validate CUDA forward/backward
   and cached/uncached agreement before training. GPU is untested locally.
   Reevaluate saved weights with method order randomized within each medium,
   record actual cold setup and warmed solve separately, all A/preconditioner
   calls, memory, failures, and median/p95 time-to-tolerance. The ensemble CLI
   presently combines final evaluation with training; a dedicated evaluator
   is a useful next implementation, not something already provided.
2. **P1, independent replication.** Train seeds 2 and 3 with the same 256/32/32,
   N16/L12, 100-epoch settings, new output directories and no warm-start. This
   changes both initialization/probes and generated datasets. Retain every
   failure; report per-seed and paired per-medium differences, not only means.
   Before a confirmatory claim, reserve fresh test seeds after design freeze.
3. **P2, stronger classical controls.** Run the saved sparse reference CLI
   first at N16 and only then N32 with a memory limit. Compare ILU and exact
   LU on the same pseudo-spectral operator, including assembly/factorization
   and amortization. These CPU controls do not directly establish a GPU speed
   ranking. Broaden HSS stage sequences and DSA/TSA tuning on validation only.
4. **P3, scale training after P0/P1.** A concrete first workload is 4,096 train,
   128 validation, 256 test materials; N32/L12; batch32; 1,000 epochs; eval every
   25 epochs; validation solver selection on 32 media. This is 128,000 updates
   and 4,096,000 presentations, not 1,000 updates. Start from scratch and include
   a separately labeled warm-start arm if desired. Do not load the small saved
   dataset directory for this new experiment. Measure GPU memory in a short
   `--stop-after 1` run using the final horizon, then exactly resume it.
5. **P4, identify the cause.** Cross material count, sources per material, and
   update budget. Compare classical fixed/global gains, learned global gains,
   learned spatial gains, and raw U-Net at matched data and training effort;
   separately report measured solve cost. More epochs alone do not identify
   an expressivity or data-coverage bottleneck. Probe amplification on solver
   residuals not used in training, with true residual rather than loss alone.
6. **Architecture work after these controls.** Test collision-rate-conditioned
   modal hypernetworks shared across angular modes, then mixed-g ensembles,
   anisotropic spatial receptive fields, and controlled macro/micro coupling.
   These are proposals, not implemented capabilities or demonstrated wins.
   Scalar conservation, zero residual response, coefficient cache invalidation
   and equal classical preparation optimizations remain audit requirements.

Example P3 command (workstation only):

```bash
python -m rte.train_collision_transport_ensemble --device cuda:0 \
  --train-size 4096 --val-size 128 --test-size 256 --selection-count 32 \
  --size 32 --n-angles 12 --regimes mixed --g 0.65 \
  --epochs 1000 --batch-size 32 --eval-every 25 --seed 101 --stop-after 1 \
  --output rte/workstation_runs/ct_n32_data4096_seed101
```

Resume with all the same options, replace `--stop-after 1` with
`--resume rte/workstation_runs/ct_n32_data4096_seed101/last.pt`. Scaling the
dataset also changes the joint coefficient/source distribution and reuse;
add a fixed-update control before calling data size the causal bottleneck.

## Code And Report Map

- `rte/models.py`: physical operator, phase normalization, reference resolvents.
- `rte/collision_transport.py`: CT/HSS factors, coefficient networks and caches.
- `rte/train_collision_transport_ensemble.py`: shared-field training, selection,
  checkpoint/warm-start logic, saved-data loading, paired test evaluation.
- `rte/transport_baselines.py`, `rte/audit_transport_baselines.py`: tuned
  classical periodic references, including zero shift and discrete transport DSA.
- `rte/spectral_sparse_reference.py`: independent sparse assembly, ILU/LU controls.
- `rte/run_large_angular_spectral_experiments.py`: corrected U-Net ensemble
  generator/training harness; remains a separate experiment family.
- `docs/research/PROJECT_AUDIT_2026_09_12.md`: evidence and bottleneck audit.
- `docs/research/RTE_RESIDUAL_SUBSPACE_NOTE.tex`: self-contained mathematical
  note, with compiled PDF in `output/pdf`.
- `docs/research/RTE_COLLISION_TRANSPORT_ARCHITECTURE.md` and
  `RTE_COLLISION_TRANSPORT_RESULTS.md`: detailed earlier CT design and experiments.

The CUDA U-Net files arriving from the remote are preserved but were not
compiled or numerically verified on this non-CUDA laptop.
