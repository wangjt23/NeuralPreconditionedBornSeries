# RTE Krylov-tail research, September 26, 2026

This continues the [source-initialization review](RTE_SOURCE_INITIALIZATION_REVIEW_2026_09_26.md).
The research objective remains genuine time-to-solution improvement over tuned
classical controls, at the unchanged physical tolerance. An oracle improvement,
training loss improvement, or smaller iteration count alone does not establish it.

## What was implemented

The solver now accepts one optional residual-conditioned direction after four
TSA8 applications, preserving the accumulated search space. It jointly minimizes
over the old directions and the proposal; it does not restart at injection.
Already-converged cases skip inference. Zero, nonfinite, dependent, or
residual-increasing proposals are rejected. The final residual is independently
checked against the original physical right-hand side.

The projected training objective removes **span(A Z)**, not all Arnoldi vectors.
An exact tail can lie inside the Arnoldi space while adding an independent search
image. Rejecting such a direction as a breakdown would incorrectly discard the
perfect answer. Tests cover this case, disabled-injection parity, rejected
directions, original-RHS stopping, and differentiable least-squares equivalence.

The new training contexts contain the actual four-step residual, exact remaining
correction, homogeneous reference action, and real-orthonormal search-image basis.
Of 128 training media, 112 need injection; of 16 validation media, 13 need it.
Easy cases remain in every full-solver evaluation. Training and validation
teachers are inherited from the earlier independently generated datasets.

## The integration gate passes, but it is not an AI result

On all 16 validation media, three shuffled, synchronized timing repeats give:

| Integration control | Mean case-median solve, ms | Mean classical calls |
|---|---:|---:|
| Legacy TSA8 | 128.156 | 8.0000 |
| New solver, no injection | 127.686 | 8.0000 |
| Homogeneous-reference tail | 128.475 | 7.8125 |
| TSA8 tail | 128.754 | 7.1875 |
| Exact teacher tail | 65.564 | 3.8750 |
| Exact teacher tail plus actual network forward cost | 71.743 | 3.8750 |

The gate excludes common analytical initialization and setup. It establishes
roughly 44% information-ceiling headroom with the base network's cost charged;
teacher acquisition is excluded and the teacher is unavailable at deployment.
It does not show a trained speedup. All gate solves meet tolerance.

Artifacts: `rte/workstation_runs/tail_injection_gate_20260926/`.

## Networks and bounded training

The base `TailResponseNetwork` reuses the global/local Fourier network and
fine-grid characteristic filters, but normalizes output by reference RMS instead
of the positive-source intensity scale. This matters for small, sign-changing
complex tail residuals. It has 670,811 real-valued parameters.

The first run uses 2,000 updates, batch 8, learning rate 0.001: 800 updates of
normalized teacher-direction supervision, then projected physical residual loss
with a small teacher regularizer. Validation selects step 800. The later projected
loss stage does not improve the full-solver work metric. The measured training
loop, including periodic validation/checkpoints, takes 100.78 seconds.

The four-moment variant decomposes a proposal into scalar, cosine, sine, and
remaining angular components. Four physical actions and a 4-by-4 projected
least-squares solve choose their coefficients. This is a classical coefficient
optimization, not four learned networks. The analytical-reference-only version
is mandatory as a non-AI control. A validation probe finds no useful runtime
gain from coefficient adjustment alone.

The `OpticalTailNetwork` adds material-only queries/keys and residual-conditioned
values to axial attention on the half-resolution grid. Its attention bias uses
the shorter periodic optical path along each axis. It retains the global/local
and characteristic paths, has 675,683 real parameters, and starts from an exact
zero-output upgrade of the base checkpoint. Optical distance itself is cheap
classical information; the hypothesized AI contribution is the learned
heterogeneous nonlocal coupling, not the distance calculation.

The attention continuation uses 2,000 updates at learning rate 0.0003, batch 8,
with the four-moment projected loss. Its loop takes 307.69 seconds. The validation
work proxy selects step **0**, i.e. no improvement over the starting weights.
The final trained step 2,000 is separately predeclared as an architecture candidate
for timed validation, so the benchmark actually tests trained attention rather
than silently evaluating a zero attention branch.

The checkpoint-selection proxy is classical calls plus 0.5 times injection calls.
It is not a runtime result, and is optimistic for attention/moment projection.
The final pipeline selection uses measured validation time instead.

Artifacts: `tail_direction_pilot_20260926/`, `tail_moment_probe_20260926/`,
and `tail_optical_moment_20260926/` under `rte/workstation_runs/`.

## Why the immediate loss is insufficient

The eligible-prefix audit distinguishes physical residual reduction from field
accuracy. The residual ratios below include optimal projection against the old
search images and the proposed direction, but no subsequent classical steps.

| Validation proposal | Immediate residual ratio | Optimally scaled field error | Phase-equivariance defect |
|---|---:|---:|---:|
| Analytical reference | 0.7843 | 0.5788 | 0 |
| Selected base tail | 0.7869 | 0.3685 | 0.2450 |
| Trained attention, final step | 0.5531 | 0.4542 | 0.1872 |
| Exact tail | 0.00000150 | approximately 0 | 0 |

These are means across 13 eligible validation media. Moment recombination changes
the attention ratio only from 0.5531 to 0.5494. Its train ratio is 0.4765, so
there is a generalization gap, but failure cannot be described simply as a
complete inability to learn residual reduction. Conversely, better teacher
field error does not reliably improve immediate residual reduction.

The phase defect measures `||N(i r) - i N(r)|| / ||N(r)||`, keeping the material
fixed and rotating the reference too. Its nonzero value is a measured mismatch
with the linear complex physical problem, despite phase augmentation. It does
not by itself prove that enforcing linearity will improve runtime.

The working inference is that training must reward components that survive the
subsequent classical trajectory, not only a good immediate correction. The
default injection also changes the next classical Arnoldi input; a separate
validation-only frontier probe checks that local trajectory choice.

Artifacts: `rte/workstation_runs/tail_generalization_20260926/`.

## Held-out large-grid protocol

All grids use L=12, anisotropy g=0.65, epsilon=0.1, the original FP32 physical
geometry promoted exactly to FP64, and relative physical tolerance 1e-5 against
the original RHS. Maximum budget is 100 accepted directions, restart 25.

Twelve classical configurations and three neural pipelines are evaluated on
16 validation media at each resolution, with one warmup and two measured,
shuffled repeats. The fastest zero-failure method in each family is frozen for
each grid **before any test data generation**. The classical candidates include
TSA depths 4/8/16/32, reference depths 16/32, HSS depths 16/32, zero versus balanced
initialization, and non-neural tail/moment controls.

The test cohort is 32 new independent media, seed 8626, evaluated on nested
64-squared, 128-squared, and 256-squared grids. These are 32 physical media at
three resolutions, not 96 independent media. Three measured repeats follow a
warmup; repeats are reduced to a case median before averaging and paired
media bootstrap confidence intervals. Neural inference, reference initialization,
moment projection, and final checks are included. Physical/preconditioner setup
is separately recorded; setup-plus-warm-solve is not cold-start latency.

Selections are `reference_tail` versus `neural_tail` at N64/N128, and
`tsa16_zero` versus `neural_tail` at N256. Matched TSA8 and non-neural controls
are retained in test evaluation. Training is only at N64; larger grids test
resolution transfer, not high-resolution retraining.

All 1,440 measured held-out solves complete with zero tolerance failures.
The three resolutions share the same 32 independent media; repetitions are not
additional problems. The result is negative for the validation-selected AI:

| Grid | Frozen classical, ms | Selected neural, ms | Neural time saving | Paired 95% interval |
|---|---:|---:|---:|---:|
| N64 | 133.210 | 139.579 | -4.78% | [-7.43%, -2.42%] |
| N128 | 134.713 | 142.584 | -5.84% | [-8.15%, -3.74%] |
| N256 | 142.968 | 157.333 | -10.05% | [-12.68%, -7.30%] |

Positive savings would favor AI. The intervals use 10,000 paired bootstrap
resamples of media, not individual timing repeats, and do not eliminate shared
workstation measurement limitations. Setup-plus-warm-solve means are respectively
140.113/146.482 ms, 142.135/150.006 ms, and 153.104/167.434 ms for classical/AI.

Matched TSA8-balanced means are 139.211, 141.306, and 152.777 ms. Thus restricting
the baseline to TSA8 would make the network look approximately tied on smaller
grids; tuning classical controls exposes the loss. The trained optical/moment
pipeline takes 146.119, 153.522, and 185.466 ms. Its extra work does not pay off.
At N64 the reference-tail and learned-tail methods use 7.750 and 7.719 classical
calls on average: almost all apparent improvement over plain TSA8 is available
without a network.

The benchmark additionally contains 1,440 measured validation solves. Warmups
are excluded from these counts. The observed GPU compute-process lists contain
only this experiment and an 8-MiB desktop process; timings remain shared-host
measurements, not a claim of hardware isolation or deployment-grade latency.

Artifacts: `rte/workstation_runs/tail_solver_benchmark_20260926/`, including
the frozen selection, all case timings, dataset tensors, per-case medians,
paired confidence intervals, source/checkpoint hashes, and setup accounting.

## Short-horizon continuation follow-up

The immediate-loss audit motivates a bounded follow-up, not a post-hoc change
to the completed held-out comparison. The differentiable loss reconstructs the
old Arnoldi span from `span(Q, r)`, where `Q` spans the old search images.
It then follows the injected image and two actual TSA8 applications, with the
same real inner product and retained image-space projection. This avoids saving
all old solution directions merely to evaluate a training loss.

Tests verify its first step against the earlier projected objective, its later
steps against the deployed augmented FGMRES trace (including mixed-precision
transport), finite-difference gradients, and independent batched behavior.
The follow-up changes offline training cost, not the one-network-call inference
path. The bounded follow-up does not change the frozen results above.

The frontier probe is negative. Using the pre-injection residual as the next
classical input, then resuming ordinary Arnoldi, increases base-tail classical
calls from 7.625 to 8.000, and attention-tail calls from 7.6875 to 7.9375.
The corresponding validation times are 131.821 to 138.292 ms and 134.275 to
138.709 ms. This probe uses two measured repeats and excludes common analytical
initialization/setup. It is a local trajectory ablation, not an implementation
of exact block GMRES; the deployed default is unchanged. All solves pass.
Artifacts: `rte/workstation_runs/tail_frontier_probe_20260926/`.

After a successful one-update smoke check, the continuation run completes 1,000
updates, batch 4, learning rate 0.0003, warm-starting the selected base-tail model.
Validation selects step 1,000. The work proxy improves from 8.03125 to 7.78125,
corresponding to classical calls falling from 7.625 to 7.375 across the same 16
validation cases. The training loop takes 139.47 seconds, plus 0.28 seconds for
the fixed analytical reference cache. CPU unit tests run during part of this
training loop; training duration is not an isolated hardware-performance claim.

Before evaluation, the continuation gate is set to zero failures and at least
5% mean time savings at **both** N128 and N256. It retains the earlier frozen
classical choices. The gate uses all 16 validation cases, one warmup and two
measured repeats; it would generate a fresh 32-case test cohort with seed 9626
only after passing. The measured outcome is:

| Validation grid | Frozen classical, ms | Lookahead-trained tail, ms | Neural time saving |
|---|---:|---:|---:|
| N64 | 132.410 | 132.182 | +0.17% |
| N128 | 132.158 | 134.468 | -1.75% |
| N256 | 146.032 | 159.733 | -9.38% |

All solves meet tolerance. The gate fails, and **seed 9626 is not generated or
tested**. These are validation results, not a new held-out claim. There are
288 measured validation solves in this gate. Only the two-step pilot improves
the selected work metric, but this is not a controlled loss-only ablation:
seed, batch size, and training schedule also differ. It still does not deliver
the desired runtime advantage.

Resolution transfer is specifically weak: old versus lookahead-trained classical
calls are 7.625 versus 7.375 at N64, 7.6875 versus 7.625 at N128, and 7.6875 versus
7.6875 at N256. Thus the N64 gain cannot be assumed to survive refinement. In a
follow-up eligible-prefix audit, the lookahead model's N64 validation immediate
residual ratio is 0.6815 and phase-equivariance defect is 0.1766. It outperforms
the attention model in full-solver work despite having a worse immediate ratio
(attention: 0.5531), reinforcing the need to measure the later trajectory.

Artifacts: `tail_lookahead_smoke_20260926/`, `tail_lookahead_20260926/`, and
`tail_lookahead_gate_20260926/` under `rte/workstation_runs/`. Together with the
base and attention pilots, the campaign performs 5,000 substantive optimizer
updates plus the separately retained one-update smoke check. No training run
uses held-out test fields or test-teacher solutions.

## Paired larger-resolution bank prepared

`rte/workstation_runs/tail_multiresolution_bank_20260926/` now contains accurate
N128 teachers and actual four-step contexts for the same 128 training and 16
validation material/source draws as the N64 bank. Nested material values and
source/geometry parameters are verified exactly. There are 112 eligible train
tails and 13 eligible validation tails; these are paired resolutions, not 144
additional independent draws.

Maximum teacher residuals are 9.952e-10 (train) and 9.745e-10 (validation).
Maximum normalized residual overlap with the search-image basis is 2.787e-11
and 1.421e-11, respectively. The saved parent/context artifacts occupy about
3.4 GiB. This is preparation only: **mixed-resolution neural training has not
yet been run**. It enables the next controlled comparison without redoing
teacher acquisition or touching the unused test seed 9626.

## Research cross-check and next architectural constraint

Differentiating through later Krylov steps is established research, not a new
claim here. Dimola, Coclite and Zunino explicitly contrast a static residual
objective with a differentiable flexible-GMRES objective; their experiments are
on different PDE systems and do not establish an RTE runtime advantage.
[Neural Preconditioning via Krylov Subspace Geometry](https://arxiv.org/html/2507.15452v1).

A recent controlled architecture study separates source linearity from how
inputs are projected onto the output basis. It supports testing an operator
that is nonlinear in material but linear in residual, with compatible
restriction/prolongation. It also warns that an inadequate fixed coarse space
cannot be repaired merely by changing the surrogate architecture. Its
diffusion/advection results and preliminary runtime comparison do not transfer
automatically to this periodic RTE.
[Melchers, Dolean and Abdelmalik, July 2026 revision](https://arxiv.org/html/2605.19867v3).

The next architecture hypothesis is therefore a material-conditioned,
complex-linear action on a direction-aware multiscale basis, trained against
post-classical error. This is distinct from learning a dense raw inverse or
repeating the unsuccessful fixed radial coarse-space trials. The nonlinear
material branch may learn global multiple-scattering coupling; the residual
branch should not spend capacity relearning phase and amplitude symmetry.
This remains a hypothesis, not a demonstrated winner.

### Concrete next study, not yet run

First isolate the demonstrated resolution-transfer problem: continue the same
cheap tail architecture and continuation loss on paired N64/N128 contexts,
select on both validation grids, and test N256 only after a runtime gate.
Compare with N64-only training at the same update/sample budget. Do not change
architecture simultaneously and then attribute any gain to structure alone.
The paired N128 data preparation is saved separately and is not neural training.

Keep one-shot injection and exact final residual checks. The next architecture
ablation, if needed after this controlled data comparison, is a factored
material-conditioned action

`z_theta(r, sigma) = B0 r + P_sigma C_theta(sigma) R_sigma r`.

The material encoder may be nonlinear. The entire residual path must be
complex-linear, with no residual-dependent normalization, biases, or activation
functions inside that path. Use compatible projection and reconstruction on
a direction-aware multilevel basis; let its effective dimension grow with grid
size instead of reusing the failed tiny fixed-rank/radial spaces. Learn the
compressed action directly, not a dense inverse assembled during deployment.

The potentially expensive information to amortize is heterogeneous, global
multiple-scattering coupling in the hard residual space. The neural encoder
should predict that interaction without per-medium fine-grid eigensolves or
inverse construction. Known streaming, local collisions, homogeneous inverses,
and optical distances remain analytical; none is an AI contribution by itself.

Use the new differentiable continuation to train against the *actual tuned*
classical continuation, including TSA16/zero initialization where appropriate.
Do not train exclusively behind TSA8 and then imply parity with a stronger
baseline. Compare two-step and longer, tolerance-aware horizons before paying
for a larger model. Enforce phase, complex-amplitude, and additivity tests by
construction, then use a four-case fit control to check representational
sufficiency before generating a larger mixed-resolution training bank.

Only scale the data/architecture after that control succeeds. A subsequent
runtime gate should charge all material-encoder/setup costs as well as online
application, retain a matched non-neural compressed-space control, and require
a clear win over tuned classical methods on larger grids. The present results
do not justify another generic increase in CNN width or another optical-attention
training run with the same immediate-loss objective.

## Preservation and reproducibility

All earlier checkpoints and negative runs are preserved. The old main handoff
has a byte-identical snapshot at
`rte/workstation_runs/tail_solver_report_20260926/previous_handoff_snapshot.md`,
SHA-256 `c1c8099a2182d562aec2f2112d393e6f836179ac189548cac5c9ac17b9b02ebd`.
The historical characteristic-report manifest is not rewritten: its handoff
entry resolves explicitly to that snapshot after the main handoff is updated.
The original first tail training script likewise has an exact execution snapshot
in its run directory. Later trainer extensions do not erase that provenance.

`python -m rte.report_tail_solver` verifies the earlier manifests and executed
source snapshots, then creates the new artifact manifest and provenance audit.
No commit, push, deletion, or external write is part of this campaign.

## Final workstation state

These are single-run pilots, not replicated architecture/loss ablations. The
same 16 validation cases are reused for development; the completed held-out
comparison uses 32 unseen cases, while the later continuation gate remains
validation-only. All physics tests use one synthetic periodic family, fixed
angular count, anisotropy, and stiffness. Neither general RTE performance nor
boundary-condition transfer follows from these experiments.

All bounded runs, larger-grid tests, diagnostics, and paired-bank acquisition
are complete. No RTE research process remains running. The final suite passes
144 tests and 66 subtests. Original manifests verify at 178, 49, and 77 entries;
the prior 168-entry characteristic manifest verifies with its one explicit
handoff-to-snapshot mapping (the other 167 targets remain in place).

The objective of a significantly faster AI-enhanced solver is **not achieved**.
The most informative progress is a verified injection mechanism, an objective
that improves small-grid Krylov work, a measured loss of that gain with grid
refinement, and paired higher-resolution data ready for the next controlled
training study. Keep these distinctions when resuming; do not relabel the
teacher ceiling or the analytical-reference tail gain as an AI result.
