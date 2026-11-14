# September 19: review, stronger controls, and the next AI target

Status: completed bounded research continuation. The sections below distinguish measured
development results from proposed learned architectures. No new neural model
has been trained in this continuation.

## Review of the previous conclusions

The original 178-file handoff and the September 13 299-file report manifest
both verified unchanged before new experiments. The two-medium teacher-space
result reproduces from its saved table: every tested rank and control took
five iterations on all twelve held-out source translations. That negative
result remains valid **for that POD construction and source family**, not for
all global spaces, all hierarchical operators, or all physical regimes.

The earlier comparisons tested classical HSS only through depth eight. They
establish an advantage over that candidate grid, not over unrestricted
classical solvers. Deeper HSS and stronger heterogeneous/coarse controls must
be considered before promoting an AI gain. The previous GPU contention caveat
also remains; new measurements do not retroactively clean old timings.

The suggestion to vary physical stiffness instead of only pixel count is
supported by the new matched-grid survey. The suggestion that a small global
rank would suffice is not established. Nor does capturing solution-error
energy establish residual reduction or fewer Krylov iterations.

## A more structured learning target: even/odd Schur response

Pair direction l with l+L/2 and use the orthonormal variables

\[
e_l=(\psi_l+\psi_{l+L/2})/\sqrt 2,\qquad
o_l=(\psi_l-\psi_{l+L/2})/\sqrt 2.
\]

For antipodal directions and parity-preserving symmetric collision, the
operator has blocks

\[
\begin{bmatrix}C_e&D\\D&C_o\end{bmatrix},\quad D^*=-D.
\]

The odd collision block is local in space and invertible. Eliminate it:

\[
H e=b_e-D C_o^{-1}b_o,\qquad
H=C_e-D C_o^{-1}D=C_e+D^*C_o^{-1}D,
\quad o=C_o^{-1}(b_o-De).
\]

Thus H is Hermitian positive definite, with six angular unknowns per cell
instead of twelve in the present L12 family. The odd field is recovered, not
discarded. This is a classical structural reformulation, not an AI invention;
even/odd transport preconditioning and CG have precedents such as
[Li and Wang, 2016](https://arxiv.org/abs/1602.00746). The formulas above are the
specific periodic steady-state construction implemented here.

The stored FP32 quadrature/phase has small antipodal and symmetry rounding
defects. `parity_transport.py` therefore constructs an explicitly symmetric
**reference**, never silently replaces the physical operator. All benchmark
convergence tests use the original operator promoted to FP64. Inner PCG with
an iteration cap is a nonlinear residual map; it is used inside flexible
GMRES, not assumed to be a fixed linear preconditioner.

This caveat is numerically consequential. Direct high-accuracy Schur solves
on the four epsilon=.1 survey media leave physical residuals about
1.6e-5–3.5e-5, above the required 1e-5, despite small Schur residuals. Outer
physical correction is necessary; a converged symmetric-reference solve is
not automatically a converged solve of the original stored matrix.

### Classical controls now available

- A homogeneous matrix-valued Fourier inverse for H, with local collision
  blocks retained exactly in the action.
- A cheap analytic congruence scaling sqrt(sigma_t/mean(sigma_t)) around that
  inverse, tested rather than assumed to help.
- An additive positive exact Galerkin coarse correction retaining all six
  even angular components, not only a scalar flux.

Crucially, exact coarse assembly need not apply every basis vector on the
fine grid. For a strict m-grid Fourier band, band-limit the entries of C_e
and C_o^-1 to a 2m product grid. Derivatives commute with the transfer and
the retained convolution is alias-free. This compressed action agrees with
direct fine-grid R H P in the unit tests. The excluded coarse Nyquist
complement gets a positive diagonal extension. No Gibbs-clipped surrogate
coefficient or unreported physical-operator change is introduced.

This cheap classical construction removes one proposed source of AI setup
advantage. Learning a fixed-grid coarse matrix is not compelling if its exact
classical construction is already cheap. Material-adapted spaces, coupling
compression, or a much cheaper inverse action remain the possible targets.

## Same-medium stiffness/refinement survey

`parity_stiffness_survey_20260919` evaluates four continuous mixed-regime
material/source realizations at N64 and N128, L12, g=.65. Nested-grid material
samples agree bit-for-bit. Each is transformed using

\[
\sigma_t^\epsilon=\sigma_t/\epsilon,\qquad
\sigma_s^\epsilon=\sigma_t^\epsilon-\epsilon(\sigma_t-\sigma_s),
\quad\epsilon\in\{1,.25,.1\}.
\]

These are **four independent media**, not 24 independent observations: grid
and epsilon measurements are paired/correlated. Source realizations and
physics are saved. Each method has preparation plus first solve and two
warmed same-RHS repetitions; method order is randomized. All failures and
the 100-step cap are retained. This is development screening, not tuning on
one set followed by an untouched confirmation set.

Mean outer iterations for HSS8 with the original analytic shift:

| epsilon | N64 | N128 |
|---|---:|---:|
| 1 | 4.50 | 4.50 |
| .25 | 16.50 | 16.50 |
| .1 | 39.00 | 40.00 |

The frozen N64 SCT checkpoint, with two repetitions, takes 3.25 mean steps
at epsilon=1 on both grids. At epsilon=.1 it fails all four media on both
grids within 100 steps. This is **out-of-distribution transfer without
retraining**, not evidence that an appropriately trained network cannot help.
At epsilon=.25 the transferred learned iteration counts differ across grids;
the near-resolution-invariance conclusion applies to the classical HSS
observations, not universally to the neural model.

For N128, epsilon=.1, selected illustrative median warmed timings are:

| Method | Mean outer iterations | Median warm ms | Failed media / 4 |
|---|---:|---:|---:|
| HSS8, shift 1 | 40.00 | 388.56 | 0 |
| HSS16, shift 2 | 15.50 | 233.16 | 0 |
| Parity, 16 FP64 inner PCG steps | 8.00 | 218.90 | 0 |
| TSA8 | 9.75 | 167.46 | 0 |
| Full-reference8 | 36.25 | 165.58 | 1 |
| Frozen SCT | 100.00 | 1109.21 | 4 |

Full-reference8's attractive median does not make it the robust winner:
one material failed. Parity is faster than the weak HSS8 choice here, but
**not** faster than the successful TSA8 control. Positive-definiteness alone
does not solve heterogeneous conditioning. No parity speedup over the
strongest robust classical solver is claimed.

An earlier two-medium pilot uses a different dataset RNG draw because the
generator depends on the requested count. Its lower iteration counts must
not be substituted for the four-medium survey. Both directories are retained.

## Response compression: positive representation evidence, then a failed solver test

The dense block audit uses N16/N32 versions of the same continuous first two
survey media at epsilon=1/.1. It inverts the even Schur operator accurately;
the worst dense inverse residual is below 8e-13. Each field has six angular
components. Spatial patches occupy one quarter of the domain width, with
periodic same/adjacent/separated targets. These are sampled blocks, not an
entire hierarchical solver. The separated target is offset by half the
domain in both coordinates.

At 1% relative **block Frobenius** error, mean required ranks are:

| Grid / block dimension | epsilon | Full inverse far block | Inverse-minus-Fourier-reference far block |
|---|---:|---:|---:|
| N16 / 96 | 1 | 28.0 | 42.0 |
| N32 / 384 | 1 | 49.5 | 79.5 |
| N16 / 96 | .1 | 1.0 | 15.5 |
| N32 / 384 | .1 | 1.0 | 19.5 |

There is regime-dependent compression, not a resolution-independent global
rank-eight structure. Subtracting the reference cancels dominant smooth
responses and can make the remaining block less relatively compressible.
The diffusive rank-one observation is not an AI advantage by itself; a
classical scalar diffusion mechanism can already exploit part of it.

An end-to-end N16 oracle then keeps all same/adjacent patch blocks of H^-1
exact and truncates **all** separated blocks. It evaluates three source
translations on each of the two media, with the original full-angle physical
residual and tolerance. This oracle computes/stores dense matrices; its
iteration counts are not a deployable timing result.

Naive 1% Euclidean far-block truncation stalls on all six moderate-regime
RHS and three of six stiff RHS. Even a small relative inverse error can be
amplified by H. Therefore the block singular spectrum alone did **not**
justify starting a large neural training campaign.

### Residual-weighted compression is the more promising target

For output patch I, factor H[:,I]=Q_I R_I. Compress R_I B_IJ, then map back
with R_I^-1, where B=H^-1. This minimizes the output-block contribution in
the Schur residual norm at fixed rank. Directed blocks are compressed
independently, so Hermitian positivity is **not** guaranteed; the experiment
uses outer FGMRES, not CG with this approximate inverse.

At a tighter .1% local truncation tolerance:

| epsilon | HSS8 mean steps | Full-reference8 | Euclidean oracle | Residual-weighted oracle | Exact Schur oracle |
|---|---:|---:|---:|---:|---:|
| 1 | 4.33 | 5.83 | 16.83 | 5.33 | 1.00 |
| .1 | 38.50 | 15.00 | 8.50 | **4.00** | 2.00 |

All methods in this table converge on all six RHS per regime. A looser 1%
residual-weighted truncation still fails three of six RHS in each regime.
The metric matters, and accuracy requirements are substantially tighter than
the initial low-rank plots suggested. The moderate-regime weighted oracle
still does not beat HSS8 even by iteration count.

There is a major storage limitation: estimated block-factor storage remains
96.3% of dense at epsilon=1 and 76.3% at epsilon=.1 for the successful weighted
oracle. These estimates leave near blocks exact and do not count a symmetry
storage bonus; the actual diagnostic application is dense. The stiff result
identifies useful information to learn, **not** a compressed fast solver.
A recursive, solver-aware hierarchy and cheaper near-field representation
must demonstrate better scaling before neural training is promoted.

## Coarse and direction-aligned classical controls

On the first two N64/N128 survey media, adding the exact strict-band N8 coarse
inverse to the homogeneous Schur preconditioner gives little or no benefit.
For N128 epsilon=.1, mean outer iterations change from 15.5 to 15.0, while
extra application/setup work makes the observed solve slower. Analytic
sqrt(sigma_t) congruence scaling is harmful and frequently reaches the
100-step limit. It is retained as a rejected candidate, not assumed useful.
Compressed coarse actions match direct R H P to at most 8.4e-16 relative
error on the saved probes, so inconsistency is not the explanation here.

A second control selects 49 Fourier modes per even angle by the score
(Omega dot k)^2 + gamma |k|^2, gamma=.001/.01/.1. This includes weak-streaming
high-transverse-frequency components omitted by radial coarsening. Its
294-by-294 Galerkin matrix is assembled directly from Fourier coefficients
of C_e and C_o^-1; no full-grid basis-response solves are required. Projection
agrees with direct action to below 7.4e-16 on the probes.

On the two N64 media, gamma=.001 reduces mean outer iterations from 13 to
10.5 at epsilon=1, and from 15 to 12.5 at epsilon=.1, versus the unaugmented
Schur preconditioner. It remains slower than the strong full-angle controls.
This is useful evidence for phase-space directionality, not a speedup claim.
The radial grid has 294 active strict-band modes plus a 90-dimensional
positive Nyquist extension, whereas the directional factor stores only its
294 active modes; timing differences are not a pure directionality ablation.

## Expanded-depth audit of the original 24-case comparison

`depth_review_n128_gpu_20260919` reuses the exact saved-comparison fingerprint
from September 13. Six separate selection media choose settings from 58
classical candidates and four frozen CT/SCT repetition candidates. HSS now
includes depths 16/32; TSA and full reference extend to 16. There is no new
training. All eight selected methods converge on all 24 comparison cases and
all repetitions.

| Selected method | Mean iterations | Median warm ms | Preparation + first ms |
|---|---:|---:|---:|
| Compiled HSS16, shift 1 | 2.4167 | **44.84** | **54.60** |
| HSS16, shift 1 | 2.4167 | 49.10 | 63.88 |
| CT, two repetitions | 3.4167 | 46.58 | 62.30 |
| SCT, two repetitions | 3.2500 | 46.68 | 64.32 |

Against compiled HSS16, CT and SCT each win only 10/24 warmed pairs. Median
paired warm savings are **-3.43% and -3.90%**, respectively; first-solve
savings are -10.16% and -17.38%. Extra classical stages reduce expensive
outer-Krylov work even when total factor stages do not decrease.

These timings are contended and provisional. Still, the old measured AI
advantage is **not robust to expanding the classical candidate grid**.
The data do not support a current general claim that CT/SCT beat classical
methods. They also do not prove that classical HSS16 universally dominates:
the same cases are reused, the device is shared, and CPU/other regimes need
their own tuning. Keep the earlier raw results, but narrow their conclusion.

## What will justify a larger network

The candidate is a coefficient-conditioned **hierarchical inverse of H**,
with local/full-angular classical physics and learned inter-region coupling.
The nonlinear material encoder runs once per medium. The cached residual
action should be linear and preferably positive, for example an additive
multilevel form with matched restriction/prolongation and positive coarse
factors. Parameterizing a factor is safer than predicting an arbitrary dense
inverse, but does not by itself guarantee good conditioning or low cost.

The encoder needs global context at coarse levels and direction-dependent
features at finer levels. Its learned outputs are adaptive transfer/response
factors across spatial scales, not just a scalar shift or pointwise gain.
The odd angular response is recovered analytically; full original-range
corrections remain available through outer FGMRES.

Before a substantial training run, require a useful classical/oracle
hierarchy on held-out RHS and a measured cost budget. Use the exact cheap
Galerkin construction above and stronger adaptive classical coarsening as
controls. Hierarchical compression is itself classical; for related RTE
integral formulations, see [Fan, An and Ying](https://arxiv.org/abs/1802.03061).
An AI contribution would be amortizing material-dependent setup or achieving
a cheaper accurate action across new media, not merely using a hierarchy.

In the diffusive regime, a variable-coefficient DSA/AMG comparison is still
required. Heterogeneous DSA has substantial classical developments beyond
the averaged-coefficient control in this repository; see
[Southworth, Holec and Haut](https://arxiv.org/abs/2001.09196). That control is
not yet implemented here. Do not promote an AI claim without it.

Training should target short true-residual Krylov trajectories, with
projector/response distillation only as an auxiliary objective. Include
physical stiffness and contrast in the training distribution; retain
separate material and source splits. Count encoder/setup costs, distinguish
one RHS from multiple distinct RHS, and preserve the prior 30% paired
time-to-tolerance promotion criterion against the strongest successful
control. Do not launch a larger generic network solely because a response
block has a decaying singular spectrum.

## Decision and next bounded steps

Prioritize the **physically stiff, residual-weighted hierarchical response**
direction. Do not train a bigger generic correction network, a flat global
POD space, or an ordinary low-frequency Schur coarse inverse based on these
results. None has demonstrated the needed deployment advantage.

1. Build a recursive energy/residual-aware response oracle with inexpensive
   near-field factors; test N32/64/128 and multiple source shapes/directions.
   It must retain the stiff oracle's iteration benefit while reducing storage
   and measured action cost substantially. Current 76% dense storage is not
   enough. Include relative-energy/factor formulations if an SPD action is
   desired; the current residual-weighted construction is not SPD-guaranteed.
2. Compare with variable-coefficient DSA/AMG and a classical adaptive
   hierarchy, in addition to deeper HSS/TSA. Maintain the original physical
   operator and all failures.
3. Only if that oracle leaves time-to-tolerance headroom, train a multiscale
   material encoder to predict its expensive response/transfer factors on
   independent media. Use short physical-residual FGMRES loss. Separate
   teacher acquisition, neural inference/setup, action cost and multiple-RHS
   amortization. A new model must beat both matched CT and the expanded
   successful classical controls, not merely the old HSS8 baseline.

## Timing limitations, artifacts, and verification

The GPU was initially idle. Unrelated process 569434 appears in four
stiffness-survey snapshots, starting at N128 epsilon=1 medium 1. Process
571430 appears in the final survey snapshot: five of 24 snapshots in total
contain a competing research process. The latter process is also present in
every multilevel/directional snapshot and all 397 expanded-depth snapshots.
A direct device check observed roughly 6 GiB use and substantial utilization
during the later campaign. Neither process was stopped or modified.
Endpoint sampling cannot exclude brief interference between
snapshots, so do not describe the full campaign as uncontended. Iteration,
identity, residual and compression conclusions are the primary evidence.

Artifacts under `rte/workstation_runs/`:

- `parity_stiffness_pilot_20260919`, `parity_stiffness_survey_20260919`:
  saved materials/checkpoint, randomized first/warm samples, traces and
  physical-versus-reference diagnostics.
- `parity_response_blocks_20260919`: N16/N32 block singular spectra and
  verified dense-inverse checks.
- `parity_multilevel_control_20260919`,
  `parity_characteristic_control_20260919`: positive coarse/scaling controls,
  exact projection checks, failures and telemetry.
- `hierarchical_response_oracle_20260919`,
  `hierarchical_residual_weighted_oracle_20260919`: full-solve compression
  tests, including all capped/failed cases.
- `depth_review_n128_gpu_20260919`: separate selection and original 24-case
  comparison, frozen checkpoint copies and full candidate records.
- `parity_research_report_20260919`: normalized summaries, paired depth-review
  comparisons, provenance and hashes of the completed continuation.

The report hashes record the final reproducibility state, not a retroactive
execution-time Git commit. Existing dirty-worktree edits and all historical
results remain preserved. No RTE experiment is left running after handoff.
The full RTE suite passes **105 tests**, including parity/block recovery,
Hermitian positivity, exact radial/directional Galerkin identities, cache
invalidation and residual-weighted compression checks.
