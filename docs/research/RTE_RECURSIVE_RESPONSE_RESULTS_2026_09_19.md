# Recursive response continuation, September 19, 2026

This follows the [parity/structure review](RTE_PARITY_STRUCTURE_REVIEW_2026_09_19.md)
and the user's instruction to pursue the most promising direction. It tests
the **actual online representation**, rather than estimating storage for an
approximation still applied as a dense matrix. These are development oracles,
not trained AI. The goal of a generally faster AI-enhanced solver is not met.

## Scope and controls

The physical operator is unchanged: periodic Fourier RTE, L=12, g=.65,
complex fields, original FP32 material and geometry promoted unchanged to
FP64 for outer FGMRES. All reported convergences require an independently
recomputed full-angle physical relative residual <= 1e-5. Maximum iterations
are 100, restart length 25. Odd angles are recovered, not discarded.

The two original development media are indices 0/1 of a **four-medium** draw
with seed 1919. Keeping the draw count is necessary to reproduce the material
fields. N16/N32 response experiments use three source translations per medium;
coarse-composition pilots use one source per medium at N64/N128. These are two
independent media, not independent samples for every source, precision, grid,
or regime. There is no held-out-material neural evaluation in this campaign.

Unrelated jobs occupied the GPU at the resource checks. These runs use CPU:
four threads for oracle acquisition, one for online timing. Methods are
shuffled within each repetition. Main response runs use three measured
repetitions after a warm solve; energy/coarse pilots use one. The host is
shared and some short correctness checks ran concurrently, so timings are
exploratory, not an isolated performance benchmark. Setup plus warmed solve
is explicitly an **estimate**, not a measured cold end-to-end solve.

## Implemented recursive representation

`rte/hierarchical_response.py` implements a periodic spatial quadtree with all
six even-angle components in each cluster. Same/adjacent patches are refined;
separated patches are compressed. Unprofitable far blocks are refined too,
down to 2x2 spatial leaves. Directed low-rank blocks use ranks padded to four;
padding, dense leaves and int64 gather/scatter indices are all counted.

The online action contains only grouped batched products and gather/scatter,
not a hidden dense inverse. Dense and low-rank blocks are grouped by shape
and rank. Complex adjoint action, multiple RHS, FP32 caches, and stale-material
guards are implemented. Teacher H/H^-1 are not retained by the hierarchy.
Dense teacher generation remains a diagnostic restricted to N<=32.

Residual weighting compresses R_I B_IJ, where

\[
H[:,I]=Q_I R_I,\quad B=H^{-1}.
\]

Consequently the block truncation measures
||H[:,I](B_IJ-Btilde_IJ)||_F. This is **not** a global defect bound: errors from
different blocks can accumulate coherently. Weighted directed compression
does not guarantee Hermitian positivity; outer FGMRES is essential.

### Main result: the N16 benefit does not scale unchanged to N32

Times below are means across six RHS cases of each case's median warmed CPU
time. The classical reference is the fastest successful *reported* candidate
pooled across the regime, not independent validation tuning. All candidates
and failures remain in the CSV files.

| Grid/regime | Method | Mean iterations | Warm ms | Failed RHS |
|---|---|---:|---:|---:|
| N16, epsilon=.1 | residual hierarchy, FP32 | 4.33 | 30.86 | 0/6 |
| N16, epsilon=.1 | full reference8 | 15.00 | 137.42 | 0/6 |
| N16, epsilon=.1 | TSA8 | 9.83 | 144.04 | 0/6 |
| N32, epsilon=.1 | residual hierarchy, FP32 | 7.17 | 232.97 | 0/6 |
| N32, epsilon=.1 | full reference8 | 14.67 | 222.26 | 0/6 |
| N32, epsilon=.1 | TSA8 | 9.33 | 204.60 | 0/6 |
| N32, epsilon=1 | residual hierarchy, FP32 | 53.67 | 2435.71 | 3/6 |
| N32, epsilon=1 | HSS8, shift1 | 4.33 | 65.43 | 0/6 |

The hierarchy tolerance is .001. Its stiff-regime stored coefficient fraction
falls from **72.59% at N16 to 41.86% at N32**, but the N32 FP32 cache still
occupies about **127.2 MiB per medium including indices**. The dimension rises
fourfold and the cache grows much faster than fourfold. Two grids do not
establish an asymptotic complexity law. A smaller fraction of a dense matrix
is not by itself adequate evidence of scalability.

N32 teacher acquisition takes about 11-12 seconds and residual compression
about 21 seconds per stiff medium on four CPU threads. Thus even the N16
warmed benefit does not establish a first-solve advantage for this classical
oracle. An eventual learned setup would have to be measured, not assumed free.

At N32 the harder stiff medium has random-probe inverse-action relative error
about 9.3e-4, yet residual-defect ratio about .206. In the moderate harder
medium, the corresponding figures are about .00159 and .375, with every
translated-source solve failing at 100 steps. FP64 caches do not remove this
failure. This is predominantly a representation/metric problem, not simply
FP32 roundoff. The exact dense reference needs only 1-2 outer iterations.

## Two-sided energy metric control

The optional energy metric uses R_I^* R_I = H_II and compresses
R_I B_IJ R_J^*. It measures the Frobenius energy error of an individual block
embedded in the full space. Unit tests check that identity explicitly.
It does not certify a positive global approximate inverse.

On the same six stiff N16 RHS:

| Metric/tolerance | FP32 iterations | Failures | Coefficient fraction |
|---|---:|---:|---:|
| residual .001 | 4.33 | 0/6 | .726 |
| energy .003 | 5.67 | 0/6 | .709 |
| energy .01 | 54.00 | 3/6 | .547 |

This bounded energy control does not solve the compactness/accuracy problem.
It is not evidence against all factor-based or recursively preconditioned
methods. [Feliu-Faba, Ho and Ying](https://arxiv.org/abs/1808.01364) specifically
use block preconditioning before hierarchical skeletonization to improve
robustness in ill-conditioned elliptic problems; the independently truncated
inverse blocks tested here are not that algorithm.

## Fixed coarse-response deployment check at N64/N128

`rte/coarse_response_transport.py` combines a classical fine HSS correction
with an exact strict-band Galerkin Schur response at m=8 or m=16. Coarse
construction uses the existing exact 2m product-grid method, not rediscretized
averaged coefficients. The m16 FP32 inverse cache is **18 MiB**, independent
of fine N. This tests whether fine-grid storage can be avoided altogether.
The initial pilot includes both Cholesky solves and cached inverses so that
an unnecessarily expensive inverse-building baseline cannot create an AI win.

The first composition recovered the complete odd RHS after the coarse even
correction. All 12 such coarse configurations failed on both N64 media.
Exact Cholesky and cached-inverse variants agree on the failure. Resetting
the fine odd response is an additional operation, not merely a correction in
the coarse space; it can spoil a useful fine correction.

The follow-up isolates harmonic-only lifting:

\[
Z e=(e,-C_o^{-1}D e),\qquad A_0 Z e=(H e,0).
\]

It preserves the odd residual image of the fine complement, as checked by a
unit test. This is substantially better than the full odd reset, but still
not competitive. Each row below contains two media, one source each:

| Grid | Method | Mean iterations | Warm seconds | Failures |
|---|---|---:|---:|---:|
| N64 | HSS8 + m16 harmonic correction | 44.0 | 2.718 | 0/2 |
| N64 | HSS4 + m16 + HSS4 | 45.5 | 2.925 | 0/2 |
| N64 | TSA8 | 9.5 | .576 | 0/2 |
| N128 | HSS8 + m16 harmonic correction | 45.5 | 11.077 | 0/2 |
| N128 | HSS4 + m16 + HSS4 | 46.0 | 11.910 | 0/2 |
| N128 | TSA8 | 9.5 | 1.907 | 0/2 |

HSS4 with a coarse correction and no post-step also fails on the harder
medium at both grids. A fixed Fourier low-frequency response is therefore
not a supported neural target for these particular compositions. This does
not rule out properly designed adaptive/directional multilevel methods.

A physical minimum-residual scalar safeguard is implemented and unit-tested:
it costs one additional A application, chooses the optimal real damping of
the proposed correction, and cannot increase that stage's physical residual
norm in exact arithmetic. It makes the preconditioner nonlinear. It was not
used in the tables above, and is not claimed to improve wall time.

## Global coherence repair

The last bounded test asks whether independently compressed blocks leave a
small, globally coherent residual defect. With E=I-H K0, estimate leading
right singular vectors V of E using a randomized sketch, the actual grouped
action and its adjoint. The dense teacher then supplies the repair

\[
K=K_0+(H^{-1}V-K_0V)V^*,\qquad
I-HK=(I-HK_0)(I-VV^*).
\]

For orthonormal V this cannot increase the Schur residual-defect spectral or
Frobenius norm in exact arithmetic. It does not guarantee faster outer
full-physics FGMRES. The teacher inverse used to obtain H^-1 V is charged to
setup; this is not an online learning or calibration shortcut. Rank8/32 add
only 2kD coefficients to the hierarchy. This differs from the earlier failed
global rank8 augmentation of HSS: the target here is the compression defect
of a substantially different response operator.

The matched N32 stiff-regime repeat gives:

| Method, FP32 cache | Mean iterations | Warm ms | Failed RHS |
|---|---:|---:|---:|
| hierarchy only | 7.17 | 271.91 | 0/6 |
| hierarchy + rank8 repair | 6.67 | 254.11 | 0/6 |
| hierarchy + rank32 repair | 6.50 | 249.74 | 0/6 |
| full reference8 | 14.67 | 230.62 | 0/6 |
| TSA8 | 9.33 | 212.79 | 0/6 |

The sketch/acquisition takes another 3.1-3.3 seconds per medium, and the
rank32 repair adds 3 MiB to each FP32 cache. On the harder medium its probe
defect falls only from .2061 to .1896. The tested repair is helpful but small:
it does not produce a wall-time advantage. It is a randomized rank32 test,
not a proof that all globally shared/nested bases are ineffective. Its
timings are compared with the **controls in this same run**, not with faster
numbers from an earlier measurement window.

## Architectural implications

The expensive information remains the **material-dependent nonlocal
response**, but predicting every block independently is not enough. A
candidate learned architecture must share global material context and
cross-level response bases, retain exact local/angular physics, and be trained
against the original physical residual or short Krylov solves. Entrywise
inverse/factor MSE is insufficient; SVD basis signs and rotations also make
naive factor supervision inappropriate.

[Fan et al.'s nested-basis neural architecture](https://arxiv.org/abs/1808.02376)
provides a relevant H2/FMM-inspired parameter-sharing design, including RTE
solution-map examples. That precedent does not establish preconditioning
accuracy or speed for this full-angle periodic discretization. A prospective
material encoder would cache nested restriction/lifting bases and interaction
factors once per medium, with a linear residual action afterwards. The
current code implements a directed H-matrix, **not** shared nested H2 bases.

[Xu, Li and Xi](https://arxiv.org/abs/2510.13034) learn Green-function
representations and compress them into preconditioners for elliptic PDEs.
This supports the general amortized-response idea, not a guarantee of cheap
full-angle RTE response ranks. Predicting a coefficient-conditioned inverse
across independent media and proving a net setup-plus-solve benefit remain
separate tasks.

No larger generic network should be promoted merely because a small-grid
oracle lowers iteration counts. The next training gate remains a compact,
robust action with substantial measured headroom against the best successful
classical controls. Variable-coefficient DSA/AMG remains a missing strong
control before any general AI speedup claim.

### Decision: no neural training of the tested inverse-block representation

The previous small-grid conclusion is narrowed: residual-aware response
information is valuable, but the current independent inverse-block format
has not met the cost/robustness gate. Recursive storage, lower precision,
two-sided energy weights, a fixed coarse band, and the tested small global
repair do not establish the required deployment advantage. A network cannot
be credited with eliminating application/storage costs inherent in its
chosen output representation. No new neural model was trained in this turn.

The next **untested hypothesis** is to learn a *relative operator factor*,
not raw inverse entries:

1. Keep the exact local angular physics and homogeneous anisotropic Fourier
   reference H0. Let F=H0^-1/2, computable from small per-frequency angular
   matrices. Normalize T=F H F before asking a network to approximate a factor.
2. A global material encoder supplies direction-aware, shared nested bases
   and inter-patch factor interactions. Encode absolute optical scales,
   coefficient contrast, cell size and angle directions; do not normalize
   away the stiff parameter. Cache coefficient-dependent outputs once.
3. Use a positive factor form such as
   M_H=F (I+L_theta)(I+L_theta)^* F, with a nonsingularity safeguard. This
   retains the known high-frequency anisotropic reference analytically.
   Positivity alone is insufficient, and two factor applications must be
   included in the cost budget.
4. First test an oracle factor on the *same* two difficult media at N16/N32,
   measuring factor ranks, full physical convergence and actual action cost.
   Do not silently change to an easier material or source distribution.
5. Only if it clears the headroom gate, train the material encoder with
   disjoint material/source splits and short true-physics Krylov losses,
   then test N64/N128 matrix-free deployment. Compare with classical
   recursively preconditioned factorizations and heterogeneous DSA/AMG.

There is a concrete reason for the normalization, not a guarantee of neural
success. If local collision and odd-inverse blocks are bounded above/below
by fixed multiples of their reference blocks, the same quadratic-form
bounds hold for H versus H0. Consequently T's condition bounds depend on
material contrast rather than explicit derivative growth with N. Whether
the normalized inverse factor admits cheap directional/nested compression
is an open empirical question; it has **not** been measured in this turn.

## Artifacts and reproduction

All directories below are under `rte/workstation_runs/`:

- `recursive_response_n16_20260919`: residual/Euclidean, both precisions,
  both regimes, 396 measured solves.
- `recursive_response_n32_20260919`: residual, both precisions/regimes,
  324 measured solves, including failures.
- `recursive_energy_n16_20260919`: stiff energy-metric control,
  66 measured solves.
- `coarse_response_pilot_n64_20260919`: full odd-reset composition,
  exact Cholesky and inverse controls, 32 measured solves.
- `coarse_response_harmonic_20260919`: pure coarse-space correction at
  N64/N128, 28 measured solves.
- `recursive_response_repair_n32_20260919`: bounded global-repair study,
  234 measured solves.

Saved artifacts include actual coefficient/source datasets, settings,
physical residuals, failures, iteration counts, shuffled timing order,
teacher/setup costs, block-level ranks and actual cache/index byte counts.
No prior result directories were overwritten. The report script produces
pooled and paired summaries and a SHA-256 artifact manifest.

Example commands:

```bash
python -m rte.run_recursive_response_study --size 32 --count 2 \
  --metrics residual --epsilons .1 --repair-ranks 8 32 \
  --output rte/workstation_runs/NEW_UNIQUE_DIRECTORY
python -m rte.run_coarse_response_study --sizes 64 128 \
  --coarse-sizes 16 --count 2 --sources 1 --repeats 1 \
  --harmonic-only --cache-kinds dense \
  --output rte/workstation_runs/ANOTHER_NEW_UNIQUE_DIRECTORY
```

The earlier 178-file handoff and September 19 pre-continuation manifest were
reverified unchanged. The six runs contain **1,080 measured solves**, including
repetitions and failed methods, not 1,080 independent physical problems.
All bounded experiments completed; no RTE experiment is left running.
The full RTE suite passes **118 tests and 66 subtests**. The subsequent
construction cleanup explicitly breaks a recursive Python closure cycle so
temporary teacher buffers can be released promptly; it does not change the
stored factors or arithmetic. Reported experiment timings were not rerun
after that memory-lifetime cleanup.
