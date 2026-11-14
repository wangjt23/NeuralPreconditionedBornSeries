# Resumed high-resolution experiments and structural audit

## Decision

**Do not promote a larger neural operator or the new characteristic-coordinate
variant as a significant improvement over matched CT.** There is a moderate
learned-versus-classical advantage in this problem family, but adding neural
expressiveness has not produced the requested larger wall-time gain. The next
research target should be physical stiffness and heterogeneous coupling, not
resolution alone; see [the structure-driven plan](RTE_STRUCTURE_DRIVEN_PLAN_2026_09_13.md).

The suspended September 12 CPU processes were identity-checked and terminated
before fresh timing runs in new directories. Their partial output files were
preserved. No suspended-time measurements were reused. All runs described below
are complete; no RTE experiment from this task remains running.

## Completed comparisons

The September 12 frozen checkpoints were compared at N64/N128, L12, g=0.65.
Each comparison uses 24 saved material/source pairs (seed 100843), one first
solve and three warmed repeats, randomized method order, and tolerance 1e-5.
Outer physics/FGMRES arithmetic is FP64 with unchanged FP32 geometry promoted
to double; preconditioners remain FP32. This addresses the previously measured
FP32 residual floor without relaxing the requested physical tolerance.

Classical tuning considered 42 candidates: angular-FFT and compiled-matrix
HSS at depths 2/4/8 and shift factors .5/1/2; TSA, DSA, transport-DSA and full
reference at depths 2/4/8 and eta scales 0/.03. Learned four-stage blocks were
also tuned for one/two repetitions, so selected CT/SCT have eight effective
factor stages with shared weights/cache. The deeper HSS8 control is important.
CPU classical selection used six seed-542 media; CPU learned repetition used
six seed-843 media. GPU selection used six seed-843 media for all candidates.

All compared methods converged on every case and repetition. These are new
coefficient cases for the frozen September 12 models, not additional training
seeds. CPU/GPU reuse the same cases. The independently generated resolutions
are not a controlled same-medium refinement sequence.

### Median timings, milliseconds

Each value is a median over media; warmed values first take a median over the
three repetitions within a medium. First-solve values include preparation.

| Device/grid | SCT warm / first | Matched CT warm / first | Fastest observed classical warm / first |
|---|---:|---:|---:|
| CPU N64 | 88.30 / 103.41 | 91.57 / 105.08 | HSS8: 118.90 / 123.81 |
| CPU N128 | 345.23 / 389.18 | **327.75 / 361.51** | HSS8: 432.76 / 462.97 |
| GPU N64, shared | 31.76 / 41.31 | 34.04 / 41.20 | Compiled HSS8: 40.54 / 45.09 |
| GPU N128, shared | 31.63 / 39.18 | 35.24 / 40.59 | HSS8: 42.16 / 46.23 |

SCT's aggregate-median warm savings versus these classical controls are about
26%, 20%, 22%, and 25%, respectively. The **median paired savings** are only
22.8%, 19.2%, 16.2%, and 19.3%. These are different statistics: none passes the
previously stated 30% paired-median promotion gate. Preparation reduces the
advantage further. At CPU N128, the simpler matched CT is faster than SCT.

The denser NTO is not the preferred deployment model: its N64 CPU median warm
time is 176.34 ms with 328.30 ms preparation-plus-first, versus CT's 91.57 /
105.08 ms. Its N64-to-N128 GPU transfer needs 9.04 mean iterations versus SCT's
3.25 with two repetitions. N128 adaptation of SCT did not improve mean
iterations over its N64 weights on these cases: both are 3.25.

**GPU limitation:** the device was idle at the first workstation check, but
another process, PID 3453759, appeared before evaluation started. It is present
in every one of the 445 N64 and 541 N128 telemetry snapshots; a separate process
utilization sample observed it using GPU compute. Therefore neither run is an
uncontended GPU confirmation. No unrelated process was stopped. CPU runs use
one Torch thread on the shared 36-vCPU host, not an exclusive host reservation.

Warm repeats use the same RHS, not an independently tested multi-RHS workload.
Preparation-plus-first excludes process startup/model loading/initial operator
construction. These measurements do not establish process-cold deployment cost.

![Paired resolution comparison](../../rte/workstation_runs/resolution_structure_report_20260913/comparison.png)

## What the operator structure explains

### 1. Grid size is not the main HSS convergence difficulty

In the ideal discretization, A=S+C has skew-Hermitian streaming S and positive
Hermitian collision C, with local angular eigenvalues

\[
\lambda_m(x)=\sigma_t(x)-\sigma_s(x)p_m,
\]

where p_m comes from the actual discrete circular phase matrix. The classical
HSS spectral-radius bound depends on C's spectrum, not the streaming spectrum;
see [Bai, Golub and Ng](https://hub.hku.hk/handle/10722/276739). Floating-point
rounding and Euclidean nonnormal transients still matter; this is not a bound
on every one-step Euclidean residual or on learned variants.

For the optimal constant shift and condition ratio kappa_C, the bound is

\[
q=(\sqrt{\kappa_C}-1)/(\sqrt{\kappa_C}+1).
\]

On eight development media, mean kappa_C is 42.49 at N64 and 44.22 at N128;
mean q^8 is about 0.082 at both resolutions. HSS8 also takes exactly 4.458 mean
iterations in both frozen comparison sets. More pixels increase arithmetic,
but do not automatically create a harder preconditioned problem or more
opportunity for neural acceleration.

An **analytic, not solver-performance**, scaling check sets
sigma_t' = sigma_t/epsilon and sigma_s' = sigma_t' - epsilon(sigma_t-sigma_s).
At N128, epsilon=.25 raises mean kappa_C to 703 and mean q^8 to .508;
epsilon=.1 raises them to 4395 and .759. Those physically stiffer regimes are
a more relevant next challenge. No learned performance on these transformed
cases has yet been measured.

### 2. Residual energy and solution-error energy are different targets

`audit_transport_structure.py` uses four existing development media per
resolution, FP64 reference solutions verified below 1e-9, and three-step
FGMRES approximations. The characteristic mask is
abs(Omega_l dot k) <= mean(sigma_t); the coarse band is abs(k_i)<16.

At N128, about 91–96% of surviving **solution-error** energy is near weak
streaming directions. Yet only 0.8–1.7% lies outside the N32 spatial band for
HSS/CT/SCT; NTO transfer has about 9.0% outside. Thus the original hypothesis
that missing high spatial frequencies alone explains coarse-grid failure is
not supported. The corresponding residuals put substantially more energy
outside the coarse band (roughly 5–6% for HSS/CT/SCT). Residual-only losses do
not directly reveal the geometry of solution error.

At N128, scalar-angular error fractions average .381 for HSS, .195 for CT,
.217 for SCT and .155 for NTO. A scalar-only correction cannot be assumed to
capture all slow error in this moderate-anisotropy family. Retain the full
angular fine-grid correction in any macro/coarse design.

HSS is also not automatically a conventional high-frequency smoother. For
angle-zero manufactured errors on the four N128 media, the mean one-application
error ratio is .181 for k=(0,32), but **1.382** for k=(32,0). A convergent HSS
iteration can show one-step nonnormal amplification. An elliptic-style spatial
coarsening argument is insufficient for this transport operator.

### 3. The coarse operator and its inverse both matter

The previous two-level trial rediscretized cell-averaged coefficients. On the
actual surviving errors, this differs from R A P by **5.7–10.5%** in relative
action norm. A new dealiased Galerkin action uses a 2m product grid for a strict
m-grid Fourier band and agrees with direct fine-grid R A P to roughly 2.4e-15
relative error after preserving the fine quadrature/wavenumber tensors.

For four N128 development cases, starting from three HSS4-preconditioned
Krylov steps:

| Coarse correction | Remaining solution-error norm ratio | Residual ratio after two fine HSS steps |
|---|---:|---:|
| Unavailable oracle: exact band component of solution error | .0688 | .1291 |
| Four approximate coarse HSS steps | .2529 | .1750 |
| Converged rediscretized coarse solve | .1242 | .1455 |
| Converged Galerkin coarse solve | .0961 | .1286 |

Galerkin consistency helps, but is not a practical speedup here: a projected
full-reference preconditioner needs **63 mean inner iterations** to reach 1e-9.
Using the rediscretized HSS8 preconditioner instead caps at 120 iterations with
residuals between 5.9e-5 and 6.2e-4. The original diagnostic labeled these rows
`galerkin_exact`; that label was not justified by their residuals. Preserve the
first artifact as exploratory and use `coarse_consistency_refined_20260913`
for conclusions. The refined file distinguishes converged and capped solves.

An earlier compressed action also regenerated slightly different quadrature;
its discrepancy was about 2.2e-7. The final implementation copies the fine
geometry and passes direct-action tests at 1e-11 tolerance. No benchmark claim
uses that earlier approximation as an exact Galerkin identity.

## Tested structural neural variant: CCT

`characteristic_collision_transport.py` replaces separate Cartesian spectral
heads with a direction-shared MLP in normalized along-ray and transverse
coordinates. The amplitude is even under frequency reversal, phase odd, and
the DC amplitude shared across angles. This continuous-frequency parity is not
a claim about a real-preserving even-grid Nyquist derivative. Filters are still
cached; no extra FFT is added to each CT stage.

CCT (4,708 parameters), a newly initialized SCT head (9,020 parameters), and
plain CT (1,340 parameters) start from the **same selected N128 CT weights**.
All receive 200 additional updates, the same 256 training materials, batch 2,
lr 3e-4, three-step solver-aware loss, and validation-only checkpoint selection.
The architectures are not parameter-matched; CCT also changes the DC constraint,
so this is a candidate comparison, not an isolated causal coordinate ablation.

A separate frozen comparison uses 24 new media, seed 100947, with repetition
and classical settings selected on six seed-947 media. All methods converge.
GPU timings are again shared-device observations.

| Candidate | Mean iterations | Median warm ms | Preparation + first ms |
|---|---:|---:|---:|
| CCT | 3.1667 | 30.40 | 39.32 |
| Matched CT | 3.2083 | **29.71** | **35.19** |
| Matched new SCT head | 3.2083 | 30.39 | 37.60 |
| Compiled HSS8 | 4.3333 | 36.53 | 39.48 |

CCT is faster than matched CT on only 2/24 warm pairs. Its p95 is 45.88 ms
versus CT's 40.54 ms. The extra structure produces a tiny iteration change,
not a deployment win. **Retain the implementation and negative result, but do
not promote CCT or extrapolate a breakthrough from its architecture.**

## Artifacts, verification, and limits

The five completed comparison directories are
`resolution_n64_final_{cpu,gpu}_20260913`,
`resolution_n128_final_{cpu,gpu}_20260913`, and
`structure_n128_fresh_gpu_20260913` under `rte/workstation_runs/`.
They retain datasets, selected checkpoint copies, full candidate tuning, raw
first/warm repetitions, failures and process telemetry. Three new training runs
are `cct4_structure_n128_seed402_20260913`,
`ct4_structure_control_n128_seed402_20260913` and
`sct4_structure_control_n128_seed402_20260913`.

The consolidated
[tables, conditional paired intervals, figure and hashes](../../rte/workstation_runs/resolution_structure_report_20260913/comparison.csv)
are generated by `report_resolution_structure.py`. Its hashes describe the
current reproducibility state, not a retroactively captured execution-time Git
tree. Bootstrap intervals condition on fixed trained weights and resample
paired media; they do not quantify training-seed uncertainty or cure hardware
contention. No claim of statistical significance across training runs is made.

The complete test suite passes **94 tests**. The original **178 handoff
artifacts verify unchanged**. No experiments, additional seeds, diffusion-regime
solver comparisons, or multi-RHS studies beyond those listed were run.
