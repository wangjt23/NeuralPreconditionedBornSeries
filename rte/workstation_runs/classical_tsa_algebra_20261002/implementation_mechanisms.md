# Cheapest distinct mechanism: remove redundant TSA streaming work

The recommended first check is an algebraic optimization of the strongest
classical trajectory, not another learned/coarse architecture. Keep the exact
existing TSA reference, all outer physical-defect updates, and the prescribed
depth/initialization. Eliminate only the streaming FFT pair used to form the
inner defect after the constant-coefficient transport action. This is an
implementation opportunity, with an exact-arithmetic identity and a finite
precision parity question. No novelty or measured speedup is asserted.

The isolated prototype is `deliverables/stepback/algebraic_tsa.py`, class
`AlgebraicTSA(ClassicalTransportStack)`. It overrides `single` only. The parent
still owns preparation, eager reference-inverse caching, coefficient refresh,
relaxation and the outer repeated-defect loop. Original project files remain
unchanged. This inspection and local CPU correctness testing use no new
training, teacher acquisition, held-out data, remote writes or solver benchmark.

## Actual discrete objects and identity

Let `S` be the stored periodic Fourier streaming operator, including its stored
directions, wave numbers and even-grid Nyquist convention. Let
`K_ij = phase_matrix_ij * weights_j` be the actual stored angular action, and
let `t(x)=sigma_t(x)`, `s(x)=sigma_s(x)`. The physical operator is

`A = S + t I - s K`.

The reference transport action is `G=(S+gamma I)^(-1)`, with spatially constant
`gamma = base.reference.sigma0 = mean(t)+eta` for each material. The full-angle
homogeneous transport-scattering action `R` is the same cached inverse used by
the original baseline. The original TSA stage is

`M r = G r + R(r-A G r)`.

With `z=G r`, exact arithmetic gives

`r-A z = (gamma-t) z + s K z`.

Therefore the conservative implementation is

```python
z = base.reference.G(r)
defect = (base.reference.sigma0.to(z.dtype) - rte.sigma_t.to(z.dtype))*z
defect = defect + rte.scattering_op(z)
return z + base.reference.reference_transport_solve(defect)
```

This identity does not require `R` to be exact, dense, symmetric or full-angle;
it leaves whichever original `R` action is selected intact. It does require
that the streaming symbols in physical `A` and reference `G` are the same and
that `G` is the inverse of that shifted streaming operator in exact arithmetic.
Use `base.reference.sigma0`, not an assumed mean or the physical operator's
possibly different `sigma0`. The identity also holds for positive reference
shift; restricting the initial project check to the selected `eta_scale=0`
baseline does not justify dropping this implementation detail.

The physical angular action is preserved exactly as code, with its rounded
phase and quadrature. No scalar-flux replacement, Fourier-diagonal phase
approximation, artificial symmetrization or low-rank truncation is introduced.
The prototype caches a pointer/version/grid/setup-generation guard, compares
the actual stored streaming symbols exactly on cache misses, checks equal
stored phase/quadrature, and retains its input tensors to prevent pointer reuse.
No streaming comparison is repeated across the 16 unchanged stages.

Code evidence: `remote_reference/rte/models.py:182-221` defines `sK`, `S`, `A`,
`V` and `G`; `models.py:244-274,315-333` defines the cached dense angular
reference and its application. `transport_baselines.py:25-50` constructs and
refreshes the reference; `:52-69` shows the current inner and outer defects.

## Cost identities, not elapsed-time predictions

Count calls to a forward or inverse **spatial** FFT on the full angular field.
The untruncated TSA baseline uses a dense angular matrix action, so no angular
FFT is included in this count. Reference setup/inversion happens before these
application counts and is unchanged.

| Application | Original | Conservative optimization |
|---|---:|---:|
| Single TSA: `G`, inner physical `A`, `R` | 6 spatial FFT calls | 4 spatial FFT calls |
| Depth `d`: transport plus reference pairs | `4d` FFT calls | `4d` FFT calls |
| Depth `d`: inner and outer streaming pairs | `2(2d-1)` FFT calls | `2(d-1)` FFT calls |
| Complete depth `d` application | `8d-2` FFT calls | `6d-2` FFT calls |
| Complete depth16 application | 126 FFT calls | 94 FFT calls |
| Physical `A` calls per depth16 application | 31 | 15 |
| Physical angular-scattering actions per depth16 | 31 | 31 |
| Cached full-angular reference matrix applications | 16 | 16 |

The 32 fewer spatial FFT calls at depth16 are a 25.40% reduction in that
operation count, not a 25.40% runtime claim. Launches, memory traffic, angular
matrix products, least squares, host/device synchronization and complete solve
behavior remain relevant. A cache miss also pays the explicit physics guard.

For `B` materials, `L` angles and an `N x N` grid, the unchanged reference
construction is `O(B N^2 L^3)` and stores `B N^2 L^2` complex entries. At L12,
N256 and complex64 this is 72 MiB per material. Each reference application is
one spatial FFT pair plus `O(B N^2 L^2)` matrix work. Each physical angular
scatter is also `O(B N^2 L^2)` in the current dense implementation. This first
change neither solves the dense-angular setup cost nor changes its storage.

The unchanged repeated stage obeys

`z_(k+1) = z_k + omega M(r-A z_k)`.

Thus after `d` stages its exact linear action is

`sum_(j=0)^(d-1) omega M (I-omega A M)^j r`.

The optimized and original stages have the same exact-arithmetic action and
conditioning. Any change in software trajectories must be explained by roundoff
or a failed implementation assumption, not a claimed new approximation space.

## Why this is not an optimization already used in the saved TSA

The project already knows the Born identity: `models.py:210-213` implements the
local `V_op`, and `verify_rte_npbs.py:42-49` checks `A+V=L_eta` and the residual
identity. However, the searched TSA paths still compute `r-A(G r)` explicitly:
`transport_baselines.py:58`, `models.py:419`, `global_source_predictor.py:98`,
`npbs_bottleneck_harness.py:89`, and multiple `npbs_unet_ablation.py` controls.
The `V_op` calls found implement Born iteration or identity tests, not this TSA
stage optimization. This is a scoped source observation, not proof that no
earlier external implementation exists.

`FrozenTSA` already caches both the homogeneous inverse and streaming inverse,
but still calls physical `A` on the transported field. A future corresponding
optimization there must carry the actual `gamma`; its current constructor only
receives the inverse multiplier, which is insufficient provenance for silently
assuming the reference shift. Optimizing the deployment baseline first avoids
changing trained-loss numerics and confusing the speed contribution with AI.

## Finite precision is the decisive correctness issue

For the *computed* transport result define `e_G = r-(S+gamma I)z`. Then, before
other rounding effects,

`old_inner_defect = new_inner_defect + e_G`.

The old and new TSA outputs consequently differ by `R e_G`. FP32 reference
inversion can amplify this small defect, especially at low absorption. The
operation elimination removes a cancellation-heavy route but is not a bitwise
substitution. A small single-stage difference alone does not establish a small
depth16 difference or unchanged stopping decisions.

Do not initially use the more folded expression `R(I-EG)r`, with
`E=(t-gamma)I-(s-mean(s))K`. Its derivation additionally uses exact inverse
relations for the full reference `R`, whereas the saved software stores an
FP32 inverse and can offer truncated/angular variants. It adds a second
roundoff/representation question without eliminating another spatial FFT pair.

The correct parity comparison keeps the same stored operator, dtype, reference
cache and RHS. In particular, do not compare a low-precision physical streaming
geometry promoted to FP64 with independently regenerated FP64 reference
directions: they need not be bit-identical. The exact guard rejects such a
mismatch rather than silently redefining the physical matrix.

## Cheapest gate before any timed full solve

1. Use small native-FP64 and native-FP32 `PeriodicRTE2D` fixtures with L12,
   homogeneous and heterogeneous subcritical coefficients, zero and positive
   reference shifts, and fixed complex RHSs. Check original physical inner
   defect against the local identity and compare parent/optimized stage actions
   at depths1 and16, reporting relative errors and unchanged finite status.
2. Check complex phase and additivity of the optimized action, exact zero
   response, rejection of changed stored streaming/phase data, and rerunning
   the guard after a tensor replacement or in-place mutation. These are actual
   operator identities and cache checks, not tests that restate the code.
3. Mock only spatial FFT entry points while running actual operator code,
   with reference setup completed first. Verify 6 versus4 calls for a single
   stage and126 versus94 calls for depth16. This is the mechanism test.
4. On FWI, freeze the already-used development coefficient/RHS data. Apply
   original and optimized single/depth16 preconditioners on one easy and one
   eligible hard TSA16 context (existing IDs0 and3), keeping existing FP32
   physics/cache. Record output, physical-image and transport-inverse defects;
   do not acquire teachers or generate another corpus. This is an application
   parity check, not a new solver benchmark.
5. Only if those checks pass should the root choose a bounded, separately timed
   original-physical full-solve comparison with TSA16-zero, all cases retained,
   unchanged true-residual acceptance, and setup charged. FFT counts alone do
   not justify deployment or a performance claim.

The owned focused test file is `deliverables/stepback/test_algebraic_tsa.py`.
Local numerical checks use this small manufactured-fixture protocol only;
actual-coefficient application checks and any benchmark are not executed by
this delegated inspection.

## Existing mechanisms that should not be proposed as fresh directions

The following outcomes are from preserved reports, not remeasured here.

| Mechanism already present | Concrete evidence and limitation |
|---|---|
| Cached material-conditioned complex-linear full-angle neural maps/symbols | `neural_transport_operator.py:96,148,172`. Residual linearity and material caching already exist. Later N64 CPU cost was176.34ms versus CT91.57ms; N128 transfer required9.04 versus SCT3.25 iterations (`RTE_STRUCTURE_AND_RESOLUTION_RESULTS_2026_09_13.md:57`). Another generic linear map is not a new mechanism. |
| Full-angle two-level spatial correction | `multilevel_neural_transport.py:16,59,82`. Strict bands and HSS fine complements exist; rediscretized coarse action differed from actual `RAP` by5.7–10.5% (`same report:138`). |
| Exact dealiased full-A Galerkin action | `galerkin_transport.py:15,43`. Stored geometry and2m coefficient products give direct-action agreement near2.4e-15, but converged coarse correction needed63 mean inner iterations. An earlier capped diagnostic was explicitly relabelled as inexact (`same report:140`). |
| Cached parity-Schur coarse inverse | `coarse_response_transport.py:13,36`. Full odd reset caused all12 configurations to fail on both N64 media. Harmonic lifting repaired composition but N128 HSS8+m16 remained11.077s versus TSA8 1.907s; m16 inverse cache was18MiB (`RTE_RECURSIVE_RESPONSE_RESULTS_2026_09_19.md:116`). Its current default odd recovery is not evidence of successful composition. |
| Direction-aligned algebraic dictionaries | `parity_coarse.py:74`. The existing294-mode dictionary improved Schur iterations but lost timing; its radial comparator had an extra Nyquist dimension and was not an equal-cost directionality ablation (`RTE_PARITY_STRUCTURE_REVIEW_2026_09_19.md:214`). |
| Recursive residual/energy-weighted inverse and rank32 repair | Stiff N32 cache127.2MiB with11–12s teachers plus about21s compression; warm232.97ms versus TSA8 204.60ms. Moderate regime failed3/6. Rank32 repair added3.1–3.3s and3MiB and still lost runtime (`RTE_RECURSIVE_RESPONSE_RESULTS_2026_09_19.md:64,182`). |
| Nonlinear one-shot tail or four-forward phase averaging | Completed October2 direction diagnostic records useful images behind TSA8 but no convincing full-solve gain against frozen classical, and no mean TSA16 call saving at N128/N256 (`deliverables/followup/RTE_DIRECTION_DIAGNOSTIC_2026_10_02.md`). More training of that mechanism is unsupported. |

A tiny original-stored-A Fourier/angular dictionary behind frozen TSA16-zero,
with equal active dimension for radial and directional comparisons, would
isolate a different coarse-composition question from old parity-PCG studies.
It still introduces assembly/factorization/application work and is less cheap
and less certain than the exact TSA redundancy check. A low-rank angular
reference approximation is also not free: the g=.65 full-angle action is not
rank-one, its near-conservative modes must remain correct, and truncated moment
controls already exist. No such approximation is selected here.

The normalized relative positive-factor `F H F` proposal remains explicitly
untested in `RTE_RECURSIVE_RESPONSE_RESULTS_2026_09_19.md:241`. It is different
from independent inverse-block truncation but requires a compactness/action-cost
oracle before learning. It is not the next cheap implementation gate.

## Decision boundary

Select only the conservative TSA inner-defect optimization now. Passing its
parity/counter gate demonstrates removable classical work, not new learning,
improved Krylov geometry, or a general RTE preconditioning theorem. Failure
under actual stored/dtype conditions is a retained negative result. Training,
new coarse construction, larger datasets, full benchmark selection and remote
project edits remain outside this delegated implementation inspection.

## Actual FWI outcome: the FP32 physical-image parity gate failed

The root executor subsequently ran the frozen helper on FWI, preserving its
SHA256 `85ca74f338091c645ca77b198d74f011eca09ec62b86d27d732ba92126de5e81`.
Six focused tests and 58 subtests passed in 4.412 seconds. The bounded actual
operator check completed 48 applications in 7.893 internal wall seconds
(11.629 seconds for the process). Its exit code was 2 because the predeclared
all-cases parity gate failed, not because of a crash or nonfinite field.
There was no subsequent full-solve benchmark, timing comparison, gate change,
code modification, training or test-data access.

The fixed protocol used existing development IDs 0 and 3 at N64/N128/N256,
depths 1 and 16, both complex64 and complex128 inputs, and two RHS probes:
the existing physical source and a deterministic grazing Nyquist stripe.
Existing development tensors were verified against their saved hashes and
all seven material/source fields. The Nyquist probe is supported in angle 0,
whose stored direction is along x, while its stripe varies along y: it is a
grazing mode for that ordinate. It is an algebraic probe, not a new source
distribution or held-out sample.

Every field was finite, all 48 cases verified the exact spatial-FFT counts,
and 43 of 48 cases passed both numerical criteria. All single-stage cases,
all grazing-stripe cases and all complex128-input cases passed. The five
failures were exclusively **depth16, complex64, existing physical source**:

| Grid | Development ID | Relative field difference | Relative original-physical-image difference |
|---|---:|---:|---:|
| N64 | 0 | 2.924031e-7 | 1.397991e-4 |
| N128 | 0 | 7.761876e-7 | 2.566321e-4 |
| N128 | 3 | 3.402960e-7 | 1.049790e-4 |
| N256 | 0 | 4.851632e-7 | 5.386726e-4 |
| N256 | 3 | 1.829392e-7 | 2.121982e-4 |

The frozen complex64 gate required both relative quantities to be at most
`1e-4`; each failure was in the physical-image criterion alone. Across all
complex64 cases, the largest field difference was `5.039647e-6`, which arose
from a passing single-stage grazing probe. The maximum physical-image
difference was `5.386726e-4`. Thus the largest field difference and the failed
physical-image difference are different cases; they must not be conflated.

The complex128-input gate required both quantities to be at most `1e-9`.
All 24 such cases passed, with maximum field difference `9.391071e-15` and
maximum original-physical-image difference `1.429498e-12`. These comparisons
retain the stored FP32 coefficients, geometry and cached reference inverse;
only input/application precision changes. They establish parity of these two
implementations under that application precision, not accuracy of a fully
rebuilt FP64 inverse or improved performance of a double-precision solver.

Here the field metric is `||z_fast-z_original||/||z_original||`. The image
metric is

`||A_physical z_fast-A_physical z_original|| / ||A_physical z_original||`,

where each output is promoted to complex128 and `A_physical` is the unchanged
original physical operator. This tests the mathematical effect of the actual
rounded output fields; it does not hide their difference inside a low-precision
physical residual evaluation. It is not the solved-field residual
`||q-A x||/||q||`, and it must not be compared directly with the solver's
`1e-5` convergence threshold.

The result is consistent with accumulated FP32 reassociation/inverse-FFT
roundoff and its amplification by the physical action. For any nonzero field
difference `delta_z`, the ratio of these two relative metrics is

`(||A delta_z||/||delta_z||) * (||z_original||/||A z_original||)`.

The five failures have observed ratios between about 308 and 1160. Fourier
derivative growth and an inverse-like field with a relatively small physical
image can make such amplification plausible. The present receipts do not
separate those factors, identify the Fourier spectrum of `delta_z`, or isolate
the eliminated transport defect `e_G` from all subsequent rounding. Therefore
this is an interpretation supported by precision dependence and the metric
identity, not a measured attribution of every discrepancy to a specific kernel.
Grazing probes passing while physical-source depth16 probes fail also prevents
claiming that the constructed Nyquist stripe itself caused the failure.

**Decision:** retain this failed FP32 equivalence gate and keep the original
classical implementation as the validated baseline. The algebraic identity and
the verified 126-to-94 FFT count reduction remain valid. Neither demonstrates
that the optimized FP32 preconditioner would fail FGMRES, lose the final true
residual guarantee, or improve elapsed runtime: no complete solve was run.
Any future investigation would need a separately authorized, precision-aware
bounded solve/cost protocol with original-RHS stopping and unchanged strongest
classical controls, charging conversion/cache/application costs. Merely
loosening this failed gate or promoting inputs and assuming a net speed win
would not establish that result. No such further experiment was performed.

Receipt sources are under
`receipts/rte/workstation_runs/classical_tsa_algebra_20261002/`:
`focused_tests.json`, `operator_checks/settings.json`,
`operator_checks/cases.json`, `operator_checks_execution_receipt.json`, and
the frozen `code/check_frozen_tsa.py`. This appended analysis independently
reaggregated the saved rows; it changed documentation only.
