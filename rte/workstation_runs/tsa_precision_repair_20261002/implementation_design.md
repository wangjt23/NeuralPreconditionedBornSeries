# Finite-precision gap: diagnosis before one isolated repair

The frozen FP32 algebraic TSA optimization removes an FFT pair but does not yet
pass the unchanged physical-image parity gate. The completed diagnostic below
identifies differential reference-action rounding and outer-addition rounding;
it does not support local-defect cancellation as the main explanation. This
note audits one bounded selective-sum precision candidate chosen by the root.
That candidate has now completed the unchanged48-case gate and fails the same
five complex64 physical-source/depth16 cases. The negative result is retained;
no full solves or solver timing were run.
No numerical code is changed by this note. The original operator, TSA code,
physical FGMRES and acceptance semantics remain untouched.

## What the retained 48 checks establish

The five failures are exclusively existing physical source, depth16,
complex64 input: N64 ID0, N128 IDs0/3, and N256 IDs0/3. Their relative field
differences are only `1.829e-7` to `7.762e-7`, while original-physical-image
differences are `1.050e-4` to `5.387e-4`. All fields are finite. All single-stage
checks and all constructed grazing Nyquist probes pass. The largest field
difference, `5.040e-6`, belongs to a passing single-stage grazing probe.

All24 complex128-input checks pass with maximum field difference `9.391e-15`
and physical-image difference `1.429e-12`. These retain the FP32 stored
coefficients, geometry and reference inverse; they do not rebuild an FP64
reference. All work counts are exact: 6 to4 spatial FFT calls per stage and
126 to94 per depth16 application. The predeclared complex64 gate remains
`1e-4` for both field and physical-image differences; complex128 remains
`1e-9`. Full-solve convergence and elapsed cost have not been measured.

Receipts: `receipts/rte/workstation_runs/classical_tsa_algebra_20261002/`,
especially `operator_checks/cases.json`, `operator_checks/settings.json` and
`code/check_frozen_tsa.py`. Frozen helper SHA256:
`85ca74f338091c645ca77b198d74f011eca09ec62b86d27d732ba92126de5e81`.

Passing double-input parity and failing only accumulated source trajectories
support a precision-dependent implementation difference. They do not by
themselves identify transport FFT error, local cancellation, reference action
rounding or outer accumulation as the dominant cause. Nor does a failed
preconditioner-image parity gate establish failed FGMRES convergence.

## Two different stored-symbol notions must be kept separate

`PeriodicRTE2D._stream_symbol` first multiplies/sums directions and wave numbers
in `self.float_dtype`, then casts to the requested complex dtype
(`remote_reference/rte/models.py:189-195`). Thus on an FP32 operator,
requesting a complex128 symbol still uses the assembled FP32 streaming values.

For a reference-transport inverse error, compute the FP32 assembled symbol
once and then promote it to complex128. Do not recompute its products from
promoted direction/wave-number factors and call that the same matrix.
`high_precision_copy` preserves the stored factors but uses FP64 product/sum
arithmetic to apply its physical streaming action
(`mixed_precision_transport.py:10-19`). Its `A_op` remains the correct,
unchanged **gate image** used in the retained experiment. A diagnostic must
explicitly report any difference between that original physical image and
the promoted, previously assembled low-precision symbol.

The angular action also has a fixed operation order:
`weighted_psi = weights * psi`, then the phase-matrix product, then multiplication
by `sigma_s` (`models.py:182-187`). Promoting a precomputed FP32
`phase * weights` matrix is not identical to multiplying the separately stored
phase/weight constants in FP64. Use the actual operation definitions and label
which quantities are recomputed, which constants are promoted and which cached
inverse is shared. No rounded phase is replaced by an ideal circulant,
symmetrized or scalar action.

## Nominal trajectories and common-input replays answer different questions

Retain the original and frozen fused loops for all16 actual stages. Before
each update capture their own accumulated field `x`, outer defect `d`, transport
field `g=G(d)`, inner defect `f`, reference correction `c=R(f)`, stage update
`u=g+c`, and next accumulated field. Preserve the existing expression order:

```python
d = rhs if stage == 0 else rhs - low.A_op(x)
g = reference.G(d)
# Original:
f_original = d - low.A_op(g)
# Frozen fused:
f_fused = (reference.sigma0.to(g.dtype)-low.sigma_t.to(g.dtype))*g
f_fused = f_fused + low.scattering_op(g)
c = reference.reference_transport_solve(f)
u = g + c
x_next = x + relaxation*u
```

Trace the two nominal branches independently: once their `x` differs, their
outer `d` and every later `g` also differ. Comparing only their `f` thereafter
does not isolate the optimized single stage.

Additionally replay both inner formulas on the same original `d` and its same
computed `g`. Reuse the original cached `R` and record whether independently
constructed caches are bit-identical before nominal comparisons. This
common-input replay separates an inner-stage discrepancy from amplification
through different outer states. Extra diagnostic actions must not update either
nominal state. All component work/captures are diagnostic overhead; their
wall time is not an unprofiled application benchmark.

For cheap cohort coverage, the diagnostic can retain the six existing physical
source/depth16 cases (including the passing N64 ID3 control), one worst-grid
grazing control and the corresponding complex128 source control. The root owns
the exact bounded diagnostic cohort. No teacher, training bank or new source
distribution is required. The complete unchanged48-case gate is mandatory
after the root chooses a repair; a passing diagnostic subset cannot replace it.

## Stage receipts and scales

At each stage record absolute norms as well as the previous gate ratios:

| Quantity | Purpose |
|---|---|
| `||x_fused-x_original||/||x_original||` | Accumulated field discrepancy |
| `||A_phys(x_fused-x_original)||/||A_phys x_original||` | Physical effect of the rounded field discrepancy |
| Relative and absolute differences of outer `d`, common-input `f`, `R(f)`, and stage `u` | Locate the first amplified component |
| `||d||`, `||g||`, `||f||`, `||R(f)||`, `||u||`, `||x||`, `||A_phys x||` | Prevent misleading ratios when late defects or images are small |
| Finite flags and exact nominal-transform counts | Detect numerical failures and preserve work-count provenance |

Compute reporting norms/images in complex128 from captured fields. Use the
same original `A_phys` as the prior gate. Also report
`||A_phys(x_fused-x_original)||` directly and the difference of separately
evaluated `A_phys x` values: their small difference checks FP64 diagnostic
linearity/reassociation. The primary gate still uses its original metric.

For the nominal accumulation, evaluate the following identity on stored fields
promoted to complex128:

`delta_acc_round = (x_f_next-x_o_next) - [(x_f-x_o)+omega*(u_f-u_o)]`.

This reveals the difference of update-rounding errors; it does not confuse
different stage updates with the final addition. With the actual omega=1 no
additional scalar rounding convention is needed. For other relaxations use
the scalar value actually consumed by the FP32 update.

## Inner-defect decomposition

Use `e_G = L0 g - d` in this note, where `L0` uses the **assembled FP32**
reference symbol promoted exactly and its stored `gamma`. This sign is the
negative of the `r-L0 G r` convention in the earlier mechanism note.
For common input, exact arithmetic gives

`f_fused - f_original = e_G`.

In finite precision, define complex128 attribution values on the same captured
FP32 `g`: `f_o_star = d-A_stored_symbol_exact(g)` and
`f_f_star = (gamma-t)g+C_exact(g)`, using the stored coefficient/phase/weight
constants and the promoted assembled streaming symbol. Then

`f_f32-f_o32 = e_G + (f_f32-f_f_star) - (f_o32-f_o_star)`.

Measure every term and the closure residual. Decompose the local errors further
only if necessary, separating actual rounded products, scattering and final
subtractions/additions. Ratios such as
`(||d||+||A_low(g)||)/||f_o32||` and
`(||(gamma-t)g||+||C(g)||)/||f_f32||`
describe cancellation; they do not prove which term dominates the final image.

Apply the shared cached reference in complex128 to each decomposition term,
then apply the original `A_phys`. This measures which inner differences have
large physical effects after reference amplification. It does not reconstruct
the exact FP32 `R` action: compare the sum against the actual common-input
`R(f_f32)-R(f_o32)` to report that action's own rounding contribution. Keep
linear attribution, actual FP32 outputs and the closure residual distinct.

## Transport FFT/division attribution

For the actual FP32 `G` operation retain its forward transform `dhat32`, divided
spectrum `ghat32`, and rounded inverse-transform output `g32`. Let
`lambda` be the denominator **actually assembled** for that call, promoted to
complex128. With backward FFT normalization throughout this ledger,

```text
lambda * FFT64(g32) - FFT64(d32)
  = lambda * [FFT64(g32)-ghat32]
    + [lambda*ghat32-dhat32]
    + [dhat32-FFT64(d32)]
```

The terms distinguish inverse-FFT/output roundoff, division/spectrum roundoff,
and forward-FFT roundoff on the same rounded input. Their FP64 inverse
transforms reconstruct `e_G`, up to FP64 reporting error. A direct
`IFFT64(ghat32)` comparison with `g32` is another view of inverse-transform
rounding. Also retain the actual FP32 round-trip
`FFT32(g32)-ghat32`, which enters the original `A_low(g32)` route.
Do not silently mix unitary diagnostic transforms with the actual backward
normalization when evaluating this identity.

These transforms are diagnostic additions, not part of the proposed optimized
cost. A blanket conclusion that promotion of `G` will satisfy parity is not
warranted: the control itself contains FP32 FFT round-trip noise absent from
the algebraic formula. A more accurate `G` can change its own rounded output
and thereby move away from that particular noisy control.

## Spectrum of the discrepancy

Use an orthonormal spatial FFT of `delta_x` only for reporting. Preserve stored
wave numbers, count all angles, and report both delta-field energy and
physical-image energy. A small set of frozen masks suffices: strict low band
`|mode_x|,|mode_y|<16`, its complement, the exact Nyquist rows/columns, and a
directional grazing subset defined with the actual stored `Omega dot k` and
mean material scale. Masks overlap where stated; do not sum overlapping energy
fractions and treat them as disjoint partitions.

Record the high-frequency contribution of the **difference field**, not merely
the spectrum of the original solution. The observed ratio

`image_relative/field_relative = (||A delta_x||/||delta_x||)*(||x||/||A x||)`

can be large through derivative amplification, a large inverse-like solution
scale, or both. Without the spectra/absolute norms, assigning all308–1160
observed amplification factors to a particular band or kernel is speculative.

## Candidate ledger considered before diagnosis

All counts below refer to a depth16 application, before instrumentation, with
the original output dtype and actual FP32 reference cache retained. They are
operation ledgers, not latency predictions.

| Possible isolated change | Spatial transform ledger | Additional work and decisive limitation |
|---|---|---|
| Stable affine ordering of the fused local defect | 94 complex64 transforms | No new FFT or cache. Can change multiplication/addition roundoff; cannot restore the omitted `e_G` or original streaming round-trip noise. Choose only if common-input local cancellation dominates. |
| Promote only the local affine products/sum, retaining actual FP32 scattering, then cast the defect to complex64 before `R` | 94 complex64 transforms | FP64 elementwise work on full fields and cast buffers. Avoids paying for a new FP64 dense angular action. Can reduce affine rounding; cannot repair an error already present in `g` or accumulated state. |
| Complex128 outer accumulator, each outer physical action evaluated on its complex64 cast; original FP32 `G`, local defect and `R`; final cast to original dtype | 94 complex64 transforms | Additional accumulator/cast storage and memory traffic; scalar addition uses FP64. Changes accumulated rounding only. Choose only if update-rounding dominates after common-input inner attribution. Final FP32 quantization can still dominate the image gate. |
| Promote `G` spatial transforms and division, then cast `g` back to complex64 before the unchanged local defect/`R` and outer loop | 32 complex128 plus62 complex64 transforms | Same number of transforms, different precision cost and FFT workspace. Material symbol/denominator constants must be promoted from their actual assembly. Choose only if transport inverse/round-trip error dominates; cast-back noise and the noisy FP32 control can still prevent parity. |
| Promote outer residual action and accumulator | 30 complex128 plus64 complex64 transforms if `G`/`R` remain FP32 | FP64 physical angular action and casts as well as outer FFTs. It changes a larger part of the trajectory; it is not the cheap default. |

At L12/N256/batch1, a complex64 field occupies6 MiB and a complex128 field
12 MiB. These sizes describe tensors only, not peak memory: temporaries,
reference casts and FFT workspaces must be measured. Promoting the whole
`reference_transport_solve` input would also promote its two transforms and
cast its72 MiB cached inverse to144 MiB for use. That accidentally expands
the precision intervention and must not be labelled “G FFT only.” Rebuilding
the reference inverse is a separate, unselected change.

The first choice should be the smallest change explaining the measured
dominant component, not the variant giving the best number in an exploratory
multi-candidate sweep. Freeze its scope/source before running the unchanged48
checks. In particular, do not correct a local-cancellation explanation with
whole-operator FP64 or assume that an FP64 accumulator addresses `G` error.

Do not presume that subtracting the stored nearby positive coefficients
`gamma-t` is itself the unstable step. Expanding it into `gamma*g-t*g` introduces
two larger products and their cancellation and can make the situation worse.
Separate coefficient subtraction, multiplication and addition diagnostics
before selecting an ordering change. Likewise, an FP64 accumulator whose outer
action sees only its FP32 cast must report the physical image of the difference
between the retained state and the cast state; a more accurate hidden sum is
not automatically the state seen by the preconditioner or final output.

## Observed diagnostic outcome and interpretation

The completed extended receipt covers the five retained failed cases, all16
stages each, and common-input replays at stages0,7,15. It uses actual FP32
assembled streaming symbols promoted to complex128 for the inverse ledger,
the same FP32 reference inverse, and the unchanged original high-precision
physical gate image. `receipts/precision/diagnosis_extended/settings.json`
reports80 stages,15 common-input decompositions,3.5551 seconds of diagnostic
wall time and no reserved-seed access. That time includes attribution actions
and is not a preconditioner or solver benchmark.

For common-input stage0, let `delta = f_f32-f_o32`,
`r_delta = R128(delta)`,
`r_round = [R32(f_f32)-R32(f_o32)]-r_delta`, and
`s_round = [(g32+R32(f_f32))-(g32+R32(f_o32))]-[R32(f_f32)-R32(f_o32)]`.
Subtractions in this definition are evaluated on stored values promoted to
complex128. The following are absolute original-physical-image norms, rather
than relative gate values:

| Existing case | `A_phys r_delta` | `A_phys r_round` | `A_phys s_round` | Actual common-single difference | Stage15 outer-addition rounding difference |
|---|---:|---:|---:|---:|---:|
| N64 ID0 | 2.056e-5 | 1.265e-3 | 5.039e-5 | 1.266e-3 | 1.379e-3 |
| N128 ID0 | 6.537e-5 | 5.226e-3 | 2.563e-4 | 5.232e-3 | 5.152e-3 |
| N128 ID3 | 6.585e-5 | 3.296e-3 | 2.276e-4 | 3.306e-3 | 2.610e-3 |
| N256 ID0 | 2.018e-4 | 1.518e-2 | 6.207e-4 | 1.520e-2 | 2.018e-2 |
| N256 ID3 | 2.121e-4 | 1.028e-2 | 7.245e-4 | 1.030e-2 | 1.015e-2 |

The reference-action differential-rounding norm is about49–80 times the
linear image of the changed defect at stage0. It is also substantially larger
than stage-addition rounding. This is an observed arithmetic difference
between two nearby FP32 applications of the same cached reference; it does
not establish that the cached inverse itself is mathematically inaccurate.
The vector decomposition closes with physical-image norms from7.77e-23 to
5.57e-20 across the15 common-input checks. Component norms cannot be added
as causal percentages: their vectors can interfere or cancel.

The outer-addition column measures the **single stage15 difference of update
rounding errors**, using the stored independent trajectory states. It is not
the cumulative contribution of all previous additions, and the common-input
stage0 and nominal-stage15 columns must not be directly summed. Its size
nevertheless makes retained FP64 sums a concrete, cheap hypothesis to test.
Removing this component alone does not remove the larger common-single
reference-action difference or guarantee closer parity to the FP32 control.

The local-defect cancellation ratios at the common replay stages are modest
(original1.212–1.317; fused1.056–1.672), and the local-rounding contribution is
smaller than the inverse/round-trip and physical-action arithmetic terms.
The G FFT ledger separates actual inverse-FFT/output, division and forward-FFT
terms, closing at3.49e-10 to8.83e-10 relative error. For N256 ID0 stage0 their
post-reference physical-image norms are respectively1.238e-4,1.128e-5 and
3.730e-5. These observations do not support spending an initial repair on
local-affine reassociation or on a G-only promotion while the much larger R32
differential roundoff is untreated.

At the final stage, high-frequency power fractions of the physical-image
difference range0.895–0.948; exact Nyquist-row/column fractions range0.224–0.279.
The executed spectrum helper defines high frequency by
`max(abs(mode_x),abs(mode_y)) >= N/4`; this is a grid-relative band rather than
the earlier proposed fixed16-mode diagnostic cutoff. Its Nyquist mask is
`max(abs(mode_x),abs(mode_y)) >= N/2` on these even grids.
This supports high-frequency amplification of tiny rounded field differences.
It does not identify one FFT or material coefficient as the sole cause. The
retained complex64 parity failures, complex128 passes and exact FFT reduction
remain the previously reported bounded results; this diagnosis establishes
neither FGMRES failure nor a runtime benefit.

Receipt hashes for a reproducible reading of this note:

- Initial diagnosis `cases.json`: `9538f70f02cb7109d413883c73278e390c3f49c460f1c67800ab17c44b52046e`.
- Extended diagnosis `cases.json`: `6c9c71f45c47e7714c14e9f1035d9df54af83a09c97108bbf90e49b50ced60f4`.
- Extended diagnosis `settings.json`: `f7fa8bf6344dec1e33d24803b6f80e40ac3c7f4f9bfa2fa8367499d1193c8d2d`.

## Audit of the one selected retained-sum policy

The root-selected candidate is a **combined stage-sum and outer-sum precision
intervention**, represented by the isolated
`deliverables/precision/sum_stable_tsa.py`. It retains the actual input dtype
through `G`, coefficient subtraction/multiplication, scattering, local-defect
addition, cached `R`, and all outer physical defect actions. It promotes the
already computed `g` and correction only for their sum and retains the outer
accumulator and returned result in complex128:

```python
g32 = reference.G(d32)
f32 = (gamma32-t32)*g32 + low.scattering_op(g32)
c32 = reference.reference_transport_solve(f32)
u128 = g32.to(complex128) + c32.to(complex128)
x128 = x128 + relaxation*u128
# The next stage deliberately sees exactly the input-dtype physical action:
d32 = rhs32 - low.A_op(x128.to(complex64))
return x128
```

This is a legitimate selective arithmetic change, preserving the stored
physics guard, cache construction, outer physical action and true-residual
authority. It is not equivalent to the frozen fused FP32 numerical map: the
next FP32 defect can see a different rounded accumulated field, and the final
returned field retains low-order bits instead of rounding back to FP32.
The deployed wrapper already returns complex128; the changed numerical values
must still pass the exact same physical-image gate against the original
control. The solver may consume the resulting preconditioned direction
without changing its physical operator or acceptance tolerance.

The explicit state cast is essential. Passing `x128` directly to `low.A_op`
would also promote its FFT, coefficient and angular arithmetic, creating a
broader unselected intervention. Likewise, feeding the promoted stage defect
to `R` would change dense angular multiplication, inverse casts and FFT
precision. The current candidate's casts avoid both changes. For complex128
input, every `.to(complex128)` is already in the input dtype and the algorithm
follows the existing fused complex128 arithmetic; the unchanged1e-9 gate
continues to test it. This audited policy is restricted to complex64 and
complex128 inputs. Applying the same state cast to a real input dtype would
discard imaginary components and is outside the stated contract.

Computing the addition in complex128 and immediately casting the result back
to complex64 would normally return the same correctly rounded componentwise
binary32 sum of two binary32 values. Retaining the complex128 stage update and
accumulator is the intended source of additional effective precision. Special
subnormal or exceptional arithmetic must still be judged by actual finite and
zero/linearity checks; there is no generic bitwise claim.

The case for this candidate is **cost first and falsifiable**, not a claim that
it repairs the dominant common-single cause. It removes stage/outer summation
rounding without changing FFT/GEMM precision or rebuilding the inverse, and
targets the sizeable observed outer-addition component. It leaves FP32 G,
local-defect and reference-action roundoff fully present. Furthermore, the
original FP32 control contains its own roundoff, so a more accurate returned
sum can move away from that noisy target. A failed gate is therefore a useful
negative result and must not trigger an adaptive promotion sweep.

For complex64 input at depth16, the ledger stays94 complex64 spatial transforms
(47 forward and47 inverse), with zero extra complex128 FFTs. The FP32 inverse
cache stays72 MiB at L12/N256/batch1. The complex128 accumulator occupies12 MiB
instead of6 MiB; promoted `g`, correction, sum and old/new accumulator
temporaries add storage and memory traffic depending on tensor lifetimes.
Straightforward code performs32 input-to-complex128 field casts and15
accumulator-to-input field casts per depth16 call. At N256 this is846 MiB of
conceptual cast read/write traffic (47 times18 MiB), before any allocator,
cache or kernel effects. Relative to a wrapper that already casts the final
FP32 result to complex128, the net added cast count is one lower. Counts and
tensor sizes are not latency predictions; GPU peak memory and complete
unprofiled solve cost still require measurements if the gate passes.

For the retained eta0/relaxation1 cases, no relaxation representation ambiguity
is present. At another relaxation value, multiplication of the retained
complex128 update also occurs in complex128; that belongs to this sum/carry
policy and must be disclosed rather than silently claimed to reproduce FP32
scalar-update rounding. There is no change to mathematical relaxation.

## Required boundary for the selected repair

Retain the original, the failed frozen fused implementation and all existing
receipts. Place the chosen repair in a new isolated helper with explicit
dtype/cast rules, unchanged stored-physics guard/cache refresh, zero and
complex-linearity checks, and exact intended transform counts. The original
preconditioner is the numerical control; the original high-precision physical
operator and FGMRES true-residual checks remain the acceptance authority.

Run all48 frozen cases with the same `1e-4`/`1e-9` gates and compare separately
against both the original and failed fused actions. A failing repair remains
a negative result. Passing would establish this bounded application-parity
result, not generic safety or speed. Only a subsequently authorized, bounded
full-solve/cost protocol could establish unchanged failures/accuracy and actual
cost benefit; charge conversions, setup, cache storage and changed-precision
kernels. No threshold, cohort or acceptance semantics changes are implied.

## Executed candidate outcome: the unchanged gate still fails

The FWI execution receipt under `receipts/precision/executed/` confirms that
the chosen candidate completed all48 cases and exited with code2 for a failed
gate. There were zero full solves, zero warmed solver timing runs, no training,
and no test access. This delegated review independently rechecked the unique
case keys against the original48, every finite/pass flag, both limits, transform
counts, staged source hashes and frozen policy identity. The staged candidate
implements the audited sum/carry policy exactly; its `G`, local defect,
scattering, cached reference solve and outer action receive the input dtype.

The result remains43/48 passing, with the exact same five complex64
physical-source/depth16 failures:

| Existing case | Prior fused image relative | Retained-sum image relative | Retained-sum field relative |
|---|---:|---:|---:|
| N64 ID0 | 1.397991e-4 | 1.380606e-4 | 6.737993e-7 |
| N128 ID0 | 2.566321e-4 | 2.575576e-4 | 8.603625e-7 |
| N128 ID3 | 1.049790e-4 | 1.012756e-4 | 2.193436e-7 |
| N256 ID0 | 5.386726e-4 | 5.180030e-4 | 3.340666e-7 |
| N256 ID3 | 2.121982e-4 | 2.308752e-4 | 1.912005e-7 |

The worst image discrepancy decreases by3.837% but remains5.180 times the
unchanged1e-4 limit. Two individual image discrepancies increase, so this is
not a uniform improvement. All complex128-input checks still pass with maximum
field9.391e-15 and image1.429e-12; all finite and exact-count checks pass. The
selective sums alone therefore do not close the finite-precision parity gap on
the predeclared cohort. This rejects the one cheap repair hypothesis under the
current gate, without establishing a full-solver failure, absence of an FFT
work benefit, or universal cost disadvantage for other precision policies.

## Closed trajectory budget: late outer residual rounding remains

`diagnosis_closed` retains the original and failed frozen fused FP32
trajectories, rather than replacing either by the sum candidate. The independent
reference caches are asserted bit-identical in all five cases. Let `A_star`
apply the promoted **assembled FP32** stream symbol and stored coefficients
with complex128 arithmetic, and let `F` be the complex128 algebraic single
using the same FP32 cached inverse. For relaxation1, define

```text
delta = x_f - x_o
tau_o = d_o - [b-A_star(x_o)]
tau_f = d_f - [b-A_star(x_f)]
beta_delta = [single_f32(d_f)-F(d_f)]-[single_o32(d_o)-F(d_o)]
alpha_delta = delta_next-[delta+single_f32(d_f)-single_o32(d_o)]
delta_next = (I-F A_star)delta + F(tau_f-tau_o) + beta_delta + alpha_delta
```

All stored values are promoted before these reporting subtractions. `tau`
includes both original input-dtype outer physical-action arithmetic and RHS
subtraction error against the specified `A_star`; it is not a diagnosis of one
particular kernel. The acceptance image in this budget is still the unchanged
original high-precision physical operator, distinct from `A_star`.

All15 recorded budgets at stages0,7,15 close in physical-image norm at or below
1.9684e-11. At stage0 there is no previous state or outer residual/accumulation
rounding, and native-inner error gives the observed common-single difference.
At stage15 the norms are:

| Existing case | Transported outer-residual rounding `A_phys F(tau_f-tau_o)` | Outer addition `A_phys alpha_delta` | Native inner `A_phys beta_delta` | Actual accumulated image difference |
|---|---:|---:|---:|---:|
| N64 ID0 | 4.940e-3 | 1.379e-3 | 1.673e-5 | 5.184e-3 |
| N128 ID0 | 1.829e-2 | 5.152e-3 | 3.097e-5 | 1.903e-2 |
| N128 ID3 | 9.460e-3 | 2.610e-3 | 8.476e-6 | 9.841e-3 |
| N256 ID0 | 7.724e-2 | 2.018e-2 | 6.159e-5 | 7.990e-2 |
| N256 ID3 | 3.846e-2 | 1.015e-2 | 2.342e-5 | 3.979e-2 |

The transported outer-residual rounding is the largest individual term in all
five late-stage budgets, about3.5–3.8 times the addition-rounding norm and much
larger than late native-inner error. This explains why removing the measured
addition component was an incomplete intervention. The budget closes a vector
identity on original/frozen fused trajectories; it does not prove the selected
candidate has the same error decomposition, and component norms must not be
summed as independent percentages. A further candidate would require a newly
declared scope and evidence; no R/G/A promotion sweep follows from this result.

## Verified cost ledger and limits

The completed gate confirms126 to94 spatial transform calls per depth16
application (63/63 versus47/47 forward/inverse), a25.397% transform-count
reduction. The staged source keeps all of these transforms in complex64 for
complex64 input. There are no extra complex128 FFTs or precision-cache rebuilds
for that input. Both original and candidate N256 inverses occupy72 MiB;
persistent extra precision-cache bytes are zero. There are32 field promotions
and15 outer-state casts per candidate depth16 call,846 MiB of conceptual
N256 cast read/write traffic, and a6 MiB accumulator-storage increase. The
actual gate-process peaks are442.657 MiB allocated and610 MiB reserved; these
are whole-process receipt values, not isolated candidate memory increments.

The two N256/depth16 construction samples give median original setup5.665 ms
and candidate setup including the guard6.011 ms. These are incidental gate
construction timings from two media, with order/cold-start effects and no
dedicated timing protocol. They neither establish a setup regression nor an
application speed benefit. The full48-case gate's8.535 seconds includes
construction, counted applications and physical images; it is not a warmed
preconditioner latency measurement. The predeclared policy stopped all solve
and timing work when the gate failed.

No result here establishes that R-only or outer-A precision is universally
uneconomic, or estimates an unmeasured FP64 throughput ratio. What is established
is narrower: this one sum-only implementation preserves the FFT/cache ledger
but fails the frozen parity target. Retaining original TSA for deployment and
keeping the failed isolated candidate and diagnosis receipts is warranted.

Verified execution identities:

- Candidate SHA256: `274c94dd0ca5fa6e36ffd51ab9b0036a2a7df32b40041d3e9200457378460b6b`.
- Gate script SHA256: `30acea5481c605ff7fcd1655784638a6e8c6ebfba2368e86d6e8e8dc136ece42`.
- Frozen policy SHA256: `671e4bd3fcc0737b1d19af209a7ef00d423e6e745fb7eeca66a72e8bd4836ff4`.
- Repair `cases.json` SHA256: `b17b1b91e615f343a57fde5f66be4f3f3d8fcf2b422797e946a66694340deb64`.
- Repair `settings.json` SHA256: `c228292497f25bf0647ff490cdca699bf02b083d710c9b30a72617876d0d7b4c`.
- Closed-diagnosis `cases.json` SHA256: `e523866ea2e717be2c15b02c6ce5f90d129231252636d45516d35f9d050a8b63`.
- Closed-diagnosis script SHA256: `d8aae8caaeb581d2d43883cab5d2d63eff29485004505d7143e454886a527146`.
- Execution-receipt SHA256: `dedd9ed70f90e739326c3109693655fac634b60651f3f12b6753d48c9277468f`.

Status of this delegated design: source/receipt inspection and documentation
only. Diagnosis, one candidate policy and its failed unchanged gate were
independently audited. No numerical helper was edited, remote command issued,
or experiment executed by this delegated continuation.
