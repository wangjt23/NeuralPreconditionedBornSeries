# Static mathematical review of the primitive driver

2026-10-02. Reviewed the new `primitive_driver.py`, `exact_rational_rte.py`, `error_model.py` and cached original RTE APIs. This review did not import or execute the driver, call any physical action, use GPU/SSH, generate new fields/data, export original sources, or edit the driver.

**Decision:** no mathematical target-mapping blocker was found for the frozen finite probes. The exact toy comparisons are valid for their expressly separate 4×4 stored-symbol targets. Full-grid NumPy comparisons remain ordinary floating-point diagnostics. The shape/finiteness guards and explicit exact-call count check are now resolved. This is static approval of the mapping, not evidence that execution will finish inside its budget or that any numerical error gate will pass.

Final driver SHA256 reported by the owning parent: `d7149407412392cbdbdf548af907ee4348d242e709c6f882b39036f2462027c7`. The final guard resolution below records the parent's completed changes; no execution or additional review scope was undertaken for this update.

## Targets and cast-shadow comparisons

Write \(D_{32}\) for the captured original low complex stream symbol and \(D_{64}\) for the captured physical high symbol assembled from promoted stored geometry. Both include the factor \(i\). Let

\[
L_{32}=\mathcal F^{-1}D_{32}\mathcal F+T-S\Phi\operatorname{diag}(w),
\qquad
H_{64}=\mathcal F^{-1}D_{64}\mathcal F+T-S\Phi\operatorname{diag}(w),
\]

with exact interpretation of the fixed stored collision operands. These nominal maps differ only in their frozen stream symbols. They are distinct from the actual rounded applications and from the preassembled reference inverse.

For each full-grid field, the driver uses

\[
x_o=\operatorname{promote}_{128}(\text{field}),\quad
x_c=\operatorname{cast}_{64}(x_o),\quad
x_s=\operatorname{promote}_{128}(x_c).
\]

The three actions and references match:

| Actual action | Actual input | Ordinary NumPy128 comparison target |
|---|---|---|
| original low complex64 | \(x_c\) | \(L_{32}x_s\) |
| original low native complex128 | \(x_s\) | \(L_{32}x_s\) |
| original physical high complex128 | \(x_o\) | \(H_{64}x_o\) |

The native low128 action remains on the low object's FP32 stream-formation path. Promoting its field does not reconstruct the physical high symbol. Its collision operands are promoted when acting on the wide field; the NumPy target correctly applies the same stored operands separately in wide arithmetic. The low64/native128 difference therefore isolates application-precision differences on the same numerical input and same nominal low symbol. It does not measure an ideal high preconditioner or an original full-solver trajectory.

The NumPy implementation uses backward normalization on the last two axes, multiplies by the captured symbol, performs the inverse with the same convention, and then evaluates `streaming + t*x - s*scatter`. Scattering is `phase @ (weights*x)`, with weights on phase **columns**. It does not replace those two products with a separately rounded preassembled `phase*weights` matrix. Its FFT, contraction, products and norms nevertheless round independently; it is not a certified reference at full grid size.

## Exact 4×4 mapping

The fixture explicitly constructs a separate unit-square 4×4 periodic grid with `dx=.25`, signed index order \((0,1,-2,-1)\), copies the current context's angular buffers, and samples the current material arrays at \((0,N/4,N/2,3N/4)\) on both axes. The high fixture is obtained through the original `high_precision_copy` API. This is a labelled toy, not a spatial restriction or a replacement of the original full-grid physical problem.

The mapping to the exact helper is correct:

- Drop only the singleton batch dimension; the exact field is \(L\times4\times4\).
- Broadcast the four `kx` entries along the second spatial axis and `ky` along the first. The model's first spatial axis is its `kx` axis despite the generic `height` name.
- Supply copied directions, weights and phase as Python-list values. Entrywise promotion of stored FP32 buffers preserves their exact binary values.
- Supply sampled `t[0,0]` and `s[0,0]` as real 4×4 grids.
- Supply `D32[0]` or `D64[0]` as the **complex** `stream_symbol`. Omitting that argument would instead form the exact bilinear products of the stored direction/frequency operands, which changes the nominal target.

The exact helper's forward DFT has unnormalized roots \(1,-i,-1,i\); its inverse conjugates the roots and divides by 16. Thus both transforms and all collision algebra are rational operations on fixed binary-valued operands. No approximation to \(\pi\) or regenerated ideal angular data enters the comparison: the fixture frequency buffers and assembled symbols are already frozen.

The angle-zero x-Nyquist field is \((-1)^x\), constant in the second axis, with Fourier support only at \((2,0)\equiv(-2,0)\). Its streaming part is exactly \(D[0,2,0]x\). With the usual actual angle-zero direction \((1,0)\), the two captured stream coefficients agree after promotion because this mode does not exercise a sum of competing direction products. It remains a useful application control. Its complete image includes heterogeneous total attenuation and scattering through the entire phase column zero, so that image need not remain confined to angle zero or be real.

## Exact error ledger and its limits

The driver asserts that each frozen toy dyadic field survives its C64 cast exactly. It can therefore use one exact numerical input \(x\) for both targets. Each paired case makes two exact `apply_rte` calls, one for each captured symbol. Four contexts times three fields give 12 paired cases and **24 exact action evaluations**, consistent with the authorization.

The stored GPU output is converted directly to Python binary values and then to exact Fraction pairs. Subtraction from the rational oracle occurs before any floating conversion of the oracle field. Accordingly, `norm_bounds` supplies an outward enclosure of the error against this fixed toy target. The normalized upper metrics divide an upper error norm by a lower positive input norm using exact rational scalar arithmetic and outward conversion. The operator scale is the specified physical nominal norm upper bound, used consistently as a normalization scale; it is not a newly proved norm bound for the low rounded action.

For the low toy output \(\widehat L_{32}x\), the exact decomposition is

\[
\widehat L_{32}x-H_{64}x
=(\widehat L_{32}x-L_{32}x)+(L_{32}x-H_{64}x).
\]

The stored `matched_exact_nominal_error`, `exact_nominal_symbol_difference` and `overall_exact_physical_target_error` correspond to these three terms' norms. The symbol-difference diagnostic subtracts the exact difference from zero; this reverses the difference's sign but preserves its norm. There is no input-cast term because the driver explicitly asserts its absence for the toy cases. A nonzero-cast future protocol would need a separate \(H_{64}(x_s-x_o)\) ledger and matched low references built from the actual cast input.

With normal default FP32 random construction, the full-grid source, mixed source, Nyquist, random and strong-factor output probes are also C64 fields before promotion. Their input-cast discrepancy is therefore zero by construction. The recorded input dtype/cast norm should make this limitation visible. These probes do not challenge a nonrepresentable C128 outer Arnoldi frontier's preconditioner-input cast. This does not invalidate the same-input precision comparison, and this review recommends no change to the frozen fields or call budget.

Full-grid norms, reference errors, symbol/cast allowance products and the cancellation-condition indicator are ordinary evaluated scalars, even when a coefficient-only upper scale came from exact rational accounting. They cannot certify a uniform CUDA contract or the conditional global bounds. Exact toy enclosures certify only the labelled finite toy cases and cannot be extrapolated to the original full-grid FFT or to unobserved solver directions.

## Integrity checks: resolved

1. **Shape/finiteness checks: resolved in the final driver.** Ordinary diagnostics now require equal actual/reference shapes. `exact_difference` checks angular counts and every 4×4 plane/row length before its nested subtraction. `toy_diagnostics` requires actual output shape `(L,4,4)`, equal reference/input angular counts, and finite output before Fraction conversion. These fail-closed guards prevent a malformed result from receiving an exact certificate for a truncated portion and give a direct nonfinite failure before rational conversion. The previously identified `zip` truncation concern is closed.

2. **Exact-call count check: resolved during review.** The updated final check now asserts `exact_fixture_apply_calls == policy['exact_reference_actions']`. With the frozen approved value 24, this protects the distinction between 12 paired cases and 24 exact evaluations. The current two-call loop is consistent with that count.

No solver promotion, precision change, field change, extra action call, threshold relaxation or source export is justified by this static review. Preserve the prior stopped execution and the new primitive protocol's distinct finite-probe scope.
