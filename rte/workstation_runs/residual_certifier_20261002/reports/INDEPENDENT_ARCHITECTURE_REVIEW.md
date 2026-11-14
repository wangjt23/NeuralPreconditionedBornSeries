# Input-specific residual certification: independent architecture review

2026-10-02. This is a prospective proof/protocol review, not an executed certificate. Implementation and field inventory are pending. No SSH/GPU, RTE action, full solve, training, seed9626 access, source export or Git mutation was performed.

## What a direct reference can establish

An independently enclosed **exact nominal original** `H*x` for one immutable stored field can certify its residual without a uniform cuFFT error contract or an Arnoldi-trajectory bound. The target is the original promoted-buffer discrete operator, with its **actual assembled D64 symbol** interpreted exactly and the original stored phase, weights and material coefficients. A mathematically specified FFT encloses the exact DFT with the original signed-mode/axis/backward-normalization convention. It must not silently substitute exact geometry products for assembled D64, continuous transport, a low symbol or a new inverse.

For a residual enclosure containing `q−H*x`, obtain a Euclidean upper bound `R_U`, source/field lower norm bounds `Q_L,X_L`, and a positive nominal coercivity lower bound `m_L`. Sufficient joint acceptance is

    R_U <= min(1e−5*Q_L, [1e−3/(1+1e−3)]*m_L*X_L).

This implies absolute solution error at most `R_U/m_L` and relative error at most `R_U/(m_L*X_L−R_U)`. The positive-denominator requirement is essential. Preserve any separate original operational finite/status/physical-residual and paired-field gates. Two1e−3 certificates do not imply the paired1e−4 field criterion; certify the direct difference or retain its separately qualified check.

The1% reserve must be defined before measurements. A direct residual enclosure can use its own upper bound as authority; it need not infer a global action-error constant. If the policy instead adds a check-error allowance to an ordinary center residual, it must pay **all** action/subtraction/norm uncertainty within the source-relative reserve and ensure the upper bound meets the unchanged1e−5 target. A certificate-width budget is an engineering cost/fidelity condition, not an enlargement of the original tolerance or a new solver stopping rule. Failure to enclose tightly is `NO_CERTIFICATE`, not a demonstration that the candidate is physically inaccurate or that FGMRES failed.

## Exact input identity comes first

Scalar receipts cannot reconstruct a previously returned field, its complex128 mantissa bits, or its nominal residual. Need an existing saved field (or a separately authorized newly saved one), immutable byte hash, dtype/shape, source/material identity and exact D64 target identity. A checkpoint, residual norm, seed or solver status is insufficient. Regenerating the original source from sealed data can establish source identity, but recomputing a solver would create a new candidate. Do not describe a regenerated candidate as the old accepted field.

The prior primitive receipts contain scalar/hash metadata only. Their4×4 dyadic fixtures can be regenerated from explicitly frozen fixture definitions and sampled coefficients for a new finite certificate; they do not recover an old full-grid output or certify a nonsaved complex trajectory. Root-owned FWI inventory must resolve the saved-output question before a production-field admission protocol.

## Proof obligations for interval FFT and twiddles

- Every binary64 endpoint operation must enclose its exact real result. `nextafter` around a correctly rounded finite result is sufficient; undocumented FTZ/DAZ or an unhandled overflow/NaN is not. A result flushed from a sizable subnormal to zero can lie far outside one-nextafter bounds. Either justify the normal/gradual-underflow contract, use exact integer/rational fallback, or fail closed. Reject unordered/nonfinite intervals rather than silently masking invalid results.
- Corner multiplication must handle all sign combinations. `0*infinity`, signed zero, subnormal products and cancellation are adversarial cases. Array dtype, broadcasting, alias ownership and real/imaginary operations require explicit guards; a wider-looking float32 interval is not an audited float64 interval.
- Machin `pi=16*atan(1/5)−4*atan(1/239)` can be enclosed with exact rational alternating-series remainders. Negative coefficients swap interval endpoints. Conversion to endpoint floats must be outward, not nearest by assertion alone.
- Sine/cosine Taylor truncation needs a rigorous remainder for the entire argument interval, plus interval argument/range-reduction uncertainty. Exact quadrantal values should be enclosed correctly. No unchecked ordinary `math.pi`, `sin`, `cos`, complex exponential or NumPy FFT may supply a purported exact twiddle.
- Butterfly, permutation, transform sign, two spatial axes and inverse normalization must match the declared DFT exactly. Normalization rounding belongs in the interval. Tests need both signs, round trips, impulses, dense complex fields, Nyquist and non-power-of-two rejection.
- The scattering target is original `phase*(weights*x)`, then sigma_s; interval arithmetic must enclose the exact stored-coefficient algebra, regardless of floating evaluation grouping. Coercivity may safely use `||P||2<=sqrt(||abs(P)||1*||abs(P)||infinity)` with exact/outward evaluation and nonnegative sigma_s. Streaming skew-Hermitianity follows from the actual real stored symbol, not a real-field invariance claim.
- Norm upper/lower bounds must enclose exact norms of stored inputs and interval residuals. For each real/imaginary component use its smallest/largest possible absolute magnitude; lower bounds may be zero. Sum/square/square-root arithmetic and exceptional range are part of the proof.

Finite rational-oracle tests can expose implementation errors and validate fixed fixtures. They do not by themselves prove every primitive's hardware contract. A proof plus explicit runtime assumptions and fail-closed guards must accompany the word "certificate".

## Cost and admission

Account for input/source/material reads, target assembly/identity, twiddle generation, interval transforms, scattering, coercivity/norm bounds, fallback/rejected work, output serialization and peak memory. Do not report only butterfly time. Any mandatory future certification is charged identically to real/complex controls, while extra candidate-specific precision work is additionally charged. Nothing is amortized atQ1 without a stated reuse assumption. An interval FFT can be correct yet too slow or too wide; that is an accuracy/cost result, not a reason to weaken its gate.

**Current decision:** architecture is mathematically viable for an input-specific nominal certificate. No implementation or saved-field admission is granted yet. No full complex-Krylov comparison is admitted until the interval implementation, exact target/input identity, prospective budget, resources and failure semantics pass independent review. The old frozen auxiliary-cap failure remains unchanged. No training or broad performance conclusion follows from this work.
