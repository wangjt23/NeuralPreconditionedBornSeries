# Fresh input-specific certificate for the original stored RTE operator

This is a prospective mathematical design, not an executed certificate or solver admission. It uses an independent CPU enclosure on an **already saved candidate field**, with no new solve, teacher or learned model. The preserved failed low-H engineering entry remains failed with zero full solves. The consuming parent owns any source regeneration, backend implementation, staging, bounded execution and private artifact materialization.

## Target and necessary inputs

Let x be the exact complex vector represented by the saved candidate's finite binary field entries; q is the exact represented source supplied to this certificate. Use the **actual assembled original physical D64** in original FFT index order, and the original stored t, s, phase and weights promoted exactly. Define

    Hx = F^{-1} D64 F x + t*x - s*(phase @ (weights*x)).

F is the exact unnormalized spatial DFT on each angular channel; its inverse divides by N². The target is the nominal original stored discrete operator, including its Nyquist entries, not a regenerated ideal angular geometry or the rounded GPU execution as a nonlinear black box. The latter's unknown error no longer enters: the independent reference encloses nominal Hx directly. No projection to real fields or omission of imaginary components is allowed.

Candidate, q and each coefficient/symbol payload must have recorded hashes, shapes, dtype and exact-bit load semantics. Scalars from previous receipts cannot reconstruct x or certify it. If q is reconstructed through the original CUDA source API, record the exact resulting q hash and reconstruction provenance. Certify that **represented q**, and do not call it a recovered historical saved source unless that identity is verified. Reconstructing q with CPU exp/cos can silently change it. Likewise do not substitute exact bilinear streaming for actual D64; the FP64 assembly rounding is part of the original nominal target.

The theorem assumes finite real nonnegative s, positive t, one material scalar per spatial point shared across angles, and exactly purely imaginary D64. A nonzero real part invalidates the skew-streaming step below; abort or derive a separately reviewed bound, rather than rounding it away.

## Coercivity can reuse the prior exact-rational method

Let P_ij = phase_ij * weights_j denote exact products of the represented inputs. Define exact rational absolute row and column sums R and C, and choose a certified rational/binary upper bound K_U >= sqrt(R*C). Then

    ||P||₂ <= K_U,
    m_0 = min_spatial (t - s*K_U),
    m_L <= m_0,
    Lambda_U >= max |D64| + max t + max s*K_U.

All extrema/products/subtractions and conversions must be exact rational or outward. The prior `stored_scales` method computes K_U through a rational square-root ceiling and m_L through downward scalar conversion. Its **coercivity component is reusable** when all material and angular operands are identical and hashes/provenance are verified. Recomputing this small angular bound plus material minima from the current exact buffers is preferable to trusting an old scalar alone. Its old streaming formation allowance is unnecessary when actual D64 entries are loaded directly: use max abs(imag(D64)) for the purely imaginary captured symbol.

For every complex v, skew-Hermitian streaming and the shared angular material structure give

    Re <v,Hv> >= m_L ||v||².

No exact symmetry/circulant property of the rounded phase is needed. If m_L>0, Cauchy–Schwarz implies sigma_min(H)>=m_L; H is invertible on this finite-dimensional complex space and ||H^{-1}||<=1/m_L. This is a statement about the stored periodic discretization only.

## Enclosure theorem using rectangular intervals

Represent real/imaginary parts by closed binary64 endpoint intervals. Every primitive addition, subtraction, product and scaling must contain its exact real-arithmetic result. Twiddle rectangles must contain the **mathematical** roots exp(±2πij/k), not merely a libc sin/cos approximation. Start x and all represented coefficients as point intervals.

For an exact radix-two butterfly with old inputs u,v and twiddle z, compute both outputs from the same old inputs:

    w = z*v;       u_new = u+w;       v_new = u-w.

Complex interval multiplication forms the four real products with outward bounds **before** their two add/subtract combinations. Induction over each stage, then over the two spatial axes, proves that the resulting forward intervals contain Fx. Multiply by the point D64, apply the inverse with conjugated certified twiddles and outward N^{-2} scaling, and obtain an enclosure of F^{-1}D64Fx. Weighted angular contraction must retain the exact nominal operands `phase @ (weights*x)` followed by multiplication by s; a once-rounded preassembled phase*weights matrix is a different target.

Combining the streaming, t*x and weighted-scattering enclosures, and subtracting from q, gives a rectangle r_box containing the exact nominal residual r=q-Hx. Shared variables and twiddles are treated independently by interval operations; this loses correlations and can widen the result, but cannot invalidate inclusion. Neither a global GPU error constant nor a preconditioner-linearity assumption occurs in this argument.

For a real interval [a,b], its squared magnitude upper uses max(|a|,|b|)²; the lower uses0 when a<=0<=b, otherwise min(|a|,|b|)². Sum real and imaginary squared bounds outward, then enclose their square roots. This provides R_L<=||r||<=R_U, Q_L<=||q||<=Q_U and X_L<=||x||<=X_U. Exact or outward norm reduction is part of the certificate; an ordinary norm of interval midpoints is insufficient.

## Fixed accuracy and one-percent reserve

The original mathematical thresholds remain exactly

    tau = 1/100000,    epsilon = 1/1000,    paired_delta = 1/10000.

The candidate's nominal residual and relative solution error are certified if

    Q_L>0, m_L>0, m_L*X_L-R_U>0,
    R_U <= tau*Q_L,
    R_U/(m_L*X_L-R_U) <= epsilon.

Proof: ||x-x_*||<=R_U/m_L and ||x_*||>=X_L-R_U/m_L, where Hx_*=q. The last quotient therefore bounds error relative to the exact solution. Final comparisons can use exact rational cross-products of returned binary endpoints, avoiding a new floating division or rounded threshold. Equivalently test (1+epsilon)*R_U <= epsilon*m_L*X_L with a strictly positive denominator.

To retain a prospectively frozen one-percent uncertainty ledger, choose a represented residual center c and a certified ball radius E_R covering ||r-c||. For a rectangular enclosure, take outward component distances from c and their Euclidean norm; for the stagewise design below use the propagated radius plus final subtraction error. Require

    ||c||_U <= (99/100)*tau*Q_L,
    E_R <= (1/100)*tau*Q_L,
    R_U = ||c||_U + E_R <= tau*Q_L,

with the sum outward or exact. The norm uncertainty is already paid inside ||c||_U and source-norm uncertainty through Q_L. This preserves tau; it is not a one-percent relaxation. These are direct input-specific residual conditions, rather than a refit of old homogeneity caps or an empirical global action constant. A direct tight rectangle residual bound may certify tau while the separate reserve condition fails; report those classifications separately, and do not silently waive the reserve.

For a paired baseline b with its own saved field and exact-bit source/operator provenance, also certify

    ||x-b||_U <= paired_delta * ||b||_L.

Individual1e-3 solution-error certificates do **not** imply the1e-4 paired condition. The weaker residual-based sufficient test (R_x,U+R_b,U) <= paired_delta*m_L*||b||_L can also be used, but can be much less decisive than directly enclosing the saved field difference. If no paired baseline field is available, certify the individual field only; make no matched-solver claim.

A failed upper-bound test is inconclusive, not proof of inaccurate x. A residual lower bound R_L>tau*Q_U proves a nominal residual failure. Missing/invalid backend assumptions, provenance, m_L or enclosures give a separate invalid/inconclusive classification. A certificate for a saved teacher is evidence that the **checker works on that teacher**; it is not evidence for a future complex Krylov run, convergence, OOD behavior or savings.

## Fallback if rectangular widths exceed the fixed reserve

A tighter center-plus-global-norm-radius method is available without a GPU contract. In each unnormalized radix-two stage, the exact block-diagonal butterfly map has norm sqrt(2), because its true twiddles have modulus1. If current center c has ||v-c||<=E, independently enclose the exact stage applied to the **point stored center c**, compare that local enclosure with the actual next stored center, and obtain a certified local error norm e. Then

    E_next <= sqrt(2)*E + e.

Use a certified upper sqrt(2), outward products/sums, and account bit-reversal/permutation as exact. Crucially use the known modulus of the true roots for the propagation factor, rather than taking an interval-corner twiddle norm. This avoids propagating rectangular wrapping through all stages. The local center can be computed by ordinary CPU arithmetic; its precise stored bits are compared to the local exact-stage enclosure, so no fitted FFT-wide CPU constant is assumed.

A two-dimensional unnormalized forward transform has norm N, and its normalized inverse has norm1/N. Let E_F bound the own forward-center error, e_D bound local multiplication error on the center spectrum, d=max|D64|, and E_I be the local inverse-center error on its point input. Then streaming-center error is at most

    E_stream <= E_I + (d*E_F+e_D)/N.

Pay any inverse-normalization rounding in E_I. Compute independent local collision and final combination/subtraction enclosures, turn their distances from stored centers into norm radii, and add them outward. This yields an input-specific E_R for the same target and unchanged reserve. Correct scaling, old-input retention, local norm aggregation and every rounding allowance need fresh source review. It is a fallback theorem, not a measured assertion that this implementation passes or is fast.

## Trust boundary and cost

The claim moves from an unknown GPU-wide error contract to an audited native interval kernel plus mathematical twiddle proofs. It still trusts exact-bit I/O, the integer/rational runtime, correctly rounded/faithful elementary CPU operations under the stated format, and reliable adjacent-float endpoints. Record compiler/version/options, binary/source/twiddle hashes, binary64 properties, rounding mode and process-local denormal state. No fast-math/reassociation or contraction across endpoint expressions is allowed. Require a supported native format/evaluation path and disabled FTZ/DAZ; otherwise outward-neighbor arithmetic need not include underflowed exact results. Reject NaNs and unhandled infinities.

With correctly rounded binary64 operations, nextafter toward each infinity encloses a finite primitive result, including gradual underflow. Expand each elementary operation separately: one nextafter after the compound expression m*X-R is not enough. Compute the lower product first, then the lower subtraction, or use exact final cross-products. Dynamic rounding is optional; nearest-plus-outward-neighbor avoids requiring frequent rounding-mode changes. GCC explicitly qualifies `-frounding-math`; a flag alone is not an implementation proof. [GCC optimization documentation](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html).

Native C++ endpoints are a practical first bounded implementation. At L12/N256, one forward-plus-inverse2D action has12,582,912 complex butterflies and9,437,184 angular contraction terms, before endpoint expansion. Python Fraction is suitable for the **small reusable twiddle table, coercivity and independent tiny fixtures**, not a credible fast full-grid transform: intermediate rational sizes and interpreter operations become costly. Already available MPFR directed primitives are a credible alternative if present; no installation is implied. Its rounding semantics are explicit, but availability and full-application cost must be checked. [MPFR manual](https://www.mpfr.org/mpfr-current/mpfr.html#Rounding).

Measure twiddle construction/verification, compilation/preflight, coefficient/provenance loading, norms, H enclosure, final comparisons and owned-process total separately. Cache only with input-independent table hashes or fixed-material coefficient hashes, charge cold Q1 setup honestly, and expose any Q16 amortization as conditional on reuse. Any future comparative solver protocol must charge required certificates to every arm and candidate-specific work additionally. No useful width, runtime, deployability or improved economics is established here.
