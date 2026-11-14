# Prospective1% input-specific reserve policy

2026-10-02. Mathematical protocol recommendation only; no certificate or RTE action was executed by this reviewer.

Freeze tau=1/100000 and reserve fraction b=1/100 before measurements. Let a complex rectangular interval field contain the exact nominal residual `r=q−H*x` for the immutable stored candidate and original assembled-D64 target. Choose a finite stored center c lying inside every real and imaginary component interval. Let E_U bound the Euclidean norm of the entire difference box `rbox−c`, C_U bound the exact norm of c from above, R_U bound the exact norm of every residual in the original box from above, and Q_L>0 bound the stored source norm from below.

The prescribed three source-residual predicates are

    C_U <= (1−b)*tau*Q_L,
    E_U <= b*tau*Q_L = 1e−7*Q_L,
    R_U <= tau*Q_L.

Evaluate products/comparisons exactly on the returned binary scalar bounds, or round thresholds downward and numerators upward. The first two imply `||r||<=||c||+||r−c||<=C_U+E_U<=tau*Q_L<=tau*||q||`. The third is redundant mathematically when all bounds are valid, but independently checks direct-box acceptance and is appropriate as a fail-closed predicate. This is certificate slack; it does not alter the original1e−5 stopping or acceptance tolerance.

**No separate scalar-width surcharge is necessary** when the bounds are genuinely outward: norm inflation is already charged through C_U/E_U/R_U, and source-norm uncertainty through Q_L. Adding the same norm uncertainty again to E_U double-counts it. Describe E_U as **vector-enclosure uncertainty**, with scalar/norm/source/threshold arithmetic separately enclosed, rather than claiming E alone includes every scalar uncertainty. If ordinary computed norms are used instead, their errors must be paid explicitly and the above proof is incomplete.

For component `[l_i,h_i]` and chosen `c_i`, a sufficient radius is `max(c_i−l_i,h_i−c_i)` with outward evaluation. Combine real and imaginary radii in the Euclidean norm. The norm interval width `R_U−R_L` is **not** this radius. Example: center `(9.9e−6,0)` and imaginary uncertainty `[-1e−6,1e−6]` have a norm width about5.04e−8, although vector radius is1e−6. WithQ=1 the norm width fits the1e−7 reserve, while the vector radius violates it tenfold. The direct residual can still be below1e−5; failure here means the declared reserve is not established, not that the field is physically inaccurate.

Also require finite/bound-valid data, certified m_L>0 and X_L<=||x|| with

    R_U < m_L*X_L,
    R_U/(m_L*X_L−R_U) <= 1e−3.

Equivalently `R_U <= [1e−3/(1+1e−3)]*m_L*X_L` implies both. The first strict inequality protects the denominator. It establishes a relative error bound with respect to the unknown unique nominal solution; no continuum or intensity positivity follows. The separately prescribed paired1e−4 field gate does not follow from two1e−3 solution-error bounds.

## Candidate and fixture scope

Root's metadata inventory identifies saved q(complex64) and target(complex128) for sealed N64/N128 development banks. Certify their actual bytes, promoting q exactly; do not rerun a solver or silently regenerate the target. Hash and check the material/source/geometry and actual D64 identity. There is no saved N256 candidate from prior scalar benchmarks.

The proposed N256 `x=M_source`, `q_fixture=stored GPU H(x)` is a newly fixed manufactured discrete problem, not a solved original source-family case. Its known x need not be the exact nominal solution, since q_fixture contains GPU rounding; the input-specific interval residual/error bound determines that discrepancy. Original q withx=0 is an expected negative control and must fail the source-residual predicates. Such failure is not a solver failure.

All certification, source/target assembly, twiddle generation, enclosures, exceptional-range fallback, negative/rejected checks and serialization costs must be retained. Future mandatory certification must be charged to both matched controls without assumedQ1 amortization. This round admits no new full solve even if saved-field/fixture certificates pass. Architecture and scalar reserve proof do not replace independent implementation/primitive-contract admission.
