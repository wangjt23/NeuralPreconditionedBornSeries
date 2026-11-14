# Stagewise center plus Euclidean norm radius for fresh original-H certification

Prospective source-level theorem and implementation obligations; no solver, original numerical action, GPU/SSH, teacher generation or source export was performed by this author. The parent owns the numerical implementation and any bounded checker experiment. The old rectangular checker, its NO_CERTIFICATE outcomes and the failed auxiliary low-H entry remain preserved. Persisted new N256 fixture pairs must be reused by exact input hashes; an earlier transient pair is not retrospectively replayable from scalar receipts.

## Target and representation

The target remains the exact nominal original stored periodic operator

    Hx = F^{-1} D64 F x + t*x - s*(phase @ (weights*x)),

with the actual assembled physical D64, exact represented t/s/phase/weights/source/candidate bits, original Fourier ordering, both spatial axes and backward normalization. No ideal geometry, low D32, rounded preassembled angular matrix, real-field projection or padded transform substitutes for it. D64 must be finite and exactly purely imaginary for the original coercivity argument. Fields retain both complex components, including Nyquist effects.

A ball (c,E) means c is a finite represented complex128 field and the exact intermediate v satisfies ||v-c||₂<=E, where E is a nonnegative certified scalar upper bound. The norm is over the **whole supplied tensor**, including all angular channels/rows/batch entries. Exact saved inputs start with E=0. A local point-center rectangle is temporary; only its represented center and scalar norm radius pass to the next stage. This removes rectangular dependency propagation without changing the target.

## Butterfly-layer lemma

For a true unit-modulus twiddle z, the unnormalized complex butterfly is

    B_z = [[1,z],[1,-z]],      B_z* B_z = 2 I.

An entire radix-two layer is a direct sum of these blocks and exact permutations. Therefore its Euclidean operator norm is exactly sqrt(2), independent of the number of butterflies, rows or angles. The inverse-sign layer has the same norm. Use one sqrt(2) factor per **disjoint layer**, not one per butterfly and not an extra angular/row factor.

Suppose ||v-c||₂<=E. Let a local interval evaluation on the point represented field c enclose Bc, using certified rectangles containing the mathematical roots. Choose any finite represented next center d, and enclose the difference Bc-d componentwise. For each real/imaginary component, take an outward maximum absolute deviation; outward square/sum and a certified square-root upper give e>=||Bc-d||₂. Then

    ||Bv-d||₂ <= ||B(v-c)||₂ + ||Bc-d||₂
               <= sqrt(2)*E + e.

Thus return E_next>=sqrt(2)*E+e. This is an exact triangle-inequality proof. The true root's known modulus1 justifies the propagation factor even if its enclosing rectangle has corners outside the unit circle. The root rectangle is still required in the **local defect** calculation; a rounded point twiddle is insufficient.

Compute both outputs from the same old u,v before writing either new field. Bit reversal, axis moves, reshape and permutations must preserve exact field entries and indexing. When local defects are split into independent chunks, aggregate their squared upper bounds and take a square root; taking only the maximum chunk error does not bound the whole-stage norm. Summing chunk errors is also safe but looser. Never reuse a local e from another field.

The represented center d need not itself be a certified exact FFT result. Its actual stored bits are compared with an enclosure of Bc, so even an approximate center is covered. Only the enclosure/arithmetic/norm path supplies proof. Choosing the clipped represented midpoint of the local rectangle is a practical way to keep e small. A center coordinate equal zero does **not** remove its incoming norm radius.

## Two-dimensional normalization

For N_x=2^{p_x}, N_y=2^{p_y}, compose p_x+p_y layers, with exact bit permutations, to obtain the unnormalized two-dimensional transform. Its norm is sqrt(A), A=N_x*N_y. The backward-normalized inverse has norm1/sqrt(A).

If normalization is applied after each inverse axis, first propagate that axis's unnormalized layers. Enclose exact point scaling c/N_axis, choose the actual stored scaled center d, obtain local error e_scale, and return

    E_scaled >= E_before/N_axis + e_scale.

Two such passes give inverse norm1/sqrt(A). Alternatively scale after both unnormalized axes: E_scaled>=E_before/A+e_scale. For square N these factors are forward N and inverse1/N, while the final unnormalized inverse scaling is1/N². Mixing1/N and1/N² changes the target.

Power-of-two normalization is exact only when the scaled represented center is representable; gradual-underflow rounding must be enclosed. General safe interval scaling avoids assuming exactness. Scalar-radius products, sums and divisions must be exact rational then outward, or use correctly outward primitives at each operation. A certified upper sqrt(2) is needed; an ordinary downward-rounded sqrt(2) is not sufficient. No fixed gamma FFT model or measured global FFT constant is introduced.

## Streaming, collision and residual composition

Let c_F be the own forward center and E_F its certified error relative to Fx. Set d_D=max|D64|; for a purely imaginary symbol it is the maximum exact magnitude of the represented imaginary entries. Enclose exact point D64*c_F, choose stored center c_D and derive local defect e_D. Then

    ||D64 Fx-c_D||₂ <= d_D*E_F+e_D.

Apply the own inverse ball transform to that uncertain input. Equivalently, if E_I is the inverse's local error for point input c_D, the streaming error satisfies

    E_stream >= E_I + (d_D*E_F+e_D)/sqrt(A).

Pay all center normalization errors in E_I. The proof uses the fixed diagonal symbol, not a homogeneous reference inverse or a newly assembled bilinear stream. A supplied real component must not be silently dropped.

Collision can be enclosed directly on the immutable original **point x**, since its input has no incoming transform radius. Retain exact stored t, s, phase and weights; compute weighted angular input, dense phase contraction and material multiplication with outward endpoints. An interval product cache for P_ij=phase_ij*weights_j is valid, but a once-rounded point P changes the nominal target.

For the implementer's economical fused local-box path, enclose

    r_local = q - c_stream - t*x + s*(phase @ (weights*x)).

Choose a represented residual center c_r and let e_local bound ||r_local-c_r||₂ from this point-center local rectangle. Since the exact residual differs from r_local only by the streaming error,

    ||q-Hx-c_r||₂ <= E_stream+e_local = E_r.

All local collision/addition/subtraction rounding is already paid in e_local. Recenter once, then add the incoming streaming radius outward. Do not discard that radius because c_stream or some source/collision coordinates vanish. An alternative with separate attenuation, scattering, combination and subtraction defect balls is also valid if every local error is charged once.

## Norms and fixed original acceptance

Let C_L<=||c_r||<=C_U be independently enclosed point-center norms. The residual ball gives

    R_L = max(0,C_L-E_r),      R_U = C_U+E_r,

with downward/upward conversions. A direct rectangular bound may be retained as a separate diagnostic; it is not the propagated global radius. Outward norm reduction must cover each square and positive accumulation, followed by rational square-root bracketing or a separately certified square root. One nextafter after an unchecked ordinary dot/norm is insufficient.

Recompute or validly reuse the exact-rational coefficient bound

    K_U >= sqrt(|| |phase*diag(weights)| ||₁
                * || |phase*diag(weights)| ||∞),
    m_L <= min(t-s*K_U),      m_L>0.

Pure-imaginary streaming is skew-Hermitian under the exact Fourier similarity. Real nonnegative s and materials shared across angles give Re<v,Hv>>=m_L||v||² and ||H^{-1}||<=1/m_L. Exact phase symmetry is unnecessary.

Freeze tau=1/100000, reserve alpha=1/100 and solution epsilon=1/1000 before results. Require positive source lower Q_L and candidate lower X_L, valid finite enclosures/provenance/native guard, and

    C_U <= (1-alpha)*tau*Q_L,
    E_r <= alpha*tau*Q_L,
    R_U <= tau*Q_L,
    m_L*X_L-R_U > 0,
    R_U/(m_L*X_L-R_U) <= epsilon.

Evaluate thresholds/cross-products exactly on the represented bound scalars. Norm uncertainty is already inside C_U and E_r and source uncertainty inside Q_L. The first two inequalities put all vector uncertainty inside the fixed one-percent reserve; they preserve the original tau. They are not a revised homogeneity cap, empirical GPU contract or one-percent accuracy relaxation. If R_U is reported after an outward float conversion, that converted upper remains the accepted conservative bound; do not round it downward for a boundary case.

Proof of the solution bound: with Hx_*=q, ||x-x_*||<=R_U/m_L and ||x_*||>=X_L-R_U/m_L. The positive denominator yields the displayed relative error bound. A separate paired baseline requires the direct certified gate

    ||x-b||_U <= (1/10000)*||b||_L.

Two individual1e-3 solution-error certificates do not imply that1e-4 field difference. A certificate on a saved teacher or a manufactured represented q proves individual nominal accuracy only; it supplies no new full-solver convergence, OOD, GPU linearity, continuum or learned-speed claim. Negative zero candidates and reserve failures remain independent controls. A failed upper-bound admission is NO_CERTIFICATE/inconclusive, except when a rigorous residual lower bound proves an actual threshold failure.

## Fast paths that preserve proof

- A finite exact point0, +1, -1, +i or -i coefficient gives exact zero/sign/swap products, with appropriately ordered endpoints. Exact point-zero addition is an identity. Branch only on exact endpoint equality and exact values; no isclose, clipping of small values or learned cutoff is permitted. Product defect0 does not remove later butterfly add/subtract rounding.
- A point scalar times an interval needs only the two sign-selected endpoint products; point-by-point multiplication needs one correctly rounded product plus outward neighbors. General interval multiplication still needs the extremal products. Preserve the local inclusion proof when specializing it.
- A zero center may permit exact zero center output and e=0 for a homogeneous local transform, but its **incoming E propagates** with the true operator norm. Only a point input with E=0 is known to be the exact zero field.
- Sign/cardinal transforms are exact even for finite subnormals under the supported boundary. General powers-of-two need verified representability or an enclosed underflow/overflow path. Return owned output or safely immutable views; an in-place fast path must not mutate old butterfly/source/cache inputs.
- Cache immutable twiddles by shape/sign/code proof hash, and coefficient-only products, d_D and m_L by exact target operand hashes/dtype/layout. Check the native boundary each certificate. Never cache a field-dependent e, center norm, residual, source norm or radius under only a material key. Optional source-norm reuse needs the exact q hash and the same bound/proof version.

These reduce redundant interval operations without replacing H by a new rounded target. Stagewise radii can improve enclosure width while costing more local norm reductions than a rectangular FFT. Neither successful width nor faster verification follows from the theorem; record cold construction, coefficient/provenance work, all local norm/FFT/contraction work, final acceptance, serialization and owned-process total. Future matched solver arms must pay the same mandatory checker costs, with conditional amortization stated explicitly.

## Native boundary and primary context

The interval proof retains the authored correctly rounded RN-even binary64 add/multiply boundary, separate endpoint operations, no FMA contraction/reassociation across their expressions, valid exact-bit loading and adjacent-value expansion, and no FTZ/DAZ. Guard IEEE format, FE/hardware control states before and after; guard flags that repeat register checks are not arithmetic probes. Finite primitive tests support source review without proving the compiler. The GPU FFT/contraction/norm error contract remains unnecessary because this own checker encloses nominal Hx directly.

The butterfly/operator-norm identities and ball-composition proof above are self-contained. [GCC's primary optimization documentation](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html) specifies contraction controls and qualifies rounding-sensitive flags; [NumPy1.26's nextafter specification](https://numpy.org/doc/1.26/reference/generated/numpy.nextafter.html) supplies the adjacent-value API semantics used by the boundary. [INTLAB's primary page](https://www.tuhh.de/ti3/rump/intlab/) establishes verified FFT and interval linear algebra as classical infrastructure. None proves this new source/binary's correctness or performance. The previously audited exact-rational Machin/Taylor/Lipschitz twiddle construction is reused unchanged, with its proof/hash and actual generated constants sealed separately.
