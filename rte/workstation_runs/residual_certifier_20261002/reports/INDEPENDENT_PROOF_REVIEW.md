# Independent proof review: saved-field CPU interval RTE certifier

2026-10-02. Theory and static source review only. No backend was imported or executed, no field/data was generated, no project action or remote/GPU operation was performed, and no implementation file was edited. This note reviews the proposed proof design, not completed numerical certificates or runtime measurements.

**Conclusion.** An input-specific CPU interval evaluation can certify the exact nominal discrete residual of a saved candidate without establishing a uniform error bound for the GPU action. The design is sound if every input and twiddle is enclosed exactly, every elementary interval operation encloses its real-arithmetic result, and the FFT/indexing/collision construction represents the accepted original operator. The strongest outstanding risks are endpoint-arithmetic assumptions, target/provenance mismatch, interval width at full grid size, and interpreting individual solution-error bounds as the tighter paired-field gate.

## 1. What is being certified

Fix the saved binary-valued candidate \(x\), original source \(q\), captured complex streaming symbol \(D_{64}\), and exact promoted stored material/angular buffers \(t,s,\Phi,w\). Interpret each finite stored binary value exactly. The target is

\[
H=\mathcal F^{-1}\operatorname{diag}(D_{64})\mathcal F
  +T-S\Phi\operatorname{diag}(w),\qquad r=q-Hx. \tag{1}
\]

The spatial DFT is forward unnormalized and inverse divided by \(n_xn_y\), on the two spatial axes independently for every angle. The actual captured symbol already includes \(i\). The cached original [physical action](../../remote_reference/rte/models.py) first forms `weights*x`, contracts `phase` against the input-angle index, multiplies by the local scattering material, and returns `streaming + t*x - scattering`. Formula (1) uses the exact products of those stored collision operands; it must not substitute a separately rounded preassembled phase-times-weight matrix, reference inverse, ideal quadrature, low FP32 symbol, or modified Nyquist convention.

The certificate proves a property of this **saved input and fixed nominal discrete target**. It does not enclose the GPU implementation's complete computation for arbitrary inputs, prove a global GPU error constant, or certify continuum/real-intensity accuracy. It can bypass an Arnoldi residual-gap estimate because it evaluates the complete saved field directly. It does not change the prior stopped execution, gate, or solver status.

Required target checks are finite inputs; exact shapes and angular order; the same original source and material/geometry generation for candidate and baseline; immutable hashes of every relevant buffer; no loss of binary bits during saving/loading; and explicit symbol convention. A binary file or exact IEEE bit encoding is preferable to undocumented decimal parsing. If decimal serialization is used, the reader's exact round-trip contract must be checked. The saved candidate is a fixed field, not an interval representing uncertainty about an ideal iterate.

For the skew/coercivity proof below, check the captured symbol's real part is exactly zero and its imaginary part finite. Otherwise the streaming Hermitian part must be bounded explicitly. Real-source Nyquist images can be complex; projecting the field or zeroing a Nyquist coefficient changes (1).

## 2. Interval inclusion proof

Represent a complex scalar by a rectangle \([a,b]+i[c,d]\). Saved binary-valued components begin as singleton intervals. For real intervals, use outward addition and subtraction and the minimum/maximum of all four outward endpoint products for multiplication. Complex multiplication follows

\[
(A+iB)(C+iD)=(AC-BD)+i(AD+BC). \tag{2}
\]

Every term in (2) is an interval operation. Dependencies between repeated quantities can widen the rectangle, but do not invalidate inclusion. Multiplication cannot use only two endpoint products unless the signs prove that shortcut valid. Division needed for FFT normalization is by a known positive power of two; include underflow rounding rather than assuming every scaling operation is exact.

For a power-of-two axis, bit-reversal followed by radix-two butterflies computes the exact DFT algebraically. Each butterfly uses

\[
u'=u+\zeta v,\qquad v'=u-\zeta v,
\]

where the interval \(\zeta\) contains the exact mathematical root of unity. Both outputs must use the old \(u,v\); in-place aliasing or updating \(u\) before computing \(v'\) breaks the algebra. Induction over butterfly stages proves the transform enclosure. Applying this proof to rows and columns gives the rectangular transform enclosure. Reject non-power-of-two axes unless a separate algorithm/proof is supplied. Length-one axes are the identity.

Use negative-angle roots for the forward transform, their conjugates for the inverse, and the exact matching inverse normalization. A correct proof requires indexing and array strides to agree with the source's last-two-axis convention. An FFT with the wrong sign, inverse scale or output permutation can still produce very tight intervals around the wrong target.

Multiplying the forward enclosure by the captured \(D_{64}\), applying the inverse enclosure, and evaluating the exact stored collision path with interval operations gives \(Hx\in\mathcal H\). Finally, interval subtraction of the saved source gives \(r\in\mathcal R=q-\mathcal H\). This is a direct input-specific inclusion theorem under the stated arithmetic contracts. It uses no independence, stochastic roundoff, preconditioner linearity, or solver-convergence assumption.

### Certified twiddles

Do not insert ordinary `sin`, `cos`, `acos`, `M_PI` or a NumPy/PyTorch root as a singleton certified coefficient. One elementary construction is Machin's identity

\[
\pi=16\arctan(1/5)-4\arctan(1/239). \tag{3}
\]

For \(0<z<1\), a partial sum of \(\arctan z=\sum_{j\ge0}(-1)^jz^{2j+1}/(2j+1)\) and that sum plus its next signed term bracket the exact value. Compute these endpoints rationally. The lower \(\pi\) endpoint uses the lower \(\arctan(1/5)\) and **upper** \(\arctan(1/239)\); the upper endpoint reverses those choices. Identity (3) follows from the arctangent addition formula: \(\tan(2\arctan(1/5))=5/12\), \(\tan(4\arctan(1/5))=120/119\), and subtracting \(\arctan(1/239)\) yields tangent one in the first quadrant.

Reduce the integer twiddle indices by exact quadrant symmetries. Give the roots \(\pm1,\pm i\) exact singleton values. A reduced angle can lie in \([0,\pi/4]\), represented by a rational enclosure derived from (3). Evaluate sine/cosine with rational Taylor sums plus a proved remainder bound, or a proved endpoint-monotonicity bracket. A simple valid bound is the Lagrange remainder \(|R_p|\le\theta_{\max}^{p+1}/(p+1)!\), because real sine/cosine derivatives have magnitude at most one. Interval evaluation of the polynomial plus this symmetric remainder avoids assuming monotonicity across an unreduced quadrant boundary. Alternating-series tightening is valid only after checking the decreasing-term conditions on the angle domain.

Convert the rational lower endpoint downward and upper endpoint upward, preserving inclusion. Reuse the immutable certified twiddle table. Generating later twiddles by an ordinary floating recurrence is not covered; either generate them independently or propagate a rigorous interval recurrence and accept its extra width. Fix precision/remainder choices prospectively, or prescribe a deterministic bounded refinement rule whose changes are reported.

## 3. Endpoint arithmetic is part of the proof

For finite IEEE binary64 operations correctly rounded under the declared environment, applying `nextafter` toward the relevant infinity to the correctly rounded result encloses its exact result. The proof needs separate elementary operations with the intended precision and reliable adjacent-float stepping. A compiler/backend contract must establish this; source syntax alone does not.

Record compiler/version/options, endpoint format, rounding mode, platform and denormal behavior. Exclude unsafe reassociation and unproved contractions across endpoint expressions; handle overflow, subnormals, signed zero, NaN and any flush-to-zero behavior explicitly. A power-of-two scale can round near underflow. A finite positive lower bound that underflows to zero must become zero, not remain a spurious positive number. Infinite endpoints can be valid enclosures, but cannot support a finite acceptance certificate; undefined endpoint operations must fail closed.

The [GCC optimization documentation](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html) describes contraction controls and the transformations enabled by fast-math. It also states that `-frounding-math` alone does not guarantee suppression of every rounding-sensitive optimization. Compiler flags are therefore a necessary recorded part of an audited arithmetic contract, not a complete proof by themselves. A fixed round-to-nearest plus explicit adjacent-float enclosure design can avoid repeated rounding-mode switches, provided its individual-operation contract is established. This review does not recommend installing another package or changing system arithmetic settings.

Do not assume that `long double` means binary128 on every machine, or that an ordinary libc square root/transcendental automatically gives a directed enclosure. Either establish correctly rounded primitive square roots and expand outward, or use integer/rational square-root bracketing for the relatively few final scalar bounds. The [C++ floating-environment interface](https://eel.is/c++draft/cfenv) exposes rounding-mode controls; it does not by itself establish this application's complete floating implementation contract.

## 4. Norm and coercivity bounds

For a real interval \([a,b]\), set

\[
\operatorname{minabs}([a,b])=
\begin{cases}0,&a\le0\le b,\\ \min(|a|,|b|),&\text{otherwise},\end{cases}
\quad
\operatorname{maxabs}([a,b])=\max(|a|,|b|).
\]

Sum the squared minimum magnitudes downward and squared maximum magnitudes upward over every real and imaginary component. Enclose the square root outward. This yields norm lower and upper bounds. Do not square the midpoint or only one endpoint when the interval crosses zero. The Euclidean norm includes all angles and spatial cells; no quadrature weighting or source normalization may be inserted silently.

Compute \(Q_{\rm lo}>0\) for the original source and \(X_{\rm lo}>0\) for the saved candidate. These fields are exact stored points, so exact dyadic squared-norm accumulation followed by rational square-root bracketing is also possible. Ordinary binary64 norm reduction is not a lower-bound certificate. Compute \(R_{\rm up}\) from the residual interval; a lower residual bound \(R_{\rm lo}\) is useful for distinguishing a proved failure from an inconclusive enclosure.

Let \(P=\Phi\operatorname{diag}(w)\). One conservative certified bound is

\[
\|P\|_2\le p_{\rm up}
 :=\operatorname{up}\sqrt{\|P\|_1\|P\|_\infty}.
\]

Use exact or outward sums of the absolute **column-weighted** phase products. For finite real total material \(t\), nonnegative scattering \(s\), shared across angles at each cell, and a pure-imaginary captured stream symbol, an outward lower bound is

\[
m_{\rm lo}\le\min_j(t_j-s_jp_{\rm up}),\qquad m_{\rm lo}>0. \tag{4}
\]

Then \(\operatorname{Re}\langle v,Hv\rangle\ge m_{\rm lo}\|v\|^2\), so \(\|Hv\|\ge m_{\rm lo}\|v\|\). This proof does not require \(P\) to be symmetric or exactly normalized. The Fourier similarity is unitary after matching normalization, so its pure-imaginary multiplier is skew-Hermitian even at unpaired Nyquist frequencies. If (4) is nonpositive, the sufficient norm bound is inconclusive; it does not prove the actual collision is noncoercive. A tighter Hermitian-part bound would require its own certificate. Negative or angular-varying scattering needs a different bound rather than silently reusing (4).

## 5. Acceptance and the separate paired-field gate

With a certified \(m_{\rm lo}>0\), the exact nominal problem has a unique solution \(x_*\). The saved candidate satisfies

\[
\|x-x_*\|\le R_{\rm up}/m_{\rm lo},
\qquad
\frac{\|x-x_*\|}{\|x_*\|}
\le\frac{R_{\rm up}}{m_{\rm lo}X_{\rm lo}-R_{\rm up}}, \tag{5}
\]

when the denominator is positive. For the frozen \(\tau=10^{-5}\), sufficient certified acceptance is

\[
R_{\rm up}\le\tau Q_{\rm lo},\qquad
\operatorname{down}(m_{\rm lo}X_{\rm lo}-R_{\rm up})>0,\qquad
\operatorname{up}\frac{R_{\rm up}}
 {\operatorname{down}(m_{\rm lo}X_{\rm lo}-R_{\rm up})}
\le10^{-3}. \tag{6}
\]

Every product, subtraction, division and comparison used for these thresholds must preserve its one-sided direction. In particular, implement the denominator as `p_lo = down(m_lo*X_lo)` followed by `d_lo = down(p_lo-R_up)`, not one adjacent-float expansion after an expression with two unaccounted roundings. The same requirement applies to constituent endpoint products before complex multiply's additions/subtractions. Exact rational scalar comparisons against the declared decimal limits avoid ambiguous last-bit decisions. State whether the frozen threshold is the exact decimal rational or the original stored binary tolerance; do not mix conventions near equality.

For a saved baseline \(x_b\) on the same target, the original paired gate uses \(\|x-x_b\|/\|x_b\|\). Certify this separately by interval or exact dyadic subtraction of the **saved fields**, a difference norm upper bound \(D_{\rm up}\), and baseline norm lower bound \(X_{b,\rm lo}>0\):

\[
\operatorname{up}(D_{\rm up}/X_{b,\rm lo})\le10^{-4}. \tag{7}
\]

Both fields need their own residual/error certificates and matching target/source hashes for a paired accepted-solver statement. Individual relative solution-error certificates of \(10^{-3}\) do **not** imply (7). A stronger but potentially much more conservative sufficient condition is

\[
\operatorname{up}\frac{R_{c,\rm up}+R_{b,\rm up}}
 {\operatorname{down}(m_{\rm lo}X_{b,\rm lo})}\le10^{-4}. \tag{8}
\]

It follows by bounding \(H(x-x_b)=r_b-r_c\). Direct difference certification in (7) can succeed even when (8) is inconclusive because common errors cancel. If a stronger true-solution-normalized difference is desired, use \(D_{\rm up}/(X_{b,\rm lo}-R_{b,\rm up}/m_{\rm lo})\) with a certified positive denominator; this is an additional condition, not the original paired denominator.

### The fixed one-percent allowance

Freeze its meaning before evaluating certificates. The previous prospective reserve \(\rho=.99\tau\) for the measured residual and a separately allocated \(.01\tau\) uncertainty budget is one coherent policy. A direct interval certificate can instead enforce (6) and a separately specified enclosure-tightness diagnostic, such as \(R_{\rm up}-R_{\rm lo}\le .01\tau Q_{\rm lo}\). That width inequality measures enclosure uncertainty; it is not itself the residual acceptance proof and is not the norm of the CPU/GPU image difference. Do not redefine the one-percent denominator or reserve after seeing a result. No reserve, interval precision or acceptance threshold is instantiated or relaxed by this review.

A failure of the sufficient upper-bound tests is generally **inconclusive**. The actual residual may be smaller than the interval upper bound. To certify residual rejection, a norm lower bound must exceed a source norm upper bound times the threshold. Likewise, a solution-error upper bound above \(10^{-3}\) does not establish the actual error exceeds it. Separate proved acceptance, proved rejection where available, arithmetic failure, resource timeout and inconclusive width.

## 6. Credible cost scope and remaining proof obligations

A radix-two forward and inverse 2D FFT over \(L\) angular fields needs

\[
L n_xn_y(\log_2n_x+\log_2n_y)
\]

complex butterflies in total. At \(L=12,n_x=n_y=256\), this is 12,582,912 butterflies before interval expansion of complex arithmetic. Angular collision additionally has \(L^2n_xn_y\) weighted real-coefficient complex terms: 9,437,184 at that shape. These are operation counts, not runtime estimates. Copies, certified twiddle construction, norm reductions and field-pair checks add work.

C++ using bounded-size interval endpoints is a credible implementation route for these tens of millions of structured operations. Python `Fraction` over the entire full-grid FFT has large per-operation allocation, normalization and integer-arithmetic overhead, so it is better suited to twiddle construction, tiny independent oracles and final scalar checks. Fraction cost cannot be equated to native scalar FLOP cost. No time claim is justified without an authorized measured run; certification time and data acquisition must be reported separately from solver time.

Natural rectangular interval FFTs discard correlations between paths, so widths can grow substantially through forward/inverse transforms and cancellation. Correct inclusion does not guarantee a useful one-percent enclosure at binary64. A timeout or excessive width must remain visible. Predetermined higher endpoint precision or a bounded refinement policy would need its own arithmetic contract and resource authorization; do not silently tune a passing precision from outcomes.

Before a certificate claim, discharge and record: exact immutable target loading; interval primitive arithmetic including range behavior; rational twiddle/remainder construction; FFT indexing/normalization; collision operand order; outward norm and coercivity bounds; all final one-sided threshold calculations; direct paired-field comparison; and negative/inconclusive status handling. The present design has a valid mathematical route, but neither backend validation nor an actual saved-field certificate has been observed in this review.
