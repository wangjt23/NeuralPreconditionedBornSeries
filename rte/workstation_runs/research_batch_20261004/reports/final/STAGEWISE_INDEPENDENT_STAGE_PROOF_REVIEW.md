# Independent review: stagewise FFT centers and Euclidean radii

2026-10-02. Mathematical derivation and cached-source review only. No numerical code, field/data generation, project action, backend execution, GPU or SSH operation was performed. This note does not assess a completed implementation or claim a numerical admission result.

**Conclusion.** Replacing propagated component rectangles with a represented point center and a certified global Euclidean radius is mathematically sound. Every exact FFT stage has norm \(\sqrt2\), so a local enclosure of the defect on the point center gives the recurrence \(E'\le\sqrt2 E+e\). The local interval boxes must enclose the exact-root butterfly applied to the old point center and be compared with the actual represented next center. They must not be substituted for the incoming uncertainty. This distinction avoids repeated global rectangular wrapping while retaining a direct inclusion proof.

## 1. One-stage theorem

Let \(u\) be the exact current transform state and \(c\) its stored finite point center, with

\[
\|u-c\|_2\le E.
\]

An unnormalized radix-two layer is a permutation and direct sum of butterflies

\[
B_\zeta=\begin{pmatrix}1&\zeta\\1&-\zeta\end{pmatrix},
\qquad |\zeta|=1.
\]

Since \(B_\zeta^*B_\zeta=2I\), the exact complete layer \(B\) has \(\|B\|_2=\sqrt2\). A permutation has norm one. The proof uses the mathematical root's modulus, not the norm of its interval rectangle. A certified rectangle may include points off the unit circle; it is still a valid coefficient enclosure for the following local defect calculation.

Compute the next center \(c'\) by the prescribed finite numerical procedure. It can use rounded arithmetic or approximate point twiddles: the theorem only requires the **actual stored** finite \(c'\) used in the defect comparison. Separately evaluate exact-root interval butterflies on singleton intervals representing the old center \(c\), obtaining component rectangles \(\mathcal Y_j\) that contain \((Bc)_j\). Let

\[
\mathcal D_j=\mathcal Y_j-c'_j,
\quad
\mu_j^R=\max(|\inf\operatorname{Re}\mathcal D_j|,
                 |\sup\operatorname{Re}\mathcal D_j|),
\quad
\mu_j^I=\max(|\inf\operatorname{Im}\mathcal D_j|,
                 |\sup\operatorname{Im}\mathcal D_j|).
\]

With every square, sum and square root enclosed upward, set

\[
e\ge\left[\sum_j((\mu_j^R)^2+(\mu_j^I)^2)\right]^{1/2}.
\]

The sum includes the **entire layer**, across all transformed lines, angles and batches. It is an unweighted Euclidean norm, not an average or quadrature-weighted norm. Then

\[
\boxed{\|Bu-c'\|_2\le\sqrt2 E+e.} \tag{1}
\]

**Proof.** \(Bu-c'=B(u-c)+(Bc-c')\). The first term has norm at most \(\sqrt2 E\); the component interval calculation bounds the second by \(e\). Apply the triangle inequality. No stochastic independence or independence of rounding errors is assumed.

Use a certified upper \(\sqrt2_{\rm up}\), then outward multiplication and addition: `product_up = up(sqrt2_up*E)` and `E_next = up(product_up+e)`. One outward step after an expression containing two unaccounted roundings is not enough. Zero terms can be preserved exactly under the proven fast paths below.

## 2. Layer organization and implementation pitfalls

- Both outputs of a butterfly must use the old \(a,b\). If `a` is overwritten before computing `b`, the center/defect pair no longer represents the claimed layer. Save old values by value or use separate buffers. The interval calculation must use those same old centers.
- The factor \(\sqrt2\) is applied once per complete disjoint butterfly **layer**, not once per butterfly, row, angle or sequential loop iteration. The direct-sum operator norm remains \(\sqrt2\) for all those independent blocks together.
- Bit-reversal, transposes and exact by-value permutations preserve the radius. A cast or numerical conversion during a permutation needs its own defect allowance.
- Use negative-angle roots for the forward DFT and conjugates for the inverse. Exact indexing, sign and stride errors are not roundoff and invalidate the target proof.
- Local boxes enclose \(Bc\), not \(Bu\). Propagate the previous global radius through (1). Treating the center as the exact state discards accumulated uncertainty; propagating global coordinate rectangles instead gives a different, often wider algorithm.
- Do not discard defect terms because they are below a numerical tolerance. A small certified upper bound is still part of the radius. Nonfinite centers, invalid intervals or unaccounted range exceptions must fail closed.

The center's computation need not satisfy a separate forward-error theorem, because its realized error is enclosed by the local exact-root comparison. The interval endpoint arithmetic still needs its own strict inclusion contract. Center contraction/FMA is therefore not automatically a mathematical problem when the actual stored next center is compared correctly; unproved contraction or reassociation of the **endpoint enclosure** expressions is a problem.

## 3. Forward, multiplier and normalized inverse

Let \(n=n_xn_y\) and \(p=\log_2n_x+\log_2n_y\). Apply (1) to the two spatial axes, viewed as \(p\) layers on the complete field. For initial radius \(E_0\),

\[
E_F\le\sqrt n\,E_0+
 \sum_{j=1}^{p}(\sqrt2)^{p-j}e_j. \tag{2}
\]

This formula states the ideal exact-arithmetic radius recurrence. Its implemented counterpart uses an outward upper radius for each layer and therefore bounds the state error; a rounded implemented radius need not be less than the displayed unrounded expression. The actual recurrence and any evaluated powers need outward arithmetic. A saved binary field represented exactly has \(E_0=0\).

Use the actual captured complex \(D_{64}\), including its factor \(i\), as the frozen nominal streaming coefficient. Let \(d_{\max}\ge\max_j|D_{64,j}|\) be certified. For point multiplier center \(c_D\), enclose the local defect of \(D_{64}c_F-c_D\) on singleton \(c_F\), giving \(e_D\). Then

\[
E_D\le d_{\max}E_F+e_D. \tag{3}
\]

For pure-imaginary captured symbols, \(d_{\max}\) is the maximum absolute imaginary component; do not reconstruct ideal stream products or substitute the low FP32 symbol. For general complex symbols, certify their magnitudes. A pure-imaginary check is also needed for the original skew/coercivity proof.

The unnormalized inverse has the same \(p\) layers and norm \(\sqrt n\). Its radius before normalization satisfies

\[
E_{I,\mathrm{un}}\le\sqrt n\,E_D+
 \sum_{j=1}^{p}(\sqrt2)^{p-j}e_{I,j}. \tag{4}
\]

Let \(\alpha=1/n\), and let the actual stored streaming center be \(c_S=\operatorname{fl}(\alpha c_{I,\mathrm{un}})\). Enclose its point-scaling defect \(e_\alpha\ge\|\alpha c_{I,\mathrm{un}}-c_S\|\). Then

\[
\boxed{E_S\le\alpha E_{I,\mathrm{un}}+e_\alpha.} \tag{5}
\]

Equations (3)–(5) likewise describe ideal scalar radius updates; their implemented versions round each update upward and certify the corresponding state error. For a square \(N\times N\) grid, the final scale is \(N^{-2}\), not \(N^{-1}\). A rectangular grid uses \(1/(n_xn_y)\). Normalizing each inverse axis separately is also valid, but each scaling's center defect and radius scaling must be included at its actual location. Division by a power of two is exact only when the scaled binary value is representable; subnormal rounding and overflow need proof or enclosure.

As a consistency check, the input-radius contribution of the streaming composition is \(d_{\max}E_0\): the forward \(\sqrt n\), inverse \(\sqrt n\), and normalization \(1/n\) cancel. A formula producing an extra \(n\) or missing the multiplier factor is inconsistent with the exact streaming operator norm.

## 4. Collision and final residual

Preserve the original cached collision target \(T x-S\Phi\operatorname{diag}(w)x\): weight the input angle, contract phase against that angle, and multiply by local scattering material. Do not silently replace it with a separately rounded preassembled phase-times-weight matrix. For an exact saved input, local interval evaluation of the finite angular contraction can directly bound its point-center defect without propagating the FFT radius into that separate branch.

For centers \(c_T,c_C\) with errors \(e_T,e_C\), and actual represented total-action center \(c_H\), enclose

\[
e_+\ge\|c_S+c_T-c_C-c_H\|.
\]

Then \(\|Hx-c_H\|\le E_H\), where \(E_H\) is an outward upper sum of \(E_S+e_T+e_C+e_+\), enclosing each addition. These component-branch errors occupy the same output space, so the triangle sum is safe. Replacing it by the square root of the sum of their squared norms requires an additional orthogonality proof; statistical independence is insufficient.

For actual residual center \(c_r\), enclose the point defect \(e_r\ge\|q-c_H-c_r\|\). The nominal residual satisfies

\[
\|r-c_r\|\le E_R:=\operatorname{up}(E_H+e_r),
\qquad
R_{\rm up}:=\operatorname{up}(\|c_r\|_{\rm up}+E_R),
\qquad
R_{\rm lo}:=\max(0,\operatorname{down}(\|c_r\|_{\rm lo}-E_R)),
\qquad R_{\rm lo}\le\|r\|\le R_{\rm up}. \tag{6}
\]

Source and field norm lower bounds must still be certified. Reuse the original fixed residual, solution-error, paired-field and one-percent reserve policy; a new representation of uncertainty is not permission to change those gates. Report the point residual norm, certified radius and their final nominal bound separately. Neither the radius nor interval width is itself the nominal residual or a uniform GPU action error.

The established sufficient tests remain \(R_{\rm up}\le\tau Q_{\rm lo}\) and a positive denominator in \(R_{\rm up}/(m_{\rm lo}X_{\rm lo}-R_{\rm up})\le10^{-3}\), with every scalar operation outward and \(\tau=10^{-5}\) frozen. The paired \(10^{-4}\) saved-field difference needs its own direct bound. An excessive upper radius is inconclusive unless a lower bound proves failure. This note supplies no numerical gate value or passing result.

## 5. Proven coefficient fast paths

Fast paths must be triggered by exact finite singleton values, not closeness to those values:

- A singleton zero coefficient has an exactly zero product with any finite represented point. A coefficient \(+1\) or \(-1\) acts by exact identity/sign reversal. Coefficients \(+i\) or \(-i\) act by exact component swap/sign reversal. Endpoint order must be reversed correctly under negation.
- These eliminate **product** defect only. Surrounding butterfly additions/subtractions or material-sum operations still need their rounding-defect enclosures. The whole cardinal butterfly is not generally exact.
- A zero point-center coordinate does not mean the unknown exact transform coordinate is zero. Its point-product defect may vanish, but the incoming radius is still propagated through the exact operator norm. Only an exactly zero operator coefficient annihilates the corresponding true uncertainty.
- Powers of two can use an exact scaling fast path only after proving the result is representable, including subnormals. Otherwise enclose the actual scaling error. Signed zeros do not change the mathematical point value; they still require consistent interval ordering and finite/range guards.

The exact cardinal roots should come from certified integer/quadrant structure, not `cos` or `sin` output rounded close to zero/one. Other twiddles retain certified exact-root interval enclosures. Their midpoint/point values may guide the center, but do not replace the nominal mathematical root.

## 6. Coefficient reuse and certified norm wrappers

Frozen material/geometry hashes can justify reusing **coefficient-only** bounds: angular matrix norm bounds, coercivity lower bounds, streaming coefficient norms, and immutable certified twiddle tables. Keys must include all relevant bytes, dtype, shape/angular ordering, grid/index convention, twiddle precision and proof/implementation version. A material hash alone does not cover a changed phase, weight or streaming symbol. Do not modify or regenerate the target to make cached bounds apply.

Material-hash equality never justifies reuse of a field-specific layer defect, center, input norm, residual radius or field-difference bound. Those depend on the actual saved field and numerical center trajectory. Fresh-evaluation policy and accounting also remain binding even when scalar coefficient caches are mathematically valid. Cache setup/use and verification time should remain visible in the owning implementation's receipts.

A certified upper norm can use upward squaring, upward positive accumulation and an upward square-root enclosure. A lower point norm needs the corresponding lower operations. An exact dyadic/binned accumulator followed by a certified square-root bracket is an alternative. Calling an ordinary dot/norm and adding one `nextafter` at the end does not enclose all its preceding rounding errors. Ordinary IEEE arithmetic is not synonymous with exact accumulation; label an outward wrapper as a certified upper/lower bound, not an exact sum unless it is actually exact.

The endpoint contract must cover intermediate precision, compiler transformations, subnormal/flush behavior, overflow and nonfinite handling. Strict endpoint arithmetic and immutable exact input loading remain the proof obligations identified in the [earlier independent review](../certifier_20261002/INDEPENDENT_PROOF_REVIEW.md). This derivation reduces wrapping by exploiting exact stage norms; it does not waive those obligations, establish a global GPU contract or claim a runtime improvement before an authorized measured implementation result.
