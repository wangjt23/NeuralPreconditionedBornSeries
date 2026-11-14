# Complex flexible Arnoldi: finite-precision residual-gap review

Date: 2026-10-02. Independent mathematical review by the coarse-sanity agent.

**Conclusion.** Complex flexible GMRES can use the actual directions returned by a rounded, variable, nonlinear or real-linear preconditioner. Its residual identity requires a fixed complex-linear physical operator acting on those same directions. Preconditioner error changes the approximation space; outer physical-image error, Arnoldi representation error and field-assembly error create the residual gap. A bound containing the observed least-squares coefficients is a posteriori. A genuinely prospective bound additionally needs a prescribed coefficient bound or a justified lower singular-value bound. The failed FP32 low-operator homogeneity cap is not a necessary hypothesis of this identity, and this theory does not waive the previously frozen execution gate.

This note proves an elementary conditional bound for the original discrete problem. It does not claim a new convergence theorem, a numerical error certificate for CUDA FFTs, continuous radiative-transfer accuracy, or an observed complex-solver improvement. No solve, training, new original data, source export or remote operation was performed for this review.

## Scope and predecessors

Let \(H\in\mathbb C^{n\times n}\) be the **nominal original physical discrete operator**, with the project's stored directions, wave-number tensors, phase matrix and material coefficients frozen. To agree with the accompanying THEORY.md, freeze the real streaming symbol assembled in FP64 from the promoted stored geometry and then interpret those fixed coefficients exactly. This is the nominal operator represented by the accepted high-precision physical action. It is not the separately assembled rounded FP32 symbol and is not an idealized operator obtained by rebuilding geometry. Alternatively, one may define a nominal operator using exact bilinear products of the promoted geometry, but then the FP64 symbol-assembly discrepancy must be included in the outer-image error; the two conventions must not be mixed. Let \(b\ne0\) be the original stored source. Use the complex Euclidean norm and Hermitian inner product throughout.

Saad's flexible GMRES allows a changing preconditioner, including iterative inner methods. This is the appropriate algorithmic predecessor; flexibility does not imply convergence for arbitrary returned directions. [Saad, *A Flexible Inner-Outer Preconditioned GMRES Algorithm*, SISC 14(2), 461–469 (1993)](https://epubs.siam.org/doi/10.1137/0914028).

Simoncini–Szyld give the coefficient-weighted inexact-image residual gap in Proposition 4.1 and Corollary 4.2. Their §9.1 distinguishes forming the solution with the actual preconditioned basis from applying an ideal inverse after the iteration. Their analysis principally assumes exact arithmetic; the explicit rounded Arnoldi, initialization and field-assembly terms below are an independently checked extension of that algebra. [Simoncini–Szyld, *Theory of Inexact Krylov Subspace Methods and Applications to Scientific Computing*, SISC 25(2), 454–477 (2003)](https://epubs.siam.org/doi/10.1137/S1064827502406415). The full article text was checked in the [author-uploaded article copy](https://www.researchgate.net/profile/Valeria-Simoncini/publication/2853528_Theory_of_Inexact_Krylov_Subspace_Methods_and_Applications_to_Scientific_Computing/links/09e41509364bee4245000000/Theory-of-Inexact-Krylov-Subspace-Methods-and-Applications-to-Scientific-Computing.pdf).

Bouras–Frayssé investigate monitored inexact outer products and a relaxation strategy. That work motivates allocating outer-image accuracy against a residual-gap budget; it supplies no universal justification for this project's fixed low-factor homogeneity threshold. [Bouras–Frayssé, *Inexact Matrix-Vector Products in Krylov Methods for Solving Linear Systems: A Relaxation Strategy*, SIMAX 26(3), 660–678 (2005)](https://epubs.siam.org/doi/10.1137/S0895479801384743).

## Exact identity on the actual stored objects

Consider one cycle and its actual stored starting field \(x_0\). Suppose \(k\) directions have been accepted. Define

\[
\widehat Z=[\widehat z_1,\ldots,\widehat z_k],\qquad
\widehat W=[\widehat w_1,\ldots,\widehat w_k],\qquad
\widehat V=[\widehat v_1,\ldots,\widehat v_{k+1}].
\]

Each \(\widehat z_j\) is the actual returned, cast, normalized if applicable, owned direction. It may come from the classical preconditioner or an accepted injection. No ideal direction \(M v_j\) is substituted. The direction must remain unchanged between its physical image and the field reconstruction; an overwritten or aliased direction introduces an additional error term.

Represent the rounded outer application and rounded Arnoldi factorization by their exact defects:

\[
\widehat W=H\widehat Z+E,
\quad E=[e_1,\ldots,e_k],
\qquad
\widehat W=\widehat V\widehat{\bar H}_k+F,
\quad F=[f_1,\ldots,f_k]. \tag{1}
\]

Thus \(H\widehat Z=\widehat V\widehat{\bar H}_k+F-E\). Here \(E\) contains the error of the **outer original physical action on actual directions**. The matrix \(F\) contains the residual of the stored Arnoldi representation, including rounded inner products, subtraction and normalization. Orthogonality loss is a separate issue; (1) remains an identity even if \(\widehat V\) is not orthogonal. If breakdown prevents storing a final basis vector, one can pad the missing column with zero and let \(F\) absorb the omitted representation. The norm-only bound remains applicable, but that padded basis cannot satisfy the \(\eta<1\) orthogonality condition below.

Let \(\widehat\beta\), \(\widehat v_1\), and the coefficient vector \(\widehat y\) be the actual stored values. Define

\[
d_0=(b-Hx_0)-\widehat\beta\widehat v_1,
\qquad
\widehat x_k=x_0+\widehat Z\widehat y+d_x,
\qquad
q=\widehat\beta e_1-\widehat{\bar H}_k\widehat y. \tag{2}
\]

All expressions in (1)–(2), including \(q\), are interpreted in exact arithmetic on the stored numbers. The initialization defect \(d_0\) includes initial residual evaluation and normalization. The field defect \(d_x\) includes multiplication, summation, and addition when building the actual stored candidate. A rounded small least-squares calculation is allowed: its output is simply \(\widehat y\). Its error affects minimization and coefficient size; it is not an additional residual-gap term once this actual vector is used consistently.

**Identity.** The true nominal physical residual satisfies

\[
\boxed{
b-H\widehat x_k
=\widehat Vq+d_0+(E-F)\widehat y-Hd_x.
} \tag{3}
\]

**Proof.** Substitute the field expression in (2), use the initial-residual expression in (2), and replace \(H\widehat Z\) by (1):

\[
b-H\widehat x_k
=\widehat\beta\widehat v_1+d_0
 -(\widehat V\widehat{\bar H}_k+F-E)\widehat y-Hd_x.
\]

Collecting the \(\widehat V\) terms gives (3). Complex coefficients are valid because \(H\) is complex-linear. This proof uses no linearity, stationarity or complex homogeneity of the preconditioner. It also uses no assertion that the returned directions span a Krylov space of one fixed matrix.

Consequently,

\[
\|b-H\widehat x_k-\widehat Vq\|
\le \|d_0\|
 +\sum_{j=1}^k |\widehat y_j|(\|e_j\|+\|f_j\|)
 +\|H\|\|d_x\|. \tag{4}
\]

If \(\widehat q\) is a rounded evaluation of \(q\), comparing against \(\widehat V\widehat q\) adds \(\|\widehat V\|\|q-\widehat q\|\). A scalar computed norm of \(\widehat q\) also needs its norm-evaluation error budget in a formal certification.

If \(\eta=\|\widehat V^*\widehat V-I\|_2<1\), then

\[
\sqrt{1-\eta}\,\|q\|\le\|\widehat Vq\|
\le\sqrt{1+\eta}\,\|q\|. \tag{5}
\]

Combining (4) and (5) gives upper and lower true-residual bounds, with the lower bound clipped at zero. Without this orthogonality estimate, the upper bound \(\|\widehat V\|\|q\|+\text{gap}\) still holds. A small coordinate residual alone is insufficient when the basis norm or the gap is uncontrolled.

## Conditional a priori theorem

Suppose, before relying on a cycle's stopping result, valid bounds are supplied for every admissible execution under consideration:

\[
\begin{aligned}
\|d_0\|&\le\epsilon_0,&
\|e_j\|&\le\epsilon_{H,j},&
\|f_j\|&\le\epsilon_{A,j},\\
\|d_x\|&\le\epsilon_x,&
\|H\|&\le L_H,&
\|\widehat y\|_2&\le Y,&
\|\widehat V\|_2&\le\nu.
\end{aligned} \tag{6}
\]

Let \(\epsilon_D=[\sum_j(\epsilon_{H,j}+\epsilon_{A,j})^2]^{1/2}\). Then

\[
\boxed{
\|b-H\widehat x_k-\widehat Vq\|
\le B_k:=\epsilon_0+Y\epsilon_D+L_H\epsilon_x,
\qquad
\|b-H\widehat x_k\|\le\nu\|q\|+B_k.
} \tag{7}
\]

This follows from (4) by Cauchy–Schwarz. Prescribed individual bounds \(Y_j\ge|\widehat y_j|\) give the potentially sharper \(B_k=\epsilon_0+\sum_jY_j(\epsilon_{H,j}+\epsilon_{A,j})+L_H\epsilon_x\).

Equation (7) is a **conditional a priori theorem** only when the bounds in (6) are established independently of the eventual observed coefficients and errors. Plugging the observed coefficients into (4) is a useful a posteriori bound, not an a priori guarantee. No valid numerical value of \(B_k\) is currently established for the project.

One sufficient coefficient condition is a full-column-rank small matrix with a justified \(\sigma_{\min}(\widehat{\bar H}_k)\ge\sigma_*>0\). Exact small least squares then gives \(Y=\widehat\beta/\sigma_*\). If the computed output solves least squares exactly for \(\widehat{\bar H}_k+\delta H\) and \(\widehat\beta e_1+\delta b\), with \(\|\delta H\|<\sigma_*\), the corresponding bound is

\[
Y\le\frac{\widehat\beta+\|\delta b\|}{\sigma_*-\|\delta H\|}. \tag{8}
\]

For a truncated SVD, the analogous condition concerns the smallest **retained** singular value, with a justified decomposition/solve error allowance. A dimension-times-epsilon relative cutoff alone does not supply a useful uniform prospective bound without scale and error bounds. The cached companion uses exactly such a cutoff, so it cannot be described as already enforcing a chosen absolute coefficient budget.

Physical coercivity by itself does not close this gap. If \(\operatorname{Re}\langle v,Hv\rangle\ge m\|v\|^2\), \(\sigma_{\min}(\widehat Z)\ge\zeta>0\), and \(\|F-E\|\le\delta\), then (1) yields

\[
\sigma_{\min}(\widehat{\bar H}_k)
\ge\frac{m\zeta-\delta}{\nu},\qquad m\zeta>\delta. \tag{9}
\]

Indeed, for unit \(c\), \(\|\widehat V\widehat{\bar H}_kc\|\ge\|H\widehat Zc\|-\delta\ge m\zeta-\delta\), while \(\|\widehat V\widehat{\bar H}_kc\|\le\nu\|\widehat{\bar H}_kc\|\). The additional transversality bound on \(\widehat Z\) is essential: an arbitrary flexible preconditioner can return dependent directions even when \(H\) is coercive. It can also return zero and make no progress. Equations (3)–(7) promise neither convergence nor lower runtime.

## What must be budgeted in this transport problem

For the nominal periodic Fourier transport discretization,

\[
H=S+D_t-D_sP,
\qquad
S=\mathcal F^{-1}\operatorname{diag}(i\,s_{64}(\ell,k))\mathcal F,
\]

with the actual stored factors, an available nominal norm upper bound is

\[
L_H=\max_{\ell,k}|s_{64}(\ell,k)|
 +\|\sigma_t\|_\infty+\|\sigma_s\|_\infty\|P\|_2. \tag{10}
\]

Here \(s_{64}\) is the original FP64 assembly from the promoted stored factors. The Fourier similarity is unitary in the Euclidean norm after the usual matching forward/inverse normalization. Formula (10) does not replace the accepted high-precision action with the low assembled FP32 symbol. A formal outward bound also requires safe evaluation of its factors; an ordinary FP64 value is not a directed-rounding certificate.

Under an appropriate scalar floating-point model and the actual accumulation order, a field-assembly budget can have the form

\[
\epsilon_x\le\gamma_{c(k)}
\left(\|x_0\|+\sum_j|\widehat y_j|\|\widehat z_j\|\right),
\qquad \gamma_p=pu/(1-pu). \tag{11}
\]

The operation-count factor \(c(k)\) must cover complex multiplication and the actual reductions; no specific CUDA reduction constant is asserted here. A prospective version of (11) needs prescribed direction and coefficient norm bounds. Likewise, \(\epsilon_{H,j}\) needs an absolute or action-wise forward-error bound for the outer FP64 Fourier/collision computation on \(\widehat z_j\), not a ratio from a low-factor homogeneity probe. This review has not established such a vendor FFT error bound.

For a restarted method, apply the identity to each actual cycle start. A fresh residual at the stored start resets the initial-defect accounting; accumulated update error already resides in the stored field. Preserve normalization by the original \(\|b\|\). If a cycle uses separately evaluated initial and correction fields, budget that split residual rather than treating it as the fresh residual of the rounded assembled field.

## Cached implementation and original physical acceptance

The source inspected was the already cached [complex companion](../../receipts/complex_execution/code/complex_augmented_fgmres.py). It stores actual preconditioned directions, applies the supplied physical action to each, uses Hermitian modified Gram–Schmidt, solves a small complex SVD least-squares problem, and assembles candidates from those directions. Accepted injection columns are covered by the same proof. Rejected proposals do not enter \(\widehat Z\).

There is a finite-precision qualification: it forms `b = rhs - action(initial)`, then cycle and candidate residuals as `b - action(correction)`. The corresponding cycle start in the original physical problem is the rounded assembled `initial + cycle_start`. Although splitting is valid for the nominal linear \(H\), the two rounded evaluations and rounded addition need not equal a fresh physical evaluation of that assembled field. This discrepancy belongs in \(d_0\) and \(d_x\). The final result is assembled and checked freshly as `rhs - action(result)`; that check is the authority for the returned operational acceptance.

The existing original-field residual gate is an ordinary FP64 evaluated relative residual of at most \(10^{-5}\), using the original source and original high physical action. The separately prescribed evaluated discrete relative-error bound of at most \(10^{-3}\), paired final-field difference of at most \(10^{-4}\), and finite/success checks also remain unchanged. The theorem does not authorize replacing the physical residual with a Hessenberg norm or a residual against a low operator. Independently evaluating the final field also protects against an inaccurate Arnoldi representation, but it is still a rounded evaluation.

For a formal guarantee for the nominal discrete problem, if an independent evaluation satisfies

\[
\widehat r_{\mathrm{eval}}=b-H\widehat x+e_{\mathrm{eval}},
\qquad \|e_{\mathrm{eval}}\|\le\epsilon_{\mathrm{eval}},
\]

then the sufficient condition is

\[
\|\widehat r_{\mathrm{eval}}\|+\epsilon_{\mathrm{eval}}
\le10^{-5}\|b\|. \tag{12}
\]

Norm evaluation uncertainty must also be included, or one may use a verified upper residual norm and lower source norm. This is a mathematical sufficient condition, not a proposed change to the authorized operational gate. Finite outputs, consistent immutable coefficients/geometry/source, owned directions used for both images and reconstruction, and explicit handling of rank loss and unsuccessful breakdown are also needed. A positive coercivity bound \(m\) would imply \(\|\widehat x-x_*\|\le(\|\widehat r_{\mathrm{eval}}\|+\epsilon_{\mathrm{eval}})/m\); evaluated coercivity and residual values alone remain ordinary discrete diagnostics unless their rounding uncertainty is enclosed.

## Meaning of the failed FP32 homogeneity cap

The [existing result analysis](../complex_krylov/RESULT_ANALYSIS.json) reports an intentional entry obstruction: 108 admissibility records, eight failed `low_H64` records with maximum normalized defect approximately \(5.0361\times10^{-6}\) against the frozen \(2\times10^{-6}\) cap, and **zero full solves**. The original physical FP64 action and the complete deployed mixed-precision preconditioner passed their separately prescribed finite-probe caps. These are observed probe results, not uniform mathematical error bounds.

The low action belongs inside the preconditioner. Its rounded homogeneity defect can alter directions, their conditioning and costs, and indirectly amplify the outer-image or field-assembly budgets. It is not itself the \(e_j\) of (1), because (1) compares the outer original physical action against \(H\widehat z_j\), with the actual low-precision direction already fixed. Introducing a separate discrepancy against an ideal preconditioner would double-count or analyze a different field unless the reconstructed solution used that ideal basis instead.

Therefore the \(2\times10^{-6}\) low-factor homogeneity cap is an engineering consistency surrogate and an explicit execution-policy requirement, **not an algorithmically necessary assumption for complex flexible Arnoldi's residual identity**. The probes do not bound every future returned direction, all outer FP64 image errors, or the coefficients in (7). Passing them would not prove convergence or physical acceptance; failing them does not establish a complex-specific impossibility. The recorded failures include real scaling as well as complex phase tests, and repeated source-based witnesses are not independent media evidence.

The prior discriminator correctly stopped under its frozen policy. A theory-only derivation supplies no permission to relax that cap, reinterpret the stopped run as a solve experiment, or launch another run. The remaining mathematical gap is to supply valid prospective budgets for outer action, Arnoldi, coefficient size and final-field construction, or to rely on the established ordinary final physical check with its stated precision scope. No numerical gap certificate or cost conclusion follows from the available stopped-run receipts.
