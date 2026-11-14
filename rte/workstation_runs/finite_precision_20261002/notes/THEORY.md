# Conditional finite-precision model for the stored RTE and flexible Arnoldi

Prepared 2026-10-02 from existing cached project sources and primary references. **The executed complex-entry failure remains unchanged: eight low-H records failed the frozen 2e−6 cap and zero full solves ran.** This note derives an a priori model with explicit unresolved contracts; it does not fit the observed defect, change a threshold, instantiate a new GPU experiment, or promote a solver. No SSH, GPU/action call, training, new field/data, original-source export, Library or Git mutation occurred. Only this new directory was written.

The answer to the narrow algorithmic question is clear: **a relative homogeneity cap on the auxiliary FP32 low-H action is not necessary for flexible GMRES's algebraic validity, and passing it is not sufficient for physical accuracy.** The appropriate residual-gap quantities concern actual FP64 outer images, orthogonalization, candidate construction and final physical residual evaluation. This distinction does not waive the separately required old entry rule.

## 1. Fixed data, three operators and two precisions

Use the original stored directions, frequencies, phase matrix Φ, weights w and material fields. Define P=Φ diag(w), T=diag(σt), C=diag(σs)P (with its spatial/angular tensor extension), and U as the unitary spatial DFT. All norms are Euclidean/operator 2-norms on the complete complex field. The code uses an unnormalized forward FFT and a 1/(Nx Ny)-normalized inverse; U is an analysis device, not a precision or normalization change. This convention agrees with the [version-pinned PyTorch FFT source](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/fft/__init__.py).

For fixed stored real geometry, let

    s32(l,k) = the original FP32-assembled omega_x*kx + omega_y*ky,
    s64(l,k) = the physical copy's FP64 assembly from promoted stored geometry,
    L = U* diag(i s32) U + T - C,
    H = U* diag(i s64) U + T - C.

L and H interpret these fixed assembled symbols and stored coefficients exactly; the executed actions are denoted Lhat32 and Hhat64. This separates **stored-symbol differences** from subsequent FFT/product/action roundoff. `high_precision_copy` promotes buffers and recomputes the original stream products in64; it does not simply cast the completed FP32 streaming symbol. Physical collision data are promoted copies, so their exact nominal P,T,C coincide. Material means/reference inverse caches belong to the preconditioner and are not part of physical H.

Both nominal actions are complex-linear. Their streaming is skew-Hermitian because s32/s64 are real, including at Nyquist. Neither continuum accuracy nor preservation of real-valued fields follows. The previously observed imaginary images of real sources are compatible with complex linearity.

With B=max(l,k)(|omega_x*kx|+|omega_y*ky|) on stored data, unit roundoffs u32=2^−24 and u64=2^−53 for binary round-to-nearest, and ordinary correctly rounded real multiply/add arithmetic without exceptional range effects, the conditional assembly bound is

    Δs = max |s32-s64|
       <= [gamma_2(u32)+gamma_2(u64)] B,
    gamma_k(u) = k*u/(1-k*u),       k*u<1.

The two-product/add paths have at most two rounded operations on each summand; FMA can reduce that count. The condition must be checked against the actual kernel, precision and underflow/overflow behavior; it is not a cuFFT assertion. Since U is unitary, ||L-H||=Δs for this symbol-only difference. NVIDIA describes the differing multiply/add and FMA rounding paths in its [CUDA floating-point reference](https://docs.nvidia.com/cuda/archive/12.4.1/floating-point/index.html).

## 2. Explicit conditional forward bound for one low-H application

Fix β=max|s32|, a=||σt||∞, b=||σs||∞ || |Φ| diag(|w|) ||2 and ΛL=β+a+b. Then ||L||<=ΛL. Assume uniform error contracts on the actual relevant inputs:

- Forward FFT, expressed in U coordinates: error <=εF ||x||; normalized inverse: error <=εI ||v||. In the literal unnormalized coordinates the bounds are εF sqrt(Nx Ny)||x|| and εI||v||/sqrt(Nx Ny). Inverse scaling roundoff is included in εI.
- Symbol multiplication: error <=μD β||v||.
- Real coefficient/weight products: componentwise error <=μt |σt x|, μw |w x| and μs |σs v|.
- Angular contraction: componentwise error <=μP |Φ| |v|, including its actual multiply/reduction precision and order.
- Final stream-plus-total-minus-scatter additions: norm error <=μA times the sum of the three computed term norms.

These are assumptions to discharge, not measured fit parameters. Constants can depend on fixed shape/backend/precision, not be inferred from the stopped source's 5.036e−6 ratio. Appropriate additive underflow terms must be included if relative scalar contracts fail near subnormals. A γk value for the angular contraction requires a verified elementary-operation/accumulation contract; tensor dtype alone does not establish it.

Define

    θS = (1+εI)(1+μD)(1+εF)-1,
    θC = (1+μs)(1+μP)(1+μw)-1,
    EL = β θS + a μt + b θC
         + μA [β(1+θS)+a(1+μt)+b(1+θC)].

**Conditional forward-error theorem.** Under the preceding contracts and fixed data, for every input in their validity domain,

    ||Lhat32(x)-L x|| <= EL ||x||,
    ||Lhat32(x)-H x|| <= (EL+Δs)||x||.

Proof: propagate the forward FFT error through the diagonal symbol, then the inverse. The streaming contribution is β[(1+εI)(1+μD)(1+εF)-1]||x||. Propagate the weight, phase and σs componentwise envelopes using nonnegative absolute coefficient matrices, yielding b θC||x||. Add the total-coefficient and final-addition bounds. The second inequality adds ||L-H||. No coercivity, preconditioner linearity or FFT stochastic independence is needed.

Replacing32 contracts/s32 by64 contracts/s64 yields EH and ΛH for ||Hhat64(x)-Hx||<=EH||x||. **No numerical EL or EH has been certified here.** [cuFFT12.4.1 §2.13](https://docs.nvidia.com/cuda/archive/12.4.1/cufft/index.html#accuracy-and-performance) describes Cooley–Tukey error growth with log2 transform size and fixed-plan reproducibility. It provides no numeric, norm-specific worst-case constant sufficient to instantiate εF/εI for this plan. Classical FFT accuracy analyses likewise depend on algorithm and twiddle accuracy; see [Schatzman1996](https://epubs.siam.org/doi/10.1137/S1064827593247023). These sources support the model's form, not a universal cuFFT bound. [PyTorch2.5.1's numerical-accuracy note](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/docs/source/notes/numerical_accuracy.rst) also qualifies operation ordering, backends and reduced precision. Do not assume an unrecorded runtime flag or a default establishes an effective kernel contract.

## 3. What the old homogeneity diagnostic can and cannot bound

Let fl(cx)=cx+ηin with ||ηin||<=μc |c| ||x||, and fl(c Lhat32(x))=c Lhat32(x)+ηout with ||ηout||<=μc |c| ||Lhat32(x)||. Here μc includes native representation of the declared scalar as well as its complex multiply; it is not silently treated as one real multiplication. The same forward theorem gives

    ||Lhat32(fl(cx))-fl(c Lhat32(x))||
      <= |c| [μc ΛL+(2+μc)EL]||x|| + ||ηout||,

and hence, for the **old reference-output denominator**,

    δc(x) <= μc + [μc ΛL+(2+μc)EL] ||x||/||Lhat32(x)||.

For c=i on the finite probes, component swap/sign is exact, so μc=0 and δi(x)<=2EL||x||/||Lhat32(x)||. For general phase/scaling, both native input construction and predicted-output rounding contribute. A two-input combination has the analogous noncancelling weighted sum of input norms and image norms; cancellation of the combined image is not used as its denominator.

The relevant factor is source/action conditioning: ΛL||x||/||Lhat32(x)||. It can grow with spectral differentiation scale even when each FFT has small normwise error. A source-dependent defect or its growth with N is therefore consistent with forward roundoff, but the receipts do not determine EL or isolate its causal stage. There is no derivation of 2e−6 from these formulas and no adjustment to it.

For the original real σt and nonnegative σs material fields, if the stored collision has

    m = min_x [σt(x)-σs(x) λmax(Herm(P))] > 0,

then Re<v,Lv> and Re<v,Hv> are at least m||v||², so ||Lv||,||Hv||>=m||v||. If additionally EL<m, one can replace the observed-image denominator by (m-EL)||x||, giving δi<=2EL/(m-EL). This is a conditional global sufficient bound, often pessimistic; positivity and error bounds need certified quantities for a certificate. It does not transfer to a real-AZ projected image.

Necessity/sufficiency are separate: FGMRES can use a finite nonlinear auxiliary preconditioner, so auxiliary homogeneity is not necessary. A wrong fixed nonzero linear action, such as2H instead of H, has exact homogeneity yet a large forward error, so the diagnostic is not sufficient for physical fidelity either. It remains useful as the separately frozen engineering consistency screen, whose executed failure is preserved.

For the fixed TSA depth, nominal F=G+R0(I-LG), residual propagation S=I-LF, and **state-error propagation J=I-FL**. Its exact zero-start stationary action is Md=F sum(j=0..d-1)S^j=sum(j=0..d-1)J^j F. For actual stage iterates z(j+1)=z(j)+F(r-Lz(j))+τj,

    ||zhat(d)-Md r|| <= sum(j=0..d-1)||J||^(d-1-j)||τj||

(plus ||Md|| times input-cast error when r itself is cast, and ||J^d|| times any nonzero initial-state error). Noncommutativity prevents substituting ||S|| for ||J|| in this state-error bound. No contraction claim is made for either. Low-H error influences the directions produced and their effectiveness through τj; it is not an additional outer image error once the actual z is freshly imaged by Hhat64.

## 4. Flexible Arnoldi residual gap: actual directions are authoritative

This is a direct finite-dimensional derivation informed by the [flexible GMRES](https://epubs.siam.org/doi/10.1137/0914028) and [inexact Krylov](https://epubs.siam.org/doi/10.1137/S1064827502406415) distinction. Freeze a cycle and its actual finite returned directions Z=[z1,...,zk], which can arise from nonlinear, varying or rounded preconditioning/injection. Let

    W = [Hhat64(zj)] = H Z + E,
    W = V Gbar + F,
    b-Hx0 = β v1 + f0,
    xbar = x0+Zy+dx,
    g = βe1-Gbar y.

Gbar is the stored small Arnoldi matrix, F its complete orthogonalization/reconstruction defect, f0 includes initial-residual/normalization arithmetic, and dx is actual candidate-field construction error. They are definitions on stored finite vectors interpreted exactly. V must have the row dimension of Gbar; at breakdown, an unstored final frontier can be represented by a zero-padded column with its omitted image contribution absorbed in F. An orthogonality bound with ω<1 is not asserted for such a padded basis. Then

    b-Hxbar = Vg + f0 - F y + E y - H dx.

**Conditional residual-gap theorem.** If ||ej||<=EH||zj|| and bounds fj, f0, dx are available, then

    ||b-Hxbar|| <= ||V|| ||g|| + ||f0||
                   + sum_j |yj| [||fj||+EH||zj||]
                   + ΛH||dx||.                 (RG)

If ||V*V-I||<=ω, then ||V||<=sqrt(1+ω). Small-LS solution error affects the actual y and g; (RG) is valid for any stored y and does not assume an exact SVD minimizer. A computed tiny residual/norm needs its own error allowance. Coefficients can be large near rank loss, so neither25 columns nor100 total iterations implies a universal residual-gap constant. One-pass MGS is not automatically orthogonal to machine precision.

The coefficient-weighted expression (RG) is **a posteriori in the actual trajectory**, even when EH is independently prescribed. An a priori sufficient budget requires bounds fixed independently of those results, for example |yj|<=Yj, ||zj||<=Zj, ||fj||<=Fj, ||f0||<=F0, ||dx||<=Dx, ||V*V-I||<=ω and ||g||<=G0 for every admitted cycle. It then gives

    Rprior = sqrt(1+ω)G0 + F0 + sum_j Yj(Fj+EH Zj) + ΛH Dx.

A bound Rprior<=ρ||q|| reserves a specified nominal residual budgetρ. Coercivity of H bounds physical solution error but **does not bound the conditioning of the particular direction columns or their coefficients**. A certified retained singular-value bound plus a justified minimum-norm small-LS error bound could supply a coefficient bound; a computed rank cutoff by itself is not that certificate. No such ex ante trajectory constants are supplied by the stopped run, and none are fitted here.

There is **no standalone term measuring nonlinearity of M or of its auxiliary low-H**: those operations choose the actual columns Z. Changing them can hurt convergence, rank, finiteness and economics, but stored actual columns still give the stated identity when outer H is nominally complex-linear. For real stacking the same identity holds over the real field; this supplies no restarted complex-versus-real dominance theorem.

The original/companion code forms b=fl(q-Hhat64(initial)), then cycle residuals b-Hhat64(cycle_correction), and combines initial/correction at the end. Thus f0 must include both physical application/subtraction errors and any split-field construction discrepancy; do not silently identify this residual with a fresh Hhat64(initial+cycle_correction) image. The companion and original refresh candidate residuals during iteration and **freshly evaluate q-Hhat64(final_field)** at termination. That final check controls false convergence from the split/Arnoldi representation. It is more direct than stopping only on ||g||. Default code does not retain all raw W solely for this analysis, and no unbilled diagnostic capture is proposed.

## 5. Budget for fresh FP64 physical acceptance and field error

For the actual final stored xbar, let rhat=fl64(q-Hhat64(xbar)). If subtraction error is bounded by μr(||q||+||Hhat64(xbar)||), then

    ||q-Hxbar|| <= ||rhat|| + EH||xbar||
                   + μr [||q||+(ΛH+EH)||xbar||] = Rnom.       (FA)

This a posteriori candidate expression uses **a priori operator/error constants**; it is not a refit of the source homogeneity cap. A certified residual-norm upper bound and source-norm lower bound are needed to certify Rnom/||q||<=1e−5. For example, an independently valid relative norm contract ν<1 converts a recorded norm nhat into nhat/(1-ν) as an upper bound and nhat/(1+ν) as a lower bound. Torch reduction order and exceptional range behavior must be covered; ordinary evaluated norms are not interval enclosures.

The original measured criterion remains exactly its source-normalized FP64 residual<=1e−5, finite/success status, evaluated discrete relative-error bound<=1e−3 and paired final-field difference<=1e−4. This note changes none of them. A mathematical assertion about the **exact nominal** H at the same1e−5 boundary would additionally need (FA)'s error reserve; passing a rounded residual exactly at the boundary alone does not prove that assertion. No numeric reserve or new acceptance gate is prescribed here.

If a certified m>0 holds for H and Rnom is certified, then the unique nominal solution x* satisfies

    ||xbar-x*|| <= Rnom/m,
    ||xbar-x*||/||x*|| <= Rnom/[m||xbar||-Rnom]

when its denominator is positive. Bounds computed without rounding enclosures remain ordinary evaluations. Two small residuals alone need not meet the separate paired1e−4 field threshold: conditioning matters, so retain its actual field comparison. No continuum, OOD, positivity, complex-span performance or economic result follows.

## 6. What is established and what remains missing

The failed auxiliary2e−6 cap is a surrogate rather than a necessary FGMRES condition; the old stopped execution is still valid under its own stricter rule. Conditional low-H forward and Arnoldi/final-acceptance inequalities identify explicit quantities that would support a different justified analysis. They also show why source-image-normalized scalar identities cannot stand in for an outer residual-gap budget.

The concrete missing contracts are numeric normwise FFT/twiddle bounds for the actual fixed cuFFT plan/version, effective angular contraction and scalar arithmetic bounds, range/denormal handling, norm/reduction errors, and Arnoldi/field-combination bounds or accounted diagnostics. Current receipts supply no complete trajectories because zero full solves ran. Vendor logarithmic growth and passing finite probes do not discharge these assumptions. If they cannot be justified, the strongest honest conclusion remains a conditional theorem with a contract gap—not a certified new cap, a proven physics defect, or permission for more compute. See [SOURCE_ASSUMPTIONS.md](SOURCE_ASSUMPTIONS.md), [PRIMARY_BIBLIOGRAPHY.md](PRIMARY_BIBLIOGRAPHY.md) and the independent [ARNOLDI_REVIEW.md](ARNOLDI_REVIEW.md).
