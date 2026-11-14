# Source audit: low transport roundoff and the physical outer residual

2026-10-02. This audit reads the existing local original-project snapshot, the frozen complex companion and existing scalar receipts. It runs no model, experiment, solver or SSH command and changes no prior file. `SOURCE_CONTRACT.json` records source hashes, function spans, dtype/order contracts and unresolved runtime assumptions; every original-source hash matches the recorded FWI staging hash.

**Conclusion:** the source supplies no mathematical need to promote or repair the FP32 TSA preconditioner before complex flexible GMRES can use it. The outer algorithm images each actual returned direction with the original FP64 physical action and freshly verifies the final complete field. FP32 preconditioner errors can change convergence and cost, but are not silently substituted low-operator residuals. The prior eight failed low-H action/probe records (32 individual identity comparisons) remain a preserved obstruction under their old mandatory protocol. They are not a universal mathematical condition for a new prospective protocol. No GPU full complex solve has yet measured an outer residual gap.

## Stored target, assembly and application are different layers

The source/data family is FP32. The material transformation forms total and scattering coefficients in that data dtype before constructing the low operator. Geometry, quadrature and phase normalization are generated/stored in FP32. These rounded coefficients are the actual discrete target operands; promoting them does not recover an ideal continuum material or freshly generated double quadrature.

`high_precision_copy` promotes the low directions, weights, phase, wavenumber arrays, material arrays and source entrywise, then performs high setup in FP64. Its physical action uses those stored operands. However, the stream symbol is **assembled after promotion**: two direction–wavenumber products and one addition use the operator's `float_dtype`. Thus:

- low complex64: FP32 stream formation and complex64 application;
- low native128: the same FP32 stream formation, followed by complex128 application;
- physical high128: FP64 stream formation and complex128 application.

Passing a wide field to the low object does not make it the high object. The high target is not merely promotion of an already assembled low matrix. Physical scattering first weights the field, applies the phase contraction, then multiplies by the scattering coefficient. The complete action first adds streaming and total collision, then subtracts scattering. In contrast, the cached reference matrix first rounds phase-times-weight in FP32, uses FP32 means and complex64 matrix assembly/inversion. Its complex64 inverse is promoted for native128 application, without reinversion. That reference matrix/cache is a third object and must not be called the physical high matrix.

For exact algebra on fixed stored operands, let P=phase·diag(weights), T and S denote pointwise material multiplication, and F be the backward-normalized spatial DFT. Freeze the actually assembled real stream arrays s32 and s64. Define nominal maps

    H32 = F^-1 diag(i s32) F + T - S P,
    H64 = F^-1 diag(i s64) F + T - S P.

Here both physical maps use the same exact products of stored collision operands; application rounding is separate. Consequently

    ||H32-H64||2 = max |s32-s64|.

This equality does not compare either map with the preassembled reference matrix, whose collision products and means were separately rounded.

## Constants that can be justified, and constants that cannot

Use unit roundoff u32=2^-24 and u64=2^-53, distinct from the machine epsilon 2u used by the companion's SVD rank threshold. Let gamma_j(u)=j u/(1-j u), for j u<1. Under round-to-nearest, normal-range scalar arithmetic with no overflow/underflow or lower-precision multiplication, the two-products-plus-addition stream assembly obeys

    |s32-s64| <= [gamma_2(u32)+gamma_2(u64)]
                 (|omega_x k_x|+|omega_y k_y|).

This is an absolute stored-coefficient formation bound: cancellation cannot justify a relative bound against |s|. Entrywise promotion from32 to64 is exact. Casting a normal complex128 vector to64 has Euclidean error at most u32||v||, since its two real components are rounded separately; subnormal/flush behavior needs an additional allowance. A normal real coefficient–complex field multiplication has componentwise error at most u times the product magnitude. Combining three already computed action terms in the stated add/subtract order contributes at most gamma_2(u) times the sum of their magnitudes, componentwise. These conditional bounds do not cover the preceding FFT or angular contraction.

If an L-term real-coefficient angular dot is implemented by ordinary precision-p products and additions, its componentwise absolute error is bounded by gamma_L(u_p) times the sum of product magnitudes. Including the preceding weight multiplication and following scattering-coefficient multiplication gives the conditional gamma_(L+2) bound (L=12 gives gamma_14). The source does not identify a CUDA contraction/reduction arithmetic certificate, so this is an explicitly stated arithmetic model, not a verified bound for the deployed `einsum` kernel. It likewise supplies no unconditional reduction, inverse, complex division or SVD backward-error constant. No generic gamma_(log N) is assigned to cuFFT.

An exact streaming ledger can avoid inventing such a constant. For n=Nx·Ny, ||F||2=sqrt(n), ||F^-1||2=1/sqrt(n). Write the actual FFT as Fv+f(v), the actual symbol multiplication as i s(Fv+f)+mu(v), and the actual inverse FFT as the exact inverse plus iota(v). Then

    Dhat(v)-D(v) = F^-1[diag(i s) f(v)+mu(v)] + iota(v),
    ||Dhat(v)-D(v)|| <= max|s| ||f(v)||/sqrt(n)
                        + ||mu(v)||/sqrt(n) + ||iota(v)||.

The normalization constants are exact DFT facts; f, mu and iota remain input/backend-dependent unknowns until bounded or measured. Local collision/product/addition errors and the assembled-symbol difference are additional terms. Native scalar construction also matters: a measured consistency defect such as Hhat(fl(c v))-fl(c Hhat(v)) equals

    H delta_input + e_H(c v+delta_input) - c e_H(v) - delta_prediction.

It therefore does not isolate an operator-only error. The prior failed low-H cap combines these effects and is not a uniform action-error bound. Its source-dependent/grid-dependent population is consistent with spectral amplification, but those receipts do not locate the dominant primitive.

## The actual preconditioner and outer arithmetic

Deployment casts the normalized outer complex128 vector to64, performs the original TSA8 or TSA16 recurrence and returns an owned128 field. Each TSA primitive applies G, one low A and R0. Later stationary stages apply another low A before the primitive; zero addition, relaxation one and all original rounding remain. A depth-d application contains d G, d R0, 2d-1 low-A calls and 4d-1 spatial FFT/ifft pairs: TSA8 has31 pairs, TSA16 has63. Native128 diagnostic fallback is not deployment and can repeatedly promote a large reference cache; it is not a cost-free precision remedy.

Every outer returned direction z_j, including a nonlinear or rounded preconditioner's actual return, is independently imaged by physical high128. No low H(z_j) is used as the outer column. The real algorithm concatenates real/imaginary float64 channels; this conversion is an exact Euclidean isometry and keeps imaginary fields. Its MGS products and LS coefficients are real. The companion uses complex128 conjugate MGS, complex coefficients and a tiny SVD pseudoinverse with rcond=max(shape)·eps. Its field products/reductions and norm kernels differ from the real path. Norm results are float64, but neither norm path is compensated or interval certified. Rank truncation is part of the declared algorithm, not a proof of a kernel error bound.

For a cycle, let H be the nominal fixed high target and Hhat(v)=Hv+e_A(v). Retain actual returned directions Z and actual images What=HZ+E. Define the realized MGS defect by

    What = Vhat Hsmall + E_K.

Let the realized candidate be c_hat=c_start+Z yhat+delta_c. With r_start=q-H(x0+c_start), d_start=r_start-beta Vhat_1 and r_true=q-H(x0+c_hat) before the final initial-plus-correction addition,

    r_true = Vhat(beta e1-Hsmall yhat) + d_start
             - E_K yhat + E yhat - H delta_c.

Thus the residual-gap allowance contains start/normalization error, MGS error, weighted physical-image error and update error. In particular, the image term is at most sum_j |yhat_j| ||e_A(z_j)||. **There is no extra standalone error of the FP32 preconditioner:** it determines Z, whose actual fields are then imaged. If low images had instead been substituted, an additional (H-high minus H-low)Z y term would appear; the source does not do that. LS numerical error changes the realized y and residual fit. Using that same y in this ledger avoids double-counting it as another gap. Large cancelling coefficients can nevertheless amplify image and combination errors.

Vhat need not be exactly orthonormal after one-pass MGS. If ||Vhat*Vhat-I||<1, the norm of Vhat times the small residual lies between sqrt(1-||Vhat*Vhat-I||) and sqrt(1+||Vhat*Vhat-I||) times its norm. A Hessenberg residual alone is therefore insufficient. Both augmented algorithms instead evaluate every candidate correction with the high action, and restart from a freshly evaluated correction residual.

## The remaining initial-field/final-field residual gap

Their iterative test uses the rounded initial defect b_hat and the physical action on the correction c_hat. It does not apply H to the complete initial-plus-correction field at every candidate. Let x0 be the stored initial field, and define subtraction errors d_b,d_r,d_f and final addition error delta_x by

    b_hat = q-H x0-e_A(x0)+d_b,
    r_corr_hat = b_hat-H c_hat-e_A(c_hat)+d_r,
    x_hat = x0+c_hat+delta_x,
    r_full_hat = q-H x_hat-e_A(x_hat)+d_f.

Exact bookkeeping gives

    r_full_hat-r_corr_hat = e_A(x0)+e_A(c_hat)-e_A(x_hat)
                            -H delta_x + d_f-d_b-d_r.

This is FP64 high-action/addition/subtraction bookkeeping, not a low-TSA consistency term. With zero start it simplifies, but physical evaluation rounding remains. Both solvers freshly test q-Hhat(x_hat) at return, and the driver independently repeats complete-field residual/error/finite acceptance. A correction residual crossing tolerance can therefore stop and subsequently return failure if the complete-field check fails; it cannot produce a reported successful solve from the small residual alone. No current full complex-solve receipt measures this gap. Exact nominal solution error <=||q-H x_hat||/m requires an exact positive coercivity lower bound m. The ordinary evaluated residual/eigenvalue-based bound previously reported lacks an application-roundoff allowance and directed margins; it was correctly labelled an evaluated discrete bound, not a rigorous certificate.

## Minimum prospective correction and a meaningful primitive discriminator

No kernel promotion, inverse rebuild, reassociation, extra reorthogonalization or solver change is presently justified by the source audit. The minimum action is to record the distinction between coefficient formation, low application, physical application and residual bookkeeping, then measure the primitives without changing the original target. If a later solve exhibits a material correction-versus-complete residual gap, the smallest reliability intervention would be a fresh complete-field residual replacement/restart under the same total budget. It must be charged and tested; it is not implemented here. Promoting low kernels may improve direction quality or a future engineering cap, but is not required merely because flexible GMRES uses complex coefficients.

The root's prospective no-solve primitive plan has consistent counts: four contexts × six full-grid probes × three original GPU actions gives72 H calls; four separate4×4 fixtures × three dyadic probes × two GPU actions adds24, for96 total. Two existing strong-factor calls per context gives8 M calls; two independent NumPy FFT128 references per full-grid probe gives48; there are12 exact fixture cases. Low64, low native128 and physical high128 must retain their actual symbol/cast paths. Compare references with the **captured assembled low and high symbols** to separate coefficient formation from application error. Separate local products/sums where possible, retain ordinary reference uncertainty and compare using fixed scales. A true angle-zero streaming Nyquist probe varies along the first spatial axis; the old final-axis stripe was grazing.

For a prospective, tolerance-derived physical-action allowance, let L bound ||H|| and m>0 be a conservative exact coercivity lower bound. One valid choice is

    ||H||2 <= L := max|s64|+max(t)+max(s)||P||upper,
    Re<x,Hx> >= m||x||2^2,
    m := min_x[t(x)-s(x)||P||upper] > 0,
    ||P||upper = sqrt(max_row_abs_sum(P) max_column_abs_sum(P)).

Use exact/outward arithmetic to make these upper/lower bounds; an ordinary floating square root or fixture margin is not a full-grid certificate. The prospective model keeps the exact source-relative target **tau=1e-5 unchanged**. Ordinary solver stopping and independently evaluated acceptance remain at tau. A hypothetical measured residual ceiling rho_hat=(1-f)tau, f=0.01, allocates slack to a conditional certificate; it does not change solver stopping or revise the prior failed protocol.

Assume a uniform physical-action error ||e_H(x)||<=epsilon_H||x|| and a residual-subtraction error at most mu_sub(||q||+||Hhat(x)||). Assume relative norm contracts |n_hat-n|<=nu_r n for the measured residual and <=nu_q n for the measured source, with nu_r<1. Then

    rho_upper = rho_hat (1+nu_q)/(1-nu_r),
    a = mu_sub,
    b = epsilon_H + mu_sub (L+epsilon_H).

Writing R=||q-Hx||/||q||, coercivity gives ||x||/||q||<=(1+R)/m. Thus R<=rho_upper+a+b(1+R)/m, and the sufficient condition for the unchanged exact target R<=tau is

    b <= m (tau-rho_upper-a)/(1+tau),
    epsilon_H_allow = [m (tau-rho_upper-a)/(1+tau)-mu_sub L]
                      /(1+mu_sub).

A positive allowance is required. The fixed one-percent reserve pays **all** action, subtraction and norm uncertainty. There is no additional allowance outside that reserve. The previously recorded (tau+delta)/(1-delta) expression describes an enlarged-target sensitivity when delta=epsilon_H/m; it is historical sensitivity only and is not the prospective gate.

The current `error_model.py:prospective_budget` uses exact fractions for this slack calculation and outward conversion, assumes mu_sub=u64 and nu_r=nu_q=gamma_(4D+16)(u64), where D is the complex field count, and searches for a sufficient equal forward/inverse FFT relative contract in its declared action model. That effective norm/reduction contract and the FFT contracts are assumptions, not established CUDA backend theorems. The requested FFT contract is derived before observing defects, not fitted to them. A primitive reference comparison is evidence on those probes; its own reference uncertainty does not become a certified global application-error bound.

Finite probes and comparisons between two floating FFT libraries provide engineering evidence for those probes, not the hypothesized uniform bound. The new4×4 rational DFT oracle, using roots±1,±i and exact stored-value fractions, can rigorously check that labelled toy's results and norm brackets. It is neither the full-N target nor a global cuFFT guarantee. The new prospective protocol must be frozen before its separate primitive execution; the prior failed protocol/results remain unchanged.
