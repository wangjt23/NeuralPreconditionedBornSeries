# RTE finite-precision error model and bounded references

## Workspace, preservation and scope

The authorized SSH alias FWI reaches host FWI-Standard-PC-i440FX-PIIX-1996 as wjt. Workspace `/data/wjt/NeuralPreconditionedBornSeries` is on main at commit `028caea3f35d100225f656e2b1e910c1366aaafe`. RTE means radiative transfer equation: the project implements a periodic two-dimensional discrete-ordinates steady-state radiative-transfer operator with FFT streaming and angular scattering. This is separate from the paused OL_preconditioner project.

Before the new work, all 458 prior preserved files were SHA256 checked, including the prior complex-Krylov policy, execution records and private Library receipt. All nine existing tracked dirty files matched the prior status. The A6000 had49140 MiB total,37 MiB used and0%utilization, with only the desktop integration process and no other research jobs. Existing SSH credentials and host-key checking were retained; configured forwarding was disabled. No training, seed9626 access, source/model/data export, Git write, package install, system change or other-job cancellation is part of this work.

The frozen complex-Krylov entry screen remains a failure:8 of108 records failed solely at original low-H64, whose fixed2e−6 homogeneity cap was exceeded by a maximum5.036133085028879e−6. It completed **zero full solves**. This report neither raises that cap nor retroactively waives its entry rule. A separately defined diagnostic addresses the mathematical relevance of that surrogate; its results are not a new solver-performance study.

## Main conclusion

The auxiliary low-H homogeneity screen is **neither necessary nor sufficient** for flexible-GMRES accuracy. Generic scalar illustrations use outer H=1 and preconditioner F(r)=r+(r−a(r)). Auxiliary a(r)=r³ fails complex homogeneity, yet F(1)=1 gives an exact flexible step. Auxiliary a(r)=2r has exact homogeneity and nonzero images, yet F≡0 cannot advance. These illustrate the auxiliary-screen logic; they are not the project's transport operator. Actual returned preconditioner outputs are retained as columns Z, and the original physical operator applies its FP64 action to those columns. Flexible Arnoldi permits varying, nonlinear and rounded preconditioners. Low-H rounding changes Z and can hurt convergence and cost, but does not independently add a homogeneity-defect term to the exact residual-gap identity.

No selective precision repair of the low-H action is justified by the present mathematics. Native-complex128 application of the low object still forms its streaming symbol in FP32 and is not the original physical H. Original physical H promotes the stored geometry/material buffers and assembles the symbol in FP64; it does not rebuild ideal continuum coefficients. These distinctions are explicit in the model and references.

**Conditional mathematical model: GO. Certified full-solver admission/performance: NO-GO.** A finite set of NumPy128 or exact4×4 comparisons cannot establish a uniform cuFFT, contraction or norm error bound. The original finite discrete operator is the target; exact nominal complex linearity does not assert that its rounded execution is exactly linear. A verified fresh final physical-action/error enclosure could certify an individual returned solution without a global Arnoldi-trajectory bound. Internal-coordinate admission or prospective convergence claims would additionally need trajectory bounds. No full solve is authorized or executed in this round.

## Exact nominal operator and conditional implementation bound

Let H be the exact-arithmetic operator using the actual stored physical streaming symbol D64, promoted material buffers and angular matrix P=phase·diag(weights). Let L use actual assembled D32. FFT forward/inverse normalization is paired so the nominal Fourier similarity is unitary. Streaming is skew-Hermitian, including Nyquist modes; real-field invariance is not assumed.

From stored binary coefficients, exact rational arithmetic gives

\[
\|P\|_2\le\sqrt{\|P\|_1\|P\|_\infty}=K,
\quad \|H\|_2\le\Lambda=\|D_{64}\|_2+t_{max}+s_{max}K,
\quad \Re\langle v,Hv\rangle\ge m\|v\|^2,
\]

where m=min(t−sK)>0 is conservative. Rational square roots are rounded upward and final scalar upper/lower bounds outward. The stored-geometry product/sum formation bound pays both32 and64 paths when comparing L with H. It is a conditional standard-arithmetic assembly bound, not a claim about regenerated geometry or ideal physics.

For an implemented action, write epsilon_F/epsilon_I for **assumed uniform** normwise forward/inverse FFT contracts. Under normal-range IEEE operations and explicitly stated contraction budgets, the source-level composition bounds symbol formation, FFT multiplication, collision, weighted angular contraction and the final two additions. The authored `error_model.py` computes an absolute error epsilon_H satisfying

\[
\|\widehat H v-Hv\|\le\epsilon_H\|v\|.
\]

The full conditional derivation is in THEORY.md. Angular count8L+4 is a stated conservative elementary-operation assumption, not a proof of the CUDA contraction's reduction route. CUDA12.4.1 cuFFT documentation describes qualitative accuracy and fixed-plan reproducibility; it does not provide the required numerical uniform normwise constant. FP32 field casting and fixed-symbol coefficient discrepancies must also be paid when their target differs. Observed error ratios cannot supply a missing global contract.

## Flexible-Arnoldi residual-gap identity

For the actual stored columns and computed images,

\[
W=HZ+E=V\bar G+F,\qquad b-Hx_0=\beta v_1+f_0,
\qquad \bar x=x_0+Zy+d_x,
\]

with g=beta·e1−Gbar·y, exact algebra gives

\[
b-H\bar x=Vg+f_0-Fy+Ey-Hd_x.
\]

Thus, with measured/verified orthogonality defect omega,

\[
\|b-H\bar x\|\le\sqrt{1+\omega}\|g\|+\|f_0\|
+\sum_j|y_j|(\|f_j\|+\epsilon_H\|z_j\|)+\Lambda\|d_x\|.
\]

There is no independent FP32-preconditioner homogeneity term. A priori trajectory admission needs ex ante bounds on the coefficient-weighted terms; coercivity alone does not bound y or Z conditioning. Restart, tiny least squares, rank loss and original real-stacked versus complex coefficient arithmetic do not alter this exact identity when actual columns/images are used, but require appropriate implementation error accounting. Original split correction residuals can differ from a fresh combined-field residual; the final fresh physical check is authoritative.

For TSA state error propagation, the correct noncommuting state map is J=I−F·L, not the residual map I−L·F. The separate theorem retains that distinction and does not turn finitely observed contractivity into a uniform bound.

## Prospective budget, independent of measured defects

The original residual target stays tau=1e−5. A fixed1% reserve, chosen before new action measurements, proposes a measured-ratio ceiling0.99tau for a future certified check. Norm, subtraction and physical-action uncertainty all fit inside that reserve; it is not a1% relaxation of the target.

Let rho_upper bound the exact norm ratio of the rounded fresh residual, a=mu_sub and b=epsilon_H+mu_sub(Λ+epsilon_H). Coercivity yields the sufficient unchanged-target budget

\[
b\le m\,\frac{\tau-\rho_{upper}-a}{1+\tau}.
\]

Relative norm contracts convert a measured ratio by rho_upper=rho_measured·(1+nu_q)/(1−nu_r). The implementation uses exact Fractions and outward final conversion, avoiding cancellation of1+u64 to1. Norm count4·field_complex_count+16 is another stated standard-reduction assumption. The equal forward/inverse FFT contract required by this sufficient model is solved from coefficient scales and this unchanged-target budget **before** comparisons; it is not fitted to5.036e−6 or any new discrepancy. A historical sensitivity formula allowing(1+allocation)tau is explicitly not the prospective admission rule.

For direct interval inputs Rhat_upper, X_upper, Q_lower and X_lower, a fresh physical check must instead use

\[
R_{upper}=\widehat R_{upper}+\epsilon_HX_{upper}+e_{sub,upper},
\quad R_{upper}/Q_{lower}\le10^{-5},
\quad \frac{R_{upper}}{mX_{lower}-R_{upper}}\le10^{-3},
\]

with a strictly positive solution-error denominator, finite original field/status and original paired field difference≤1e−4. Ordinary torch norms are measurements, not automatically verified enclosures. An empirical estimated allowance is explicitly not a certificate.

## Bounded check protocol and evidence

The new authored policy freezes four original development contexts: N128/256, IDs0/3, seed4626 and their original hashed parents. Six fields per context are source, complex mixed source, true x-Nyquist stripe in angle0, private unit complex random field, and actual unchanged direct-segmented strong-TSA outputs on source/random. Each receives original low32, low-native128 and original physical128 actions, plus independent NumPy128 fixed-symbol references. Separate4×4 fixtures use exact roots±1,±i, original angular buffers and sampled material coefficients, three fixed binary-dyadic complex fields, actual assembled symbols, low32/high128 GPU actions and exact Fraction reference evaluation. No fixture substitutes for the N128/256 target.

The maximum is96 GPU H-actions,8 M-actions,48 NumPy references,12 exact fixture cases,60s soft/75s hard owned-process limits,4GiB reserved-memory ceiling and **zero full solves**. Counts include explicit diagnostic actions; capture setup and the internal H applications of TSA are disclosed separately, so96 does not mean the total number of primitive FFTs or internal operator calls. Operator differences, nominal-versus-physical coefficient differences, input-cast discrepancies, setup/reference acquisition costs, backend settings and memory are recorded as scalar metadata. Reference arrays and original source remain on FWI.

The diagnostic completed successfully with every frozen count, **zero full solves**, all four ownership/source checks passing, and no timeout. All 29 meaningful CPU tests also passed on FWI in0.63854s:5 root model tests,12 independent rational-fixture tests and12 independent scalar audit tests. These validate the model arithmetic/domain/reserve and exact-reference implementation; they do not establish a CUDA contract or convergence.

Policy SHA256: `426fe29b012c7e275a1587e315f0f2f23efe2989f63cf00b13a2a02869f67d08`. Driver SHA256: `d7149407412392cbdbdf548af907ee4348d242e709c6f882b39036f2462027c7`. The final guard review was received before execution and its SHA is pinned in the execution receipt. An earlier implementation-note snapshot was already captured in the staging payload; it remains preserved alongside the final local review, rather than being silently overwritten.

The next table reports worst observed physical-H128 discrepancy against separately rounded NumPy128 per input norm, and the independent coefficient-derived conditional allowance. The last column is an engineering ratio, **not a certified gap**.

|Grid/ID|Coercivity lower m|Operator upper Λ|Allowed ε_H|Observed FP64 discrepancy/‖v‖|Observed/allowance|
|---|---:|---:|---:|---:|---:|
|128/0|0.01700810363|788.854596|1.700706e−9|1.210402e−13|7.1171e−5|
|128/3|0.04077052552|836.788978|4.076919e−9|1.206813e−13|2.9601e−5|
|256/0|0.01700810363|1338.166037|1.700645e−9|2.614385e−13|1.5373e−4|
|256/3|0.04077052552|1386.100419|4.076858e−9|2.612032e−13|6.4070e−5|

The sufficient equal forward/inverse FFT64 contract required by the model ranges7.73076e−13–3.70913e−12 across contexts. These are requirements obtained from stored scales and the fixed accuracy reserve; none is established from the observations. The assumed non-FFT64 terms are1.70100e−12–2.28500e−12. Their smallness also remains conditional on the contraction/scalar contracts.

Low32 matched nominal discrepancy per input norm is at most7.62169e−5 on N128 and1.43985e−4 on N256, or at most9.66172e−8 and1.07599e−7 after division by Λ. At N256/ID0 M_source, low32 has3.80239e−4 image-relative error but3.59356e−8 error/(Λ‖v‖), with an ordinary image-cancellation indicatorΛ‖v‖/‖Lv‖≈10581.1. These norms are NumPy measurements; that indicator is not a certified condition number. This different forward-error measurement neither recomputes nor replaces the old homogeneity failure.

Low-native128 agrees with its matched D32 NumPy target to1.20820e−13 on N128 and2.61364e−13 on N256, yet differs from the original D64 physical target by up to7.53404e−6 and1.50100e−5 per input norm. Actual symbol discrepancies are4.90929e−5 and1.05366e−4. Thus native low128 is a different fixed-symbol target; promoting its execution alone would not make it original physical H.

All six full-grid fields originate in C64 and the three fixture fields are binary dyadic. Every cast ledger is exactly zero by construction. Native low128 applies to the actual cast64 field promoted exactly to128. No nonrepresentable C128 casting accuracy is tested or claimed.

In the separate exact4×4 fixtures, the maximum rigorous finite-input forward-error upper bound per input norm is7.43666e−6 for low32 and1.41572e−14 for physical128. GPU output bits are converted directly to exact Fractions, subtracted from exact nominal references, and their rational norms enclosed outward. This statement certifies those particular4×4 differences only; it is not a uniform N128/256 or cuFFT guarantee.

Launcher-owned process time was52.07058s; the settings timestamp was50.66212s and precedes its final receipt flush. Peak allocated memory was361,017,856B and peak reserved731,906,048B (698MiB). Captured component costs include physical setup0.27839s, base setup0.08631s, capture setup0.09738s, explicit H/M action timing0.38322s, NumPy references3.05875s, exact references3.03284s, scalar full-grid modeling3.51212s and previously completed JSON writes28.02609s. These are research-process components with partial/nested scopes, not solver costs, and must not be summed into a solve ledger. The owned total includes final flushing and teardown; SSH startup, preservation scans and later downloads are separate bridge costs. Additional low-H work inside M/capture is included in those action/setup times, not in the96 explicit-H count.

Read-only settings were CUDA matmul TF32=false, cuDNN TF32=true, float32 matmul precision=highest, FP16/BF16 reduced-precision reduction flags=true. The latter flags do not establish an FP32/FP64 kernel contract or imply these paths were used. No precision flag was changed. CUDA12.4, Torch2.5.1 and NumPy1.26.4 were recorded; exact cuFFT plan/library error constants remain unverified.

Post-run preservation verification passed all 490 expectations:458 prior files plus32 new payloads. All nine original tracked edits remained unchanged, no research jobs remained, and the A6000 returned to37 MiB/0%utilization. The original failed entry remains unchanged. Independent raw-result audits and the stdlib reduction are included in the evidence bundle.

## Directions and next decision

1. **Certifiable fresh original-H acceptance** is the most decisive next direction. Establish uniform physical-action/reduction bounds or a verified interval backend on the actual discretization, then apply direct residual and solution-error enclosures. It closes the mathematical admission gap without changing preconditioner precision or fitting homogeneity caps. Its significance is reliable scientific accuracy, not a learned-preconditioner speed claim.
2. **Complex flexible Arnoldi with unchanged actual factors** remains a plausible future bounded numerical comparison. It needs separately reviewed admission, original physical acceptance and fully charged real/complex ledgers. Existing zero-solve screen supplies no convergence or cost evidence.
3. **Selective low-H precision correction** has lower priority unless an actual convergence/quality failure is traced to a specific component. Any symbol/FFT/scattering correction must be minimum necessary, preserve the original physical target, be frozen prospectively and be charged to both arms. No such repair is selected now.

Training and stopped gain/scalar variants stay stopped; seed9626 stays sealed. Finite diagnostics cannot overcome these current gaps: global backend contracts, full-solver trajectory/convergence evidence, out-of-distribution reliability and matched fully charged economics. Prior learned methods did not pass the unchanged10% fully charged improvement gate; the successful direct-capture improvement was classical. No new learned or physical-continuum claim follows from this report.

Primary references and explicit source assumptions are in PRIMARY_BIBLIOGRAPHY.md and SOURCE_ASSUMPTIONS.md. The key framework is Saad's flexible-GMRES formulation and Simoncini–Szyld's inexact-Krylov analysis, applied to actual source paths rather than a linearity test alone. Exact code, hashes, independent reviews, scalar raw records, preserved failure and reproducibility instructions accompany the evidence bundle. The private Library verification receipt pins the final report/bundle versions and checks downloaded bytes independently.


## Reproduction and continuation

The scalar analysis is self-contained. From an extracted evidence bundle, run `python3 deliverables/finite_precision_20261002/analyze_primitive.py receipts/finite_execution reproduced.json` and compare the output byte-for-byte with RESULT_ANALYSIS.json. Run the29 scalar/reference tests with `python3 -m unittest discover -s deliverables/finite_precision_20261002 -p 'test_*.py'`. These commands require no CUDA, original source or project data. The independently authored raw-result audit also reconstructs the scalar aggregates.

A future FWI primitive rerun requires the preserved original repository, original internal development parents, matching source hashes and a fresh empty owned run directory. The frozen driver fails if its result directory already exists; it must not overwrite this run. Original source/data/arrays are deliberately absent from the Library bundle, so the bundle does not claim to reproduce the entire GPU oracle without access to that workspace. Current diagnostics are complete; repeating or expanding them is not the next authorized action in this handoff.

The research decision remains conditional-model GO and certified solver NO-GO. Establish a justified full-grid physical-action/check bound before proposing any new accuracy admission. If fresh original-H acceptance can be enclosed rigorously, it may certify a returned solution directly; coordinate-based/convergence claims additionally need Arnoldi error and coefficient bounds. Any future actual precision repair or mandatory certification acquisition must be prospective and fully charged to both real and complex controls. Preserve the unchanged10% improvement gate, original residual/error/paired-field limits, old failed2e−6 screen, stopped scalar variants and sealed9626.

`REVIEW_PROVENANCE_SUPPLEMENT.json` identifies both implementation-note versions and the final pre-execution review pinned in execution_receipt.json. The final local review is saved under a new reports path; the older staged note is preserved. `library_artifact_receipt.json` records exact private Library version IDs and downloaded SHA256 checks after saving; its separate existence avoids a circular report/bundle hash.
