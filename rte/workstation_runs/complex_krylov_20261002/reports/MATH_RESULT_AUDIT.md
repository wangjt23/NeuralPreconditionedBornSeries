# Mathematical audit of the executed complex-Krylov entry screen

The frozen entry rule failed in all four grid/material contexts. **No full solve proceeded.** This is a measured consistency-cap obstruction of the original FP32 auxiliary action, not a complex-FGMRES convergence failure, solution-accuracy failure, economic result, or theorem that the RTE model is nonlinear.

This audit uses preserved scalar receipts and existing local source only. [MATH_WITNESSES.json](MATH_WITNESSES.json) hashes its inputs, reproduces every one of the 756 recorded identity ratios and cap decisions with zero discrepancy, and retains all failed witnesses. No operator calls, new numerical experiment, original-source export or threshold waiver were made. Policy SHA256 is `acd73134c63839726d3b699c60e795b8da90d14709a9529fb94e905ce59ebd93`.

## Exactly what failed

Nine actions × three frozen pairs × four contexts produced 108 records and 1,080 action evaluations. Eight records failed, all for `low_H64`, with two failures per context. All 112 action-parity/ownership/cache checks passed. Every action was finite, exact on zero, correctly typed/shaped, input preserving and free of recorded errors; every identity denominator was eligible. `solver.json` is empty and settings record `full_solves=0`.

For scalar homogeneity the frozen metric is

\[
\delta_a(x)=\frac{\|\widehat L(\operatorname{fl}(ax))-\operatorname{fl}(a\widehat L(x))\|_2}{|a|\|\widehat L(x)\|_2}.
\]

For the fixed combination \(a=.73+.29i,b=-.41+.37i\), its denominator is \(|a|\|\widehat L(x)\|_2+|b|\|\widehat L(y)\|_2\). Input construction and output predictions retain their actual native dtype; differences and norms promote to complex128. These are finite-probe ordinary floating-point measurements, not interval certificates or worst-case operator-norm bounds.

The sharpest low-H witness in each context is below; the cap is **2e−6** throughout. `phase` means \(e^{.37i}\), `real` means −1.7, and the mixed source is the frozen rolled/scaled source.

| N / ID | Identity and actual field | Numerator | Reference denominator | Defect | Cap multiple |
|---|---|---:|---:|---:|---:|
| 128 / 0 | phase, source | .002313248355 | 1051.028959 | 2.200936839e−6 | 1.100468 |
| 128 / 3 | phase, mixed source | .003710380947 | 1544.897077 | 2.401701059e−6 | 1.200851 |
| 256 / 0 | real, mixed source | .020097452002 | 4261.332259 | 4.716236796e−6 | 2.358118 |
| 256 / 3 | phase, mixed source | .015871038620 | 3151.433521 | 5.036133085e−6 | 2.518067 |

The 32 failed scalar rows comprise six i-scalings, twelve phase-scalings, twelve real scalings and two complex combinations. Ten rows repeat exactly the same source identity, leaving 22 distinct field/identity cases. `source_mixed.x` and `nyquist_source.y` are the same source. Thus eight failed records represent four grid/material contexts and two development media, not eight independent sources or population evidence.

All failures in a `nyquist_source` record concern its **source y**, not the grazing/Nyquist x. The latter's scalar-identity maxima are 4.92e−8–8.75e−8; all four random-complex low-H pairs also pass, maxima 2.40e−7–2.63e−7. The observed defect depends on these structured fields. This sampling cannot establish a uniform linearity norm or attribute the failure to high-frequency/Nyquist content.

| Action/path | Failed records / 12 | Maximum defect | Own frozen cap |
|---|---:|---:|---:|
| Physical H128 | 0 | 9.780659e−15 | 1e−12 |
| Original low H64 | 8 | 5.036133e−6 | 2e−6 |
| G64 / R064 | 0 / 0 | 2.622050e−7 / 7.625112e−7 | 5e−6 |
| Strong eager64 / deployed backcast128 | 0 / 0 | 1.526864e−5 / 1.546164e−5 | 1e−4 |
| Strong eager128 / segmented native128 | 0 / 0 | 2.865096e−15 / 2.865096e−15 | 1e−12 |
| Physical H after deployed factor | 0 | 5.897304e−4 | 1e−3 |

Actions, output normalizers and caps differ across these rows. Their maxima are not an interchangeable action-accuracy ranking. Passing the composed factor or physical image neither proves the auxiliary action satisfies its separate cap nor repairs it. Symmetric-output ratios are additionally recorded as diagnostics; using their roughly doubled homogeneity denominator to rescue failures would change the frozen criterion after observing results.

## Nominal linearity and literal rounded behavior

The original fixed-material operator is \(H=S+T-P_s\). Its stored streaming symbol is imaginary times a real direction/frequency product; streaming is a fixed Fourier multiplier. Collision uses fixed material fields, weights and phase matrix. Each is complex-linear in exact arithmetic on the stored data. The inspected operator source code contains no field conjugation, magnitude operation or real projection. Fixed G, cached R0 and the finite TSA stages likewise have nominal complex-linear algebra. A rounded inverse can be inaccurate and still define a nominal complex-linear matrix action.

Write the executed action as \(\widehat L(v)=Lv+e(v)\), where L is its nominal stored-data linear action. With \(\operatorname{fl}(ax+by)=ax+by+\eta\) and rounded prediction error \(\zeta\), the measured combination defect has the decomposition

\[
L\eta+e(ax+by+\eta)-a e(x)-b e(y)-\zeta.
\]

It can be nonzero without nonlinear physics or a structural anti-linear operation. The recorded low-H failures establish that this literal rounded path exceeds the prescribed cap at these probes; they do not separate the terms in that decomposition.

There is one sharper statement. For the finite source fields at N256, multiplication by i is a component swap/sign change; the predicted i·H64(source) has the same property. The failed i-identities, approximately 3.30e−6–3.53e−6, therefore cannot arise **solely** from scalar-input rounding. Arithmetic or operation-order effects inside H64 are implicated. The receipts contain no stagewise or spectral error control, so they do not determine whether FFTs, streaming assembly, scattering reduction or final cancellation cause the defect. Real −1.7 scaling also fails, which rules out describing the evidence as exclusively a complex-phase issue. Native128 passes are consistent with a precision explanation but do not prove its cause or a forward-error bound.

For complex flexible GMRES, nominal complex linearity of the outer H and use of each actual returned direction z and image Hz allow \(H\sum_j z_jc_j=\sum_j(Hz_j)c_j\) in exact arithmetic; a flexible preconditioner need not itself be complex-linear. Consequently failure of an auxiliary low-H consistency cap is not a mathematical proof that complex FGMRES is inadmissible. **The user's frozen admission rule nevertheless explicitly requires every tested cap to pass.** No cap substitution, alternate precision, cast change or solve is licensed by these receipts.

## Reality, span and physical scope

All sources are exactly real in complex storage. Physical H128(source) has a nonzero imaginary norm fraction, 2.9266e−6–1.5583e−3, despite passing its complex-linearity cap near double precision. Real-subspace invariance and complex linearity are different properties. An even-grid FFT derivative's unpaired Nyquist frequency can prevent a real-input image from being real while its imaginary Fourier multiplier remains complex-linear. The receipts do not isolate that contribution or validate a continuous real-intensity interpretation. The selected grazing stripe has zero imaginary physical image because its variation is transverse to its angle-zero streaming direction; it does not test every real Nyquist field. The original solver retains both field components and imposes no real-field projection.

For a fixed unprojected direction-image matrix W, complex coefficients enlarge the fit space from real span(W) to real span(W,iW). That conditional inclusion gives no restarted/flexible/rounded trajectory dominance and no measured gain here. Projection through an existing real Krylov projector generally does not commute with multiplication by i, so the earlier real-AZ single-image geometry cannot provide a complex-span result. No field combination, Krylov coefficients, solver trajectory or cost comparison was produced by this stopped run.

## Cost and stop interpretation

Unique mandatory entry acquisition totals **73.558860 s** across four contexts. It includes diagnostics/metadata and receipt work and would have been charged once per context to every arm. It is research acquisition, not solver-application latency or an unavoidable operational lower bound. Runner time84.434398 s, process time87.909460 s and reserved memory2,009,071,616 bytes remain within the frozen limits. The process exited normally; no training, held-out seed9626 or source export occurred.

No accepted solves, warm solve times, Q1 savings or complex-versus-real accuracy results exist. A statement that complex coefficients failed the economic goal would be unsupported. The precise concluded barrier is: **the original native FP32 low action failed the separately required 2e−6 consistency cap on the frozen source-based probes, and the unchanged rule therefore admitted zero full solves.** A different precision/error budget or admission design would require a separately authorized protocol before measurements; this audit proposes no waiver or new computation.
