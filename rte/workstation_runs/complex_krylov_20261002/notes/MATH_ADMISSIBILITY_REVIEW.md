# Complex-coefficient Krylov: mathematical and numerical admissibility

Prepared 2026-10-02 from the existing cached original project sources and preserved local notes. This reviews the newly authorized discriminator **only after its entry checks pass**. No SSH, GPU/scientific experiment, source export, Library/Git mutation, gain variant, training or seed9626 access occurred here. The new isolated [admissibility helper](admissibility.py) and [fifteen focused CPU tests](test_admissibility.py) were separately authorized; all fifteen passed with existing /usr/bin/python3/Torch, final suite0.035s. The previous [beta2 mechanism](../gain_two/MECHANISM_AND_STOP_RULE.md) and [conditional fallback review](../gain_two/FALLBACK_REVIEW.md) are preserved. Independent mathematical review checked the lemma and floating-point qualifications.

## Nominal admissibility lemma

Freeze the actual stored geometry, quadrature, phase, material fields, reference means, inverse cache, depth and relaxation. Let H be the nominal physical finite-dimensional operator, with arithmetic interpreted exactly on those stored values:

    H = S + T - V P,
    S = Fourier^-1 diag(i omega_l·k) Fourier,

where T/V multiply by sigma_t/sigma_s, and P is the original stored phase matrix with column quadrature weights. Spatial FFTs and every fixed multiplier/matrix are complex-linear. Therefore `H(a x+b y)=a Hx+b Hy` for all complex a,b. S is skew-Hermitian under Parseval because its actual frequency multiplier is imaginary with real omega·k; the stored Nyquist frequency does not invalidate that fact. Collision need not be Hermitian for complex linearity. No accretivity or uniform convergence assumption is needed for this lemma.

The original [models.py](../../remote_reference/rte/models.py) uses no conjugation, real-part extraction, absolute value or residual-dependent coefficient in `A_op`, `G` or `reference_transport_solve`. G divides the spatial transform by the fixed stream-plus-mean-total symbol. R0 applies the original cached full twelve-by-twelve inverse at every spatial frequency. A cache that approximates the exact inverse is still a fixed complex-linear nominal map; inverse accuracy and linearity are different properties.

For [ClassicalTransportStack](../../remote_reference/rte/transport_baselines.py), the TSA primitive is

    F = G+R0(I-H_low G).

Starting from zero, each of the fixed d stages adds `F(r-H_low z)` with fixed relaxation1; sums/compositions preserve complex linearity. [build_base](../../remote_reference/rte/train_source_predictor.py) constructs the strong TSA8/TSA16 controls with eta_scale0. Their preparation/guards depend on material/cache state, not residual values. The exact nominal factors are complex-linear. Capture/segmentation must separately remain byte-identical to their own original action; capture is not a different mathematical factor.

The accepted H_h is [high_precision_copy(low)](../../remote_reference/rte/mixed_precision_transport.py): it promotes the original stored buffers, then evaluates its original FP64 action. It is not ideal geometry, a native regenerated phase or the FP32 low operator. Complex linearity of each nominal operator does not imply H_low=H_h.

The original [balance_integrated_source](../../remote_reference/rte/global_source_predictor.py) is also linear in the source/value pair at fixed material: its correction numerator is `sum(source)-sum(row*value)` and its denominator `sum(row)` is material-only. Thus R0-based balanced initialization is nominally linear when that finite nonzero denominator is fixed. It is not a fitted source-dependent scale. The reference-tail injection is the original fixed R0 map on its supplied residual; its triggering, normalization and acceptance make the finite solver trajectory adaptive. Do not require the complete solver's input/output map to be additive.

## Floating implementation versus valid flexible GMRES

FP32 casts, FFTs, matrix products, defect subtractions and repeated additions are rounded. Consequently the actual deployed action

    M_hat(r128)=promote128(TSA_d(cast64(r128)))

is not exactly additive/complex-homogeneous as a map on complex128 fields. The literal FP64 H application also has ordinary rounding. The lemma is not a bitwise floating-point theorem.

Complex flexible GMRES stores the **actual** returned z_j and their physical images w_j=H_h z_j. For nominally complex-linear H_h,

    H_h(x0+sum_j z_j c_j)=H_h x0+sum_j w_j c_j.

This image relation does not require M to be complex-linear, fixed or even linear. Flexible/nonlinear preconditioning changes which columns are generated; it does not invalidate complex fitting of actual columns when H is complex-linear. Literal floating application/field-combination error is qualified by fresh physical residual evaluation. No cached inverse theorem or Hessenberg residual estimate substitutes for that check.

This permissive FGMRES statement **does not waive the user's stricter entry condition**. A genuine structural defect or a failed/inconclusive predeclared numerical admissibility check blocks the authorized full comparison. Do not use “FGMRES permits nonlinear M” to proceed after its gate fails. Conversely, a routine nonzero rounding defect within the frozen cap is not proof of structural model nonlinearity or of an RTE obstruction. Report the distinction explicitly.

## Real coefficients, complex freedom and physical reality

For the same retained solution columns Z and unprojected physical images W=H_h Z,

    min_(c in C^k) ||r0-Wc||
      =min_(a,b in R^k) ||r0-[W,iW](a,b)||
      <=min_(a in R^k) ||r0-Wa||.

Complex rank can differ from real rank, so use conjugate inner products and rank-aware complex QR/LS. A real-independent pair w,iw is complex-dependent. A fixed real-AZ projector J_R generally has `J_R(iw)!=i J_R(w)`; no real-projected one-complex-scalar score proves this full-span claim. Rebuild the entire complex image space, or project w and iw separately for a precisely limited old-real/new-complex test. No raw coercivity bound transfers to arbitrary real-AZ projection.

For one fixed complex-linear M, same start and unrestarted exact arithmetic, the real polynomial correction space is contained in the corresponding complex Krylov space at matching degree. This does not guarantee savings under restart25, injection, adaptive decisions or rounded actions. Nor does a frozen measured-image fit establish the trajectory or an independently verified combined field.

The [source generator](../../remote_reference/rte/run_large_angular_spectral_experiments.py) produces real localized fields in complex storage. If all operators preserved a common real subspace, real/complex Arnoldi would coincide in exact arithmetic. However, the original even-grid unpaired Nyquist symbol can turn a real spatial Nyquist field into a complex image. Failure of real-field preservation is compatible with complex linearity and skew streaming. Preserve this operator; do not zero Nyquist frequencies or regenerate geometry to obtain a preferred reality test. Actual source, preconditioned-field and H-image imaginary fractions are descriptive evidence, not an admissibility gate or proof of useful complex coefficient freedom.

Real-stacked classical FGMRES can already be globally phase-equivariant when its complete state/initialization/injection rotate together. Changing coefficients to complex adds minimization freedom; it does not repair a demonstrated phase defect. Phase rotations in entry tests are aliases of the same source, not independent media/RHS workloads.

## Frozen numerical entry design

The root supplied these caps **before GPU results**:

| Actual action/dtype | Maximum normalized identity defect |
|---|---:|
|H_h, complex128|1e-12|
|H_low, complex64|2e-6|
|standalone G/R0, complex64|5e-6|
|strong TSA eager/direct-segmented, complex64 and actual128→64→128 solver path|1e-4|
|uncast native128 factors using the same original promoted cache|1e-12|
|H_h after the actual deployed128→64→128 strong action|1e-3|

These are defensible **engineering consistency caps**, not rigorous universal forward-error bounds. They allow more accumulated error for fixed-depth FP32 TSA than a single factor. In unit-roundoff terms, lowH2e-6 is about34u32, G/R05e-6 about84u32 and full TSA1e-4 about1678u32; the latter contains up to126 spatial transforms plus repeated defects/products/sums. The128 cap is about9007u64. Operation counts support the tier ordering; they do not prove those numerical constants or a worst-case FFT error bound. The separate1e-3 composite cap is appropriate only as a finite-probe **physical-image consistency** cap: H_h amplifies FP32 action/combination/input errors, so1e-4 field consistency does not transfer unchanged. It is not a solve-residual allowance or an exact/global H_h M bound. Passing also requires the source-level structural lemma and own-recipe parity, not merely a permissive numerical threshold. Every cap was proposed before measurements; none is relaxed after outcomes.

For every fixed context, test the actual original and deployed wrappers on three fixed pairs: source/mixed source, grazing-Nyquist/source, and two deterministic unit-norm random complex fields (seed91731+N+ID). Include zero, i-commutation, one general unit phase, real scaling and the predeclared combination

    a=.73+.29i,  b=-.41+.37i.

Use actual native input dtype, record scalar/dtype construction, and promote differences/norm evaluations consistently. For the cast-back solver map, keep its128 input and64 cast; test it under the full TSA1e-4 cap. The native128 cap applies to the uncast action, **not** a complex128 wrapper that intentionally casts to64. Retain the original complex64 inverse; no rebuilt wide inverse, new shifts or new physical matrix.

Normalize the combination defect by the noncancelling sum

    ||L(a x+b y)-a L(x)-b L(y)|| /
       (|a| ||L(x)||+|b| ||L(y)||),

and homogeneity by `||L(a x)-a L(x)||/(|a| ||L(x)||)`. Record numerator and denominator as well as ratios. Exact zero actions must remain zero. A zero denominator on nonzero probes is uninformative/needs explicit eligibility; a clamp must not create a pass. Do not normalize by the possibly cancelling combined output. Independent complex probes matter because real/source-only tests can miss conjugation, real extraction and cross-channel defects. Finite probes supply no full-input/global linearity certificate.

The helper also records a symmetric denominator consisting of lhs norm plus the noncancelling reference-output sum. Its ratio is diagnostic only: the pass gate uses the earlier stricter reference-output ratio, so a factor-two normalization change cannot silently relax a frozen cap. Scalar receipts include actual observed input/output dtype, shape/stride, finite/norm/imaginary fractions, mutation status, exact zero, each denominator/numerator and eligibility. Every output must have the declared complex dtype (default input dtype), consistently across all ten calls; an intentional conversion requires an explicit expected_output_dtype. Callable-internal casts are explicitly not introspected; the driver separately records the declared original/cast/native path. It clones images before later calls to avoid static-output overwrite, returns no Tensor/image values and makes ten action calls per pair. Such snapshots stabilize identity receipts but do not prove raw solver-output ownership; the driver's separate cross-call retention/guard checks remain required. Nine maps x three pairs x four contexts therefore produce108 records/1080 action evaluations. Any failed record blocks all complete solves.

Own-recipe complex64/complex128 byte parity, immutable material/cache state, output ownership and restored guards are separate prerequisites. A wide shadow using the same promoted cached factors can help attribute an FP32 discrepancy. It neither represents the actual cast32 pipeline nor overrides a failed practical cap. Do not rebuild coefficients to make a shadow pass.

For interpreting actual retained solver fields, fresh H_h images are authoritative rather than combinations inferred from separate source images. Field construction itself rounds: with the nominal bound

    L_H=max_(l,k)|omega_l·k|+||sigma_t||_infinity
          +||sigma_s||_infinity ||P||_2,

its contribution is at most `L_H ||rounded_combination-exact_combination||`. A scalar-expression roundoff allowance is proportional to `u64 (|a| ||z1||+|b| ||z2||)` with the appropriate complex multiply/add constant. Source cancellation is avoided by the denominator, but ill-conditioned physical action can still amplify field-combination error. Thus1e-12 is not a uniform theorem for arbitrary near-null combined fields. Ordinary evaluated norms/allowances are not interval certificates. The frozen H_h-after-deployed map records composite image consistency directly; no optional prefix/complex-span fit is planned. Complete solver iterations and final acceptance still freshly evaluate physical fields. Even a rounding-explained result exceeding a frozen cap stops the run; distinguish numerical obstruction/inconclusive fidelity from structural nonlinearity without result-based relaxation.

## Original complex solver and the conditional complete screen

The original [fgmres_with_trace](../../remote_reference/rte/npbs_unet_ablation.py) already uses conjugate MGS, complex RHS/Hessenberg dtype, complex tiny LS, stored actual Z and a fresh physical residual at every iterate. Its real wrapper performs the same routine on concatenated real fields. This supports the N256 complex authority. Its generic harness defaults/restart20 are not a matched restart25 replacement, and it has no reference-tail augmentation. N128 needs a reviewed isolated complex counterpart preserving the original injection timing, finite/zero/dependence rejection, physical acceptance and happy breakdown. Conjugate transposes replace real transposes; preserving source/initialization alone is not sufficient if these decisions change accidentally.

Small CPU correctness fixtures should cover genuinely complex matrices/coefficients, an unchanged real-invariant problem, complex-dependent versus real-independent images, rejection fallback and a perfect accepted direction/happy breakdown. The scalar H=iI,M=I distinguishes real versus complex coefficients algebraically; it is an algorithm fixture, not this RTE's performance evidence. No fixture was executed by this review.

Only after **every** declared entry gate passes may the root run the32 complete solves belonging to the two-arm coefficient comparison: IDs0/3 at N128/N256, one warmup/three timed repeats. The separately revised six-arm screen refreshes four other real backends, bringing the frozen total to96 complete solves; see [independent protocol audit](INDEPENDENT_PROTOCOL_AUDIT.md). Keep restart25/max100, actual original H_h/RHS, source-normalized tolerance1e-5, balanced initialization/reference injection at N128 and zero start/TSA16 at N256. Retain all attempts, easy cases and failures; no scalar variant or backend reselection. Complete fields need finite/solver success, independent original-H residual<=1e-5, evaluated discrete relative-error bound<=1e-3 and paired final-field difference<=1e-4. These remain ordinary FP64 discrete acceptance checks.

Charge all physical/material/inverse/capture/setup, cast/copy, changed complex reductions/LS, injection/rejection, continuation and physical acceptance. Root's mandatory entry-diagnostic acquisition is common-charged in full to all six arms, with operational setup/accepted costs also retained separately. This research acquisition can limit percentage savings under its chosen accounting; it is not an operational hardware floor. Preserve the existing Q1/warm economic gate and strongest successful portfolio; a complex classical success only strengthens the subsequent learned denominator. Equal direction counts do not imply equal costs. No teacher/training/source-phase amortization or new metric is introduced here.

The admissibility outcomes are: structural and numerical pass with explicit finite-precision qualifications; structural failure; or a failed/inconclusive numerical gate. Only the first permits this authorized discriminator. The latter two preserve a precise blocker and current real-solver baseline, not permission to switch algorithms, relax tolerances or declare global complex linearity impossible.
