# Independent fixed gain-two protocol audit — 2026-10-02

The user authorizes one prespecified β=2 classical control, not a gain sweep or training. This is the protocol and prepared-helper review; the driver, frozen hashes and raw results require subsequent verification. No SSH, GPU experiment, original-source export, Library write or Git action was performed by this auditor. Preserve the previous numerical/cost negatives and v1 capture failure.

## Literal action and mechanism

Use the unchanged original FP32 TSA4, physical-defect action and reference inverse:

    g = TSA4(r)
    d = r - A_low(g)
    filtered = 2*d - Pi3(d)
    p2 = g + R0(filtered).

Pi3 is the original angular-index input projection onto 0,+1,-1; R0 uses the same prepared complex64 full-angle inverse. A captured prefix's output must be cast back to the incoming dtype before forming d. Build/charge the identical projection mask outside capture, guard its complete tensor state and hold it strongly. Native128 must promote the same mask/inverse, not regenerate either. Final solver outputs own storage, with serial graph use and unchanged state/layout guards.

The independent eager GPU authority must call original `angular_project(d,1)` and original `reference_transport_solve(2*d-Pi3(d), harmonics=None)`. A single angular weighted-mask FFT expression, or `2*p_full-p_q3`, is algebraically equivalent in exact arithmetic but changes FP32 operation order. Neither may silently replace the authorized literal. Spatial transform count remains 17 forward/17 inverse per complete application, plus one angular FFT/ifft pair; counts do not establish runtime or a three-channel compute advantage.

For fixed complex-linear actions in exact arithmetic,

    M_beta = M4 + R0[Pi3 + beta(I-Pi3)](I-A_low M4)
    p_beta = p_q3 + beta R0(I-Pi3)d.

Thus β2 weights omitted input twice and retained input once. It is a fixed material-dependent complex-linear rule, not source-dependent fitting or merely global scaling already available to real FGMRES. Floating implementations retain ordinary rounding deviations. It still carries full-angle information; no optimized-Q3 range or coefficient-learning theorem follows.

Write projected unit-update error as `e_beta=e0-beta*w`, where `w=J H R0(I-Pi3)d`, using the real inner product and old search-image projector J. Gain0→1 improvement only implies `Re<e0,w>/||w||² > 1/2`; gain2 improves upon gain1 only if that ratio exceeds 3/2. Prior best-line/attained-span improvement gives no extrapolation theorem. In the ideal case A_low=H=H0 and R0=H0^-1, `r-H0*p_beta=(1-beta)(I-Pi3)d`: gain1 is exact, and gain2 has the same omitted-defect norm as gain0 with opposite sign. Amplification helping heterogeneous material remains an empirical hypothesis.

## Frozen screen and accuracy

Keep development IDs0/3 at N128/N256, all ten previous arms and one β2 arm. Fix β2 RAW at N128 and SEGMENTED at N256 from the preceding full-rule selection before results; do not reselect from this screen. Eleven arms × four contexts × one warmup/three timed repeats gives 176 full solves: 44 warmups and 132 measured. Ten wrappers × six action probes plus one guard bundle × four contexts gives 280 global checks (240 action/40 guard), all before full solves. Five measured applications per arm/context gives 220. Retain four independently verified physical prefixes and eight actual-H gain-two geometry targets; no new teacher, CGLS or two-span coefficient fit is authorized.

N128 retains original balanced initialization/full-reference tail injection; N256 retains zero start. All attempts/failures/costs remain in receipts. Each implementation is checked against its own literal action; only same-recipe backends need bitwise field/status equality. Numerical eligibility retains finite fields, zero complete-solver failures, original-H relative residual ≤1e-5, evaluated discrete relative-error bound ≤1e-3 and paired field difference ≤1e-4. The prefix's four-iteration diagnostic failure flag is not a complete-solve failure. Ordinary FP64 discrete certificates are not interval or continuum claims.

The owned launch envelope is 95 s soft, 120 s hard and 12 GiB reserved, with current hardware/jobs verified by root and unrelated work undisturbed. Preserve/hash the preceding370-file scope, preservation receipt and eleven new reports before staging; record new isolated code/policy hashes without overwriting old runs.

## Economics and interpretation

Charge each deployable arm for material/coercivity setup, the appropriate prepared base, entire graph/action/mask construction, casts/copies, original-H images, solver continuation and all rejected work, plus independent acceptance. Shared benchmark preparation is not free deployment setup. Charge geometry/prefix/H0 acquisition separately and avoid nested timing sums. Retain constructor order and first-use costs; setup plus warm accepted Q1 is not matched process-cold latency.

For this round no H0 rebuild or two-span fit is planned. The fixed user-prescribed action consumes no online oracle or geometry coefficients. Therefore primary algorithm cost may exclude diagnostic acquisition while still charging every necessary constructor/application/check. Retain total research execution costs and the preceding 0.6127944 s acquisition as historical provenance. Also freeze a conservative nonamortized per-context Q1 sensitivity that adds this round's unique prefix initialization/solve/verification plus both target diagnostic totals to β2. Nested timers must not be added again. This is a research-cost sensitivity, not mandatory operator runtime or a time floor; no superiority claim should rely solely on primary costs if the prespecified stricter sensitivity fails. Do not silently normalize prior history or imply free calibration of a deployed selector.

Independently minimize successful prior-ten Q1 and warm accepted time per case for the comparator envelope. These are analysis envelopes, not free runtime selectors. The gate is ≥10% mean improvement at BOTH grids for BOTH metrics, with worst individual Q1 regression ≤10%; all eleven successful arms then enter the future portfolio. Timing repeats summarize within media and are not independent samples.

Geometry must apply original H to the actual rounded p2, score composites against physical r, and use the same recorded real-AZ basis/orthogonality and nonzero-norm eligibility. Do not infer actual p2 from stored full/Q3 image combinations. Initial-source versus physical-prefix targets remain distinct. A scalar best-line value may be recorded as a diagnostic; it is not an acquired deployable coefficient or a full-range optimum.

A positive screen would justify freezing a new classical comparator and a bounded confirmation, not claim learning. A negative screen rejects this fixed gain/placement/cost rule; it does not reject all gains, coefficient fields, learned maps or correction placements. No post hoc gain sweep, backend choice, gate relaxation, q5 expansion, training or seed9626 use follows from either result.

## Prepared helper review

[gain_two_reference.py](gain_two_reference.py) implements the prescribed multiply/subtract before one original reference correction, after the inherited prefix backcast. It inherits the frozen v2 mask/cache/layout and backend guards, includes beta type/value in signatures, hardcodes literal gain2 without a gain argument, and binds native128 fallback to the whole compound. The fixed kind/mode changes are followed by a fresh constructor signature before use. Raw capture retains global error checking, a private pool and warmed side stream; its expected transform counts are checked. New owned complex128 outputs preserve retained solver vectors.

[test_gain_two_reference.py](test_gain_two_reference.py) compares against the original projector/solver in both precisions and backends, guards gain/mode/harmonics/mask/material/cache/layout mutations, checks original-cache preservation and operation counts, and explicitly demonstrates why output reassociation is not a parity substitute. These tests are meaningful contracts; their passing status and actual CUDA behavior still require root's preserved preflight/execution receipts. No blocking semantic, precision or guarding defect was found in this static helper review.

[gain_two_screen.py](gain_two_screen.py) now implements the counts/order above and constructs only the prescribed β2 backend. Its independent authority uses original angular projection before the weighted defect/reference solve. It adds a beta-mutation rejection to the existing ownership/cache bundle, and explicitly verifies bitwise equality of reference caches, physical source and N128 initialization across strong/shallow recipes. Constructors and base selection are billed correctly; only identical previous recipes share full-field/status parity assertions. Gain2 geometry is evaluated through the actual backend and original H, compared bitwise to its own original literal, without a new H0 build or span fit. Prefix initialization/solve/verification and target totals supply the separate acquisition sensitivity. No blocking prepared-driver issue was found; fixed mapping, thresholds and the sensitivity claim rule still require the frozen policy and later raw-aggregation audit.
