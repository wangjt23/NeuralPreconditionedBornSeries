# Fixed gain-two transport mechanism and stop rule

Prepared 2026-10-02 from existing local original-project snapshots, the preserved matched-v2 receipts and the new isolated helper. The user authorizes one fixed beta=2 control; this note does not authorize a gain sweep, training or additional compute. No SSH, GPU experiment, original-source export, Library mutation or seed9626 access was performed here. The original [TSA implementation](../../remote_reference/rte/transport_baselines.py), [transport operators](../../remote_reference/rte/models.py), [base construction](../../remote_reference/rte/train_source_predictor.py), [literal new helper](gain_two_reference.py) and [independent protocol audit](INDEPENDENT_PROTOCOL_AUDIT.md) are the source authority. An independent mathematical review checked the ordered identities below.

## Operators and the implemented action

The project is steady periodic two-dimensional radiative transfer with twelve stored angular directions. The nominal discrete transport action is

    H psi = Omega·grad psi + sigma_t psi - sigma_s P psi,

where P includes the original stored phase matrix and quadrature weights. Spatial differentiation uses the original Fourier symbol, including its actual grid/Nyquist convention. Pi denotes the angular-index DFT projection retaining modes0,+1,-1. It acts on reference **input**; R0 produces all twelve output angles.

Distinguish H_l, the nominal linear action using the low object's FP32-assembled stream/material/phase conventions, from H_h, the original promoted-buffer FP64 physical action used by FGMRES and acceptance. The stream assembly depends on the operator's float dtype; promoting an input alone does not make a low object the high operator. These nominal maps treat their fixed stored coefficients as exact for algebra. Actual FP32 FFTs, sums, casts and divisions are rounded and are not exactly linear.

R0 is the original cached full-angle homogeneous-reference inverse: its twelve-by-twelve matrix is prepared in complex64 from the unshifted stored mean coefficients, then applied with spatial FFTs. Native128 promotes that same cache, rather than regenerating a double-precision inverse. G is the original mean streaming-plus-total-coefficient inverse, retaining its division and FFT order. Neither changes the physical H in the defect.

The authorized literal is

    r_l=cast32(r_h),
    g=TSA4(r_l), d=r_l-low.A_op(g),
    projected=Pi(d), weighted=2*d-projected,
    p2=promote128(g+R0(weighted)).

Any owned128 segmented prefix result is cast back before the low defect. The angular mask is the separately validated capture-safe equivalent of the original mask. The eager parity authority must construct the same literal through the original projector and reference solve. Computing `2*p_full-p_q3`, or separately applying two reference solves, changes the floating-point path and cannot supply the candidate's acceptance evidence.

Writing C2=2I-Pi=Pi+2(I-Pi) shows that the retained inputs have coefficient1 and omitted inputs coefficient2. **The order is R0 C2, not C2 R0.** Streaming makes the reference angular response mode-coupled, so these maps generally do not commute. A positive spectrum for C2 does not establish positivity/coercivity of the preconditioner, a residual contraction or a convergence theorem. This relative weighting is not an overall scalar already optimized by real FGMRES.

## What TSA4's residual polynomial actually is

The original `single` for TSA is F=G+R0(I-H_l G). Its `forward` starts z0=0 and repeatedly executes

    d_j=r-H_l z_j,  z_(j+1)=z_j+F d_j,

with relaxation1. Define S=I-H_l F. Exact fixed-operator algebra gives

    S=(I-H_l R0)(I-H_l G),
    d_j=S^j r,
    M4=F(I+S+S^2+S^3),
    g=M4 r,  I-H_l M4=S^4.

Thus TSA4 is four stationary residual stages. It is **not F^4**, and the ordered product in S must not be replaced by a commuting/scalar approximation. The corresponding solution-error map is `I-M4 H_l=(I-F H_l)^4`.

The gain-two complete map is

    M2=M4+R0 C2 S^4,
    I-H_l M2=(I-H_l R0 C2)S^4,
    I-M2 H_l=(I-R0 C2 H_l)(I-F H_l)^4.

This is a terminal angularly weighted reference-defect stage applied after the four original TSA stages. It is generally not a degree-five polynomial in one common operator: the new R0 C2 factor need not commute with F or H_l. These identities explain the composition; they are not exact equalities for every rounded application or a prediction of FGMRES iteration counts.

Let DeltaH=H_h-H_l. The nominal accepted residual map is

    I-H_h M2=(I-H_l R0 C2)S^4-DeltaH M2.

The low residual polynomial alone therefore cannot establish high-operator acceptance. For the actual constructed low field p_l, let a_l be the promoted output of the rounded `low.A_op(p_l)`. A bookkeeping identity, exact in algebra after promoting those actual stored values, is

    r_h-H_h(promote p_l)
      = [r_h-promote r_l]
        + [promote r_l-a_l]
        + [a_l-H_h(promote p_l)],

with ordinary FP64 subtraction rounding retained in a numerical evaluation. The middle difference is formed after promotion; an additionally rounded FP32 residual needs its own subtraction-error term. This separates cast discrepancy, a residual using the rounded low image and low/high action discrepancy; it does not certify a global rounding norm. The complete proposal must be freshly scored by H_h.

## Reference-defect interpretation and the homogeneous limit

To avoid the earlier operator-attribution error, H0 denotes the **actual stored FP32-assembled reference matrix**, promoted128, not a newly reconstructed high-precision homogeneous operator. Let E_c=H0 R0-I and D_l=H_l-H0. Then

    H_l R0=I+E_c+D_l R0,
    I-H_l R0 C2=-(I-Pi)-E_c C2-D_l R0 C2.

E_c concerns inverse-cache/reference closure. D_l includes heterogeneous coefficient variation and differences between the actual stored operator constructions. For physical high scoring use D_h=H_h-H0; it likewise includes stored streaming/mean/phase arithmetic differences. Closure receipts cannot assign the whole mismatch to pure heterogeneity. Production FFT/cast rounding adds input-dependent action error beyond these nominal identities.

For any tentative g in the ideal matched limit H_l=H_h=H0 and R0=H0^-1, a general terminal rule `g+R0[Pi+beta(I-Pi)]d` leaves residual `(1-beta)(I-Pi)d`. Gain1 closes that ideal reference defect; gain2 reverses the omitted component rather than monotonically reducing it. For the actual TSA primitive in that same ideal limit, F=R0 already, so d after TSA4 is zero and M2=R0. Both statements are compatible. Homogeneous material in the stored FP32/FP64 implementation alone does not imply either exact limit.

## Why this single gain is a falsifier

The preceding [matched-v2 decision](../full_angle_v2/DECISION_REVIEW.md) found useful omitted-input signal: full input improved the actual-H complete-proposal allowed-line geometry over q3 in all four source/prefix strata, yet neither unit composition passed the strongest-portfolio economics. In the attained two-measured-image fit, normalized omitted-component gains b/(a+b) ranged1.634857–4.774191 across the eight contexts. This motivated one modest fixed beta2 before new results.

In exact arithmetic, with M_full=M4+R0 S^4 and M_q3=M4+R0 Pi S^4,

    M2=2 M_full-M_q3,
    M2-M_full=R0(I-Pi)S^4.

The old fit only used separately measured images. It did not independently apply H_h to the fitted combined field; it therefore cannot verify the new literal FP32 p2. Nor does it establish a q3/full response-range optimum. Fresh raw and prefix real-AZ images of p2 answer the fixed-direction question. Real projection can remove rank and is not a complex projector, so no raw coercivity/singular-value lower bound transfers to that projected geometry. A best-line improvement is diagnostic, not an acquired deployable coefficient or a guarantee along the new solver trajectory.

## Work and frozen economics

Relative to plain TSA4, beta2 adds one low physical defect, one full R0 application, one angular FFT/ifft pair and the local `2*d-Pi*d`. Relative to the preceding full beta1 rule it adds only that angular pair and weighting; q3 already has the same angular pair. All three terminal rules still use the full twelve-angle spatial reference solve. There is no three-channel compute claim.

Under the unchanged original operation path, TSA4 has15 spatial forward/15 inverse FFTs; each terminal composition has17/17, with beta2 and q3 additionally having one angular forward/inverse pair. Counts exclude guards/copies and do not establish latency, memory traffic, setup or total FGMRES cost. Preserve operation order; previous unsuccessful algebraic rewrites are not silently revived.

The root-owned frozen screen is eleven arms: the ten prior controls and exactly one beta2 arm, RAW at N128 and SEGMENTED at N256 chosen before its results. Development IDs0/3, four grid/material contexts, one warmup plus three measured repeats give176 complete solves. Retain280 checks (252 previous-equivalent plus28 candidate), four physical prefixes, eight geometry targets and all failed/negative receipts. The existing95s soft/120s hard/12GiB envelope is unchanged. This note executes none of that work.

Keep original FGMRES initialization, RHS, high operator, real-stacked coefficient convention and acceptance: zero failed complete solves, physical relative residual<=1e-5, evaluated relative error bound<=1e-3, paired field difference<=1e-4, finite fields and parity. These are ordinary FP64 **discrete** evaluations, not interval certification, continuum solution error or an OOD guarantee.

Primary Q1 is recorded full deployment setup plus complete warm accepted solve. Charge material/common preparation, base/inverse/mask construction, capture, casts/clones/helpers, initialization, H_h images, solve continuation and required acceptance. Record diagnostic geometry/prefix acquisition separately and do not double-count nested times. Preserve first-use/constructor ordering: the screen is not a matched process-cold estimate. There is no uncharged fit, teacher or training; assumed source-phase aliases cannot create independent Q16 amortization.

The comparator is the same-case successful prior-ten envelope, independently minimizing Q1 and warm cost. Its means are270.017832/164.766683ms Q1 and50.680837/130.545220ms warm at N128/N256. This hindsight envelope is a benchmark denominator, not a free deployed selector. Require>=10% mean improvement at **both grids for both metrics**, no individual Q1 regression>10%, and unchanged accuracy; do not reselect the candidate backend or relax the gate afterward. At the preceding selected-full setup costs, the entire-solve binding warm allowances were32.666833ms at N128 and117.490698ms at N256. New actual beta2 setup must replace those illustrative values, not be assumed equal.

## Outcome boundary

If engineering parity/capture/accuracy fails, stop scientific interpretation and preserve the failure. If literal geometry fails, this rejects the one fixed relative-weight mechanism on these contexts, not an optimum over the full response range. If geometry helps but the complete cost gate fails, the evidence separates useful input information from its current acquisition/application/outer-work economics. Stop this gain/composition family; no gain sweep, q5, FNO training, fine teachers, history-necessity claim or seed9626 access follows.

That economic negative is a concrete barrier for the tested cheap shallow/reference rules against the strengthened classical portfolio. A principled later alternative must change **representation or coefficient-field action**, rather than scalar weighting: for example, a genuinely cheaper all-angle fixed-material response factorization whose complete cost budget is established before acquisition. It would need fresh physical-image and accepted-solve evidence; current two-image geometry does not authorize it or prove its range useful. If no such independently affordable representation is specified, retain the tested-family cost boundary rather than inventing another parameter search.

If beta2 passes, freeze it as another **classical** comparator and include it in the next portfolio. A frozen same16-development-medium confirmation and genuinely independent/OOD sources remain outstanding; neither is authorized by this note. Preserve the verified direct-segmented classical20.784%/19.645% Q1 progress and all earlier neural/angular/response negatives. No novelty, learned-preconditioner success or universal convergence claim follows from this control.
