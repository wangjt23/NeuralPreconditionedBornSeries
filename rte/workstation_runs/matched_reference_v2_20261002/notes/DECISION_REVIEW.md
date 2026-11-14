# Matched-result decision and one next falsifier

Prepared 2026-10-02 from [the result audit](RESULT_AUDIT.json) and local [raw v2 receipts](../../receipts/matched_execution_v2/results/geometry.json). Exact receipt hashes, coefficient ratios, image predictions and budget arithmetic are saved in [DECISION_RECEIPT_ARITHMETIC.json](DECISION_RECEIPT_ARITHMETIC.json). This review performed receipt arithmetic only: no SSH, GPU experiment, source export, teacher, training, Library mutation or seed9626 access. **The current round stops at the completed matched160 screen.** The proposal below is a precisely specified future falsifier, not authorization to execute it or sweep gains.

## What the controlled result establishes

All252 entry checks and160 complete source solves pass the declared physical accuracy gates, including120 measured rows and40 retained warmups. The maximum true relative residual is9.8221942921e-6, evaluated relative error bound7.2498061346e-5 and paired field difference3.6305204471e-7. These are ordinary FP64 discrete evaluations. The v1 capture failure and equivalent cached-mask repair remain separate engineering provenance.

Full input improves the **complete proposal's actual-H allowed real-line geometry** over q3 in all four frozen strata:

| Grid/context | Mean reduction in squared best-line remaining residual, q3 minus full |
|---|---:|
|N128 source|.029906411|
|N128 prefix4|.034966341|
|N256 source|.029880560|
|N256 prefix4|.045522023|

Every stratum passes the frozen.02 signal. The discarded-input fraction averages.932 for sources and.967–.971 for prefixes. That fraction alone would not prove usefulness; the matched physical images provide the actual positive signal. The q3 response still has all-angle output, including higher-output harmonics, so this is **input-information evidence**, not an output-mask explanation.

Neither selected full nor q3 composition passes the unchanged10% Q1 **and**10% warm-accepted gate at both grids. Full beats q3 on the same backend: warm savings are16.04/11.26% for segmented and14.32/12.70% for raw at N128/N256. It nevertheless loses to the strongest classical actions. Full raw warm means72.280/191.313ms versus best-original warm50.681/130.545ms: **42.62/46.55% slower**. Its calls are20/13 at N128 and21/14 at N256, versus12/8 and7/5 for the strong recipes. Cheaper individual shallow actions do not pay the additional outer work.

Small reference closure/additivity defects support the matched interpretation but are not continuum or uniform guarantees: max homogeneous full closure7.8731e-6, max correction additivity8.2122e-6, max physical-image additivity8.3299e-5. The low-versus-high defect gap reaches6.4390e-4. Under the implemented H0 definition, E is reference/cache/application closure; D=H-H0 combines heterogeneity and stored-operator differences. Do not attribute all remaining failure to pure heterogeneity.

The result rejects the **economics of the tested unit-coefficient shallow compositions**. It does not establish a q3/full response-range optimum, prove homogeneous coefficient maps universally inadequate, or prove that residual-history training is needed. The measured two-column fit is not an independently H-verified combined field. The exact inverse of fixed linear H is memoryless; history-sensitive economical approximations need a separate witness.

## One precise no-training mechanism: double omitted-angle response

Freeze one future control at **beta=2**, retaining the original FP32 shallow prefix, low defect, cached projector/inverse and final owned128 output:

    g=TSA4(r), d=r-low.A_op(g),
    p2=g+R0(2d-Pi3 d)
      =g+R0(Pi3 d+2(I-Pi3)d)  [exact arithmetic].

It preserves the q3-input coefficient and doubles the omitted-input component. It needs one cached angular projection and one existing full-angle reference application, not a second inverse solve, new range, teacher, history network or learned parameter. This changes **relative angular-channel weighting**; it is not an overall rescaling that FGMRES already optimizes. Fixed material/coefficients give a complex-linear mathematical map; FP32 rounding remains explicit. No positivity, stability, speed or convergence theorem follows from beta>0.

The evidence for this single choice is the attained measured-image fit. With pq=g+zq and pf=g+zf, its coefficients a,b give, when s=a+b!=0,

    (a pq+b pf)/s
      =g+zq+[b/(a+b)](zf-zq).

All eight fitted s are positive. The normalized omitted-component gains b/s are:

|Grid/ID|Source|Prefix4|
|---|---:|---:|
|128/0|1.634857|2.492541|
|128/3|2.838853|3.996841|
|256/0|1.637478|2.504272|
|256/3|2.866155|4.774191|

The minimum is1.634857 and maximum4.774191. They consistently suggest underweighting of the omitted-input component within this **two-measured-image span**. They are not optimal response coefficients or a fitted deployable field. Beta=2 is a modest, development-informed single falsifier; do not select beta separately per case/context or tune1.5/2/3/4 after seeing more results.

Receipt algebra for `2Hpf-Hpq` predicts a smaller allowed-line remaining ratio than Hpf in all eight measured image contexts. This is an image-combination hypothesis only. The actual FP32 construction `g+R0(2d-Pi3d)` has not been evaluated through H; addition/projection/cache rounding and a changed FGMRES trajectory may defeat it. First future evidence must therefore be **H applied to the actually constructed p2**, with the same raw/prefix real-AZ scoring and original numerical gates. The pre-existing fit cannot serve as its parity or acceptance test.

If that fixed gain fails the frozen geometry/cost test, stop this gain family rather than sweeping gains, expanding q, collecting inverse teachers or training. A positive geometry check still needs an actual complete solve and unchanged cost gates; it does not reopen learning by itself.

## Strongest denominator and the size of the hurdle

All ten successful arms now belong to the future classical denominator. Their conservative per-case minimum Q1 estimates average **270.018ms at N128 and164.767ms at N256**; best warm averages remain50.681/130.545ms. The per-case minimum is hindsight accounting, not a free deployable selector. Frozen per-grid backend selection and same-backend comparisons remain distinct.

Holding the recorded full-raw setup fixed, the10% Q1 target allows mean entire warm cost

    .9 Cbest - S = 32.667ms /98.760ms,

versus72.280/191.313ms observed: approximately **54.8/48.4% reductions**, before any new cost. Full segmented's Q1-only allowances are52.232/119.101ms, but enforcing the warm target too tightens them to45.613/117.491ms versus110.948/210.959ms observed: cuts58.9/44.3%. These are conditional accounting hurdles, not timing predictions or universal lower bounds. New p2 setup/application/mask/helper cost replaces the assumption; worst-case regression constraints also remain.

Recorded setup-plus-warm Q1 has first-use/shared preparation effects and excludes process startup/IO and comparative QA. Preserve that scope. No teacher acquisition can be amortized over assumed Q16; distinct-source reuse is unmeasured. Any future learned method must also charge actual offline training/data and per-material preparation against the successful classical factors/capture reuse.

Keep the **verified classical20.784%/19.645% Q1 advantage** as classical implementation progress on16 reused development media. [Confirmation audit](../direct_capture/confirmation_audit.md) It does not count toward a neural10% goal. Seed9626 remains sealed. A future promising fixed control would require a frozen backend/policy and same16 confirmation, retaining easy cases and all regressions; that would still be development evidence. Angular refinement, new coefficient geometries, lower absorption, stronger anisotropy, boundaries and independent/OOD sources remain untested.

The scientifically supported conclusion now is: **omitted angular input contains useful response, but this unit homogeneous-reference composition is uneconomical against the strengthened portfolio**. One cheap relative-weight falsifier is concrete; no range-impossibility, novelty, learned superiority or history-data claim has been established. [Fan–An–Ying](https://arxiv.org/abs/1802.03061) and [trajectory-aware ROM](https://arxiv.org/html/2509.05001v1) remain relevant primary predecessors, without removing the cost or discretization qualifications.
