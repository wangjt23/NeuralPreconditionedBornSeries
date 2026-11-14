# Independent matched full-angle/Q3 result audit — 2026-10-02

The raw v2 receipts pass independent numerical/protocol verification. Full-angle input supplies useful direction information in this fixed composition, but neither correction meets the frozen cost gate. This is a positive information diagnostic and a negative affordable-control screen; it supplies no learned-method advantage or full-range impossibility result.

The [standard-library auditor](../full_angle/audit_matched_receipts.py) independently recomputed [this JSON](INDEPENDENT_RESULT_AUDIT.json) from [preserved raw receipts](../../receipts/matched_execution_v2). Counts are exact: 252 global entries (216 action/36 guard bundles), 160 full solves (120 measured/40 warmups), 200 measured applications, 96 same-recipe whole-solve backend comparisons, four truncated prefixes and eight geometry targets. All entries, ownership/cache guards, transform counts and same-recipe field/status checks pass. Different algorithms are not required to produce identical fields/status.

All 160 full solves pass finite-field/residual/error/field QA, with zero failures. Minimum evaluated coercivity margin is 0.0170082186; maximum original-H relative residual is 9.8221943e-6, evaluated relative-error bound 7.2498061e-5 and paired field difference 3.6305204e-7. These are ordinary FP64 evaluations for the stored discrete operator, not interval or continuum certificates. The four prefixes deliberately stop at four iterations and have `failed=1`; they are diagnostics, not failed complete solves.

The process took 65.5917885 s; runner 61.9187501 s, within 120/95 s limits. Peak allocated/reserved memory was 2,682,390,528/9,718,202,368 bytes, below 12 GiB. All eleven policy code hashes and fifteen staged payload hashes match. CPU preflight reports 25 tests passed. Policy SHA is `67e5cbdfaf734265f9b06ffb7f18cee75a6ab52c22cf8c5a46bce7cee0c30c48`; driver SHA is `c82c8050e93c48894f05c7791864ec8794237ea1497ae4862491f52a057dfe4d`. The [post-execution preservation receipt](../../receipts/matched_final_preservation.json) reports all 370 hashes matched, 155 original files preserved and nine tracked dirty files unchanged on main, HEAD `028caea3f35d100225f656e2b1e910c1366aaafe`; no other research job or new training, test9626 access, source export or Git write is recorded. The separate v1 capture failure remains preserved.

## Complete costs

Within-medium medians precede means over the two development media per grid. Warm and Q1 comparator envelopes independently choose the cheapest successful original-five implementation per case; these minima are comparison envelopes, not a free deployed chooser. Q1 means recorded algorithm setup plus warm accepted time, excluding startup/source I/O and QA.

| Grid/rule selected by Q1 | Mean Q1, ms | Mean warm accepted, ms | Q1 saving | Warm saving |
|---|---:|---:|---:|---:|
| N128 full_raw | 282.629 | 72.280 | +0.797% | −42.618% |
| N128 q3_raw | 295.849 | 84.359 | −3.843% | −66.451% |
| N256 full_segmented | 240.148 | 210.959 | −45.750% | −61.598% |
| N256 q3_segmented | 267.124 | 237.730 | −62.122% | −82.106% |

None passes 10% Q1 AND warm improvement at BOTH grids, or the individual-regression safeguard. Even faster full_raw warm time is 16.643%/39.927% slower than the warm time of the original Q1-best direct-segmented backend at N128/N256; the negative conclusion does not rely only on the conservative warm envelope. Full does beat Q3 on matching backends: warm savings range 11.261–16.041%.

First-use charges matter: N128 ID0 physical setup is 264.785 ms, margin/eigensolve setup 75.995 ms, and first strong-base construction 70.071 ms versus TSA4 3.400 ms. These charges are retained. They prevent interpreting the small N128 mean Q1 saving or backend ranking as matched process-cold performance. Full's from-start calls are 20/13 at N128 and 21/14 at N256, versus Q3 23/15 and 23/16 and strong 12/8 and 7/5. Lower application cost is outweighed by more full-H images, projections and continuation; call/application products are not exclusive timings or rigorous floors.

## Geometry and boundary

Mean reductions in squared normalized best-real-line composite residual, Q3 minus full, are 0.0299064/0.0349663 for N128 source/prefix and 0.0298806/0.0455220 for N256. All four exceed the prespecified 0.02 signal, and full improves unit and real-line residuals in all eight targets. Real-AZ Gram defect is ≤7.34e-16; physical-prefix recurrence gap ≤7.00e-11. Targets/images are nonzero; all two-proposal fits retain rank two. Unique diagnostic acquisition totals 0.6127944 s, including prefix initialization/verification and H0 construction once per context, without adding nested timing fields again.

This isolates benefit of retained full-angle correction INPUT for the tested coefficients. Shared TSA4 and N128 initialization/injection already use full-angle information; Q3 output also contains higher harmonics. It neither proves every optimized Q3 map deficient nor explains prior neural phase/runtime gaps. H0 is the FP32-assembled reference matrix promoted to128: H−H0 includes stored-operator differences and heterogeneity. Maximum low/high defect gap is 6.439e-4; no pure-heterogeneity causal decomposition was measured. Increment additivity and attained real-image fits do not certify rounded composite rewrites or deployable source-dependent coefficients.

A proposed fixed β=2 gain on omitted-input response is only a new classical hypothesis: improvement from gain 0 to 1 does not prove extrapolation improves convergence or cost. It needs a fresh literal-map/parity/accuracy/cost protocol against all ten successful controls. No q5, teacher acquisition or training is justified here. Under unchanged recorded full-action setup and zero new acquisition, the all-ten comparator permits aggregate warm budgets of only 32.667/117.491 ms at N128/N256 for a 10% goal, versus actual 72.280/210.959 ms—necessary reductions of 54.805%/44.306%, with individual constraints still required. These conditional arithmetic requirements are not attainable-headroom estimates or universal time lower bounds. The next useful question is whether a cheap fixed classical gain or limited correction placement can remove this measured continuation cost; this round did not test it.

Reproduction uses no solver execution:

```sh
python3 deliverables/full_angle/audit_matched_receipts.py receipts/matched_execution_v2 \
  --output deliverables/full_angle_v2/INDEPENDENT_RESULT_AUDIT.json \
  --expected-phase matched_reference_screen_v2 \
  --expected-policy-sha 67e5cbdfaf734265f9b06ffb7f18cee75a6ab52c22cf8c5a46bce7cee0c30c48 \
  --expected-driver-sha c82c8050e93c48894f05c7791864ec8794237ea1497ae4862491f52a057dfe4d
```
