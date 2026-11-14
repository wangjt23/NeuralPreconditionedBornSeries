# RTE: coarse correction followed by cheap full-spectrum correction

13 November 2026. Same fixed K material, N128, L12, g=0.65 and source IDs [0,4,7,14]. This is a classical CPU coupling experiment, not a neural-method result or a new generalization test.

## Question and controls

The completed second-P8-injection experiment found that most remaining residual energy was outside P8. Repeating the coarse correction did not remove classical work. This new experiment keeps one injection at the original position, avoiding a second injected column, and asks whether a cheap full-spectrum response applied to the actual coarse defect is useful.

C(r) is the existing inner solve on the 17 tangential spatial Fourier modes, with full angular/normal structure. B8(r) is eight stages of the existing TSA action, using the same method's already prepared reference. The ordinary outer action remains TSA64.

- One-shot: C(r).
- Fine only: B8(r), with no coarse factors.
- Additive: C(r) + B8(r).
- Multiplicative: C(r) + B8(r - A C(r)).

The multiplicative expression uses the residual left by the coarse correction, not the original residual twice. With fixed linear actions its residual propagation is (I - A B8)(I - A C). Here C is tolerance-truncated; no exact projector cancellation is assumed. Multiplicative two-level correction is classical, so any gain belongs to this paid coupling/control, not to learning or a novelty claim.

## Actual result

Outcome: MULTIPLICATIVE_NO_GO. All 16 solves passed unchanged original-equation criteria.

| Method/order label | Setup s | First accepted RHS s | Complete four RHS s | TSA64 calls |
| --- | ---: | ---: | ---: | --- |
| 01_E_once | 2.978 | 7.740 | 22.654 | [3, 3, 3, 3] |
| 02_fine_only | 0.132 | 6.443 | 26.100 | [5, 5, 5, 5] |
| 03_additive | 2.826 | 9.066 | 28.043 | [4, 4, 4, 4] |
| 04_multiplicative | 2.530 | 7.521 | 23.071 | [3, 3, 3, 3] |

Paired complete-cost comparisons against the original one-shot path:
- one-shot before multiplicative: 22.654 s versus 23.071 s; multiplicative saving -1.84%.

A reverse pair was predetermined only if the first multiplicative batch both removed classical work and beat every accepted control. No depth, source, tolerance or P8 sweep was performed. Two pairs, if present, are not a confidence interval or evidence for arbitrary materials.

## Paid work and verification

| Label | Coarse inner s | Eight-stage smoothing s | Additional original-A composition s |
| --- | ---: | ---: | ---: |
| 01_E_once | 3.623 | 0.000 | 0.0000 |
| 02_fine_only | 0.000 | 0.579 | 0.0000 |
| 03_additive | 3.526 | 0.581 | 0.0000 |
| 04_multiplicative | 3.573 | 0.582 | 0.0335 |

Each batch pays fresh physical/operator/reference setup and its own factor construction and all-mode qualification when C is used. Within a batch, B8 legitimately shares the already prepared reference with the outer TSA64, without modifying its depth. A separate bridge found the explicit eight-stage action bitwise equal to an actual TSA8 object on the fixed probe; the outer object remained depth 64. Every use and stage is counted. Additional original-A work needed to form the multiplicative defect is included. Diagnostics, full solves, field checks and per-source logging remain in complete wall time. Interpreter/import and shared bank loading are outside per-method timing and included in the research ledger.

The newly accepted one-shot output is the field-comparison reference, not candidate information. This avoids rerunning unchanged native baselines. Its own original-A residual and error-bound checks still apply. No accepted output is used as an initial guess, correction direction or teacher.

Maximum original relative residual: 4.71909683e-06, limit 1e-5. Maximum evaluated relative-error bound: 1.89174234e-05, limit 1e-3. Maximum relative field difference: 1.89758023e-07, limit 1e-4. Original source/material/background identities and sealed production/E hashes remain unchanged.

Recorded numerical research wall: 101.303 s. CPU0 with one numerical thread matches the existing workload. Shared GB activity is recorded; no process was preempted and no GPU lock or new paid resource was used. No GPU performance claim is made.

## Interpretation boundary

If the multiplicative action wins while additive and fine-only controls do not, the causal evidence supports correcting the coarse defect rather than merely adding the two actions or spending extra smoothing work. It does not establish an AI advantage. The result concerns this fixed material/four-source regime; material transfer and any learned surrogate need separate evidence against the strengthened classical control. Negative controls are retained.

Files: DESIGN.md, executed scripts, action bridge, raw results/results.json and ANALYSIS.json. This remains a continuation of the existing FWI checkout and source bank, not a standalone dataset distribution. No Library upload was attempted for this follow-up.
