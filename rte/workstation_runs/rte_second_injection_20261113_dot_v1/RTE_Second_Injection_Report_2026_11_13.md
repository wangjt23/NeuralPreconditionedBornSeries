# RTE follow-up: repeated P8 correction is not the missing speedup

13 November 2026. FWI CPU evidence, unchanged K material and source IDs 0, 4, 7, 14. P8 retains 17 tangential spatial Fourier modes and all 12 angular components. This is a classical diagnostic, not a learned-method result.

## Main completed comparison

All 20 method/source solves passed the original-equation criteria. A second fixed P8 injection did not remove a single classical call. Four-source cost increased from 22.611 s to 29.747 s, a 31.56% slowdown against the matched one-shot method. Classical calls changed from [3, 3, 3, 3] to [3, 4, 3, 4]. Both injections were accepted; rejection was not the cause.

| Method | Fresh setup s | First accepted RHS s | Complete four RHS s | Classical calls |
| --- | ---: | ---: | ---: | --- |
| tsa64_zero | 0.134 | 6.232 | 25.065 | [5, 5, 5, 5] |
| compact4 | 3.263 | 8.306 | 24.125 | [4, 4, 4, 4] |
| tsa16_zero | 0.136 | 6.345 | 26.111 | [17, 17, 17, 17] |
| E_once | 2.770 | 7.602 | 22.611 | [3, 3, 3, 3] |
| E_twice | 3.003 | 8.889 | 29.747 | [3, 4, 3, 4] |

The one-shot path was 6.27% below compact4 in this fresh sequence. This remains one shared-host timing sequence, not a robust universal speed claim. All methods used one numerical CPU thread to retain the existing benchmark condition. GB jobs remained active elsewhere on the host; no jobs were preempted and no GPU lock was created. This is not a GPU or multi-core throughput comparison.

Costs include fresh physical/material/preconditioner/factor construction, all 17 original-A0 qualifications and zero/tiny checks for each P8 factor, the original solves, final checks, diagnostics and per-source logging. Interpreter/import and shared source loading are outside per-method times and recorded in research wall. Single-source values are observed setup-to-first-result clocks, not four-source totals divided by four.

Maxima across all 20 rows: original relative residual 8.63348956e-06 (limit 1e-5), evaluated relative error bound 4.56527267e-05 (limit 1e-3), relative field difference 2.66769144e-07 (limit 1e-4). The new one-shot implementation matched the production solver on the real source-0 bridge to 2.056e-13, with identical iteration/call/acceptance counts. Nine focused interface/schedule tests passed. Production and sealed E source hashes were unchanged.

## What the residual diagnostics actually show

| Source | Residual energy outside P8 before second injection | P8 norm reduction factor | Outside-P8 norm ratio, after/before | Total residual ratio before -> after |
| --- | ---: | ---: | ---: | --- |
| 0 | 98.37% | 69.1 | 1.000215 | 2.37e-05 -> 2.35e-05 |
| 4 | 97.68% | 210.5 | 1.000055 | 7.53e-05 -> 7.44e-05 |
| 7 | 96.49% | 83.3 | 1.000416 | 3.35e-05 -> 3.29e-05 |
| 14 | 97.35% | 179.6 | 1.000079 | 7.54e-05 -> 7.44e-05 |

The second correction substantially cleans the already-small P8 residual, while the complementary residual barely changes. The total residual remains above stopping tolerance. This observed action is poorly targeted at the work still required. It does not prove that every P8-based or learned correction is incapable of reducing the complement.

## A separate trajectory issue, and its limits

Source inspection shows why an extra accepted direction need not preserve the original classical continuation. In the existing implementation, injection advances the Arnoldi column. The next classical call uses the newly generated vector, skipping the vector that the one-shot path was about to precondition. The two resulting spaces need not contain one another. This is not a failure of original-equation acceptance; it changes the search trajectory and complicates a pure "add a direction" interpretation.

An isolated variant retains that pending classical input across the second injection. In exact arithmetic, once the retained input has been processed, its search space contains the corresponding one-shot space plus the additional correction. Least-squares residual dominance follows only under that shared-direction hypothesis, not for the original skipping implementation. This is a standard subspace fact, not an originality claim.

The first four-source frontier diagnostic stopped at source 0 because a deliberately stringent 1e-10 input-vector identity check failed. Its partial receipt and costs are preserved. The failure was in that diagnostic assertion; it is not evidence of an original residual failure. No claim of a completed four-source frontier comparison is made.

A single-source follow-up then compared two fresh one-shot runs with the preserved-input variant, recording discrepancies instead of using the diagnostic identity as a solution-acceptance gate. Original residual and field/error criteria were unchanged. This is a distinct mechanism diagnostic, not a favorable timing retry of the original experiment.

| Source-0 diagnostic | Cost s | Classical calls | Final original residual |
| --- | ---: | --- | ---: |
| 01_E_once | 8.309 | [3] | 4.788854e-07 |
| 02_E_once | 7.738 | [3] | 4.788855e-07 |
| 03_E_preserve | 8.959 | [3] | 4.784634e-07 |

The source-0 diagnostic has only one RHS despite the inherited raw field name complete_four_RHS_seconds. The table above labels it as a single-source cost. The retained-input variant recovered the one-shot residual quality but still required three classical calls and added an inner solve. It therefore supplies no speedup evidence. Two fresh one-shot runs differed by 1.1962e-4 in their third normalized classical input despite virtually identical accepted final residuals. The preserved-input run matched the first reference inputs exactly in this diagnostic. Thus the 1e-10 intermediate-vector identity check was not a reliable numerical acceptance criterion. The reason for that intermediate sensitivity is not isolated here. The actual discrepancies remain in ANALYSIS.json; no solution tolerance was relaxed.

## Decision

Stop the fixed second-injection branch as a speed candidate on this workload. Do not train a network merely to imitate it or retune the same injection schedule. The next meaningful research question is how to act cheaply on the complementary residual while preserving useful classical directions, not how to repeat an already-effective low-mode solve. Any such method needs its own executable representation and strong classical control. No learned acceleration, general-material superiority, publication readiness or GPU improvement has been established.

Total recorded research wall for the completed study, preserved partial diagnostic and final single-source diagnostic: 230.776 s, excluding launch/import/file-transfer overhead. No worker remains intentionally running.

## Reproduction scope

The package includes isolated source changes, raw scalar receipts, tests, driver, prior input/provenance records and this independent analyzer. It is a continuation package for the existing FWI checkout, not a standalone distribution of all original project dependencies or source-bank tensors. Original dependency hashes and paths are recorded; original files, held-out seed 9626 and historical results remain unchanged.
