# RTE fixed P8-exterior direction: real work avoided, full-cost no-go

**Completed:**12/12 original-equation solves accepted. No new training, cutoff sweep or heldout claim.

## Controlled question

The prior paid error-response study, not residual-spectrum guessing, identified tangent |k|9..16 in the source0/4 tail. This experiment changes only the support of a second projected solve after classical call2: repeat P8 versus shell9..16. Both retain the original first P8 correction, original TSA64 outer action, full12 angles, original streaming/collision coefficients, inner1e-4/max8, outer1e-5/max100 and all original acceptance checks. The new shell solves S A S z=S r with real original-A actions and16 newly constructed, fully qualified background mode factors. No stored exact response is injected.

Source0/4 are explicitly the discovery set for this choice. Source7/14 were not used for support selection but are historical K development sources, not independent heldout data.

## Full paid result

| Method | Setup s | Inner s | TSA64 s | Remaining s | Total4 RHS s | TSA64 calls |
|---|---:|---:|---:|---:|---:|---|
| one_shot_P8 | 3.099 | 3.524 | 14.178 | 2.065 | 22.866 | [3, 3, 3, 3] |
| repeat_P8 | 2.865 | 7.107 | 16.627 | 2.442 | 29.041 | [3, 4, 3, 4] |
| shell_second | 6.201 | 6.845 | 12.066 | 2.355 | 27.467 | [2, 3, 2, 3] |

The shell method is 20.12% slower than the strongest incumbent. It removes one TSA64 call on source0 and source7, but not source4/14. It is insufficient to beat the already-negative repeated-P8 control alone; the one-shot incumbent remains the decision baseline.

## Why it still loses

Compared with the incumbent, added setup is 3.101s, added projected-inner work 3.322s, and remaining overhead 0.290s. TSA64 work decreases by 2.112s. The net loss is 4.601s. Even deleting all setup from both timings leaves the candidate 1.500s slower online. More-RHS factor amortization alone therefore cannot rescue this implementation.

The remaining category includes original-A outer actions, Arnoldi/least-squares work, independent checks, observer FFT/copies and logging. It is not an isolated orthogonalization timer. Timings are single-order on a shared host, but the classical-call changes and original-equation acceptance are explicit; no reverse-order confirmation was triggered because the full-cost gate failed.

## Actual second-direction effect

| Source | Before second original residual | After second original residual | Shell norm reduction | Final TSA64 calls |
|---|---:|---:|---:|---:|
| 0 | 2.373114e-05 | 3.83578e-06 | 191.92x | 2 |
| 4 | 7.528894e-05 | 1.469831e-05 | 1268.67x | 3 |
| 7 | 3.347282e-05 | 6.696454e-06 | 276.92x | 2 |
| 14 | 7.537709e-05 | 1.561467e-05 | 1330.45x | 3 |

The shell direction sharply reduces the intended component, including on source7 not used to choose the support. On source4/14 the post-injection original residual remains above1e-5, so another classical call is still required. This is why error concentration alone cannot guarantee that a fixed projected direction crosses the final stopping threshold.

Second-input differences from the independently rerun incumbent are recorded, not artificially forced to bitwise equality: the largest source-normalized discrepancy is below9e-9. The unchanged first-injection paths satisfy the1e-12 bridge. Every final result is independently checked on original A.

Maximum original residual 8.63348874e-06 (limit1e-5), evaluated relative bound 4.56526966e-05 (limit1e-3), and relative field difference 3.28987231e-07 (limit1e-4). All16 shell mode original-A0 tests and signed-boundary zero/tiny tests pass. Production and sealed E sources are unchanged.

## Cost ledger and consequence

Entire process: 84.07s wall, 76.91s user CPU, 3.74s system CPU, peak RSS 1336.67MiB. Together with the preceding59.90s diagnostic, these two processes cost 143.97s, excluding code preparation/transfer and separately disclosed observer validation. All offline references and controls remain paid research work.

The next authorized distinction is not a wider cutoff or a tolerance change. Keep the now-demonstrated shell target but replace its expensive inner solve with exactly one existing eight-stage classical action, compared against the same unfiltered action at the same late point. The shell branch must pay its own projectors, and the unfiltered branch must not be artificially slowed. Original equation and incumbent remain unchanged. Earlier B8-first-composition and all-later-stage-reduction no-go results must remain in the audit; the late one-direction question is different.

No automatic shell expansion, angle selection, source resampling or model training was performed. All workers from this experiment exited normally, GPU remained unused, other research was not stopped, and no Library upload was attempted.

Raw SHA256: `b8689101c9d5777f6c0457bff63df84a89e4007a3fca5777a56efd2613704bec`.
