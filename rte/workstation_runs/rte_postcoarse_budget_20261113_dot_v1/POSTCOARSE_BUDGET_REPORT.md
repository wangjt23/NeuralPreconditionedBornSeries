# RTE: remove unnecessary classical work after coarse correction

13 November 2026. Fixed K material, N128, 12 angles, four original RHS. Classical CPU result; not neural acceleration or material-generalization evidence.

## Why this test

Repeating P8 did not help, and adding eight transport stages to the existing correction improved residual quality without removing any full 64-stage classical call. This experiment instead asks whether the coarse correction makes long subsequent classical calls unnecessary. All original operator, source/material, inner and outer accuracy conditions remain unchanged.

Methods:
- coarse64: the incumbent, 64 transport stages in every outer classical call, with one P8 correction.
- coarse64_then8: identical first 64-stage call and P8 correction, then eight stages per classical call.
- plain64_then8: the same stage schedule without the P8 correction or its setup.
- coarse8: eight stages from the first call, plus the same P8 correction.

TSA8 is the previously bitwise-bridged action, not an unverified depth label. The source records every actual stage and leaves the stored outer object unchanged at depth 64. Raw outer iteration counts are deliberately not used as a cost proxy.

## Full paid comparison

| Label | Setup s | First accepted RHS s | Four RHS s | Outer calls | Actual transport stages per RHS |
| --- | ---: | ---: | ---: | --- | --- |
| 01_coarse64 | 3.005 | 7.773 | 22.745 | [3, 3, 3, 3] | [192, 192, 192, 192] |
| 02_coarse64_then8 | 2.834 | 7.753 | 24.292 | [12, 14, 12, 15] | [152, 168, 152, 176] |
| 03_plain64_then8 | 0.137 | 6.834 | 28.206 | [25, 26, 25, 25] | [256, 264, 256, 256] |
| 04_coarse8 | 2.709 | 8.457 | 24.625 | [18, 18, 18, 18] | [144, 144, 144, 144] |

Selected observed alternative: coarse64. Selection is exploratory on the same development material, not a fresh holdout. A predetermined reverse-order selected/incumbent pair was run only if an alternative beat the incumbent.

## Where saved transport work went

| Label | Preconditioner s | Coarse inner s | Remainder outside setup/preconditioner/inner s |
| --- | ---: | ---: | ---: |
| 01_coarse64 | 14.052 | 3.644 | 2.045 |
| 02_coarse64_then8 | 11.933 | 3.406 | 6.119 |
| 03_plain64_then8 | 18.810 | 0.000 | 9.258 |
| 04_coarse8 | 10.517 | 3.621 | 7.778 |

The remainder is directly derived by subtracting disjoint recorded timers. It includes outer original-A actions, Arnoldi/least-squares work, checks, diagnostic copying and logging; it is not an isolated orthogonalization timer. A rise cannot be attributed uniquely to one kernel.

## Acceptance and cost scope

All 16 solutions passed the original criteria. Maximum original relative residual 9.62030521e-06 (limit 1e-5); evaluated relative-error bound 4.50603232e-05 (limit 1e-3); field difference from the newly accepted incumbent 3.50769137e-07 (limit 1e-4). All original source/material/background and sealed code identities remained unchanged.

Every method pays fresh physical/operator/reference setup. Coarse variants pay fresh P8 factor construction, all-mode qualification and zero/tiny checks, inner solves, outer work, diagnostic copies and original-A acceptance. The no-coarse method pays no coarse costs. Reference sharing occurs only inside the same method. Accepted incumbent fields are used only for comparison, not for initialization or directions. Import/interpreter and common bank reading are disclosed outside method clocks.

Recorded numerical research wall: 100.908 s. One CPU thread retains the existing benchmark workload; shared GB processes were not interrupted. No GPU locks, paid resources, security changes or Library upload. No GPU performance claim.

## Interpretation

This test distinguishes reducing unnecessary transport work from adding a new correction. More outer calls can still be cheaper when each contains eight rather than 64 transport stages; the stage ledger and complete wall time resolve that directly. The matched plain64_then8 control tests whether the stage change alone suffices for the switched candidate. If coarse8 is selected, a matching plain8 baseline remains necessary before attributing its benefit specifically to the coarse correction.

The result may strengthen a classical baseline. It is not by itself a new neural architecture, evidence that correction coefficients can be learned cheaply, or a universal RTE improvement. Preserve the original negative experiments and compare future learning against the best complete-cost classical path.

Evidence: DESIGN.md, executed driver, prior input identity record, raw results/results.json and this scalar-only analyzer. Existing FWI dependencies and original source bank remain required.
