# Independent TSA algebra feasibility verification — 2026-10-02

The conservative FFT identity has a correct exact-arithmetic proof and a concrete observed reduction in work, but **the deployed-precision substitution gate failed**. Do not promote this implementation as an unchanged floating-point action, a solver improvement, or a measured speedup. No further experiment was performed by this reviewer.

I checked `receipts/rte/workstation_runs/classical_tsa_algebra_20261002/`. Its **48 unique rows** exactly cover three grids × saved IDs `[0,3]` × depths `[1,16]` × complex64/complex128 inputs × the existing physical source and a deterministic grazing/Nyquist probe. These are two physical media, not 48 independent draws. All outputs and recorded discrepancies are finite. The settings declare and save the gate before the check loops: both relative field difference and original-FP64-physical-image difference must be at most `1e-4` for complex64 input, or `1e-9` for complex128 input, with exact transform counts. I recomputed every pass/fail flag and all saved maxima.

**43/48 rows pass. All five failures concern physical-image parity of existing-source, depth16, complex64 inputs:**

| N | ID | Relative field difference | Relative original-physical image difference |
|---:|---:|---:|---:|
|64|0|2.9240e-7|1.3980e-4|
|128|0|7.7619e-7|2.5663e-4|
|128|3|3.4030e-7|1.0498e-4|
|256|0|4.8516e-7|5.3867e-4|
|256|3|1.8294e-7|2.1220e-4|

Field parity passes in all rows; the global complex64 field maximum is `5.039646566729061e-6`, arising in a different row from the image maximum `5.386726405103414e-4`. All **24 complex128-input rows pass**, with maxima `9.391070762393184e-15` in the field and `1.4294981204936225e-12` in the physical image. These inputs retain the actual stored FP32 material, geometry and cached reference inverse. They are not a newly rebuilt all-double operator or a proposed deployment cost result.

Every transform count matches the code-derived prediction: depth1 changes six to four spatial transforms; depth16 changes **126 to 94**, a **25.3968% transform-count reduction**. The helper retains the fifteen outer physical defect recalculations and the same reference/cache preparation. The exact identity uses the reference's actual `sigma0`, checks common stored streaming/phase/quadrature, and preserves the weighted phase action. It is not the more aggressive recursive-residual mechanism discussed in `operator_mechanisms.md`.

The failed N256/ID0 row changes the field by approximately `4.85e-7` while changing its relative physical image by approximately `5.39e-4`. Spectral differentiation can amplify small altered rounding components, consistent with this observation. No error-spectrum/directional decomposition was recorded, so the precise amplification mechanism is unproved. Nor does this disagreement establish which action is closer to the exact inverse. **No full FGMRES solve, final solution-error comparison, or latency comparison was run.** It therefore establishes neither whole-solver failure nor safety, nor a timing benefit. Exit code2 is the checker's intended failed-gate result after all rows completed, not an execution crash.

The four executed source hashes match the staged/retrieved bytes, and the three development-parent hashes match the preceding frozen diagnostic. Six focused tests and 58 subtests passed. `refresh_and_preservation.json` verifies 155 original sources, checkpoint identities and development parents without mismatches, but is a **preflight** receipt (`new_directory_exists=false`); final preservation should use the consuming executor's separate post-run verification. Seed9626 remains unused, and the checker writes only its isolated output files.

The justified reopening is narrow: a **new precision-aware classical implementation** may be assessed under the same predeclared parity limits, initially on these retained rows with additional independent verification fixtures. It must account for its actual extra precision/casts/memory cost, preserve the original physical operator/tolerance, and pass the gate before any frozen full-solver timing study. Retesting a changed implementation under an unchanged gate is legitimate; raising the `1e-4` limit, dropping failed rows, or selecting a different image norm after seeing this result would not establish the original substitution claim. A future proposal to change the solver mechanism instead of preserve it needs a separately declared full-solver correctness/cost protocol, not a retroactive reinterpretation of this failed test.

A learned linear or coarse campaign needs **new oracle evidence**, not this work-count identity: a classical, physically residual-aware compressed action must first retain convergence/accuracy across relevant source and trajectory directions while reducing actual setup/storage/application cost against the strongest classical control. Prior exact coarse and hierarchy results did not supply that cost headroom. No new training, teacher generation, held-out access or coarse campaign is supported by this receipt set. Preserve the failed FP32 result alongside the exact proof and positive work-count finding.
