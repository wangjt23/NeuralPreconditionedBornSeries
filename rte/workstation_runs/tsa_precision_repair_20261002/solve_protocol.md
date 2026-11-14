# Conditional six-pair accuracy and warm-cost protocol

Protocol design only, 2026-10-02. No experiment or code change was performed by this reviewer. The existing 48-case algebraic-TSA gate failed five FP32 physical-image rows; that failure remains intact. This document permits no baseline promotion, training, new source family, or access to seed 9626.

## Mandatory entry gate

Run the solve comparison **only if all 48 rows of the newly frozen stable implementation pass** the original parity limits, including finite outputs, expected spatial-transform counts, both field and unchanged original-FP64 physical-image differences: at most 1e-4 for complex64-input rows and 1e-9 for complex128-input rows. Require exactly the same six material/grid cases, two probes, depths 1/16, and stored-parent identities. Do not drop the five former failures, raise limits, or substitute a different image denominator.

Before that gate, freeze the stable implementation hash, source/parent hashes, depth16/shift/relaxation, and its complete dtype policy: residual input, G application, local cancellation, scattering, homogeneous inverse, accumulation, output, and cache. State which stored FP32 tensors are merely promoted and which computations use wider arithmetic. Regenerating ideal directions/phase or replacing the original physical operator is outside this protocol. Cache validation/refresh and the precision costs are part of the implementation.

If the 48-case receipt is missing, incomplete, mismatched, or fails any row, record `solve_gate_not_entered` and stop. Passing it supplies feasibility on those probes, not whole-solver accuracy or speed evidence.

## Frozen cases and solver

The only physical cases are development IDs **0 and 3** from the already saved seed-4626 parent, at **N64, N128, N256**: six paired case comparisons, representing only two independent media on nested grids. Verify all seven material/source fields and saved-parent hashes. No new sampling or teachers.

Main arms are original `ClassicalTransportStack(..., kind='tsa', depth=16, eta_scale=0., relaxation=1.)` and the frozen stable variant at the same settings. Use the existing zero-initialized classical solver path (`protocol.solve('tsa16_zero', ...)` / `solve_from_initial`) and its original real-stacked FGMRES, restart25, maximum100 directions, original-RHS tolerance1e-5, and independent final FP64 physical check. Replace only the preconditioner arm; keep the same original FP64 A and RHS, stored geometry, zero initial state, source norm, and solver acceptance/stopping code. Do not use an instrumented/profiled clone for timing or change the residual checks to save time.

Retain the deployed input/output conversion wrapper. Any internally wider stable arithmetic or changed output precision must be explicit and timed; it is not an all-double rebuilt physical problem. Different floating-point preconditioners may produce different trajectories or stopping iterations despite passing the operator gate. Record this instead of demanding bitwise trace equality.

For cost context, retain the already frozen per-grid portfolio: reference-tail at N64/N128 and TSA16-zero at N256. The N256 context is the main baseline itself. If contemporaneous context is needed, evaluate only the frozen reference-tail on these same four N64/N128 case inputs; no additional methods, cases, tuning, or portfolio reselection. Saved earlier context may also be shown, clearly separated by run/date and never used as a paired timing denominator.

## Accuracy measurements and acceptance

Reuse the first untimed warmup output for each main arm; accuracy analysis occurs outside timing windows. For all warmups/timed solves save status and independently evaluate finite fields and

    rho = ||rhs - A_original,FP64(x)|| / ||rhs||.

Both arms must have `failed=0` and rho <=1e-5 on every solve. A recomputed failure stops further timed cases; preserve the result and do not loosen the tolerance or reclassify it as solver success.

For each pair save final fields/traces or their verified receipts and report D_x = ||x_stable-x_original||/||x_original||, actual absolute field difference, angular-mean difference, final residual vectors/norms, iterations, classical calls, and any breakdown/stagnation. Require **D_x <=1e-4** on each case. Report an angular-mean relative difference only if its denominator is nonzero and well scaled; also report its absolute difference normalized by the full-field norm so a nearly cancelling mean cannot make the table misleading. Final-field parity is distinct from preconditioner-output/image parity.

Evaluate the existing stored-phase coercivity estimate on the unchanged operator: mu = lambda_max((P+P*)/2), m = min_x(sigma_t-mu sigma_s), delta = ||rhs-Ax||/m. Require finite m>0, ||x||>delta, and evaluated relative bound delta/(||x||-delta) <=1e-3 (0.1%) for both arms on each case. This is a predeclared conservative continuation limit, not a continuous-error or rigorous interval certificate. Preserve the theorem assumptions and lack of directed rounding. The consistency inequality ||x_stable-x_original|| <= delta_stable+delta_original, allowing disclosed FP64 evaluation roundoff, is an additional diagnostic. It is not an independent reference solution.

Do not infer that the stable arm is closer to the exact inverse merely because it uses FP64 internally or disagrees less with a probe. Both returned solutions must satisfy the same original physical checks; no teacher acquisition is authorized to rank their exact errors.

## Distinguishing amplification and cancellation

Use the existing parity rows and common input fields, especially the retained N256/ID0 physical-source depth16 case. Do these diagnostics outside solve timers; do not feed them back into candidate selection or retry multiple precision policies.

1. For the two action outputs, form delta_z = z_stable-z_original in FP64. Measure ||A delta_z||/||delta_z|| and ||z_original||/||A z_original|| separately. Their product explains the ratio between relative image and field differences in exact arithmetic. Compare A(delta_z) with A(z_stable)-A(z_original) to expose cancellation in the diagnostic itself. Record the small FP64 closure discrepancy rather than treating the identity as bitwise exact.
2. Decompose A(delta_z) into the unchanged streaming and local collision terms. Report ||S delta_z||, ||C delta_z||, ||A delta_z|| and their cross term. Inspect delta_z Fourier energy in fixed **physical** frequency bands shared across grids and in newly available bands; record actual ordinate |Omega dot k|, including grazing directions and the stored Nyquist convention. High-k energy plus a large streaming amplification factor supports high-frequency derivative amplification. A large (||S delta_z||+||C delta_z||)/||A delta_z|| indicates image cancellation. Neither alone attributes the discrepancy to a particular kernel.
3. At a **common** input r and common rounded g=G(r), evaluate the eliminated identity defect e_G = r-(S+gamma I)g using both its actual FP32 evaluation and an FP64 evaluation of those same stored/rounded fields. Compare the original defect r-A g with the local defect (gamma-sigma_t)g+sigma_s P g. Separate their difference from e_G and report a cancellation scale (sum of term norms divided by resulting defect norm, with absolute norms when the result is nearly zero). This distinguishes inverse/division/FFT identity error from local subtraction/reassociation sensitivity. Promoting the same g isolates evaluation precision; recomputing g in FP64 is a different diagnostic and must be labelled separately.
4. If the stable implementation uses compensation/promotion, record what component it actually changes and the residual disagreement it leaves. Optional one-case stage receipts can use the original common baseline stage inputs; comparing two already-diverged stage inputs cannot isolate a local rounding cause. Do not claim all five legacy failures were explained by the Nyquist stripe: that stripe passed while physical-source depth16 rows failed.

## Warm timing, setup, and resource receipts

Use one process and the existing FWI GPU/runtime. Record GPU/user-job state before and after; stop at the next safe boundary if an unrelated research job appears or contention makes the comparison uninterpretable. Do not cancel another job or repeat a campaign to obtain cleaner numbers. Use a soft **240-second** process budget checked between bounded solves; preserve incomplete receipts without a scientific pass/fail decision.

Build the two arms fairly from the same low/physical objects. Separately time physical construction, original preparation, stable preparation/precision casts/extra caches, first application, and process startup/imports. Distinguish shared cached data from method-specific work. A cache or conversion performed only on first use belongs to preparation/cold latency, even when the steady-state apply is cheap. Record working/storage dtypes, allocated/reserved/peak memory where available, and casts or copied bytes; end-of-run occupancy is not peak memory.

For each case, run **two untimed warmups per main arm**, then **five paired repetitions**. Predeclare a balanced AB/BA order across the thirty paired timing blocks, shuffled with one recorded timing-order seed unrelated to data generation. Synchronize before starting and after ending each complete original-solver invocation; include all preconditioner work, casts, physical actions, least-squares, CPU scalar synchronization, final physical checks, and temporary allocation. No per-stage synchronization or monkey-patched FFT counter belongs in these timers. Reduce repetitions to a median per case before summarizing. The main bound is 6 cases x 2 arms x (2+5) = **84 full solves**.

If contemporaneous portfolio context is taken, use two warmups and three repetitions of frozen reference-tail on the same four N64/N128 cases: at most **20 additional solves**. Keep these receipts separate from the balanced primary pairs. No new broad benchmark. Total maximum is 104 full solves; all reuse the six frozen physical inputs.

Report raw times/order and case medians, paired per-case saving, per-grid two-case mean saving, setup-plus-first/warm-solve cost, calls/iterations, and true residuals. Six grid-case rows are not six independent media. Five repeats quantify timing repeatability, not scientific sample size; no confirmatory paired-bootstrap claim or OOD claim is warranted from two reused media. Label synchronized profiles, if any already exist, as explanatory and perturbed; never substitute their sums for these warm measurements.

## Decision and stopping rule

Accuracy is mandatory before cost interpretation. Any 48-row entry failure, nonfinite field, original-RHS convergence failure, final-field difference above1e-4, or evaluated error bound above1e-3 stops this candidate under the unchanged protocol. A failed parent/operator gate or incomplete run is not evidence of speed or whole-solver correctness.

For an accuracy-passing candidate, continue only to an **exploratory implementation opportunity** if the per-grid paired warm means show at least **5% saving at both N128 and N256**, no individual case has a warm regression greater than10%, and setup-plus-one-solve means at both grids do not regress. Compare with the frozen portfolio context too: a gain over original TSA16 at N128 does not establish a gain over reference-tail. More/fewer classical calls are recorded and fully charged; fewer FFTs per call are not a timing result.

If those cost conditions fail, retain the stable variant, dtype ledger, failures/negative timings, and stop it as the current speed candidate. If they pass, retain the original validated baseline and report only six-pair development evidence; **no baseline promotion, further training, new test access, or broadened campaign follows from this protocol**. Precision widening may recover parity while erasing the saved-transform benefit. Process startup, preparation, first use, warm latency, and hypothetical repeated-distinct-RHS amortization remain separate quantities.
