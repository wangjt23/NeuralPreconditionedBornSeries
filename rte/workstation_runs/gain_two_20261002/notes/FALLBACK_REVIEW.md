# Conditional fallback: complex-coefficient classical Krylov

Prepared 2026-10-02 from existing local original-project code/notes only. This is a conditional review **if the fixed beta2 control fails**. The [frozen mechanism/protocol](MECHANISM_AND_STOP_RULE.md) is unchanged. No experiment, new metric for this round, SSH, original-source export, training, scalar/gain sweep, Library mutation or seed9626 access occurred. The proposed future discriminator below needs its own frozen implementation/authorization.

## Assessment and existing evidence

**A matched complex-coefficient classical control is a warranted missing baseline check before a broad learned-superiority claim, with unproved economic benefit.** It changes the outer minimization freedom while reusing the already prepared transport factors. It is a more immediate inexpensive question than acquiring another coefficient teacher bank or repeating an angular-range oracle. It is not logically required to complete the current fixed-real-solver beta2 comparison, and must not retroactively change that round's comparator, metrics or solver semantics.

The original [fgmres_with_trace](../../remote_reference/rte/npbs_unet_ablation.py) already implements conjugate complex inner products, a Hessenberg matrix with the RHS dtype and complex least-squares coefficients. Its `fgmres_real_stacked_with_trace` wrapper concatenates real/imaginary channels and calls that same routine on a real problem. [augmented_fgmres](../../remote_reference/rte/augmented_fgmres.py) explicitly uses real channels, real MGS coefficients and a real tiny LS problem. N256/TSA16-zero uses the original real wrapper through [source_initialization](../../remote_reference/rte/source_initialization.py); N128/reference-tail uses the real augmented solver through [benchmark_tail_solver](../../remote_reference/rte/benchmark_tail_solver.py).

This algorithm is therefore already present, not a novel proposal. The preserved [project audit](../../remote_reference/docs/research/PROJECT_AUDIT_2026_09_12.md) explains the common real-stacked protocol for earlier real-linear/nonlinear networks. The [CT architecture note](../../remote_reference/docs/research/RTE_COLLISION_TRANSPORT_ARCHITECTURE.md) explicitly denies the complex-linear CT map any special complex-Krylov advantage. The locally preserved historical-evidence metadata contains180 real-stacked rows and4 unspecified rows, all N64; no complex-mode row supplies the current matched N128/N256 comparison. I found no such comparison in the current preserved notes/receipts. This is a bounded evidence statement, not a claim that no historical remote complex run ever existed.

Complex FGMRES can also be valid with flexible/nonlinear preconditioners: each actual returned z has a physical image H z, and complex linearity of **H** supports fitting those images with complex coefficients. Complex linearity of M is needed for the simple fixed-Krylov inclusion and some symmetry arguments below, not for all flexible-GMRES validity. Thus historical preference for a common real protocol is not a theorem that a nonlinear preconditioner requires real coefficients.

## Correct mathematical mechanism

For the same physical starting field x0, r0=b-H x0 and a fixed complex-linear right preconditioner M, let T=H M. In unrestarted exact arithmetic the real coefficient method seeks corrections in

    M span_R{r0,T r0,...,T^(k-1)r0},

whereas complex GMRES can use the containing complex span. At matching degree k,

    min_complex ||r0-H delta_x|| <= min_real ||r0-H delta_x||.

This is a search-space/minimization statement, not guaranteed fewer calls or time. It does not compare divergent restarted trajectories, changing/flexible M, the N128 residual-conditioned reference injection, different initializers, absolute breakdown thresholds or rounded FP32 actions. Strong TSA16-zero is the clean nominal fixed-M control. N128/reference-tail needs an empirical matched complex-augmentation control retaining the same physical recipe; substituting an unaugmented generic solve would confound the comparison.

The real-coefficient solver is **not automatically deficient in global phase equivariance**. With H and M commuting with multiplication by gamma=exp(i theta), rotation of the complete state rotates all fields while preserving real inner products, norms, real Hessenberg entries and LS coefficients. The real solver can already be phase-equivariant, including restarts and a covariant injection. This follows the existing [acceptance/phase note](../followup/acceptance_phase_note.md). Complex coefficients add minimization degrees of freedom, rather than repairing a demonstrated symmetry defect. Four phase aliases are neither independent sources nor evidence of a new workload.

There is also an important possible null result: the original [source generator](../../remote_reference/rte/run_large_angular_spectral_experiments.py) returns real-valued localized sources in complex storage. If H, M and the initializer preserve a common real invariant subspace, all Krylov inner products/Hessenberg entries are real and the methods coincide in exact arithmetic. However, the original even-grid `fftfreq` streaming convention keeps its unpaired Nyquist mode; its imaginary multiplier can send a real spatial Nyquist field to a complex field. Stored geometry and FP32 rounding are retained. Do not assume either exact real-subspace invariance or materially useful imaginary freedom without inspecting actual physical images. No Nyquist/geometry modification is part of this fallback.

## A cheap correct span diagnostic

For retained actual solution directions Z and **unprojected** physical images W=H_h Z, solve a rank-revealing small fit

    eta_R=min_(a in R^k) ||r0-W a||,
    eta_C=min_(c in C^k) ||r0-W c||
         =min_(a,b in R^k) ||r0-[W,iW](a,b)|| <= eta_R.

This isolates missing coefficient freedom in those measured columns with no extra preconditioner/teacher solve. It is an attained frozen-span result, not the optimal complex Krylov trajectory. Complex Gram/QR uses conjugate transposes and complex rank handling: columns independent over R can be dependent over C, e.g. w and i w. A real-orthonormal image basis is not necessarily complex-orthonormal.

The existing prefix callback retains solution/residual and a small-Hessenberg-derived **real** image basis, but not Z. That basis can support an exploratory complex-span fit. It cannot independently reconstruct/verify a physical combined field. A later instrumented diagnostic must retain actual Z in memory, recompute/verify W as needed and evaluate H_h(x0+Z c) against the original b. Store additivity/rank/orthogonality and actual-field discrepancy, without interpreting reconstructed images alone as an accepted correction.

A fixed real-AZ projector J_R generally does not commute with i. Therefore `i J_R(W)` is not the physical projected image of i Z. For old real coefficients plus a new complex coefficient, project W and iW **separately**. For complex freedom on old and new coefficients, rebuild their entire complex image span; retain the original r0/x0 rather than treating the old projected residual as a new zero-start problem. The one-complex-scalar fit to an already real-projected image is not this diagnostic. No raw coercivity or singular-value lower bound transfers through the old real projection.

## Cheapest decisive future falsifier and billed cost

The span diagnostic can explain a mechanism; a paired complete accepted solve is the decisive economic falsifier. If separately authorized, freeze **one complex-coefficient implementation** against the unchanged successful real classical recipe on existing IDs0/3 at N128/N256, using the same stored H_h/RHS, source normalization, material, FP32 preconditioner/caches and chosen capture backend. Keep N128's balanced initialization and one original full-reference tail proposal after four classical calls; keep N256's TSA16-zero start. Do not replace those starts or drop the injection to make the solver comparison easier. The complex augmented path needs independently reviewed conjugate MGS/QR, complex dependence and happy-breakdown logic, physical rejection checks and original final stopping. Original generic complex FGMRES can provide the N256 authority but its default20-step harness is not a matched25-restart replacement.

Two solver arms x four contexts x one warmup/three measured repeats would be32 complete solves, contingent on future authorization and entry correctness; no all-depth/network/phase campaign is required. Complex arithmetic need not match real fields bitwise because its search space intentionally changes. It must satisfy the same finite-field, zero-failure, original-H residual<=1e-5, evaluated discrete relative-error bound<=1e-3 and paired final-field difference<=1e-4 checks. Hold restart25/max100 and the original-RHS stopping threshold fixed; retain failures, easy/already-converged cases and all attempts.

Measure accepted warm **whole-solve** time and setup-inclusive Q1. Charge unchanged physical/material/inverse/capture setup plus every changed conversion/copy, complex reduction/orthogonalization/reorthogonalization, tiny LS, injection/failed attempt, independent physical residual and field acceptance. Complex vectors have the same real storage dimension as real-stacked fields, but arithmetic/reductions and LS differ; equal calls or basis counts are not equal costs. Do not replace required physical checks with Hessenberg residual estimates to manufacture savings. Counterbalance timings and preserve first-use/setup scope; diagnostic Z/W/fits are separately billed. Reuse of prepared factors requires an actual declared multi-RHS workload, not phase aliases or assumed Q16. This does not add training amortization or process-cold claims.

Apply the existing economic standard, rather than inventing a favorable metric: >=10% mean Q1 and warm improvement at both grids, no individual Q1 regression>10%, against the current strongest successful classical portfolio, which includes beta2 if it is numerically valid even when uneconomic. Also retain the same-action real/complex comparison to attribute the mechanism. Success refreshes the classical denominator for every future learned claim; it is not a learned-preconditioner win. A negligible frozen-span gap only limits that recorded span; a failed full matched economic gate limits this solver variant/cohort, not every complex Krylov strategy.

## Ranking against another angular/compression route

Rank this outer-solver check first **conditionally on beta2 failure**, because it reuses all prepared physics and asks whether extra continuation work came from a constrained coefficient field. It does not assume that the earlier q3/full-range failures were genuine optima. The main risk is no useful imaginary freedom on the real-source cohort, or added complex solver overhead cancelling fewer calls.

A second, materially different direction would be a cheaper all-angle fixed-material factorization/coefficient representation of the response action, with a declared setup/application ceiling. Its positive witness must be literal physical-image progress and a complete accepted cost advantage; prior two-proposal fits and capped angular oracles supply no range optimum. Repeating a q3/angular CGLS acquisition, expanding q, or training compression without an affordable representation would recur to the preserved negatives. If the matched complex solver also provides no charged headroom and no independently cheaper representation is specified, retain the precise tested-family economics boundary instead of starting another campaign.
