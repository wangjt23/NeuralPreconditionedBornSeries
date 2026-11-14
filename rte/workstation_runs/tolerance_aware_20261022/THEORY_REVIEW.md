# Original-RHS tolerance awareness for the fixed TSA16 tail experiment

22 October 2026. Scope: theory and inspection of existing source/scalar receipts only. No FWI access, new PDE solve, optimizer update, model evaluation or GPU work was performed for this review. This is a proposed discriminating objective, not evidence of a speed improvement or publication readiness.

## 1. The exact mismatch in the old loss

Let `q_i` denote the original right-hand side and `r_i` the residual after the fixed four-call TSA16-zero prefix. Let `eta_i = ||r_i|| / ||q_i||`. The saved training contexts are eligible only when `eta_i > tau`, with `tau = 1e-5`.

`rte.tail_lookahead.continuation_loss` returns `rho_ik = ||r_ik||^2 / ||r_i||^2`, for k = 0, 1, 2: residual immediately after injecting the model direction, then after one and two further classical calls. It uses the real-stacked inner product, retained search-image space and two-pass image orthogonalization. The old residual objective is `mean_ik log(max(rho_ik, 1e-12))`; both objectives should retain the identical `+ 0.05 * teacher_direction_loss` term.

The original-source residual is `R_ik = eta_i * sqrt(rho_ik)`. Its stopping threshold in prefix-normalized coordinates is `rho_ik <= (tau / eta_i)^2`. Define the dimensionless signed margin

`d_ik = log(max(rho_ik, 1e-12)) + 2 log(eta_i) - 2 log(tau)`.

For an unclamped ratio, `d_ik = log(R_ik^2 / tau^2)`. Positive margin means above tolerance. These are squared residual ratios; the unsquared solver tolerance must enter as `tau^2`.

**Proposition 1 (a normalization change alone is a null ablation).** Replacing `log(rho_ik)` by `log(eta_i^2 rho_ik / tau^2)` in an unweighted mean, with fixed eta, gives exactly the same parameter gradient. The difference is a data-only additive constant. This remains true with the same clamp on rho before the logarithm.

Proof: the new objective differs by the mean of `2 log(eta_i / tau)`, whose derivative with respect to model parameters is zero. Merely renaming this expression “tolerance-aware” cannot test the proposed hypothesis.

## 2. A consequential normalization trap in the current helper

`deliverables/train_tail_multiresolution.py:normalize_and_cache` divides `q`, `rhs_exact`, `target` and `reference` by the prefix residual RMS. It does not scale the context's `original_rhs_norm` metadata, and it discards its local RMS variable. Image-basis vectors are correctly left unit-normalized.

In these context dictionaries, the field named `q` is the complex64 prefix residual proxy, not the original physical source q_i. Consequently, neither a norm of that context `q` nor `norm(normalized_rhs_exact) / original_rhs_norm` is eta. Save `eta = norm(raw_rhs_exact) / original_rhs_norm` before normalization, detach it, and retain it as a per-physical-ID scalar through restrictions, caching, sampling and phase augmentation. Alternatively retain the raw RMS and reconstruct the same ratio exactly; do not substitute an RMS for an Euclidean norm without the square root of the number of degrees of freedom. Verify eta against the previously saved prefix residual CSV by original parent ID. Its value is invariant under unit-modulus complex phase augmentation and under a common rescaling of the original equation.

The cached `inverse`/`stream` construction uses TSA8 internally but `FrozenTSA(..., depth=16)` supplies the prescribed number of TSA iterations in the TSA16 arm. The parent must verify current source hashes before treating the older copied sources as live code.

## 3. A minimal objective that is genuinely different

For a single two-arm causal experiment, keep the existing lookahead horizon exactly two and replace only the residual scalar reduction with the smooth log-excess objective

`L_tol = mean_ik beta * softplus(d_ik / beta) + 0.05 * mean_i teacher_direction_loss_i`, k = 0, 1, 2, with `beta = log(4)`.

The parent froze beta=log(4) in log-squared-residual units (a factor of two in residual norm changes the margin by one beta). This avoids a new tuning sweep, and uses the same three residual snapshots as the matched old objective. Evaluate softplus by a stable implementation; eta is detached. Keep the same ratio floor, architecture, 56 original physical IDs, batch size, phase ledger, warm weights, optimizer, learning rate, clipping, scheduler horizon, number of updates, solver validation and checkpoint rule in both arms. A matched old-objective arm is a necessary control, not a claim that rerunning it is a new research direction.

**Proposition 2 (explicit gradient reweighting).** Above the ratio-clamp floor, the residual-gradient contribution at a snapshot changes from `grad_theta log rho_ik` to `sigmoid(d_ik / beta) * grad_theta log rho_ik`. Both objectives are independent of arbitrary common equation scaling. The scalar multiplier approaches one far above tolerance, one half at tolerance and zero far below tolerance.

Proof: differentiate softplus and the definition of d. Eta and tau have zero model derivatives. Under `q,r -> c q,c r`, eta and rho are unchanged. For any d, `max(d,0) <= beta*softplus(d/beta) <= max(d,0) + beta*log(2)`, so this is a smooth log-excess surrogate, not a literal expected-call loss.

The mechanism tested is therefore precise: stop allocating comparable residual-loss gradient to snapshots already safely below the deployment tolerance, while retaining a nonvanishing gradient on difficult, unconverged snapshots. A positive result would support threshold-sensitive allocation at fixed architecture/data/horizon/budget. It would not isolate phase augmentation, model capacity or longer rollout.

Teacher regularization is intentionally unchanged, so total gradients need not vanish on solved snapshots. The residual-only gradient weights and total gradient alignment must be logged separately; otherwise an unchanged teacher term can mask whether the ablation has any effective distinction.

## 4. What a soft survival sum can and cannot claim

Let K be the number of additional classical calls after injection until the first original-RHS residual `R_k <= tau`. For a nonincreasing exact residual sequence, and cap m,

`min(K,m) = sum_{k=0}^{m-1} 1{R_k > tau}`.

If no crossing occurs by m, use the capped value m. The proof is to count the integer k strictly preceding the first crossing. Taking an expectation over physical media gives the tail-sum formula for the expected capped call count. This requires a stopping-consistent sequence: either residuals are nonincreasing or all later states are treated as absorbed after the first crossing. For a nonmonotone trajectory, `1{R_k>tau}` alone is wrong after a prior crossing; use `1{min_{j<=k} R_j > tau}`.

For nested exact least-squares search spaces without restart, the orthogonal-projection residual norm is nonincreasing. The present prefix four plus injection one plus lookahead two is below restart 25. Nevertheless, finite-precision orthogonalization, near-dependent proposals, injection rejection, breakdown and deployed explicit residual checks can break agreement between the offline continuation and the production solver. Actual traces, not this identity alone, determine production calls.

A bounded smooth surrogate is `sum_{k=0}^{m-1} sigmoid(d_k / T)`. It approaches the capped count as T tends to zero at points away from exact equality. It is not an unbiased expected-call estimator at positive T. A snapshot exactly at the stopping boundary contributes one half even though the solver accepts equality. Its scalar gradient `sigmoid(d/T)*(1-sigmoid(d/T))/T` vanishes both far above AND far below the threshold; this can starve difficult examples of gradient. Smooth log-excess avoids starvation far above tolerance, at the cost of optimizing log excess rather than bounded count.

For m=2, the capped count uses k=0 and k=1 only. Adding k=2 is a terminal-progress penalty, not a third counted call. A hard or soft capped count also treats all examples still unconverged beyond its horizon nearly alike; it cannot distinguish seven remaining calls from three. Extending the rollout to fix that limitation would be a separate horizon ablation. Do not silently compare an extended-horizon tolerance loss against the old two-step control.

## 5. Counterexamples and limits on interpretation

1. **Geometric residual improvement can rank stopping times backwards.** Consider one medium with tau scaled to one. Candidate A has residuals `(1.01, 1.005, 0.01)` and candidate B `(0.99, 0.98, 0.97)` after injection and subsequent hypothetical calls. Both sequences are nonincreasing. The old mean log-squared objective strongly favors A because of its very small terminal residual, but B stops immediately whereas A needs two more classical calls. This demonstrates an objective mismatch; it does not prove the proposed surrogate ranks every pair correctly.

2. **Even log-excess is not an exact call-ranking objective.** Candidate A with `(1.001,1.0005,0.1)` has smaller total positive log excess than candidate B with `(100,0.1,0.01)`, although A needs two additional calls and B one. Thus a log-excess loss win is insufficient to claim a speed win.

3. **A low final normalized ratio does not imply an accepted solution.** Two physical media can have the same rho but different eta. The one with a larger prefix residual can still exceed the original-source tolerance. Eta cannot be dropped from a tolerance-aware reduction.

4. **Gradient reweighting can be ineffective.** If all relevant d are large and positive, sigmoid(d/beta) is nearly one and the proposed residual gradient is almost the old gradient. If all relevant d are very negative or clamp-saturated, the residual gradient can be negligible and the unchanged teacher term can dominate. Collinear or proportional residual snapshot gradients can also make their weighted sums parallel even with visibly different scalar weights. Actual gradient angles and norms are needed.

5. **Numerical work is not wall time.** An injected direction can reduce classical calls yet remain slower after reference initialization, inference, projection, residual checks and setup. The old result of 4.500 to 4.375 mean classical calls already demonstrates that call-count improvement alone does not clear the speed gate.

## 6. Existing scalar evidence of threshold exposure

The saved `alignment_retry/train_tsa16_n128.csv` contains 56 eligible training IDs. Prefix original-source residuals range from `1.1076506e-5` to `9.1325054e-2`, median `2.9627502e-4`. The 56 matched original IDs are frozen in `alignment_retry/settings.json`.

Joining that prefix CSV by original physical ID to the 16-case N128 fixed training panel, without evaluating a model or solving a PDE, gives:

| Checkpoint | Snapshot k | Above original tau | Within [tau/2, 2*tau] | Range of R_k/tau |
|---|---:|---:|---:|---:|
| Original warm weights (step 0) | 0 | 16/16 | 1/16 | 1.689 to 604.890 |
| Original warm weights (step 0) | 2 | 6/16 | 5/16 | 0.01292 to 74.442 |
| Original warm weights (step 0) | 4 | 3/16 | 1/16 | 0.0000731 to 9.401 |
| Original warm weights (step 0) | 6 | 0/16 | 2/16 | 0.000000453 to 0.794 |
| Earlier matched TSA16 step 1000 | 0 | 16/16 | 1/16 | 1.675 to 610.909 |
| Earlier matched TSA16 step 1000 | 2 | 4/16 | 5/16 | 0.01288 to 35.768 |

These are offline projection residual estimates reconstructed from saved scalar outputs. They are not new production solver acceptance measurements. The immediate injection snapshot rarely reaches tolerance, but two-step snapshots include a mixture of solved, near-boundary and unsolved cases; that makes the proposed threshold weighting plausibly distinguishable. This is only 16 fixed training IDs, one grid and reused warm states. Snapshot k=1 was not retained in this CSV and cannot be inferred from k=0,2. No aggregate mean ratio should be substituted before multiplying by each example's own eta.

## 7. Freeze the discriminating preflight before spending the training budget

Under the existing resource controller and all required locks, the parent proposes one fixed phase-zero diagnostic on all 56 matched training IDs at N128, accumulated in bounded microbatches with the same warm weights. This covers the entire frozen training cohort at one grid without choosing favorable media. An additional N64 check is optional only if separately admitted. This preflight is numerical and must wait for admission; it is not part of this theoretical review. Do not tune its pass criteria or temperature using development timing.

Required checks:

- Assert raw eta matches saved prefix residuals by physical ID and remains unchanged after normalization and phase rotation. Assert the original-q residual conversion, eligible-ID list, source/checkpoint hashes and fixed input identities.
- On the same residual trace compute old reduction, the additive-constant null reduction and smooth log-excess. The null gradient must match the old gradient to a predeclared FP64/FP32-aware numerical tolerance. This is a positive implementation-control test.
- Log d at k=0,1,2; fractions below zero, in `[-log(4),log(4)]` and above zero; scalar weights sigmoid(d/beta); ratio-clamp saturation; teacher-only, old residual-only and new residual-only parameter gradient norms. Log residual-gradient cosine and the gradient difference relative to the old residual-gradient norm.
- Reject an uninformative run if all observed weights are indistinguishable from one, all gradients are nonfinite/zero, or the new gradient equals the old gradient up to a common scalar within the fixed tolerance. A meaningful diagnostic should show nonconstant threshold weights AND a nontrivial directional gradient change; it must not be replaced by scalar loss differences alone.
- Check offline k=0,1,2 residuals against explicit original-RHS residuals along an instrumented production trace for those same fixed examples, without changing accepted/rejected-injection handling. Near-dependent or rejected proposals must be recorded, not silently excluded. If the surrogate does not match production early-stop decisions, repair that semantic mismatch before training.

The parent freezes relative gradient difference at least 1e-3 AND `1-cosine` at least 1e-6 as a bounded operational admission criterion. These are diagnostic thresholds, not a theorem guaranteeing eventual speed or statistical power. Report both residual-only and total gradients including the common teacher term. Also report `min_{a>=0} ||g_new-a*g_old|| / ||g_old||` and the difference after the actual clipping rule. Pure common rescaling is insufficient evidence of an effective training change: all earlier observed gradients were clipped, and positive common rescaling cancels under norm clipping when both gradients exceed the threshold. Adam updates are also approximately scale-invariant away from epsilon effects. If the full frozen N128 cohort fails the direction criterion, preserve that negative diagnostic rather than searching development cases or changing beta after seeing the result.

## 8. Evaluation and evidence needed

One fixed-budget matched two-step arm versus one fixed-budget smooth log-excess arm is the minimal causal comparison. Their warm weights, 56 physical IDs, sample/phase ledger, optimizer and scheduler must be identical. Preserve step zero, final and selected checkpoints; record actual optimizer, diagnostic, validation, setup and timing costs. Do not select checkpoints with N256 timing. Reused seed-4626 media remain development data, not independent test evidence; the reserved seed 9626 stays closed unless the predeclared gate is passed.

Judge both candidates by original-source relative residual at `1e-5`, zero failures, at least 5% synchronized measured time savings at BOTH N128 and N256 versus the frozen strongest classical controls, and positive lower paired intervals. Reduce repeats to per-physical-medium medians before resampling; nested resolutions are the same media, not independent replicates. Keep the previous negative speed result. A failed gate with a discriminating preflight is useful evidence about the tested objective; a failed gate with a null gradient preflight is an uninformative implementation experiment.

## Frozen parent protocol note

The parent specified two arms from the same original mixed-grid update-800 warm weights, the exact same 56 physical IDs and TSA16 banks, seed 41022, batch four alternating N64/N128, 1000 updates at initial rate 1.5e-4 with cosine horizon 1000 and minimum rate 5% of the initial rate, unchanged teacher coefficient 0.05, and beta=log(4). New loss changes only the residual reduction. This review endorses that as a narrowly interpretable objective ablation, conditional on admitted numerical gradient discrimination and current source/data hashes. It does not authorize bypassing the Schur/pressure locks or controllers.

## Existing call-count headroom and the two-step censoring limit

The saved development TSA16-prefix CSVs at N64 and N128 each record 56 prefix classical calls across 16 media (mean 3.5), with only six media eligible for injection. The remaining ten terminate by the fixed four-call prefix and are unaffected by the learned tail direction. At N256, the saved production `tsa16_zero` case medians total 72 classical calls (mean 4.5); the aligned model totals 70 (mean 4.375). Taking each classical baseline's first four calls gives the same total prefix count, 56. An ideal accepted injection that terminates every eligible case immediately could at most remove the remaining 16 baseline classical calls across this cohort (mean 1.0), compared with the observed removal of two calls. This is classical-call headroom only: neural inference, reference solves, projection, physical operator actions and acceptance checks remain, and the bound is not a predicted wall-time saving.

For N256 baseline cases 0, 3, 4, 6, 7 and 14, the additional classical calls after the fixed prefix are respectively 3, 1, 6, 1, 2 and 3. Thus the two-step diagnostic horizon censors three of the six active tails before their natural baseline stopping times. The proposed frozen comparison still has value because its scalar weights differ on the same trace, but a negative result would reject this particular two-step smooth-excess objective at this budget; it would not establish that all original-tolerance-aware objectives lack value. The injected direction can change later search spaces, so these baseline tail lengths are descriptive and are not fixed ground-truth labels for learned stopping times.

A gradient-discrimination diagnostic is not a theorem of distinct Adam trajectories. In the epsilon-negligible fresh-optimizer limit, Adam's first update is proportional to each coordinate's gradient sign. Two non-collinear gradients with the same sign pattern can therefore give almost identical first updates. A near-zero first-step weight difference must not alone be labeled a null loss implementation; retain the fixed gradient tests and later predeclared checkpoint drift instead.

## Read-only implementation review of the first prototype

Inspected `code/objective.py` and `code/train_tolerance.py` after their creation; no training, solver, model or numerical diagnostic was executed for this review. The prototype correctly computes eta before residual-RMS normalization, applies the beta-scaled softplus to the same k=0,1,2 ratios as the old objective, and includes the same 0.05 teacher term. Four autograd evaluations share each forward graph, retaining it until the last call. Accumulating each batch-of-four mean with weight 4/56 gives the exact equally weighted 56-ID cohort gradient. Splitting complex gradients into real and imaginary components gives the real Euclidean parameter geometry used for the comparison. No loss-formula or cohort-weighting error was found.

Two guards are required before treating the diagnostic as an admission decision:

1. The first `gradient_comparison` implementation checks only zero norms, not finite vectors/norms/cosines. A finite old gradient and infinite new gradient can yield an infinite relative difference; their NaN cosine can be silently converted to a finite endpoint by Python's min/max clipping. This combination can falsely pass both admission inequalities. Require all gradient components, both positive norms, the raw cosine and every decision metric to be finite before clipping or comparing. Nonfinite values must fail closed, consistent with the training loop's `error_if_nonfinite=True`.

2. The first `train()` implementation reads the saved diagnostic pass flag without independently asserting that its warm, input, cohort, objective and source hashes equal those of the subsequently loaded training stage. If the admission wrapper does not already enforce the frozen hashes, a stale diagnostic pass could admit changed code/data. Compare the saved diagnostic provenance with the frozen protocol before running either arm. A valid decision is conditional on that exact objective, bank, warm start and scalar configuration.

The diagnostic also lacks the training loop's explicit maximum-reserved-memory guard in the first prototype. Add that budget check to the diagnostic stage. Its clipping comparison implements ideal Euclidean norm clipping; the production Torch function also uses a small denominator epsilon and its original gradient dtype. The resulting metric is an appropriate geometric proxy, not a claim of bit-identical optimizer input.
