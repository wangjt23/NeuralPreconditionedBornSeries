# TSA16 tolerance-aware study: source audit

22 October 2026. This audit uses saved local source and scalar CSV snapshots only. No SSH, tensor evaluation, training, PDE operation or solver was run for this audit. Line references below refer to these snapshots; current FWI hashes must be checked before executable reuse.

## Main finding

The existing strong-prefix banks contain enough information for a source-relative tolerance objective, but the normalizer hides an important scale. Compute and preserve each physical case's prefix ratio **before** normalizing its residual. A new benchmark method must also explicitly enter the TSA16-zero deployment path; merely adding a checkpoint to the old dictionary silently selects the TSA8-balanced path.

Let `R_i = ||r_prefix,i||`, `Q_i = ||q_original,i||`, `rho_i=R_i/Q_i`, and let `s_ik` be the squared prefix-relative ratio returned by `continuation_loss`. The original-RHS relative residual after injection and k further classical calls is

`rho_i * sqrt(s_ik)`.

Thus its squared tolerance margin is `s_ik * (rho_i/1e-5)^2`, or equivalently the prefix-relative squared threshold is `(1e-5/rho_i)^2`. Taking `||normalized rhs_exact|| / original_rhs_norm` is wrong because those two quantities are in different units.

## Exact reusable interfaces and traps

| Source snapshot | Lines | Relevant behavior |
|---|---:|---|
| `deliverables/training_priority_20261004/code/train_alignment.py` | 15–42 | `prepare_strong` acquires a TSA16-zero four-classical-call prefix, checks the existing accurate teacher, retains original physical IDs and `original_rhs_norm`. Do not reacquire these banks for this study. |
| Same | 31–40 | `rhs_exact` is the unnormalized captured residual; `target=teacher-prefix_solution`; `image_basis` is a retained search-image basis, not the original source. |
| Same | 45–50 | `restrict` applies a physical-ID mapping to every bank tensor and asserts exact order. Reuse it with the saved 56 IDs rather than recomputing a possibly changed eligible intersection. |
| Same | 79–95 | The earlier 56-ID intersection, saved sampling plan, seed and phase hash are explicit. Both new objective arms must receive exactly the same plan. |
| `deliverables/train_tail_multiresolution.py` | 139–165 | `build_sampling_plan` stores eligible row IDs, original IDs and phases; `sampling_rows` exposes an auditable scalar ledger. |
| Same | 213–264 | `normalize_and_cache` divides `q`, `rhs_exact`, `target`, `reference` by per-case residual RMS at lines 240–244, leaves other scalar metadata unchanged, then discards `scale` at line 257. Add `prefix_relative_residual` to an in-memory bank copy before this call. |
| Same | 245–256 | Cached inverse/stream data are obtained from a depth-8 TSA object. These reference data do not depend on recurrence depth: `train_source_predictor.reference_cache` lines 58–61 returns only inverse and stream. Verify this property against current source; do not infer that the actual continued preconditioner has depth 8. |
| `deliverables/training_priority_20261004/code/train_adequacy.py` | 33–43 | `batch_fields` rotates residual, reference, teacher and image basis together. The phase has unit modulus, so `rho_i` does not rotate or rescale. Pass depth 16 explicitly. |
| Same | 83–135 | `run_arm` already owns fresh optimizer/scheduler, checkpoint state, budgets and scalar logs, but its loss is hardcoded at lines 117–118. Prefer a new isolated runner with one pure `loss_from_ratios(...)` seam rather than further monkey-patching this archived runner. |
| Same | 95, 115 | Its validator and batch fields default to the TSA8 study unless redirected. New runner should explicitly call the strong validator and strong batch depth for both arms. |
| `train_alignment.py` | 52–62 | `validate_strong` uses all original parent cases, TSA16-zero and one learned injection. Keep all 16 validation cases; the six eligible tails are only the fixed diagnostic panel. |
| `remote_reference/rte/tail_lookahead.py` | 26–56 | `continuation_loss` returns `[B,H+1]` squared ratios normalized by the prefix residual. Column zero is after injection; column k follows k additional classical calls. Near-dependent candidate images are safely ignored. |
| `remote_reference/rte/augmented_fgmres.py` | 32–43, 53–77 | Deployed tolerance is against the original RHS; four classical calls precede the optional injection; already-converged cases never invoke the model. |
| Same | 86–107, 137–141 | Deployment independently rejects dependent/nonfinite/worsening injections and reevaluates final physical residual. Loss geometry alone is not a convergence certificate. |
| `remote_reference/rte/tail_response_network.py` | 15–26 | TailResponseNetwork exposes normalized residual/reference features and restores reference-RMS scale. An original-RHS tolerance loss can use external `rho_i` metadata without altering architecture or input features. |

## Benchmark repair required before a new candidate is evaluated

`deliverables/training_priority_20261004/code/benchmark_adequacy.py:54–61` routes only a hardcoded list of names to TSA16-zero. Any new ordinary model key falls into `protocol.solve('neural_tail', ...)`. In `remote_reference/rte/benchmark_tail_solver.py:80–101`, that generic route defaults to TSA8 and a balanced reference initialization. This would invalidate the objective-only causal comparison.

Use an explicit common `solve_tsa16_zero(model_or_none, ...)` implementation for the frozen neural comparator, matched two-step model and tolerance-aware model. Retain the classical comparator through the frozen selection file. Require a recorded `deployment_prefix='tsa16_zero'`, depth 16 and initial `None` for both learned arms; do not infer deployment from a label suffix.

The old gate at `benchmark_adequacy.py:89–91` selects the name `continued_selected`, and its `all(...)` could pass vacuously if the expected comparison rows disappear. The new gate must first assert exactly the expected N128 and N256 rows, then require both mean savings >=5%, each paired lower bound >0 and zero failures. Record objective-vs-objective paired comparisons separately from each candidate's comparison against the strongest frozen classical control. A tiny objective improvement cannot satisfy the classical gate by itself.

Timing must still include reference construction used by each proposal, model inference, projection and final original-RHS residual acceptance. Physical/base setup stays separate. `ClassicalTransportStack.__init__` in `remote_reference/rte/transport_baselines.py:18–31` calls `prepare`, which eagerly prepares the reference inverse at lines 48–49; do not call that work free or lazy. Training budgets matched by updates do not imply equal optimizer wall time; report both.

## Discriminability from already-saved scalar evidence

The following calculation joined `alignment_retry/train_tsa16_n128.csv` or `alignment_retry/development_tsa16_n128.csv` on original physical ID with `alignment_retry/tsa16/fixed_panel.csv`. It used the published prefix original-RHS residual times the square root of the published prefix-relative squared ratio. No array fields or new solves were evaluated.

For the 56 common original training IDs, N128 prefix `rho_i` ranges from `1.1076505889e-5` to `9.1325053710e-2` (median `2.9627501941e-4`). Consequently, squared ratio thresholds range from `1.1990026070e-8` to `0.8150691104`, a factor of about 68 million. A single threshold on the old prefix-normalized ratio cannot represent the physical stopping criterion.

| N128 panel and checkpoint | After injection: above tol | After +2 calls | After +4 calls | After +6 calls |
|---|---:|---:|---:|---:|
| Training, 16 cases, warm start | 16 | 6 | 3 | 0 |
| Training, same 16 cases, old update 500 | 16 | 4 | 2 | 0 |
| Training, same 16 cases, old update 1000 | 16 | 4 | 2 | 0 |
| Development, six eligible cases, warm start | 6 | 3 | 1 | 0 |
| Development, same six, old update 500 | 6 | 3 | 1 | 0 |
| Development, same six, old update 1000 | 6 | 3 | 1 | 0 |

This is an informative contrast: many two-step residuals already satisfy tolerance, while some four-call tails do not. The old endpoint improved residual magnitudes without changing these development crossing counts. It supports evaluating an original-tolerance-aware objective, not a claim that it will achieve a speed advantage.

These tables omit odd call counts because the archived panels report only 0/2/4/6. Do not report exact first-hit calls from them. A two-step survival objective is right-censored for 6/16 warm training cases and 3/6 warm development cases. If the new arm uses a longer horizon while the control remains two-step, report that as an objective-plus-horizon contrast, not tolerance weighting alone. For an objective-only comparison, use the same horizon and explicitly acknowledge censoring; alternatively, freeze a matched-horizon control if testing a longer-horizon objective.

## Minimal implementation and admission diagnostics

The owner-selected minimal causal contrast keeps the existing horizon `k=0,1,2` in both arms. With `eta_i=prefix_relative_residual`, the proposed truncation-aware term is unambiguously

`mean_{i,k} beta*softplus((log(s_ik)+2*log(eta_i)-2*log(tau))/beta)`, with `beta=log(4)`, `tau=1e-5`, and the same `0.05*teacher_direction_loss` in both arms.

It differs from old mean-log ratio by suppressing gradients far below the physical stopping tolerance; far above tolerance its derivative with respect to `log(s_ik)` approaches the old derivative. This is a finite-horizon excess-log-residual surrogate, not the expected number of solver calls. A diagnostic must demonstrate that enough margins fall in its transition/low-weight region and that the aggregate parameter gradient differs; otherwise the new objective could be effectively the old loss plus constants on the sampled states. The complete training budget remains two matched 1000-update arms, LR 1.5e-4, cosine horizon 1000, seed 41022, subject to the owner's lock/controller admission.

1. Load the archived strong N64/N128 context banks, saved 56-ID tensor and saved phase/ID plan; hash every input and verify parent hashes. Preserve all original files.
2. In memory, compute `prefix_relative_residual=norm(raw rhs_exact)/original_rhs_norm` in FP64 before normalization. Check that it agrees with the existing prefix CSV by physical ID. Retain this new field unchanged through the cache and sample by the same eligible row IDs as the residual.
3. Expose `loss_from_ratios(ratios, prefix_relative_residual, tau, mode)` as a pure function. Both objective arms share network, warm checkpoint, optimizer, schedule, batch, IDs/phases, continuation arithmetic, teacher coefficient and deployment validator. Keep the old control numerically identical to the archived two-step formula, including its clamp.
4. Before a training budget is admitted, check only a small frozen training diagnostic: loss values, per-time scalar derivative weights, fraction above/below threshold, fraction near the smoothing transition, fraction censored at horizon and parameter-gradient cosine/norm if a backward probe is allowed. A scalar weight difference does not prove the model-parameter gradients differ, especially with all gradients clipped in the old study.
5. Require physical scale invariance of the loss: simultaneously multiplying original source, residual, reference and teacher by a positive factor must leave the source-relative margins unchanged. Phase augmentation must leave margins unchanged as well. A repeated-ID batch must not mix original source norms across cases.
6. Log stop reasons and partial checkpoints if the admitted window expires. Verify actual consumed ID/phase rows match between completed arms. Do not compare a complete arm with a shortened arm as if budgets were matched.
7. Freeze endpoint/checkpoint policy before timing; validate against all 16 development media, and preserve reserved seed 9626. The repeated development cohort still cannot establish independent generalization.

For an all-56-case N128 warm-start gradient diagnostic, stream cases in batches no larger than four and accumulate parameter-gradient sums. Do not turn the diagnostic into a batch of 56: the archived memory evidence is for batch four. Keep diagnostic phase and sample ordering fixed and record whether clipping is applied; compare the pre-clipping gradient direction/norm and the actual post-clipping update direction separately.

`train_adequacy.prepare_ops` (lines 25–31) initializes one low/high operator pair per grid with the first four cached media. `batch_fields` then calls `low.setup(t,s,r)` and `high.setup(t.double(),s.double(),r)` on every sampled batch (lines 40–42), so reusing these objects is legitimate only if both setup calls remain intact. Cached inverse/stream fields must be indexed by the very same eligible row IDs as t/s/residual. For strong validation, call `train_alignment.validate_strong` explicitly; it reconstructs each original parent operator and starts at zero. Neither a normalized residual-bank validator nor `train_tail_direction.validate` is a substitute for this deployment check.

## Current FWI files that should be refreshed or hash-verified

Core current-source reads: `rte/tail_lookahead.py`, `rte/augmented_fgmres.py`, `rte/global_source_predictor.py`, `rte/tail_response_network.py`, `rte/benchmark_tail_solver.py`, `rte/train_tail_direction.py`, `rte/train_source_predictor.py`, `rte/transport_baselines.py`, `rte/mixed_precision_transport.py`, and `rte/audit_tail_injection.py`. These are small scoped project sources; the local snapshots may have become stale since 4 October.

Archived implementation reads: `rte/workstation_runs/resume_audit_20261002/code/train_tail_multiresolution.py` and the three archived `training_priority_20261004/code` scripts audited above. Verify archived code hashes against the completed study receipt.

Remote-only data/checkpoint reads: `training_priority_20261004/alignment_retry/train_tsa16_n64.pt`, `train_tsa16_n128.pt`, development prefix banks, `matched_original_ids.pt`, `sampling_plan.pt`, and the archived original warm checkpoint. Keep tensor fields on FWI. Refresh only scalar settings, hashes, prefix diagnostic CSVs and job/lock receipts back to the Mac bridge.

Numerical admission remains the owner's responsibility: check both named Schur locks and current pressure/Schur controllers immediately before any admitted diagnostic, training or timing stage. A lock pathname or an old launcher source does not establish current ownership or availability.
