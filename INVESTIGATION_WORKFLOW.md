# Investigation Workflow

This note captures recurring investigation priorities inferred from recent experiment conversations in this repo. Future training, ablation, analysis, and reporting work should follow this workflow by default.

## Core Priorities

- Training sufficiency matters before conclusions. Do not treat a short run or a noisy mid-run snapshot as evidence unless convergence, plateau behavior, or checkpoint selection supports it.
- Architecture updates require full ablation coverage. If an update changes multiple components, isolate each component and the combined variant instead of only reporting the final bundled model.
- Solver behavior matters more than surrogate wins. Improvements in training loss or one-step residual are not enough if stationary or FGMRES behavior regresses.
- Generalization claims must be explicit. Always state whether evaluation is source-heldout, coefficient-heldout, case-heldout, or joint-case training with per-case testing.
- Mechanism matters, not just ranking. When a method helps, explain what changed operationally, for example which classical base is being favored, what weights are learned, or why stability improved.
- Definitions must appear before use. Metrics such as stationary residual, one-step residual, and FGMRES residual should be defined before the first result table.
- Negative results should be preserved and interpreted, not hidden. If a family fails, say why it appears to fail and whether the issue is objective mismatch, instability, or limited expressivity.
- Final reports must use completed artifacts. Recheck the final CSV or checkpoint rather than relying on partial output observed while a job was still running.

## Expected Experiment Workflow

1. Anchor the exact change under test.
   - Identify what was updated since the previous baseline.
   - If the change is multi-part, enumerate the parts before launching runs.

2. Decide whether training is actually sufficient.
   - Prefer a budget large enough to answer the question cleanly.
   - Use convergence behavior, solver metrics, and checkpoint drift to judge sufficiency, not step count alone.

3. Run discriminating ablations.
   - Compare against the last committed or otherwise established baseline.
   - Toggle each new architecture component separately, then test the combined variant.
   - Keep train and heldout evaluation aligned with the intended claim.

4. Validate at the solver level.
   - Record one-step, stationary, and FGMRES behavior.
   - Treat solver-tail collapse or stationary divergence as a primary regression even if the surrogate objective improves.

5. Inspect training behavior directly.
   - Check whether early checkpoints outperform the final checkpoint.
   - Check whether the initialized solver is already better than later trained checkpoints.
   - Report aggregate training loss together with the sampled training one-step metric used during training, so loss decreases can be compared against the actual one-step behavior.
   - Check whether residual diversity, rollout penalties, or schedule changes alter the failure mode.

6. Make the protocol explicit in the report.
   - State the training dataset or coefficient family.
   - State whether models are trained jointly or separately by case.
   - State what heldout means in that experiment.
   - Define every metric before the first table.

7. Explain the mechanism when a result is important.
   - For routed or mixture models, inspect learned weights and relaxations.
   - For stabilization changes, explain whether the gain comes from improved contraction, better solver compatibility, or simple damping.

8. Close with the next concrete experiment, not a vague suggestion.
   - If a method fails, identify the most likely bottleneck.
   - If a method helps, identify the next discriminating test needed to trust the result.

## Questions That Should Be Answered Before Declaring A Result

- Was training long enough to support the conclusion?
- Was every meaningful architecture update ablated separately?
- Did the method improve solver metrics, not just the training objective?
- What exact training data or coefficient family was used?
- Is heldout evaluation testing new sources, new coefficients, new cases, or only new right-hand sides on the same field?
- Are the reported numbers from the final checkpoint or the best validated checkpoint?
- If the method helps, what is it doing mechanistically?
- If the method fails, what is the most credible failure mode?

## Communication Preferences

- Be direct about whether a run is sufficiently trained.
- When reporting training progress, include both aggregate loss and the sampled training one-step metric instead of loss alone.
- Say clearly when a report is case-specific and should not be overread as a broader generalization claim.
- Avoid first-define problems for terms and metrics.
- Compare new results against the previous committed or established reference point.
- If asked to push or commit, say explicitly whether that happened.
- Prefer analysis that explains why a model helps or hurts, not just that it does.

## Default Future Workflow For This Repo

When working on NPBS or related RTE experiments in this repository, consult this note before launching large runs or writing reports. The default expectation is: sufficient training, architecture-complete ablation, solver-level validation, explicit generalization protocol, mechanism analysis, and artifact-backed reporting.

For NPBS U-Net retraining unless the scheduler itself is being ablated, prefer `--scheduler onecycle` so the run uses OneCycle with cosine decay, and keep the training one-step metric visible in the live logs and final artifacts.