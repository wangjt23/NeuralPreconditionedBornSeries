# Repo Investigation Instructions

Follow [INVESTIGATION_WORKFLOW.md](../INVESTIGATION_WORKFLOW.md) as the default workflow for experiment, ablation, and report work in this repository.

## Required Defaults

- Do not claim improvement from short or noisy runs without checking convergence, checkpoint selection, or solver behavior.
- For architecture updates, ablate each new component separately and then the combined model.
- Prefer solver-level conclusions over surrogate-loss conclusions. One-step or training-loss gains are not enough if stationary or FGMRES metrics regress.
- State the training and generalization protocol explicitly in reports: source-heldout, coefficient-heldout, case-heldout, or joint-case training.
- Define metrics before the first result table.
- Compare against the last established or committed baseline, not only against weaker internal variants.
- Re-read completed artifacts before reporting. Do not rely on partial live output from unfinished runs.
- When a method helps or fails, explain the mechanism or likely failure mode.
- Say explicitly whether a commit or push happened when that is discussed.

## Repo-Specific Emphasis

- In NPBS and RTE studies, prefer sufficient training budgets and checkpoint-aware evaluation over fast but undertrained comparisons.
- If routing or mixture models are used, inspect learned weights, active candidates, and solver cost, not only final residual numbers.
- When the question is iteration-count reduction, verify whether the residual bank covers harder states such as rollout, Krylov, and restart residuals rather than only source residuals.