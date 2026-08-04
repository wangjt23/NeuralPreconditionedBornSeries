# Quantum PDE research harness: latest state

- Run: `20260804T111027Z-63ccfea8`
- Turns: 1
- Local compactions: 0
- API reasoning context: `None`

## Fundamental question

Which structural, objective, discretization, and experimental differences causally prevent the current RTE learned preconditioners from matching stable classical TSA/DSA behavior?

## Claims

- `C-OPERATOR-CORRECT` **OBSERVED** — The periodic RTE operator and native FGMRES evaluator pass independent dense-matrix and residual checks on small grids.
- `C-RTE-PER-MEDIUM` **OBSERVED** — The current RTE ablation trains a new model for each fixed coefficient medium and evaluates held-out sources, so it does not establish cross-medium coefficient generalization.
- `C-ANGULAR-RESTRICTION` **OBSERVED** — The structured moment models and moment U-Net restrict learned corrections to a fixed retained angular Fourier subspace Q_K.
- `C-UNET-PERIODIC-MISMATCH` **OBSERVED** — The RTE learned U-Net uses circular padding for ordinary convolutions but zero-boundary transposed-convolution and resize operations, producing a measurable learned-correction shift defect.
- `C-WIDTH-NOT-SOLE` **OBSERVED** — In a matched small mixed control, width 32 lowered sampled loss but did not improve held-out one-step residual and worsened capped FGMRES relative to width 16.
- `C-LOSS-MISMATCH` **OBSERVED** — A lower sampled training objective is not equivalent to lower held-out one-step error or better FGMRES because the sampled residual distribution and solver Krylov directions differ.
- `C-HELMHOLTZ-DATA-DIFFERENCE` **OBSERVED** — The Helmholtz training loop varies coefficient media across batches and uses fresh random right-hand sides, unlike the current per-medium RTE ablations.

## Open obligations

- `O-PERIODIC-TRANSFER` **OPEN** — Replace zero-boundary decoder transfer by periodic, preferably phase-aware, restriction and prolongation and rerun the symmetry test.
- `O-LARGER-ANGULAR-SPACE` **OPEN** — Implement K_candidate greater than K_solve with bounded learned restriction/prolongation and exact scalar-mode preservation.
- `O-CROSS-MEDIUM` **OPEN** — Train on randomized coefficient fields and test on held-out media, sources, and resolutions.
- `O-KRYLOV-OBJECTIVE` **OPEN** — Use teacher imitation, diverse later residuals, and held-out FGMRES/time-to-tolerance checkpointing.
- `O-WALL-TIME` **OPEN** — Compare learned, TSA2, and TSA4 at equal base calls and equal wall time including setup and network overhead.

## Rejected paths retained across compaction

- `R-OPERATOR-BUG-AS-MAIN` — Dense operator, exact splitting, direct solve, and FGMRES checks do not support an operator implementation bug as the primary explanation.
- `R-WIDTH-ONLY` — The width control lowers training loss without improving held-out one-step or capped FGMRES, so increasing width alone is not a sufficient fix.

## Next actions

- Implement periodic transfer and rerun the shift-defect test.
- Implement K_candidate > K_solve learned angular R/P with scalar preservation.
- Create a small coefficient-ensemble training/evaluation split.
- Use held-out FGMRES and time-to-tolerance for checkpoint selection.

This report is a projection of `state.json`. Full turn evidence is append-only in `events.jsonl` and `run.sqlite3`.
