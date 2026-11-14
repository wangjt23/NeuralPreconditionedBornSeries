# Higher-resolution continuation: protocol and decision gates

## Scope

User request: run the next research steps on larger-resolution datasets and
seek a materially stronger AI wall-time benefit. This continuation uses N64 and
N128 (4x and 16x the original N32 spatial unknown count), L12, g=0.65, and the
same periodic Fourier RTE/material-source family. It is not a new boundary
condition, S2 angular model, or real-world benchmark.

## Ordered investigations

1. Train NTO4 and CT4 on N64, seed 402, 512 train / 48 validation media,
   batch 8, 800 updates, old objective, first-eight solver selection. Both
   budgets complete. FP32 validation is close to its numerical floor and is
   retained as exploratory evidence, not used to assert large-grid accuracy.
2. Audit precision on N64/N128. Separate FP32 physical residual evaluation
   from FP64 outer/Krylov arithmetic with the same FP32 preconditioner. Copy
   the original discrete quadrature/frequency tensors into the FP64 operator.
   Do not loosen the true residual tolerance, 1e-5, or suppress failed cases.
3. Screen all-angle coarse N32 neural correction plus fine HSS against the
   identical non-neural two-level graph. This screen is negative: do not
   assume multilevel learning is effective because its cost is lower.
4. Warm-start N64 NTO4 and CT4 from their old-objective selected weights and
   train 400 updates, batch 4, lr 3e-4, on the same 512 materials. Differentiate
   through three real-inner-product FGMRES steps (two-pass MGS; relative
   1e-12 ridge in the tiny normal solve). Minimize mean log squared physical
   residual. Outer arithmetic is FP64, model applications FP32. Residuals are
   75% sources and 25% the previous probe mixture. Select on eight validation
   media by mixed-precision FGMRES failures/iterations/true residual.
5. Add spectral CT (SCT4): the same CT kernels plus a coefficient/frequency
   MLP, 12 -> 48 -> 48 -> 2dL, producing bounded complex modulation compiled
   into the existing streaming filters. Its zero spectral head equals the
   selected CT warm start. Train the same N64 400-update solver-aware budget;
   compare against CT4 with that same budget. This avoids dense local angular
   matrices; no extra FFT/matvec is added per preconditioner stage.
6. Resolution adaptation: initialize SCT4 and CT4 from their N64 solver-aware
   selected weights and run N128, 256 train / 32 validation, batch 2, 200
   updates, lr 3e-4, seed 402, with the same three-step loss and eight-medium
   checkpoint selection. This is warm-start adaptation, not an independent
   initialization replication. This protocol is fixed before the final new
   comparison data are generated.

## Comparison contract

Classical tuning uses seed 542 and six independent selection media per
resolution. Candidate depths are 2/4/8; HSS (angular FFT and compiled local
matrix implementations) tries shifts .5/1/2, and TSA/DSA/transport-DSA/reference
try eta scales 0/.03. This is 42 candidates, not unlimited optimal tuning.
Failure count comes before time. CPU and GPU tuning should be separated;
if a device uses the other's settings that limitation must be explicit.
Frozen learned blocks also receive a bounded one-versus-two repetition choice
on the selection data (effective depths four/eight, shared weights/cache). This
matches the deeper classical controls and can reduce expensive outer FP64 work.
No repetition choice is selected from the final comparison data.

After the models and protocol are frozen, the final comparison uses 24 new
media per resolution, generator seed **100843**, and three warmed repeats plus
one first solve. No training/selection uses this seed. Every model sees the
same saved tensors and source in randomized order. The cases are independently
generated at each grid size, not a documented same-medium refinement sequence.
Tests already drawn for the initial screens are development data only.

Primary gate: at least 30% reduction in paired median warmed time versus the
strongest tuned classical control, no more failures, and no p95 regression.
Also report preparation-plus-first cost and the comparison with matched CT4.
Passing a classical comparison does not establish that the extra AI capacity
helps if SCT4 does not beat its CT4 control. This is a target, not a result.

The GPU currently has an unrelated active process. No such process may be
stopped. Training and correctness checks can run; contended GPU timings must
not be called an uncontended confirmation. CPU uses one Torch thread on this
shared 36-vCPU host; neither host nor device is exclusively reserved.

## Research relationship

The new objective is related to differentiating solver convergence in
[Neural Preconditioning via Krylov Subspace Geometry](https://arxiv.org/abs/2507.15452).
Our short Arnoldi/regularized normal solve is not a reproduction of that paper's
loss/Givens implementation, and its PDE results do not establish RTE speedup.
Separating arithmetic precision follows the broader motivation of
[Carson and Higham](https://epubs.siam.org/doi/10.1137/17M1140819), but the present
implementation is FP64 outer FGMRES with FP32 preconditioning, not their full
three-precision iterative-refinement algorithm or an inherited error theorem.

## Reproducibility

All new experiments live in unique `rte/workstation_runs/` directories. Preserve
failed screens, selected-step checkpoints, raw repetitions, actual datasets,
process telemetry and SHA-256 manifests. Keep the N32 and original handoff
artifacts unchanged. End with a results/decision document and a handoff update;
do not describe additional seeds or larger training budgets as completed unless
they actually ran.
