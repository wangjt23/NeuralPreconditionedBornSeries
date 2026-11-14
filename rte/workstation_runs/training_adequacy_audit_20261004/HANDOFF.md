# RTE training adequacy audit — 4 October 2026

**The present evidence does not establish adequate training of the recent learned Krylov-tail model. The recommendation to stop all additional epochs or TSA16-aligned training was premature.** The measured result that the existing frozen checkpoints do not deliver a useful speed advantage remains valid. Additional training is an unresolved experiment, not a promised remedy.

This audit supersedes the broad stopping recommendation in `docs/research/RTE_DIRECTION_DIAGNOSTIC_2026_10_02.md`, while retaining its timings, failures, controls and narrow negative results. The October 4 four-job batch contained **zero training**; its analytic local/coarse-response screens and certificate engineering results cannot establish saturation of a learned model. Verification overhead is a supported engineering opportunity, but it does not close the learning research question.

## Access, identity and scope

The authorized SSH alias FWI succeeded with forwarding disabled, existing credentials, strict host-key checking and a bounded connection timeout. Observed identity: `wjt@FWI-Standard-PC-i440FX-PIIX-1996`; workspace `/data/wjt/NeuralPreconditionedBornSeries`; branch `main`, HEAD `028caea3f35d100225f656e2b1e910c1366aaafe`. RTE means the **radiative transfer equation**, here the steady periodic 2D Fourier pseudospectral system with twelve angular ordinates and the existing coefficient/source distribution. OL_preconditioner was not accessed or resumed.

Read-only checks found the same nine modified tracked files and existing untracked research. No scoped AGENTS.md or project .agents/skills was present. The A6000 reported 49,140 MiB total, 37 MiB used and 0% utilization; no scoped RTE training process was running. Other jobs were not modified. New work consists of scalar receipt analysis and this audit. No training, teacher generation, new holdout access, solver run, source repair, package installation, Git commit/push, job cancellation or access change occurred. Existing models, banks and earlier reports remain preserved.

## Training must be assessed separately for each family

Actual FWI checkpoint metadata and scalar histories were inspected; update counts below are not inferred merely from filenames.

| Family/run | Completed budget | Selected checkpoint | What it establishes |
|---|---:|---:|---|
| Older CT, N32, seed101 | 1,000 epochs / **128,000 updates**, 4,096 training media | epoch200 | Substantial training and a late plateau for this CT surrogate; no transfer of saturation to the newer tail model |
| N64 NTO surrogate, seed402 | 800 updates | 800 | Budget ended at the selected checkpoint |
| N64 NTO/CT/SCT three-step solver adaptation | 400 updates each | 400 each | These short solver-aware runs did not establish saturation |
| N128 SCT adaptation | 200 updates | 200 | Particularly limited adaptation evidence |
| Global source predictor | 1,000 initial + 1,500 six-step updates | 1,500 in continuation | Longer-horizon continuation selected its final checkpoint |
| Characteristic source predictor | 1,000 additional updates | 800 | One architecture/seed continuation; no matched-budget causal attribution |
| Base tail-direction model | 2,000 updates, batch8 | 800 | Immediate projected-loss stage did not improve the full-solver selection metric |
| Optical/moment tail variant | 2,000 updates, batch8 | 0 | This particular surrogate/variant failed to improve its starting work proxy |
| Two-step tail lookahead | 1,000 updates, batch4 | **1,000** | Full-solver validation work still improved at the end |
| October 2 N64-only / mixed-N64/N128 tail arms | 1,000 updates each, batch4 | 800 each | One paired seed, controlled resolution comparison; insufficient saturation evidence |

The older CT history genuinely supports diminishing returns: train loss is 0.008057409 at epoch900 and 0.008057143 at epoch1000; validation loss is 0.008399424 and 0.008406284. Its selected solver score remains at epoch200. This is evidence against assuming that every project failure needs more epochs. Its historical severe test failures were largely an inference-cache bug, subsequently audited and corrected; they are not evidence of failed optimization. The earlier CT precision and contention limitations also remain.

Historical N64 U-Net result tables contain successive nominal continuation segments through `from18600...steps1800`, with a 5,949-residual bank. Late one-step metrics change little. These are a different network and task distribution; this audit did not verify that entire checkpoint ancestry or establish that 5,949 residuals represent independent materials. Neither the older CT nor U-Net budget can be added to the newer tail model's effective training history.

## Strongest evidence that the recent stopping decision was early

The **September 26 two-step lookahead run** selected its final update, and the scalar full-solver validation metric improves throughout the last part of its budget:

| Update | Mean classical calls + 0.5 × injection calls |
|---:|---:|
| 0 | 8.03125 |
| 200 | 7.96875 |
| 400 | 7.90625 |
| 600 | 7.90625 |
| 800 | 7.84375 |
| 1,000 | **7.78125** |

This is only sixteen repeatedly used development media and the improvement is small. Nevertheless, a failed runtime gate at the final improving checkpoint cannot establish that longer optimization is futile.

The October 2 paired arms used **112 eligible training material/source identities**, drawn from 128 parent media; all sixteen development media remain in full-solver evaluation. Both consumed the same 4,000 identity/phase presentations. N64-only averaged 35.7 presentations per identity; mixed averaged 17.9 per identity **per grid**. Phase augmentation changes the view, not the number of independent materials. All 112 eligible identities were sampled at each used grid, so wholesale omission of eligible cases is not the explanation.

Actual optimizer time was **118.80 seconds and 119.35 seconds** respectively, with periodic validation recorded separately. These are small continuations, although elapsed time or update count alone does not prove undertraining.

Both arms started from `tail_direction_pilot_20260926/best.pt`, selected at update800, with optimizer/scheduler/RNG deliberately reset. They did **not** continue the selected lookahead update1000. The deployed mixed checkpoint therefore carries the base run's selected 800-update weights plus 800 selected continuation updates, not all updates expended across the different tail experiments. This choice is valid for a controlled resolution ablation; it must not be presented as one long uninterrupted convergence study.

Scalar stochastic training summaries also show remaining progress:

| Arm/grid | Mean objective, first200 → last200 | Mean immediate squared residual ratio, first200 → last200 | Mean terminal squared ratio, first200 → last200 |
|---|---:|---:|---:|
| N64-only | -4.4946 → -4.8295 | 0.4896 → 0.4074 | 0.003013 → 0.002382 |
| Mixed/N64 | -4.3720 → -4.8346 | 0.4959 → 0.4083 | 0.003137 → 0.002173 |
| Mixed/N128 | -4.4625 → -4.7007 | 0.5298 → 0.4392 | 0.003626 → 0.002985 |

These are descriptive averages over sampled minibatches, **not fixed-panel convergence tests or confidence intervals**. Lower log objective is better. The ratios are squared norms normalized by the residual at injection, not the original physical right-hand side; they must not be compared directly with the final original-RHS tolerance. The mixed N128 terminal ratio actually worsens between the 601–800 and 801–1000 windows, so progress is not uniformly monotone and these curves do not prove that a longer run will help.

Pooled full-solver validation work declines from 8.0625 initially to 7.875 at update800 in both arms. Mixed remains at 7.875 at update1000; N64-only rises slightly to 7.90625. Mixed update800 wins only the residual tie-break over update1000. The score aggregates 32 grid rows but only **sixteen independent physical media**. One 200-update interval with equal discrete call counts, near the end of cosine learning-rate decay, is weak evidence of saturation. All 1,000 logged gradient norms in each arm exceeded the clipping threshold1. This is an intentional optimization choice, not proof of a bug; its effect has not been isolated by a control.

## Other unresolved explanations

1. **Training/deployment mismatch.** Training contexts and checkpoint selection use TSA8 with balanced initialization. The strongest N256 classical baseline uses TSA16 from zero. N256 has no training examples. Testing a frozen TSA8-trained network behind TSA16 establishes performance of that transfer, not the best achievable TSA16-trained network. Neither more data, longer optimization, stronger-prefix contexts nor their interactions have been isolated.
2. **Objective mismatch.** The tail loss averages log projected residual ratios through two classical continuation steps plus a small teacher-direction term. The deployment decision depends on reaching the unchanged original-RHS tolerance and paying inference/projection cost. Immediate residual, MSE and short-horizon loss can improve without removing enough expensive calls. The older 6-step source run and 2-step tail run are different tasks and cannot substitute for a tail horizon ablation.
3. **No replicated convergence study.** The paired resolution study has one shared training/sampling seed. Several source/tail runs use common banks and selected ancestors. Their aggregate compute is not independent training replication. Nested resolutions and repeated solves do not add independent physical samples. Reusing seed4626 for selection, architecture decisions and diagnostics makes it development data. The reserved seed9626 remains untouched.
4. **Capacity and useful error geometry are unresolved.** The earlier four-case source fit reduced solution error from 8.093% to 0.2667%, demonstrating limited-case fitting capacity, but reduced iterations only 4.75→4.25. It does not prove tail-model adequacy on 112 heterogeneous contexts. A fixed small-panel tail overfit control and useful projected-image/continuation diagnostics are still missing. Oracle headroom is information availability, not a trainability or deployment theorem.

## Conclusions retained and conclusions narrowed

**Retain:** the frozen October 2 mixed checkpoint fails the prescribed runtime gate. Its N256 warm mean was 155.891ms versus 145.741ms for TSA16-zero, a -6.964% saving with conditional paired-development interval [-10.377%, -2.846%]. The original paired gate required zero failures and at least **5%** saving at both N128 and N256; later 10% mechanism screens are different protocols. No learned speed advantage or out-of-distribution robustness has been demonstrated by the recent models. The extra inference cost is a real obstacle, and the four-forward phase average is an unfavorable implementation in its measured setting.

**Narrow:** “stop additional epochs or TSA16 retraining” was stronger than the evidence supports. The appropriate statement is: **the tested checkpoints and particular fixed-cost interventions do not currently clear the deployment gate; optimization saturation and the potential of a correctly aligned learned model are unresolved.** The immediate-loss optical/moment result justifies changing that objective or mechanism; it does not establish that attention cannot learn under a solver-aligned objective. Fixed analytic local-collision/coarse-source failures do not imply a learned operator impossibility result.

## A decisive, bounded next study

The next learning experiment should test training adequacy before architecture proliferation or a large data campaign. This is a proposed protocol, **not an executed experiment**.

First use fixed saved training and development panels to log objective, teacher-direction error, projected-image quality, residual ratios after 2/4/6 steps, true final failures, total physical/base calls, and warm solve time. Include the frozen network, analytical reference and exact-teacher information control with teacher cost explicitly excluded from deployment claims.

Then extend one frozen tail architecture on the same saved paired N64/N128 bank with a predeclared 3,000-update continuation schedule, recording checkpoints every250 updates and comparing its 1,000/2,000/3,000 prefixes. Keep a small declared learning-rate control and distinguish weights-only warm-start from exact state restoration. This tests additional optimization; it is not an independent initialization replication. Proceed to independent initialization/sampling replications if the fixed-panel and full-solver evidence improves. A useful ablation must change only one of training length, learning rate, horizon, prefix or data size at a time.

A separate paired study should build TSA16-zero contexts from the **existing training media and accurate labels**, and compare TSA8-trained versus TSA16-trained tail responses at matched presentations/budgets and a longer tolerance-aware continuation objective. Do not train on the sixteen development media. More N256 training or fresh materials should be considered only after these cheaper adequacy/objective controls identify a real bottleneck.

Measured 1,000-update optimizer cost is about two minutes per arm on the A6000, with about9GiB peak reserved memory for the original cached-bank study. A single 3,000-update continuation is approximately six minutes of optimizer work **by extrapolation**, plus loading, validation and timing; stronger-prefix/horizon costs have not been measured. Start serially on FWI with a hard fifteen-minute wall cap for that initial continuation, preserving partial checkpoints if the cap is reached. Recheck current unrelated jobs before launch. A larger multi-seed/alignment campaign requires a fresh concrete resource estimate. No training was launched for this audit.

Stop only after fixed-panel improvement and full-solver/cost gains plateau across predeclared checkpoints and replicated optimization controls, or after a decisive fixed-cost obstruction for the tested mechanism. A failed speed gate remains sufficient to reject deploying that checkpoint. It is insufficient to prove that the architecture was fully trained. Freeze choices before any fresh holdout evaluation; seed9626 stays reserved until the predeclared gate is met. Report negative runs and seed-level uncertainty.

## Reproducibility and remaining gaps

`analyze_training.py` reproduces this audit from scalar logs, checks 1,000 contiguous updates and4,000 consumed presentations per arm, verifies identical identity/phase schedules, independently reproduces the selected update800, validates finite logged metrics, and emits `TRAINING_EVIDENCE.json` with input SHA256 hashes. These checks passed. Original source and numerical fields remain on FWI. Read-only remote checkpoint inventories are saved locally as `receipts/training_audit_20261004_remote.json` and `receipts/training_audit_20261004_legacy_full.json`; exact remote read scripts are retained beside them.

Remaining gaps are optimizer/horizon/prefix controls, a tail small-panel capacity test, independent training seeds, fresh held-out/OOD evidence, original-source full-solver certificates for neural solutions, and complete teacher/setup/train/cold/amortized cost. There is no SSH access blocker. The audit establishes that a broad stopping inference was premature; it does not establish that more training will produce a useful solver.
