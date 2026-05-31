# NPBS UNet Routed-Base Update Report

## Scope

This report evaluates the recent routed-base UNet-family update in `rte/npbs_unet_ablation.py` against the previously committed size-64 physical-loss results.

The pulled change adds a `LearnedBaseRouter` that can mix classical base corrections (`transport`, `dsa`, `tsa`, and optionally `tsa_moment`) and can optionally condition routing on a residual summary in addition to coefficient summaries. The routed base is then injected into `NPBSUNetPreconditioner` / `MomentNPBSUNetPreconditioner` through `base_module`.

## Compared Runs

- Historical committed baseline: `rte/npbs_unet_ablation_tsa_full_physical_steps900.csv`
- Current post-pull plain rerun: `rte/npbs_unet_ablation_tsa_full_physical_size64_current_steps900.csv`
- 24-angle plain control: `rte/npbs_unet_ablation_tsa_full_physical24_plain_size64_steps900.csv`
- Routed 2x2 matrix:
  - `rte/npbs_unet_ablation_routed_base_nomoment_size64_steps900.csv`
  - `rte/npbs_unet_ablation_routed_base_nomoment_residual_size64_steps900.csv`
  - `rte/npbs_unet_ablation_routed_base_withmoment_size64_steps900.csv`
  - `rte/npbs_unet_ablation_routed_base_withmoment_residual_size64_steps900.csv`
- Committed teacher references:
  - `rte/npbs_unet_ablation_tsafull_size64_large_teacher1200.csv`
  - `rte/npbs_unet_ablation_tsafull_size64_large_teacher1800.csv`

All new ablations below use size 64, cases `anisotropic` and `mixed`, physical-only loss (`metric_weight=0`, `physical_weight=1`, `correction_weight=0`, `rollout_weight=0`), 900 steps, cosine schedule, and `fgmres_mode=real-stacked` unless noted otherwise. Where a historical CSV contains multiple protocol slices, the comparisons below use the size-64 `real-stacked` heldout rows.

## Training And Generalization Protocol

The comparison table does **not** represent a single model trained jointly on anisotropic and mixed coefficient datasets.

- `rte/npbs_unet_ablation.py` loops over `args.cases` and trains a separate model for each case.
- Passing `--cases anisotropic mixed` therefore launches two independent trainings inside one command: one anisotropic model and one mixed model.
- The `heldout` rows are source-heldout evaluations on the same case-specific coefficient field, not new coefficient realizations.

So these routed-base results measure **source generalization within each fixed case**, not cross-case generalization and not coefficient-distribution generalization.

## Metric Definitions

- `one-step` means the one-step true residual ratio `||A M r - r|| / ||r||`.
- `stationary residual` means the final relative residual `||b - A psi_k|| / ||b||` after applying the preconditioner as a stationary fixed-point iteration, `psi_{k+1} = psi_k + M(b - A psi_k)`, for up to 60 iterations or until tolerance.
- `FGMRES` means the final relative residual after preconditioned FGMRES, with up to 25 iterations in these ablations.

Lower is better for all three metrics.

## Main Comparison

| Run | Learned base candidates | Residual summary | Anisotropic one-step | Anisotropic FGMRES | Mixed one-step | Mixed stationary residual | Mixed FGMRES |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| committed full12 UNet | fixed transport+TSA base | no | 0.0698968 | 1.26946e-05 | 0.124534 | 7.82266e-06 | 7.54405e-06 |
| current full12 UNet | fixed transport+TSA base | no | 0.0451438 | 1.27416e-05 | 0.276063 | 2.49909e-01 | 7.84667e-06 |
| plain 24-angle UNet | fixed transport+TSA base | no | 0.0205849 | 1.20855e-05 | 0.303355 | 1.39156e-01 | 8.85221e-06 |
| routed 24-angle | transport,dsa,tsa | no | 0.0255104 | 1.27929e-05 | 0.193663 | 9.85935e-06 | 8.17534e-06 |
| routed 24-angle | transport,dsa,tsa | yes | 0.0221649 | 1.28996e-05 | 0.270868 | 1.48317e-05 | 1.07735e-05 |
| routed 24-angle | transport,dsa,tsa,tsa_moment | no | 0.0187267 | 1.29986e-05 | 0.210924 | 1.03064e-05 | 8.31556e-06 |
| routed 24-angle | transport,dsa,tsa,tsa_moment | yes | 0.0152942 | 1.30497e-05 | 0.255620 | 9.23117e-06 | 7.84718e-06 |

## Teacher Reference On Mixed

| Run | Mixed one-step | Mixed stationary residual | Mixed FGMRES |
| --- | ---: | ---: | ---: |
| committed full12 UNet | 0.124534 | 7.82266e-06 | 7.54405e-06 |
| teacher1200 UNet | 0.174053 | 7.55108e-06 | 7.35536e-06 |
| teacher1800 UNet | 0.192250 | 7.32759e-06 | 7.27767e-06 |
| best current routed mixed one-step (`transport,dsa,tsa`) | 0.193663 | 9.85935e-06 | 8.17534e-06 |
| best current routed mixed FGMRES (`transport,dsa,tsa,tsa_moment` + residual summary) | 0.255620 | 9.23117e-06 | 7.84718e-06 |

## Findings

1. The pulled update does not preserve the old mixed-case plain UNet behavior.

   The current plain size-64 rerun regressed badly on mixed relative to the committed real-stacked baseline. Mixed one-step rose from `0.124534` to `0.276063` (+121.7%), and mixed stationary residual rose from `7.82e-06` to `2.50e-01`. Mixed FGMRES stayed numerically close (`7.54e-06` to `7.85e-06`), so the main failure is in the direct learned correction and stationary fixed-point behavior rather than the final Krylov floor. The 24-angle real-stacked plain control shows the same pattern (`0.303355` one-step, `1.39156e-01` stationary), so this is not just a `complex`-mode reporting artifact.

2. The router recovers most of the new mixed-case collapse, but the best variant depends on which metric matters.

   Relative to the 24-angle plain control, all routed variants except `transport,dsa,tsa` with residual summary recovered most of the mixed stationary collapse. The best mixed one-step among routed models is the simpler no-moment, no-residual router (`0.193663`), a 36.2% improvement over the plain 24-angle control (`0.303355`). The best routed mixed FGMRES is the `tsa_moment` + residual-summary variant (`7.84718e-06`), an 11.4% improvement over the plain 24-angle control (`8.85221e-06`).

3. `tsa_moment` and residual summaries help anisotropic much more than mixed.

   On anisotropic one-step, the strongest routed model is `transport,dsa,tsa,tsa_moment` with residual summary at `0.0152942`, a 78.1% improvement over the committed full12 learned UNet baseline (`0.0698968`). That gain does not translate into a clear solver-tail gain: anisotropic FGMRES remains around `1.28e-05` to `1.30e-05`, close to the committed baseline `1.26946e-05`.

4. No current routed configuration establishes a new mixed learned best against prior committed results.

   The best current routed mixed one-step (`0.193663`) is still 55.5% worse than the committed full12 learned UNet result (`0.124534`) and 11.3% worse than the teacher1200 run (`0.174053`). The best current routed mixed FGMRES (`7.84718e-06`) is still 4.0% worse than the committed full12 learned UNet (`7.54405e-06`) and 6.7% worse than teacher1200 (`7.35536e-06`).

5. The residual-summary branch is currently a mixed-case tradeoff, not a clear win.

   Residual summaries improved anisotropic one-step in both candidate sets, but they harmed mixed one-step in both candidate sets. For the no-moment router, adding residual summary moved mixed one-step from `0.193663` to `0.270868` and worsened mixed FGMRES from `8.17534e-06` to `1.07735e-05`. For the moment-augmented router, adding residual summary improved mixed stationary/FGMRES slightly (`1.03064e-05` to `9.23117e-06`, `8.31556e-06` to `7.84718e-06`) while still worsening mixed one-step (`0.210924` to `0.255620`).

## Interpretation

The routed-base change is directionally useful, but it is not yet a net mixed-case improvement over the previously committed learned UNet results.

- If the target metric is anisotropic one-step quality, the routed moment + residual model is the strongest new configuration.
- If the target metric is mixed one-step quality, the best current routed configuration is the simpler `transport,dsa,tsa` router without residual summaries.
- If the target metric is mixed final solver residual, the routed moment + residual model is the strongest routed variant, but it still does not beat the older committed mixed learned baseline or the committed teacher runs.

The practical conclusion is that the router should be kept as the main follow-up branch, but the current update should be treated as an anisotropic improvement and a mixed-case recovery mechanism, not as a new mixed-case state of the art.

## Reproducibility Note

`rte/npbs_unet_ablation.py` uses a fixed default CLI seed (`--seed 20260519`), but it also enables CUDA benchmarking and does not force deterministic kernels. In this session, the residual-conditioned routed variants showed noticeable mixed-case variation across repeated GPU reruns of the same command. The tables above therefore use the CSV artifacts produced by the latest completed commands, and any claim about the residual-summary branch should be validated with repeated-seed or deterministic reruns before being treated as final.

## Recommendation

For the next mixed-focused iteration, the most defensible path is:

1. Keep the routed-base branch.
2. Use `transport,dsa,tsa` without residual summaries as the mixed one-step starting point.
3. Use `transport,dsa,tsa,tsa_moment` with residual summaries only if the objective is solver-tail quality and the result is revalidated across repeated seeds.
4. Do not replace the older committed mixed learned baseline with the current routed models yet.