# NPBS UNet Scale-Aware Architecture Ablation Report

## Scope

This report evaluates the full post-pull architecture update in `rte/npbs_unet_ablation.py` after the new `ScaleAwareCoeffGate` and optional `--scale-gate-channel-mix` landed.

The goal was to answer two questions with enough training to separate transient effects from real architectural behavior:

1. Does the new scale-aware gate improve the updated UNet families at 1800 steps?
2. If it does, which family actually benefits: plain `defect2`, routed `defect2`, residual, or `moment_defect2`?

All main runs below use:

- size 64
- cases `anisotropic` and `mixed`
- `n_angles=24`
- 1800 training steps
- physical-only loss (`metric_weight=0`, `physical_weight=1`, `correction_weight=0`, `rollout_weight=0`)
- cosine schedule
- `fgmres_mode=real-stacked`
- baseline depths `1 2 3 4`

For scale-aware runs, `rte/npbs_unet_ablation.py` automatically enables the extended coefficient feature map when `--scale-aware-gates` is set.

## Training And Generalization Protocol

The main table does **not** come from one jointly trained anisotropic+mixed model.

- The driver loops over `for case in args.cases`, builds a separate `rte = make_case(case, ...)`, and trains a separate model for each case.
- In other words, passing `--cases anisotropic mixed` runs two case-specific trainings in one command: one model on the anisotropic coefficient field and one different model on the mixed coefficient field.
- The `heldout` split is also **not** a new coefficient realization. It is a new source on the same fixed coefficient field for that case.

So the reported generalization is **source generalization within a case-specific coefficient field**, not cross-case generalization and not coefficient-distribution generalization.

Concretely:

- the anisotropic rows come from a model trained only on the anisotropic coefficient field
- the mixed rows come from a different model trained only on the mixed coefficient field
- `train` vs `heldout` means train-source vs heldout-source on that same case

This matters for interpretation: the current tables show which architecture works best once the coefficient regime is fixed, but they do **not** show whether one model generalizes across anisotropic and mixed regimes.

## Compared Runs

### Joint anisotropic+mixed architecture matrix

- `rte/npbs_unet_ablation_arch_residual_legacy_size64_steps1800.csv`
- `rte/npbs_unet_ablation_arch_residual_scale_size64_steps1800.csv`
- `rte/npbs_unet_ablation_arch_residual_scale_mix_size64_steps1800.csv`
- `rte/npbs_unet_ablation_arch_defect2_legacy_size64_steps1800.csv`
- `rte/npbs_unet_ablation_arch_defect2_scale_size64_steps1800.csv`
- `rte/npbs_unet_ablation_arch_defect2_scale_mix_size64_steps1800.csv`
- `rte/npbs_unet_ablation_arch_routed_defect2_legacy_size64_steps1800.csv`
- `rte/npbs_unet_ablation_arch_routed_defect2_scale_size64_steps1800.csv`
- `rte/npbs_unet_ablation_arch_routed_defect2_scale_mix_size64_steps1800.csv`
- `rte/npbs_unet_ablation_arch_moment_defect2_legacy_size64_steps1800.csv`
- `rte/npbs_unet_ablation_arch_moment_defect2_scale_size64_steps1800.csv`
- `rte/npbs_unet_ablation_arch_moment_defect2_scale_mix_size64_steps1800.csv`

### Earlier 1800-step mixed-only controls

These are not the main joint-case comparison because they were trained only on `mixed`, but they are useful for checking the earlier "train longer" concern.

- `rte/npbs_unet_ablation_tsa_full_physical24_plain_mixed_size64_steps1800.csv`
- `rte/npbs_unet_ablation_routed_base_nomoment_mixed_size64_steps1800.csv`
- `rte/npbs_unet_ablation_routed_base_withmoment_residual_mixed_size64_steps1800.csv`

## Metric Definitions

- `one-step` means the one-step true residual ratio already used elsewhere in the NPBS reports: `||A M r - r|| / ||r||`.
- `stationary residual` means the final relative residual `||b - A psi_k|| / ||b||` after applying the preconditioner as a stationary fixed-point iteration, `psi_{k+1} = psi_k + M(b - A psi_k)`, for up to 60 iterations or until tolerance.
- `FGMRES` means the final relative residual after preconditioned FGMRES, with up to 25 iterations in these ablations.

Lower is better for all three metrics.

## Main Results

| Family | Variant | Anisotropic one-step | Anisotropic stationary residual | Anisotropic FGMRES | Mixed one-step | Mixed stationary residual | Mixed FGMRES |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| residual | legacy | 0.168393 | 1.38398e10 | 1.25668e-05 | 0.752852 | inf | 9.67823e-02 |
| residual | scale-aware | 0.151060 | nan | 2.28672e-05 | 0.760133 | inf | 2.03149e-01 |
| residual | scale-aware + mix | 0.141882 | 3.27889e15 | 1.71340e-02 | 0.775004 | inf | 2.22822e-01 |
| defect2 | legacy | 0.0195834 | 1.32092e-05 | 1.26397e-05 | 0.347401 | 1.20114e-01 | 1.07557e-05 |
| defect2 | scale-aware | 0.0224676 | 1.28983e-05 | 1.31110e-05 | 0.317152 | 5.87174e-03 | 1.06491e-05 |
| defect2 | scale-aware + mix | 0.0201178 | 1.30235e-05 | 1.29340e-05 | 0.316998 | 7.10782e-02 | 1.11016e-05 |
| routed defect2 | legacy | 0.0118051 | 1.33966e-05 | 1.27908e-05 | 0.207665 | 8.87424e-05 | 8.49612e-06 |
| routed defect2 | scale-aware | 0.0109506 | 1.29564e-05 | 1.31024e-05 | 0.228504 | 1.51374e-05 | 8.45834e-06 |
| routed defect2 | scale-aware + mix | 0.0114713 | 1.24234e-05 | 1.30038e-05 | 0.208233 | 1.13644e-05 | 8.38179e-06 |
| moment defect2 | legacy | 0.125239 | inf | 4.31155e-03 | 0.315083 | 1.09433e01 | 1.78789e-03 |
| moment defect2 | scale-aware | 0.126815 | inf | 2.89952e-03 | 0.319900 | 9.73948e00 | 1.79327e-03 |
| moment defect2 | scale-aware + mix | 0.125959 | inf | 4.67495e-03 | 0.295953 | 7.67911e00 | 1.62085e-03 |

## Mixed-Only 1800-Step Controls

| Run | Training cases | Mixed one-step | Mixed stationary residual | Mixed FGMRES |
| --- | --- | ---: | ---: | ---: |
| plain 24-angle UNet | mixed only | 0.299118 | 1.91217e-01 | 1.10656e-05 |
| routed no-moment UNet | mixed only | 0.230640 | 1.04330e-05 | 8.09792e-06 |
| routed moment + residual UNet | mixed only | 0.206627 | 2.29912e-05 | 8.50877e-06 |

These controls show that 1800 steps were enough to improve the routed family on mixed-only training, but they also show that longer training does not automatically rescue weak architectures. The joint-case architecture matrix below follows the same conclusion.

## Findings

1. The residual family remains unusable, and the new gate does not rescue it.

   The legacy residual learned model was already bad on mixed (`0.752852`, `inf`, `9.68e-02`). Scale-aware gating made mixed FGMRES even worse (`2.03e-01`), and channel mixing stayed catastrophic (`0.775004`, `inf`, `2.23e-01`). On anisotropic, the residual family also showed hard instability (`1.38e10`, `nan`, `3.28e15` stationary residuals across the three variants). This family should be removed from the main candidate set.

2. Plain `defect2` does benefit from scale-aware gating, but only in a narrow way.

   Moving from legacy to scale-aware plain `defect2` improved mixed one-step from `0.347401` to `0.317152` and, more importantly, improved mixed stationary residual from `1.20e-01` to `5.87e-03`. That is a real stability gain. However, channel mixing did not add another clear win: mixed one-step stayed flat (`0.316998`), while mixed stationary worsened back to `7.11e-02` and mixed FGMRES also regressed slightly. The scale-aware gate helps plain `defect2`; channel mixing does not.

3. Routed `defect2` is the only family where the new architecture is clearly worth keeping.

   The routed legacy model was already the strongest branch in the joint-case matrix (`mixed 0.207665 / 8.87e-05 / 8.50e-06`). Adding the scale-aware gate alone improved mixed stationary residual by almost an order of magnitude (`8.87e-05` to `1.51e-05`) and slightly improved mixed FGMRES (`8.50e-06` to `8.46e-06`), but it worsened mixed one-step (`0.207665` to `0.228504`). Adding channel mixing fixed that tradeoff: mixed one-step returned to `0.208233`, mixed stationary improved further to `1.14e-05`, and mixed FGMRES improved to `8.38e-06`.

   Within the post-pull joint-case study, `routed defect2 + scale-aware gate + channel mixing` is the best overall architecture.

4. `moment_defect2` remains a negative result.

   All three learned moment variants were much worse than routed `defect2`, and they were not competitive solver-tail models. The least bad mixed learned moment result was the scale+mix variant (`0.295953`, `7.67911`, `1.62085e-03`), but that is still about two orders of magnitude worse in FGMRES than routed `defect2` and worse than several classical `tsa_moment` baselines already present in the same CSVs. The new gate changes the numbers slightly but does not change the conclusion.

5. 1800 steps were enough to separate real architectural signal from wishful thinking.

   Longer training did not rescue residual or moment variants. It sharpened the same ranking seen qualitatively in the earlier work: routed `defect2` is the viable family, plain `defect2` is salvageable with the scale-aware gate, and residual / moment learned variants are not good use of compute under this protocol.

## Interpretation

The post-pull architecture update is not a uniform improvement across the UNet family.

- `ScaleAwareCoeffGate` is useful for plain `defect2` because it materially improves mixed stationary behavior, but it does not create a new best overall model by itself.
- `ScaleAwareCoeffGate` is most useful inside routed `defect2`, where it improves solver-tail stability.
- `--scale-gate-channel-mix` only pays off in routed `defect2`. In plain `defect2` it is neutral-to-negative, and in residual / moment families it does not help.

The most defensible practical conclusion is:

1. Keep the routed `defect2` branch as the primary learned architecture.
2. Keep the scale-aware gate.
3. Keep channel mixing only for routed `defect2`.
4. Drop residual learned variants from follow-up studies.
5. Do not spend more architecture time on `moment_defect2` until there is a separate reason to revisit its formulation.

## Relation To Earlier 1800-Step Controls

The earlier mixed-only 1800-step routed controls are still informative.

- The best mixed-only one-step remains the earlier routed moment+residual control at `0.206627`.
- The best mixed-only FGMRES remains the earlier routed no-moment control at `8.09792e-06`.
- The new joint-case routed scale+mix model (`0.208233`, `1.13644e-05`, `8.38179e-06`) does not dominate both older mixed-only controls at once, but it is the best-balanced post-pull architecture inside the harder joint anisotropic+mixed training setting.

That means the new architecture should be treated as the best current post-pull joint-case model, not as a blanket replacement for every older 1800-step routed result.

## Reproducibility Note

`rte/npbs_unet_ablation.py` fixes the CLI seed but still enables CUDA benchmarking and does not force deterministic kernels. The failures in the residual and moment families are large enough that this does not affect the conclusion. The tighter routed `defect2` comparisons are still worth rechecking with repeated seeds if the next decision is whether to replace an established reference model.