# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['mixed']`
- `sizes`: `[64]`
- `n_angles`: `12`
- `base_kinds`: `['transport_tsa']`
- `tsa_harmonics`: `None`
- `forms`: `['defect2']`
- `moment_harmonics`: `2`
- `moment_weight_scale`: `0.5`
- `moment_transport_relaxation`: `1.0`
- `moment_synthetic_relaxation`: `1.0`
- `learn_scalar_moment`: `False`
- `learned_base_router`: `False`
- `learned_base_candidates`: `['transport', 'dsa', 'tsa', 'tsa_moment']`
- `learned_base_width`: `32`
- `learned_base_blocks`: `2`
- `learned_base_temperature`: `1.0`
- `learned_base_top_k`: `0`
- `learned_base_route_scale`: `4.0`
- `learned_base_relaxation_scale`: `0.5`
- `learned_base_use_residual`: `False`
- `learned_base_tsa_harmonics`: `None`
- `defect_relaxations`: `[1.0]`
- `baseline_depths`: `[2, 4]`
- `baseline_relaxations`: `[1.0]`
- `eta_scale`: `0.1`
- `width`: `16`
- `levels`: `4`
- `gate_scale`: `0.1`
- `scale_aware_gates`: `False`
- `scale_gate_channel_mix`: `False`
- `extended_coeff_features`: `False`
- `steps`: `2400`
- `lr`: `0.0005`
- `weight_decay`: `0.0001`
- `scheduler`: `onecycle`
- `scheduler_step_size`: `0`
- `scheduler_gamma`: `0.5`
- `scheduler_eta_min`: `None`
- `scheduler_min_lr_ratio`: `0.01`
- `scheduler_pct_start`: `0.1`
- `log_steps`: `200`
- `residual_mode`: `resample`
- `residuals_per_step`: `256`
- `rollout_samples_per_step`: `8`
- `n_rollout`: `6`
- `n_random`: `6`
- `random_modes`: `5`
- `aux_sources`: `0`
- `residual_dataset_path`: `rte/mixed_router_residual_dataset_size64_nangles12_xlarge_augmented.pt`
- `residual_dataset_device`: `gpu`
- `residual_dataset_gpu_max_fraction`: `0.5`
- `krylov_basis_vectors`: `0`
- `krylov_residual_vectors`: `0`
- `krylov_restart_residuals`: `0`
- `krylov_gmres_restart`: `20`
- `krylov_gmres_max_restarts`: `2`
- `metric_weight`: `0.0`
- `physical_weight`: `1.0`
- `correction_weight`: `1e-06`
- `rollout_weight`: `0.0`
- `rollout_steps`: `3`
- `rollout_decay`: `0.8`
- `rollout_space`: `metric`
- `stability_weight`: `0.0`
- `contraction_target`: `0.95`
- `stability_space`: `metric`
- `teacher_depth`: `4`
- `teacher_relaxation`: `1.0`
- `teacher_kind`: `transport_tsa`
- `teacher_weight`: `1.0`
- `maxiter`: `80`
- `fgmres_maxiter`: `40`
- `fgmres_mode`: `real-stacked`
- `tol`: `1e-06`
- `tol_grid`: `[0.0001, 1e-05]`
- `seed`: `20260519`
- `device`: `cuda:0`
- `allow_tf32`: `False`
- `output_init`: `small`
- `init_from_checkpoint`: `rte/checkpoints/npbs_unet_full12_legacy_teacherphysical_onecycle_xlargebank_mixed_size64_from14400_lr1e3_steps3600.pt`
- `resume_from_checkpoint`: ``
- `checkpoint_path`: `rte/checkpoints/npbs_unet_full12_legacy_teacherphysical_onecycle_xlargebank_mixed_size64_from18000_lr5e4_steps2400.pt`
- `checkpoint_every`: `200`
- `output_csv`: `rte/npbs_unet_ablation_full12_legacy_teacherphysical_onecycle_xlargebank_mixed_size64_from18000_lr5e4_steps2400.csv`
- `trace_csv`: `rte/npbs_unet_ablation_full12_legacy_teacherphysical_onecycle_xlargebank_mixed_size64_from18000_lr5e4_steps2400_traces.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_FULL12_LEGACY_TEACHERPHYSICAL_ONECYCLE_XLARGEBANK_MIXED_SIZE64_FROM18000_LR5E4_STEPS2400.md`

## Terminology

- `transport` is the shifted/reference transport inverse `G_eta`.
- `transport_dsa` applies one reference transport correction followed by one scalar diffusion synthetic correction.
- `transport_tsa` applies one reference transport correction followed by one reference transport-scattering synthetic correction. `tsa_harmonics=None` keeps all angular Fourier modes; an integer such as `2` keeps only the lowest angular harmonics.
- `transport_tsa_moment` applies one reference transport correction followed by a closed Galerkin angular Fourier synthetic correction `Q_K (Q_K^* A_ref Q_K)^{-1} Q_K^*` on the post-transport residual. This differs from `transport_tsa` with `tsa_harmonics=K`, which only projects the RHS before applying the full angular reference inverse.
- `dsaK` and `tsaK` mean `K` residual-correction applications of the selected base preconditioner. Starting with `z=0` and residual `e_0=r`, each stage applies `z <- z + omega B_h e_j` and `e_{j+1} <- e_j - A_h(omega B_h e_j)`. A suffix such as `-w0.5` uses relaxation `omega=0.5`; no suffix means `omega=1`.
- `UNet-defect2-w1` uses `z0=B_h r`, `z1=B_h(r-A_h z0)`, and returns `z0 + z1 + C_theta(z0,z1,coeff)`. It costs two base applications plus the U-Net evaluation per preconditioner call.
- `--scale-aware-gates` replaces the simple per-level coefficient gate by a gate that uses encoded coefficient features, raw physical coefficient features, mean/std/min/max summaries at that U-Net scale, a level/stage code, and a learnable bounded gain. The raw features include coarse-cell optical thickness, absorption ratio, transport cross-section, albedo, gradients, and anisotropy. This is intended to let coarse levels and fine levels react differently to the coefficient field.
- `--scale-gate-channel-mix` adds a bounded coefficient-conditioned channel-mixing update inside each scale-aware gate, so the gate can rotate/couple residual feature channels instead of only reweighting them diagonally. For fixed coefficients the residual path remains linear.
- `RoutedUNet-*` replaces the fixed `B_h` inside the U-Net by a trainable soft router over classical candidates such as `transport`, `dsa`, `tsa`, and `tsa_moment`. The router is conditioned on coefficient summaries, optionally residual summaries, and learns bounded transport/synthetic relaxation factors. The metric loss still uses the fixed classical base so that a learned metric cannot be gamed.
- `MomentUNet-K` uses the same defect2 structure but restricts the learned correction to angular Fourier modes `0,+/-1,...,+/-K` before prolonging back to ordinates.
- `WeightedTSA-MK` is a tiny learned structural model initialized as closed TSA-MK; it learns bounded real weights for each absolute angular mode in the synthetic correction.
- `CoeffWeightedTSA-MK` keeps the closed TSA-MK solve fixed, then uses a coefficient-conditioned nonlinear network to learn bounded physical-space weights for each absolute angular mode. It is nonlinear in coefficients but linear in residuals for fixed coefficients.
- `CoeffMixedTSA-MK` keeps the closed TSA-MK solve fixed, then uses a coefficient-conditioned local linear mixing matrix on retained moments. It is initialized to the identity map and can preserve the scalar moment exactly.
- `LearnedRP-TSA-MK` learns coefficient-conditioned restriction/prolongation maps around the fixed closed moment inverse. It is initialized to the identity and can preserve the scalar moment exactly.
- `ResidualSubspaceTSA-MK` uses the current post-transport residual to choose a bounded global mixing matrix on retained moments before the fixed closed inverse. This is a residual-dependent angular-subspace ablation and is nonlinear in the residual.
- By default `CoeffWeightedTSA-MK` freezes the scalar `m=0` synthetic weight to preserve the scalar diffusion moment; pass `--learn-scalar-moment` to ablate this constraint. `--moment-transport-relaxation` and `--moment-synthetic-relaxation` allow stable damped variants.
- `--rollout-space` and `--stability-space` choose whether rollout and stability penalties measure contraction in the base metric space or in the physical residual norm.
- `base-call equal` means equal counts of selected base-preconditioner applications inside one preconditioner call, excluding diagnostic metric evaluations, `A_h` applications, scalar reductions, and U-Net overhead. `UNet-defect2` is base-call equal to `dsa2`/`tsa2`; `dsa3`/`dsa4` and `tsa3`/`tsa4` are deeper classical work-budget baselines, not strict equal-cost matches.
- `Eval 1-step residual` is the post-training ratio `||r - A_h M r||_2 / ||r||_2` measured on the full train or heldout source after one preconditioner application.
- `Eval 1-step base-metric ratio` measures that same one-step residual after both numerator and denominator are mapped through the selected classical base metric used for the run.
- `Train batch residual start/end` is the mean one-step physical residual ratio on the sampled residual minibatches used during optimization. It is a training diagnostic and is not directly comparable to the full-source evaluation columns.
- `Train objective start/end` is the scalar optimization objective on those sampled minibatches, including whichever metric, physical, correction, rollout, stability, and teacher terms were active for the run.

## Quick Read

- Size `64`, case `mixed`, base `transport_tsa`, split `train`: best learned method `UNet-defect2-w1` reached eval 1-step residual 1.024e-01 vs `tsa2` 1.719e-01 vs `tsa4` 9.809e-02 (4.267e-03 above `tsa4`) ; stationary final residual 4.517e-02 and FGMRES final residual 7.532e-06.
- Size `64`, case `mixed`, base `transport_tsa`, split `heldout`: best learned method `UNet-defect2-w1` reached eval 1-step residual 1.085e-01 vs `tsa2` 1.790e-01 vs `tsa4` 1.145e-01 (5.962e-03 below `tsa4`) ; stationary final residual 1.075e-01 and FGMRES final residual 7.356e-06.

## Analysis

- This lower-LR continuation improved the one-step target again, but only incrementally. Relative to the previous continuation, train eval 1-step residual moved from `1.033e-01` to `1.024e-01`, and heldout moved from `1.093e-01` to `1.085e-01`.
- The remaining train gap to `tsa4` is now about `4.27e-03` (`1.024e-01` vs `9.809e-02`), down from about `5.23e-03` on the previous continuation. Heldout remains comfortably better than `tsa4`, and the heldout margin widened further from about `5.13e-03` to about `5.96e-03`.
- Base-metric evaluation also improved slightly on both splits (`1.990e-01` to `1.989e-01` on train and `2.418e-01` to `2.415e-01` on heldout). The model is already better than `tsa4` in that proxy, so the remaining issue is still the true post-application residual.
- The sampled training diagnostics were no longer aligned with the evaluation improvement on this run. Train batch residual end worsened from `1.894e-01` to `2.091e-01`, and train objective end worsened from `8.159e-02` to `9.517e-02`, yet the explicit eval one-step residual still improved. That mismatch means model-selection decisions should continue to rely on the eval columns rather than the minibatch diagnostics.
- Stationary fixed-point quality remains the main cost of this one-step-focused continuation path. It is still far worse than the classical baselines (`4.517e-02` train and `1.075e-01` heldout versus about `8e-06` for `tsa2`/`tsa4`), while FGMRES final residual stays classical.
- Because the explicit eval one-step metric still improved on both train and heldout, one more smaller checkpoint-based continuation is justified. Any further push should continue from the saved model rather than restarting.

## Results

| Grid | Base | TSA harmonics | FGMRES mode | Case | Split | Method | Eval 1-step residual | Eval 1-step base-metric ratio | Stationary final residual | FGMRES final residual | Stationary time to tol | FGMRES time to tol | Train batch residual start | Train batch residual end | Train objective start | Train objective end |
|---:|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa2 | 1.719e-01 | 2.856e-01 | 7.674e-06 | 7.376e-06 | nan | nan |  |  |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa4 | 9.809e-02 | 2.049e-01 | 7.802e-06 | 7.573e-06 | nan | nan |  |  |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | UNet-defect2-w1 | 1.024e-01 | 1.989e-01 | 4.517e-02 | 7.532e-06 | nan | nan | 2.046e-01 | 2.091e-01 | 8.772e-02 | 9.517e-02 |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa2 | 1.790e-01 | 3.604e-01 | 7.829e-06 | 7.374e-06 | nan | nan |  |  |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa4 | 1.145e-01 | 2.544e-01 | 7.836e-06 | 7.953e-06 | nan | nan |  |  |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | UNet-defect2-w1 | 1.085e-01 | 2.415e-01 | 1.075e-01 | 7.356e-06 | nan | nan | 2.046e-01 | 2.091e-01 | 8.772e-02 | 9.517e-02 |
