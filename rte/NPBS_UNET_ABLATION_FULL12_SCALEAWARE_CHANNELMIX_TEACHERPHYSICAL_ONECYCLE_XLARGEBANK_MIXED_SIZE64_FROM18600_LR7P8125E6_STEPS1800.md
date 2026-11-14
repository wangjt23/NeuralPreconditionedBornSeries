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
- `scale_aware_gates`: `True`
- `scale_gate_channel_mix`: `True`
- `extended_coeff_features`: `True`
- `steps`: `1800`
- `lr`: `7.8125e-06`
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
- `init_from_checkpoint`: `rte/checkpoints/npbs_unet_full12_scaleaware_channelmix_teacherphysical_onecycle_xlargebank_mixed_size64_from16800_lr1p5625e5_steps1800.pt`
- `resume_from_checkpoint`: ``
- `checkpoint_path`: `rte/checkpoints/npbs_unet_full12_scaleaware_channelmix_teacherphysical_onecycle_xlargebank_mixed_size64_from18600_lr7p8125e6_steps1800.pt`
- `checkpoint_every`: `200`
- `output_csv`: `rte/npbs_unet_ablation_full12_scaleaware_channelmix_teacherphysical_onecycle_xlargebank_mixed_size64_from18600_lr7p8125e6_steps1800.csv`
- `trace_csv`: `rte/npbs_unet_ablation_full12_scaleaware_channelmix_teacherphysical_onecycle_xlargebank_mixed_size64_from18600_lr7p8125e6_steps1800_traces.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_FULL12_SCALEAWARE_CHANNELMIX_TEACHERPHYSICAL_ONECYCLE_XLARGEBANK_MIXED_SIZE64_FROM18600_LR7P8125E6_STEPS1800.md`

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

- Size `64`, case `mixed`, base `transport_tsa`, split `train`: best learned method `ScaleGatedUNet-defect2-w1` reached eval 1-step residual 1.136e-01 vs `tsa2` 1.719e-01 vs `tsa4` 9.809e-02 (1.554e-02 above `tsa4`) ; stationary final residual 5.704e-05 and FGMRES final residual 7.481e-06.
- Size `64`, case `mixed`, base `transport_tsa`, split `heldout`: best learned method `ScaleGatedUNet-defect2-w1` reached eval 1-step residual 1.193e-01 vs `tsa2` 1.790e-01 vs `tsa4` 1.145e-01 (4.862e-03 above `tsa4`) ; stationary final residual 8.350e-05 and FGMRES final residual 7.225e-06.

## Analysis

- This final plateau probe at `7.8125e-6` still improved the plain scale-aware branch on both train and heldout, but only marginally. Relative to the previous `1.5625e-5` continuation, train eval one-step residual improved from `1.137757078e-01` to `1.136323512e-01`, a gain of about `1.43e-04`, and heldout improved from `1.196109802e-01` to `1.193209067e-01`, a gain of about `2.90e-04`.
- Relative to the original 2400-step plain scale-aware run, the cumulative gain is now very large: train improved from `1.542639583e-01` to `1.136323512e-01`, a total gain of about `4.06e-02`, and heldout improved from `1.480529010e-01` to `1.193209067e-01`, a total gain of about `2.87e-02`.
- The remaining gap to the best legacy continuation narrowed again, but only slightly. The branch is now about `1.20e-02` above the legacy best on train (`1.1363e-01` vs `1.0159e-01`) and about `1.16e-02` above it on heldout (`1.1932e-01` vs `1.0777e-01`). The heldout gap to `tsa4` is now about `4.86e-03`.
- Solver behavior stayed broadly classical but no longer improved monotonically with one-step quality. Train stationary residual stayed flat (`5.70e-05` to `5.70e-05`), train FGMRES worsened from `7.20e-06` to `7.48e-06`, heldout stationary residual worsened again from `7.89e-05` to `8.35e-05`, while heldout FGMRES improved from `7.50e-06` to `7.23e-06`.
- The minibatch diagnostics remained weak proxies for the target metric. Train batch one-step residual improved only slightly over the continuation (`2.083e-01` to `2.068e-01`), while the train objective again worsened modestly (`9.024e-02` to `9.261e-02`). Full-source evaluation remains the right continuation signal for this branch.
- Conclusion: this exact descending-LR continuation schedule has now effectively plateaued. The `7.8125e-6` probe produced a real but very small one-step gain, and the heldout solver-tail tradeoff remains mixed. Keep this checkpoint as the best current plain scale-aware result on this branch, and do not keep extending this exact schedule further unless the optimization setup changes.

## Results

| Grid | Base | TSA harmonics | FGMRES mode | Case | Split | Method | Eval 1-step residual | Eval 1-step base-metric ratio | Stationary final residual | FGMRES final residual | Stationary time to tol | FGMRES time to tol | Train batch residual start | Train batch residual end | Train objective start | Train objective end |
|---:|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa2 | 1.719e-01 | 2.856e-01 | 7.674e-06 | 7.376e-06 | nan | nan |  |  |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa4 | 9.809e-02 | 2.049e-01 | 7.802e-06 | 7.573e-06 | nan | nan |  |  |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | ScaleGatedUNet-defect2-w1 | 1.136e-01 | 2.017e-01 | 5.704e-05 | 7.481e-06 | nan | nan | 2.083e-01 | 2.068e-01 | 9.024e-02 | 9.261e-02 |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa2 | 1.790e-01 | 3.604e-01 | 7.829e-06 | 7.374e-06 | nan | nan |  |  |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa4 | 1.145e-01 | 2.544e-01 | 7.836e-06 | 7.953e-06 | nan | nan |  |  |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | ScaleGatedUNet-defect2-w1 | 1.193e-01 | 2.489e-01 | 8.350e-05 | 7.225e-06 | nan | nan | 2.083e-01 | 2.068e-01 | 9.024e-02 | 9.261e-02 |
