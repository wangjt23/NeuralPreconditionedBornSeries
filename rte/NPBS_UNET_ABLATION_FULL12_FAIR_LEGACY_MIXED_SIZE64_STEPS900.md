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
- `baseline_depths`: `[1, 2, 3, 4]`
- `baseline_relaxations`: `[1.0, 0.5]`
- `eta_scale`: `0.1`
- `width`: `16`
- `levels`: `4`
- `gate_scale`: `0.1`
- `scale_aware_gates`: `False`
- `scale_gate_channel_mix`: `False`
- `extended_coeff_features`: `False`
- `steps`: `900`
- `lr`: `0.002`
- `weight_decay`: `0.0001`
- `scheduler`: `step`
- `scheduler_step_size`: `300`
- `scheduler_gamma`: `0.5`
- `scheduler_eta_min`: `None`
- `scheduler_min_lr_ratio`: `0.01`
- `scheduler_pct_start`: `0.1`
- `log_steps`: `0`
- `residual_mode`: `resample`
- `residuals_per_step`: `64`
- `rollout_samples_per_step`: `8`
- `n_rollout`: `6`
- `n_random`: `6`
- `random_modes`: `5`
- `aux_sources`: `0`
- `residual_dataset_path`: `rte/mixed_router_residual_dataset_size64_nangles12.pt`
- `krylov_basis_vectors`: `0`
- `krylov_residual_vectors`: `0`
- `krylov_restart_residuals`: `0`
- `krylov_gmres_restart`: `20`
- `krylov_gmres_max_restarts`: `2`
- `metric_weight`: `0.0`
- `physical_weight`: `1.0`
- `correction_weight`: `0.0`
- `rollout_weight`: `0.0`
- `rollout_steps`: `1`
- `rollout_decay`: `1.0`
- `rollout_space`: `metric`
- `stability_weight`: `0.0`
- `contraction_target`: `0.98`
- `stability_space`: `metric`
- `teacher_depth`: `0`
- `teacher_relaxation`: `1.0`
- `teacher_kind`: ``
- `teacher_weight`: `0.0`
- `maxiter`: `60`
- `fgmres_maxiter`: `25`
- `fgmres_mode`: `real-stacked`
- `tol`: `1e-06`
- `tol_grid`: `[]`
- `seed`: `20260519`
- `device`: `cuda:0`
- `allow_tf32`: `False`
- `output_init`: `small`
- `output_csv`: `rte/npbs_unet_ablation_full12_fair_legacy_mixed_size64_steps900.csv`
- `trace_csv`: ``
- `output_md`: `rte/NPBS_UNET_ABLATION_FULL12_FAIR_LEGACY_MIXED_SIZE64_STEPS900.md`

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

## Results

| Size | Base | K | FGMRES | Case | Split | Method | one-step true | one-step base metric | stat final | FGMRES final | stat ttol | FGMRES ttol | loss0 | loss1 |
|---:|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa1 | 4.039e-01 | 4.276e-01 | 4.511e-04 | 8.625e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa1-w0.5 | 6.207e-01 | 6.679e-01 | 4.840e-03 | 8.625e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa2 | 1.719e-01 | 2.856e-01 | 9.550e-06 | 7.310e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa2-w0.5 | 4.082e-01 | 4.780e-01 | 4.893e-04 | 1.272e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa3 | 1.193e-01 | 2.383e-01 | 7.653e-06 | 7.460e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa3-w0.5 | 2.814e-01 | 3.704e-01 | 5.743e-05 | 8.002e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa4 | 9.809e-02 | 2.049e-01 | 7.708e-06 | 7.519e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa4-w0.5 | 2.043e-01 | 3.088e-01 | 9.819e-06 | 7.568e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | UNet-defect2-w1 | 1.804e-01 | 1.255e-01 | 5.487e+09 | 1.165e-05 | nan | nan | 0.22630351781845093 | 0.0487653948366642 |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa1 | 2.608e-01 | 4.605e-01 | 4.056e-04 | 6.892e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa1-w0.5 | 5.624e-01 | 6.766e-01 | 4.587e-03 | 6.892e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa2 | 1.790e-01 | 3.604e-01 | 9.315e-06 | 7.603e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa2-w0.5 | 3.470e-01 | 5.132e-01 | 4.388e-04 | 9.665e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa3 | 1.432e-01 | 3.012e-01 | 7.778e-06 | 7.362e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa3-w0.5 | 2.425e-01 | 4.260e-01 | 5.432e-05 | 7.511e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa4 | 1.145e-01 | 2.544e-01 | 7.676e-06 | 7.935e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa4-w0.5 | 1.907e-01 | 3.735e-01 | 9.955e-06 | 8.014e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | UNet-defect2-w1 | 1.632e-01 | 1.488e-01 | 1.053e+10 | 9.041e-06 | nan | nan | 0.22630351781845093 | 0.0487653948366642 |
