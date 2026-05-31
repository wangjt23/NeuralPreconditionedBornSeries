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
- `steps`: `1200`
- `lr`: `0.002`
- `weight_decay`: `0.0001`
- `scheduler`: `step`
- `scheduler_step_size`: `400`
- `scheduler_gamma`: `0.5`
- `scheduler_eta_min`: `None`
- `scheduler_min_lr_ratio`: `0.01`
- `scheduler_pct_start`: `0.1`
- `log_steps`: `200`
- `residual_mode`: `resample`
- `residuals_per_step`: `128`
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
- `output_csv`: `rte/npbs_unet_ablation_full12_legacy_teacherphysical_newbank_mixed_size64_steps1200.csv`
- `trace_csv`: `rte/npbs_unet_ablation_full12_legacy_teacherphysical_newbank_mixed_size64_steps1200_traces.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_FULL12_LEGACY_TEACHERPHYSICAL_NEWBANK_MIXED_SIZE64_STEPS1200.md`

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
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa2 | 1.719e-01 | 2.856e-01 | 7.674e-06 | 7.376e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa4 | 9.809e-02 | 2.049e-01 | 7.802e-06 | 7.573e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | UNet-defect2-w1 | 1.462e-01 | 2.092e-01 | 1.607e+01 | 7.795e-06 | nan | nan | 0.35022634267807007 | 0.1419816017150879 |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa2 | 1.790e-01 | 3.604e-01 | 7.829e-06 | 7.374e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa4 | 1.145e-01 | 2.544e-01 | 7.836e-06 | 7.953e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | UNet-defect2-w1 | 1.514e-01 | 2.631e-01 | 7.958e+00 | 8.182e-06 | nan | nan | 0.35022634267807007 | 0.1419816017150879 |
