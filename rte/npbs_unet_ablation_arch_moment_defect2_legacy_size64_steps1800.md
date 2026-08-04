# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['anisotropic', 'mixed']`
- `sizes`: `[64]`
- `n_angles`: `24`
- `base_kinds`: `['transport_tsa_moment']`
- `tsa_harmonics`: `None`
- `forms`: `['moment_defect2']`
- `moment_harmonics`: `4`
- `moment_weight_scale`: `0.5`
- `moment_transport_relaxation`: `1.0`
- `moment_synthetic_relaxation`: `1.0`
- `learn_scalar_moment`: `False`
- `learned_base_router`: `False`
- `learned_base_candidates`: `['transport', 'dsa', 'tsa', 'tsa_moment']`
- `learned_base_width`: `32`
- `learned_base_blocks`: `2`
- `learned_base_temperature`: `1.0`
- `learned_base_route_scale`: `4.0`
- `learned_base_relaxation_scale`: `0.5`
- `learned_base_use_residual`: `False`
- `learned_base_tsa_harmonics`: `None`
- `defect_relaxations`: `[1.0]`
- `baseline_depths`: `[1, 2, 3, 4]`
- `baseline_relaxations`: `[1.0]`
- `eta_scale`: `0.1`
- `width`: `16`
- `levels`: `3`
- `gate_scale`: `0.1`
- `scale_aware_gates`: `False`
- `scale_gate_channel_mix`: `False`
- `extended_coeff_features`: `False`
- `steps`: `1800`
- `lr`: `0.002`
- `weight_decay`: `0.0`
- `scheduler`: `cosine`
- `scheduler_step_size`: `0`
- `scheduler_gamma`: `0.5`
- `scheduler_eta_min`: `None`
- `scheduler_min_lr_ratio`: `0.01`
- `scheduler_pct_start`: `0.1`
- `log_steps`: `0`
- `residual_mode`: `pool`
- `residuals_per_step`: `16`
- `rollout_samples_per_step`: `8`
- `n_rollout`: `6`
- `n_random`: `6`
- `random_modes`: `5`
- `aux_sources`: `0`
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
- `output_init`: `zero`
- `output_csv`: `rte/npbs_unet_ablation_arch_moment_defect2_legacy_size64_steps1800.csv`
- `trace_csv`: ``
- `output_md`: `rte/npbs_unet_ablation_arch_moment_defect2_legacy_size64_steps1800.md`

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
| 64 | transport_tsa_moment | 4 | real-stacked | anisotropic | train | tsa_moment1 | 5.676e-01 | 5.282e-01 | 7.862e+09 | 1.290e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | anisotropic | train | tsa_moment2 | 2.089e-01 | 3.321e-01 | inf | 5.793e-03 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | anisotropic | train | tsa_moment3 | 1.009e-01 | 2.217e-01 | nan | 1.217e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | anisotropic | train | tsa_moment4 | 5.950e-02 | 1.502e-01 | nan | 5.636e-03 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | anisotropic | train | MomentUNet-K4-w1 | 1.213e-01 | 6.380e-02 | inf | 1.949e-03 | nan | nan | 0.07511336356401443 | 0.007051261141896248 |
| 64 | transport_tsa_moment | 4 | real-stacked | anisotropic | heldout | tsa_moment1 | 5.669e-01 | 5.294e-01 | 7.282e+09 | 1.318e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | anisotropic | heldout | tsa_moment2 | 2.082e-01 | 3.328e-01 | inf | 5.650e-03 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | anisotropic | heldout | tsa_moment3 | 1.006e-01 | 2.223e-01 | nan | 1.301e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | anisotropic | heldout | tsa_moment4 | 5.950e-02 | 1.506e-01 | nan | 5.474e-03 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | anisotropic | heldout | MomentUNet-K4-w1 | 1.252e-01 | 6.464e-02 | inf | 4.312e-03 | nan | nan | 0.07511336356401443 | 0.007051261141896248 |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | train | tsa_moment1 | 4.459e-01 | 4.299e-01 | 2.400e-03 | 4.926e-03 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | train | tsa_moment2 | 2.339e-01 | 2.860e-01 | 8.157e-03 | 1.364e-04 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | train | tsa_moment3 | 1.543e-01 | 2.363e-01 | 7.107e+00 | 2.527e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | train | tsa_moment4 | 1.191e-01 | 2.014e-01 | 3.326e+03 | 2.756e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | train | MomentUNet-K4-w1 | 1.605e-01 | 1.371e-01 | 3.649e+01 | 1.948e-03 | nan | nan | 0.16077472269535065 | 0.07412932813167572 |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | tsa_moment1 | 2.700e-01 | 4.604e-01 | 1.734e-03 | 3.950e-03 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | tsa_moment2 | 1.879e-01 | 3.593e-01 | 1.681e-02 | 1.406e-04 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | tsa_moment3 | 1.590e-01 | 3.008e-01 | 5.236e+00 | 1.957e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | tsa_moment4 | 1.357e-01 | 2.546e-01 | 3.419e+03 | 2.528e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | MomentUNet-K4-w1 | 3.151e-01 | 2.920e-01 | 1.094e+01 | 1.788e-03 | nan | nan | 0.16077472269535065 | 0.07412932813167572 |
