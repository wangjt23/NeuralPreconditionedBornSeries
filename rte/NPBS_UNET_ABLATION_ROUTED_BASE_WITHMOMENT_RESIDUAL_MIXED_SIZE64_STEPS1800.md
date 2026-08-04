# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['mixed']`
- `sizes`: `[64]`
- `n_angles`: `24`
- `base_kinds`: `['transport_tsa']`
- `tsa_harmonics`: `None`
- `forms`: `['defect2']`
- `moment_harmonics`: `4`
- `moment_weight_scale`: `0.5`
- `moment_transport_relaxation`: `1.0`
- `moment_synthetic_relaxation`: `1.0`
- `learn_scalar_moment`: `False`
- `learned_base_router`: `True`
- `learned_base_candidates`: `['transport', 'dsa', 'tsa', 'tsa_moment']`
- `learned_base_width`: `32`
- `learned_base_blocks`: `2`
- `learned_base_temperature`: `1.0`
- `learned_base_route_scale`: `4.0`
- `learned_base_relaxation_scale`: `0.5`
- `learned_base_use_residual`: `True`
- `learned_base_tsa_harmonics`: `None`
- `defect_relaxations`: `[1.0]`
- `baseline_depths`: `[1, 2, 3, 4]`
- `baseline_relaxations`: `[1.0]`
- `eta_scale`: `0.1`
- `width`: `16`
- `levels`: `3`
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
- `output_csv`: `rte/npbs_unet_ablation_routed_base_withmoment_residual_mixed_size64_steps1800.csv`
- `trace_csv`: ``
- `output_md`: `rte/NPBS_UNET_ABLATION_ROUTED_BASE_WITHMOMENT_RESIDUAL_MIXED_SIZE64_STEPS1800.md`

## Terminology

- `transport` is the shifted/reference transport inverse `G_eta`.
- `transport_dsa` applies one reference transport correction followed by one scalar diffusion synthetic correction.
- `transport_tsa` applies one reference transport correction followed by one reference transport-scattering synthetic correction. `tsa_harmonics=None` keeps all angular Fourier modes; an integer such as `2` keeps only the lowest angular harmonics.
- `transport_tsa_moment` applies one reference transport correction followed by a closed Galerkin angular Fourier synthetic correction `Q_K (Q_K^* A_ref Q_K)^{-1} Q_K^*` on the post-transport residual. This differs from `transport_tsa` with `tsa_harmonics=K`, which only projects the RHS before applying the full angular reference inverse.
- `dsaK` and `tsaK` mean `K` residual-correction applications of the selected base preconditioner. Starting with `z=0` and residual `e_0=r`, each stage applies `z <- z + omega B_h e_j` and `e_{j+1} <- e_j - A_h(omega B_h e_j)`. A suffix such as `-w0.5` uses relaxation `omega=0.5`; no suffix means `omega=1`.
- `UNet-defect2-w1` uses `z0=B_h r`, `z1=B_h(r-A_h z0)`, and returns `z0 + z1 + C_theta(z0,z1,coeff)`. It costs two base applications plus the U-Net evaluation per preconditioner call.
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
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa1 | 4.068e-01 | 4.279e-01 | 4.264e-04 | 2.168e-04 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa2 | 1.736e-01 | 2.849e-01 | 1.022e-05 | 8.103e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa3 | 1.195e-01 | 2.360e-01 | 7.222e-06 | 7.419e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa4 | 9.690e-02 | 2.019e-01 | 7.746e-06 | 7.416e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | RoutedUNet-defect2-w1 | 1.049e-01 | 9.664e-02 | 7.480e-06 | 7.483e-06 | nan | nan | 0.1633099913597107 | 0.010915818624198437 |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa1 | 2.607e-01 | 4.611e-01 | 3.461e-04 | 1.570e-04 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa2 | 1.789e-01 | 3.605e-01 | 1.001e-05 | 7.985e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa3 | 1.436e-01 | 3.011e-01 | 7.786e-06 | 7.506e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa4 | 1.147e-01 | 2.541e-01 | 7.232e-06 | 7.239e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | RoutedUNet-defect2-w1 | 2.066e-01 | 2.136e-01 | 2.299e-05 | 8.509e-06 | nan | nan | 0.1633099913597107 | 0.010915818624198437 |
