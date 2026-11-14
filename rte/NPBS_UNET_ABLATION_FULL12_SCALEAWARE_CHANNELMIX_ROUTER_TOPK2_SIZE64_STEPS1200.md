# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['diffusive', 'heterogeneous', 'anisotropic', 'mixed']`
- `sizes`: `[64]`
- `n_angles`: `12`
- `base_kinds`: `['transport_dsa']`
- `tsa_harmonics`: `None`
- `forms`: `['defect2']`
- `moment_harmonics`: `2`
- `moment_weight_scale`: `0.5`
- `moment_transport_relaxation`: `1.0`
- `moment_synthetic_relaxation`: `1.0`
- `learn_scalar_moment`: `False`
- `learned_base_router`: `True`
- `learned_base_candidates`: `['transport', 'dsa', 'tsa', 'tsa_moment']`
- `learned_base_width`: `32`
- `learned_base_blocks`: `2`
- `learned_base_temperature`: `1.0`
- `learned_base_top_k`: `2`
- `learned_base_route_scale`: `4.0`
- `learned_base_relaxation_scale`: `0.5`
- `learned_base_use_residual`: `True`
- `learned_base_tsa_harmonics`: `4`
- `defect_relaxations`: `[1.0]`
- `baseline_depths`: `[1, 2]`
- `baseline_relaxations`: `[1.0]`
- `eta_scale`: `0.1`
- `width`: `16`
- `levels`: `3`
- `angular_mode_width`: `4`
- `angular_mix_rank`: `2`
- `angular_mix_scale`: `0.25`
- `angular_mode_mix_scale`: `0.05`
- `angular_residual_gate_scale`: `0.25`
- `angular_gate_scale`: `0.1`
- `gate_scale`: `0.1`
- `scale_aware_gates`: `True`
- `scale_gate_channel_mix`: `True`
- `extended_coeff_features`: `False`
- `rom_rank`: `0`
- `rom_width`: `0`
- `rom_scale`: `0.1`
- `rom_use_residual_summary`: `False`
- `steps`: `1200`
- `lr`: `0.002`
- `weight_decay`: `0.0`
- `scheduler`: `cosine`
- `scheduler_step_size`: `0`
- `scheduler_gamma`: `0.5`
- `scheduler_eta_min`: `None`
- `scheduler_min_lr_ratio`: `0.01`
- `scheduler_pct_start`: `0.1`
- `log_steps`: `0`
- `residual_mode`: `resample`
- `residuals_per_step`: `64`
- `rollout_samples_per_step`: `32`
- `n_rollout`: `8`
- `n_random`: `6`
- `random_modes`: `5`
- `aux_sources`: `2`
- `residual_dataset_path`: ``
- `residual_dataset_device`: `auto`
- `residual_dataset_gpu_max_fraction`: `0.5`
- `krylov_basis_vectors`: `0`
- `krylov_residual_vectors`: `0`
- `krylov_restart_residuals`: `0`
- `krylov_gmres_restart`: `20`
- `krylov_gmres_max_restarts`: `2`
- `metric_weight`: `1.0`
- `physical_weight`: `0.5`
- `correction_weight`: `1e-06`
- `rollout_weight`: `0.0`
- `rollout_steps`: `1`
- `rollout_decay`: `1.0`
- `rollout_space`: `metric`
- `stability_weight`: `0.0`
- `contraction_target`: `0.98`
- `stability_space`: `metric`
- `teacher_depth`: `4`
- `teacher_relaxation`: `1.0`
- `teacher_kind`: `transport_dsa`
- `teacher_weight`: `0.25`
- `maxiter`: `40`
- `fgmres_maxiter`: `20`
- `fgmres_mode`: `real-stacked`
- `tol`: `1e-06`
- `tol_grid`: `[]`
- `seed`: `20260519`
- `device`: `cuda:0`
- `allow_tf32`: `False`
- `output_init`: `small`
- `init_from_checkpoint`: ``
- `resume_from_checkpoint`: ``
- `checkpoint_path`: ``
- `checkpoint_every`: `0`
- `output_csv`: `rte/npbs_unet_ablation_full12_scaleaware_channelmix_router_topk2_size64_steps1200.csv`
- `trace_csv`: `rte/npbs_unet_ablation_full12_scaleaware_channelmix_router_topk2_size64_steps1200_traces.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_FULL12_SCALEAWARE_CHANNELMIX_ROUTER_TOPK2_SIZE64_STEPS1200.md`

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
- `ROMK-UNet-*` adds a ROM-style global low-rank correction branch of rank `K` to the local U-Net correction. The branch produces residual-linear rank basis maps and coefficient-conditioned bounded rank weights, is initialized to zero, and is intended to approximate a reduced ideal-correction space while retaining the classical SA base. `--rom-use-residual-summary` makes the rank weights residual-dependent and should be treated as a nonlinear-preconditioner ablation.
- `MomentUNet-K` uses the same defect2 structure but restricts the learned correction to angular Fourier modes `0,+/-1,...,+/-K` before prolonging back to ordinates.
- `WeightedTSA-MK` is a tiny learned structural model initialized as closed TSA-MK; it learns bounded real weights for each absolute angular mode in the synthetic correction.
- `CoeffWeightedTSA-MK` keeps the closed TSA-MK solve fixed, then uses a coefficient-conditioned nonlinear network to learn bounded physical-space weights for each absolute angular mode. It is nonlinear in coefficients but linear in residuals for fixed coefficients.
- `CoeffMixedTSA-MK` keeps the closed TSA-MK solve fixed, then uses a coefficient-conditioned local linear mixing matrix on retained moments. It is initialized to the identity map and can preserve the scalar moment exactly.
- `LearnedRP-TSA-MK` learns coefficient-conditioned restriction/prolongation maps around the fixed closed moment inverse. It is initialized to the identity and can preserve the scalar moment exactly.
- `ResidualSubspaceTSA-MK` uses the current post-transport residual to choose a bounded global mixing matrix on retained moments before the fixed closed inverse. This is a residual-dependent angular-subspace ablation and is nonlinear in the residual.
- `AngularSpectralDefect2` keeps all angular Fourier modes in grouped periodic U-Net channels. Its local path is mode-preserving; a bounded coefficient-conditioned low-rank mixer learns mode coupling, and a bounded residual-energy gate selects the active angular subspace. With zero output initialization it is exactly the classical defect2 map.
- By default `CoeffWeightedTSA-MK` freezes the scalar `m=0` synthetic weight to preserve the scalar diffusion moment; pass `--learn-scalar-moment` to ablate this constraint. `--moment-transport-relaxation` and `--moment-synthetic-relaxation` allow stable damped variants.
- `--rollout-space` and `--stability-space` choose whether rollout and stability penalties measure contraction in the base metric space or in the physical residual norm.
- `base-call equal` means equal counts of selected base-preconditioner applications inside one preconditioner call, excluding diagnostic metric evaluations, `A_h` applications, scalar reductions, and U-Net overhead. `UNet-defect2` is base-call equal to `dsa2`/`tsa2`; `dsa3`/`dsa4` and `tsa3`/`tsa4` are deeper classical work-budget baselines, not strict equal-cost matches.
- `Eval 1-step residual` is the post-training ratio `||r - A_h M r||_2 / ||r||_2` measured on the full train or heldout source after one preconditioner application.
- `Eval 1-step base-metric ratio` measures that same one-step residual after both numerator and denominator are mapped through the selected classical base metric used for the run.
- `Train batch residual start/end` is the mean one-step physical residual ratio on the sampled residual minibatches used during optimization. It is a training diagnostic and is not directly comparable to the full-source evaluation columns.
- `Train objective start/end` is the scalar optimization objective on those sampled minibatches, including whichever metric, physical, correction, rollout, stability, and teacher terms were active for the run.

## Quick Read

- Size `64`, case `anisotropic`, base `transport_dsa`, split `train`: best learned method `ScaleGatedRoutedUNet-defect2-w1` reached eval 1-step residual 7.132e-02 ; stationary final residual 1.267e-05 and FGMRES final residual 1.298e-05.
- Size `64`, case `anisotropic`, base `transport_dsa`, split `heldout`: best learned method `ScaleGatedRoutedUNet-defect2-w1` reached eval 1-step residual 6.752e-02 ; stationary final residual 1.257e-05 and FGMRES final residual 4.670e-05.
- Size `64`, case `diffusive`, base `transport_dsa`, split `train`: best learned method `ScaleGatedRoutedUNet-defect2-w1` reached eval 1-step residual 9.979e-03 ; stationary final residual 2.115e-05 and FGMRES final residual 2.315e-05.
- Size `64`, case `diffusive`, base `transport_dsa`, split `heldout`: best learned method `ScaleGatedRoutedUNet-defect2-w1` reached eval 1-step residual 9.163e-03 ; stationary final residual 1.630e-05 and FGMRES final residual 1.912e-05.
- Size `64`, case `heterogeneous`, base `transport_dsa`, split `train`: best learned method `ScaleGatedRoutedUNet-defect2-w1` reached eval 1-step residual 1.063e-01 ; stationary final residual 9.664e-06 and FGMRES final residual 1.049e-05.
- Size `64`, case `heterogeneous`, base `transport_dsa`, split `heldout`: best learned method `ScaleGatedRoutedUNet-defect2-w1` reached eval 1-step residual 7.403e-02 ; stationary final residual 7.596e-06 and FGMRES final residual 9.469e-06.
- Size `64`, case `mixed`, base `transport_dsa`, split `train`: best learned method `ScaleGatedRoutedUNet-defect2-w1` reached eval 1-step residual 1.625e-01 ; stationary final residual 1.153e-04 and FGMRES final residual 3.716e-05.
- Size `64`, case `mixed`, base `transport_dsa`, split `heldout`: best learned method `ScaleGatedRoutedUNet-defect2-w1` reached eval 1-step residual 1.587e-01 ; stationary final residual 1.469e-04 and FGMRES final residual 4.530e-05.

## Results

| Grid | Base | TSA harmonics | FGMRES mode | Case | Split | Method | Eval 1-step residual | Eval 1-step base-metric ratio | Stationary final residual | FGMRES final residual | Stationary time to tol | FGMRES time to tol | Train batch residual start | Train batch residual end | Train objective start | Train objective end |
|---:|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | transport_dsa | None | real-stacked | diffusive | train | dsa1 | 2.369e+00 | 8.181e-02 | 1.785e-05 | 3.288e-05 | nan | nan |  |  |  |  |
| 64 | transport_dsa | None | real-stacked | diffusive | train | dsa2 | 4.219e-01 | 2.233e-02 | 1.927e-05 | 2.885e-05 | nan | nan |  |  |  |  |
| 64 | transport_dsa | None | real-stacked | diffusive | train | ScaleGatedRoutedUNet-defect2-w1 | 9.979e-03 | 1.496e-03 | 2.115e-05 | 2.315e-05 | nan | nan | 6.532e-02 | 3.640e-03 | 8.535e-03 | 3.334e-05 |
| 64 | transport_dsa | None | real-stacked | diffusive | heldout | dsa1 | 2.133e+00 | 8.222e-02 | 1.663e-05 | 2.738e-05 | nan | nan |  |  |  |  |
| 64 | transport_dsa | None | real-stacked | diffusive | heldout | dsa2 | 3.824e-01 | 2.243e-02 | 1.738e-05 | 2.018e-05 | nan | nan |  |  |  |  |
| 64 | transport_dsa | None | real-stacked | diffusive | heldout | ScaleGatedRoutedUNet-defect2-w1 | 9.163e-03 | 1.346e-03 | 1.630e-05 | 1.912e-05 | nan | nan | 6.532e-02 | 3.640e-03 | 8.535e-03 | 3.334e-05 |
| 64 | transport_dsa | None | real-stacked | heterogeneous | train | dsa1 | 4.876e-01 | 1.371e-01 | 1.017e-05 | 1.276e-05 | nan | nan |  |  |  |  |
| 64 | transport_dsa | None | real-stacked | heterogeneous | train | dsa2 | 2.307e-01 | 7.350e-02 | 9.658e-06 | 1.106e-05 | nan | nan |  |  |  |  |
| 64 | transport_dsa | None | real-stacked | heterogeneous | train | ScaleGatedRoutedUNet-defect2-w1 | 1.063e-01 | 3.326e-02 | 9.664e-06 | 1.049e-05 | nan | nan | 1.370e-01 | 4.254e-02 | 7.696e-02 | 8.112e-03 |
| 64 | transport_dsa | None | real-stacked | heterogeneous | heldout | dsa1 | 3.710e-01 | 1.105e-01 | 9.115e-06 | 1.109e-05 | nan | nan |  |  |  |  |
| 64 | transport_dsa | None | real-stacked | heterogeneous | heldout | dsa2 | 1.516e-01 | 5.550e-02 | 8.014e-06 | 9.816e-06 | nan | nan |  |  |  |  |
| 64 | transport_dsa | None | real-stacked | heterogeneous | heldout | ScaleGatedRoutedUNet-defect2-w1 | 7.403e-02 | 2.637e-02 | 7.596e-06 | 9.469e-06 | nan | nan | 1.370e-01 | 4.254e-02 | 7.696e-02 | 8.112e-03 |
| 64 | transport_dsa | None | real-stacked | anisotropic | train | dsa1 | 6.777e-01 | 1.129e-01 | 5.755e-04 | 1.808e-04 | nan | nan |  |  |  |  |
| 64 | transport_dsa | None | real-stacked | anisotropic | train | dsa2 | 4.294e-01 | 8.233e-02 | 1.197e-05 | 1.552e-05 | nan | nan |  |  |  |  |
| 64 | transport_dsa | None | real-stacked | anisotropic | train | ScaleGatedRoutedUNet-defect2-w1 | 7.132e-02 | 1.450e-02 | 1.267e-05 | 1.298e-05 | nan | nan | 2.282e-01 | 7.049e-02 | 1.829e-01 | 3.169e-02 |
| 64 | transport_dsa | None | real-stacked | anisotropic | heldout | dsa1 | 6.774e-01 | 1.126e-01 | 5.755e-04 | 1.870e-04 | nan | nan |  |  |  |  |
| 64 | transport_dsa | None | real-stacked | anisotropic | heldout | dsa2 | 4.283e-01 | 8.195e-02 | 1.227e-05 | 1.445e-05 | nan | nan |  |  |  |  |
| 64 | transport_dsa | None | real-stacked | anisotropic | heldout | ScaleGatedRoutedUNet-defect2-w1 | 6.752e-02 | 1.396e-02 | 1.257e-05 | 4.670e-05 | nan | nan | 2.282e-01 | 7.049e-02 | 1.829e-01 | 3.169e-02 |
| 64 | transport_dsa | None | real-stacked | mixed | train | dsa1 | 9.777e-01 | 1.650e-01 | 1.009e-02 | 9.459e-03 | nan | nan |  |  |  |  |
| 64 | transport_dsa | None | real-stacked | mixed | train | dsa2 | 6.613e-01 | 1.504e-01 | 2.735e-03 | 8.782e-04 | nan | nan |  |  |  |  |
| 64 | transport_dsa | None | real-stacked | mixed | train | ScaleGatedRoutedUNet-defect2-w1 | 1.625e-01 | 3.321e-02 | 1.153e-04 | 3.716e-05 | nan | nan | 3.098e-01 | 1.430e-01 | 2.654e-01 | 4.589e-02 |
| 64 | transport_dsa | None | real-stacked | mixed | heldout | dsa1 | 8.728e-01 | 1.672e-01 | 1.005e-02 | 8.495e-03 | nan | nan |  |  |  |  |
| 64 | transport_dsa | None | real-stacked | mixed | heldout | dsa2 | 5.548e-01 | 1.134e-01 | 2.728e-03 | 9.188e-04 | nan | nan |  |  |  |  |
| 64 | transport_dsa | None | real-stacked | mixed | heldout | ScaleGatedRoutedUNet-defect2-w1 | 1.587e-01 | 2.829e-02 | 1.469e-04 | 4.530e-05 | nan | nan | 3.098e-01 | 1.430e-01 | 2.654e-01 | 4.589e-02 |
