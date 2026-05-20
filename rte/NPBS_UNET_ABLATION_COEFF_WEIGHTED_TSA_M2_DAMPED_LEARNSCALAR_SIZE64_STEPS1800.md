# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['anisotropic', 'mixed']`
- `sizes`: `[64]`
- `n_angles`: `24`
- `base_kinds`: `['transport_tsa_moment']`
- `tsa_harmonics`: `None`
- `forms`: `['coeff_weighted_moment_tsa']`
- `moment_harmonics`: `2`
- `moment_weight_scale`: `0.5`
- `moment_transport_relaxation`: `0.4`
- `moment_synthetic_relaxation`: `0.4`
- `learn_scalar_moment`: `True`
- `defect_relaxations`: `[1.0]`
- `baseline_depths`: `[1, 2, 3, 4]`
- `baseline_relaxations`: `[0.4, 1.0]`
- `eta_scale`: `0.1`
- `width`: `24`
- `levels`: `4`
- `steps`: `1800`
- `lr`: `0.001`
- `weight_decay`: `0.0001`
- `scheduler`: `step`
- `scheduler_step_size`: `700`
- `scheduler_gamma`: `0.5`
- `scheduler_eta_min`: `1e-05`
- `log_steps`: `0`
- `residual_mode`: `resample`
- `residuals_per_step`: `128`
- `rollout_samples_per_step`: `64`
- `n_rollout`: `16`
- `n_random`: `6`
- `random_modes`: `5`
- `aux_sources`: `5`
- `physical_weight`: `0.5`
- `correction_weight`: `1e-06`
- `rollout_weight`: `0.2`
- `rollout_steps`: `3`
- `rollout_decay`: `0.8`
- `stability_weight`: `0.8`
- `contraction_target`: `0.95`
- `teacher_depth`: `0`
- `teacher_relaxation`: `1.0`
- `teacher_kind`: ``
- `teacher_weight`: `0.0`
- `maxiter`: `120`
- `fgmres_maxiter`: `60`
- `fgmres_mode`: `real-stacked`
- `tol`: `1e-06`
- `tol_grid`: `[0.0001, 1e-05]`
- `seed`: `20260522`
- `device`: `cuda:0`
- `allow_tf32`: `False`
- `output_init`: `zero`
- `output_csv`: `rte/npbs_unet_ablation_coeff_weighted_tsa_m2_damped_learnscalar_size64_steps1800.csv`
- `trace_csv`: `rte/npbs_unet_ablation_coeff_weighted_tsa_m2_damped_learnscalar_size64_steps1800_traces.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_COEFF_WEIGHTED_TSA_M2_DAMPED_LEARNSCALAR_SIZE64_STEPS1800.md`

## Terminology

- `transport` is the shifted/reference transport inverse `G_eta`.
- `transport_dsa` applies one reference transport correction followed by one scalar diffusion synthetic correction.
- `transport_tsa` applies one reference transport correction followed by one reference transport-scattering synthetic correction. `tsa_harmonics=None` keeps all angular Fourier modes; an integer such as `2` keeps only the lowest angular harmonics.
- `transport_tsa_moment` applies one reference transport correction followed by a closed Galerkin angular Fourier synthetic correction `Q_K (Q_K^* A_ref Q_K)^{-1} Q_K^*` on the post-transport residual. This differs from `transport_tsa` with `tsa_harmonics=K`, which only projects the RHS before applying the full angular reference inverse.
- `dsaK` and `tsaK` mean `K` residual-correction applications of the selected base preconditioner. Starting with `z=0` and residual `e_0=r`, each stage applies `z <- z + omega B_h e_j` and `e_{j+1} <- e_j - A_h(omega B_h e_j)`. A suffix such as `-w0.5` uses relaxation `omega=0.5`; no suffix means `omega=1`.
- `UNet-defect2-w1` uses `z0=B_h r`, `z1=B_h(r-A_h z0)`, and returns `z0 + z1 + C_theta(z0,z1,coeff)`. It costs two base applications plus the U-Net evaluation per preconditioner call.
- `MomentUNet-K` uses the same defect2 structure but restricts the learned correction to angular Fourier modes `0,+/-1,...,+/-K` before prolonging back to ordinates.
- `WeightedTSA-MK` is a tiny learned structural model initialized as closed TSA-MK; it learns bounded real weights for each absolute angular mode in the synthetic correction.
- `CoeffWeightedTSA-MK` keeps the closed TSA-MK solve fixed, then uses a coefficient-conditioned nonlinear network to learn bounded physical-space weights for each absolute angular mode. It is nonlinear in coefficients but linear in residuals for fixed coefficients.
- By default `CoeffWeightedTSA-MK` freezes the scalar `m=0` synthetic weight to preserve the scalar diffusion moment; pass `--learn-scalar-moment` to ablate this constraint. `--moment-transport-relaxation` and `--moment-synthetic-relaxation` allow stable damped variants.
- `base-call equal` means equal counts of selected base-preconditioner applications inside one preconditioner call, excluding diagnostic metric evaluations, `A_h` applications, scalar reductions, and U-Net overhead. `UNet-defect2` is base-call equal to `dsa2`/`tsa2`; `dsa3`/`dsa4` and `tsa3`/`tsa4` are deeper classical work-budget baselines, not strict equal-cost matches.

## Results

| Size | Base | K | FGMRES | Case | Split | Method | one-step true | one-step base metric | stat final | FGMRES final | stat ttol | FGMRES ttol | loss0 | loss1 |
|---:|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | transport_tsa_moment | 2 | real-stacked | anisotropic | train | tsa_moment1-w0.4 | 7.421e-01 | 7.114e-01 | 1.049e-05 | 1.343e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | anisotropic | train | tsa_moment1 | 1.015e+00 | 6.841e-01 | nan | 1.281e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | anisotropic | train | tsa_moment2-w0.4 | 5.639e-01 | 5.745e-01 | 1.174e-05 | 1.231e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | anisotropic | train | tsa_moment2 | 8.231e-01 | 5.175e-01 | nan | 2.477e-01 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | anisotropic | train | tsa_moment3-w0.4 | 4.273e-01 | 4.819e-01 | 1.127e-05 | 1.242e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | anisotropic | train | tsa_moment3 | 9.354e-01 | 5.038e-01 | nan | 1.290e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | anisotropic | train | tsa_moment4-w0.4 | 3.260e-01 | 4.105e-01 | 1.264e-05 | 1.299e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | anisotropic | train | tsa_moment4 | 1.224e+00 | 6.025e-01 | nan | 3.178e-01 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | anisotropic | train | CoeffWeightedTSA-M2 | 6.947e-01 | 7.186e-01 | 1.001e-05 | 1.354e-05 | nan | nan | 0.5341932773590088 | 0.5075967311859131 |
| 64 | transport_tsa_moment | 2 | real-stacked | anisotropic | heldout | tsa_moment1-w0.4 | 7.408e-01 | 7.117e-01 | 1.071e-05 | 1.290e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | anisotropic | heldout | tsa_moment1 | 1.012e+00 | 6.853e-01 | nan | 1.335e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | anisotropic | heldout | tsa_moment2-w0.4 | 5.616e-01 | 5.750e-01 | 1.192e-05 | 1.306e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | anisotropic | heldout | tsa_moment2 | 8.224e-01 | 5.181e-01 | nan | 2.410e-01 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | anisotropic | heldout | tsa_moment3-w0.4 | 4.244e-01 | 4.825e-01 | 1.176e-05 | 1.319e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | anisotropic | heldout | tsa_moment3 | 9.354e-01 | 5.051e-01 | nan | 1.269e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | anisotropic | heldout | tsa_moment4-w0.4 | 3.229e-01 | 4.109e-01 | 1.210e-05 | 1.344e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | anisotropic | heldout | tsa_moment4 | 1.225e+00 | 6.041e-01 | nan | 3.110e-01 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | anisotropic | heldout | CoeffWeightedTSA-M2 | 6.931e-01 | 7.186e-01 | 9.410e-06 | 1.296e-05 | nan | nan | 0.5341932773590088 | 0.5075967311859131 |
| 64 | transport_tsa_moment | 2 | real-stacked | mixed | train | tsa_moment1-w0.4 | 7.190e-01 | 7.283e-01 | 4.877e-03 | 6.237e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | mixed | train | tsa_moment1 | 6.275e-01 | 4.489e-01 | 6.116e+15 | 6.250e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | mixed | train | tsa_moment2-w0.4 | 5.628e-01 | 5.537e-01 | 8.332e-04 | 7.570e-06 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | mixed | train | tsa_moment2 | 3.534e-01 | 3.022e-01 | nan | 3.243e-02 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | mixed | train | tsa_moment3-w0.4 | 4.595e-01 | 4.411e-01 | 1.461e-04 | 7.990e-06 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | mixed | train | tsa_moment3 | 2.155e-01 | 2.420e-01 | nan | 7.689e-06 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | mixed | train | tsa_moment4-w0.4 | 3.803e-01 | 3.678e-01 | 2.684e-05 | 7.657e-06 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | mixed | train | tsa_moment4 | 1.702e-01 | 2.041e-01 | nan | 2.618e-02 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | mixed | train | CoeffWeightedTSA-M2 | 6.651e-01 | 6.486e-01 | 4.760e-03 | 1.395e-04 | nan | nan | 1.1222314834594727 | 1.0903260707855225 |
| 64 | transport_tsa_moment | 2 | real-stacked | mixed | heldout | tsa_moment1-w0.4 | 6.563e-01 | 7.326e-01 | 3.634e-03 | 6.671e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | mixed | heldout | tsa_moment1 | 3.526e-01 | 4.659e-01 | 1.441e+15 | 6.677e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | mixed | heldout | tsa_moment2-w0.4 | 4.541e-01 | 5.739e-01 | 6.588e-04 | 7.879e-06 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | mixed | heldout | tsa_moment2 | 2.119e-01 | 3.553e-01 | nan | 6.800e-03 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | mixed | heldout | tsa_moment3-w0.4 | 3.338e-01 | 4.775e-01 | 1.241e-04 | 7.871e-06 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | mixed | heldout | tsa_moment3 | 1.850e-01 | 2.987e-01 | nan | 7.522e-06 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | mixed | heldout | tsa_moment4-w0.4 | 2.628e-01 | 4.159e-01 | 2.458e-05 | 7.374e-06 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | mixed | heldout | tsa_moment4 | 1.605e-01 | 2.547e-01 | nan | 4.247e-03 | nan | nan |  |  |
| 64 | transport_tsa_moment | 2 | real-stacked | mixed | heldout | CoeffWeightedTSA-M2 | 5.958e-01 | 6.636e-01 | 3.487e-03 | 1.553e-04 | nan | nan | 1.1222314834594727 | 1.0903260707855225 |
