# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['anisotropic', 'mixed']`
- `sizes`: `[32, 64]`
- `n_angles`: `12`
- `base_kinds`: `['transport_tsa']`
- `tsa_harmonics`: `None`
- `forms`: `['defect2']`
- `moment_harmonics`: `2`
- `moment_weight_scale`: `0.5`
- `moment_transport_relaxation`: `1.0`
- `moment_synthetic_relaxation`: `1.0`
- `learn_scalar_moment`: `False`
- `defect_relaxations`: `[1.0]`
- `baseline_depths`: `[1, 2, 3, 4]`
- `baseline_relaxations`: `[0.5, 1.0]`
- `eta_scale`: `0.1`
- `width`: `16`
- `levels`: `4`
- `steps`: `900`
- `lr`: `0.002`
- `weight_decay`: `0.0001`
- `scheduler`: `step`
- `scheduler_step_size`: `300`
- `scheduler_gamma`: `0.5`
- `scheduler_eta_min`: `1e-05`
- `log_steps`: `0`
- `residual_mode`: `resample`
- `residuals_per_step`: `64`
- `rollout_samples_per_step`: `32`
- `n_rollout`: `10`
- `n_random`: `6`
- `random_modes`: `5`
- `aux_sources`: `3`
- `metric_weight`: `0.0`
- `physical_weight`: `1.0`
- `correction_weight`: `0.0`
- `rollout_weight`: `0.0`
- `rollout_steps`: `1`
- `rollout_decay`: `1.0`
- `stability_weight`: `0.0`
- `contraction_target`: `0.95`
- `teacher_depth`: `0`
- `teacher_relaxation`: `1.0`
- `teacher_kind`: ``
- `teacher_weight`: `0.0`
- `maxiter`: `60`
- `fgmres_maxiter`: `25`
- `fgmres_mode`: `real-stacked`
- `tol`: `1e-06`
- `tol_grid`: `[0.0001, 1e-05]`
- `seed`: `20260519`
- `device`: `cuda:0`
- `allow_tf32`: `False`
- `output_init`: `small`
- `output_csv`: `rte/npbs_unet_ablation_tsa_full_physical_steps900.csv`
- `trace_csv`: `rte/npbs_unet_ablation_tsa_full_physical_steps900_traces.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_TSA_FULL_PHYSICAL_STEPS900.md`

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
- `CoeffMixedTSA-MK` keeps the closed TSA-MK solve fixed, then uses a coefficient-conditioned local linear mixing matrix on retained moments. It is initialized to the identity map and can preserve the scalar moment exactly.
- By default `CoeffWeightedTSA-MK` freezes the scalar `m=0` synthetic weight to preserve the scalar diffusion moment; pass `--learn-scalar-moment` to ablate this constraint. `--moment-transport-relaxation` and `--moment-synthetic-relaxation` allow stable damped variants.
- `base-call equal` means equal counts of selected base-preconditioner applications inside one preconditioner call, excluding diagnostic metric evaluations, `A_h` applications, scalar reductions, and U-Net overhead. `UNet-defect2` is base-call equal to `dsa2`/`tsa2`; `dsa3`/`dsa4` and `tsa3`/`tsa4` are deeper classical work-budget baselines, not strict equal-cost matches.

## Results

| Size | Base | K | FGMRES | Case | Split | Method | one-step true | one-step base metric | stat final | FGMRES final | stat ttol | FGMRES ttol | loss0 | loss1 |
|---:|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | transport_tsa | None | real-stacked | anisotropic | train | tsa1-w0.5 | 5.275e-01 | 7.078e-01 | 6.895e-06 | 5.808e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | anisotropic | train | tsa1 | 1.577e-01 | 4.745e-01 | 5.646e-06 | 5.808e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | anisotropic | train | tsa2-w0.5 | 2.922e-01 | 5.340e-01 | 5.633e-06 | 5.854e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | anisotropic | train | tsa2 | 8.251e-02 | 3.059e-01 | 5.824e-06 | 6.109e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | anisotropic | train | tsa3-w0.5 | 1.750e-01 | 4.212e-01 | 5.493e-06 | 5.580e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | anisotropic | train | tsa3 | 5.400e-02 | 2.061e-01 | 6.302e-06 | 5.314e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | anisotropic | train | tsa4-w0.5 | 1.158e-01 | 3.413e-01 | 5.439e-06 | 5.633e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | anisotropic | train | tsa4 | 3.645e-02 | 1.398e-01 | 5.701e-06 | 5.724e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | anisotropic | train | UNet-defect2-w1 | 6.857e-02 | 6.040e-02 | 5.534e-06 | 5.379e-06 | nan | nan | 0.09450946003198624 | 0.0009455219260416925 |
| 32 | transport_tsa | None | real-stacked | anisotropic | heldout | tsa1-w0.5 | 5.262e-01 | 7.087e-01 | 6.712e-06 | 6.392e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | anisotropic | heldout | tsa1 | 1.564e-01 | 4.778e-01 | 6.169e-06 | 6.392e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | anisotropic | heldout | tsa2-w0.5 | 2.907e-01 | 5.360e-01 | 5.270e-06 | 5.732e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | anisotropic | heldout | tsa2 | 8.236e-02 | 3.088e-01 | 6.101e-06 | 5.623e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | anisotropic | heldout | tsa3-w0.5 | 1.738e-01 | 4.237e-01 | 6.036e-06 | 6.335e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | anisotropic | heldout | tsa3 | 5.399e-02 | 2.081e-01 | 5.693e-06 | 6.109e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | anisotropic | heldout | tsa4-w0.5 | 1.150e-01 | 3.439e-01 | 6.782e-06 | 6.402e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | anisotropic | heldout | tsa4 | 3.645e-02 | 1.412e-01 | 5.661e-06 | 5.923e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | anisotropic | heldout | UNet-defect2-w1 | 5.712e-02 | 6.395e-02 | 5.873e-06 | 5.974e-06 | nan | nan | 0.09450946003198624 | 0.0009455219260416925 |
| 32 | transport_tsa | None | real-stacked | mixed | train | tsa1-w0.5 | 6.204e-01 | 6.671e-01 | 4.653e-03 | 7.228e-05 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | mixed | train | tsa1 | 4.058e-01 | 4.252e-01 | 4.249e-04 | 7.228e-05 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | mixed | train | tsa2-w0.5 | 4.083e-01 | 4.757e-01 | 4.621e-04 | 8.464e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | mixed | train | tsa2 | 1.709e-01 | 2.792e-01 | 6.058e-06 | 3.528e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | mixed | train | tsa3-w0.5 | 2.816e-01 | 3.665e-01 | 5.210e-05 | 3.568e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | mixed | train | tsa3 | 1.168e-01 | 2.315e-01 | 3.400e-06 | 3.607e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | mixed | train | tsa4-w0.5 | 2.041e-01 | 3.037e-01 | 6.861e-06 | 3.368e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | mixed | train | tsa4 | 9.576e-02 | 1.984e-01 | 3.588e-06 | 3.330e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | mixed | train | UNet-defect2-w1 | 1.572e-01 | 1.166e-01 | 1.394e-04 | 3.501e-06 | nan | nan | 0.2388678938150406 | 0.06874844431877136 |
| 32 | transport_tsa | None | real-stacked | mixed | heldout | tsa1-w0.5 | 5.625e-01 | 6.761e-01 | 4.170e-03 | 5.674e-05 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | mixed | heldout | tsa1 | 2.638e-01 | 4.607e-01 | 3.552e-04 | 5.674e-05 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | mixed | heldout | tsa2-w0.5 | 3.476e-01 | 5.125e-01 | 3.852e-04 | 5.851e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | mixed | heldout | tsa2 | 1.811e-01 | 3.595e-01 | 5.955e-06 | 3.382e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | mixed | heldout | tsa3-w0.5 | 2.436e-01 | 4.250e-01 | 4.583e-05 | 3.331e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | mixed | heldout | tsa3 | 1.442e-01 | 2.986e-01 | 3.406e-06 | 3.927e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | mixed | heldout | tsa4-w0.5 | 1.918e-01 | 3.720e-01 | 6.503e-06 | 3.281e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | mixed | heldout | tsa4 | 1.143e-01 | 2.507e-01 | 3.251e-06 | 3.297e-06 | nan | nan |  |  |
| 32 | transport_tsa | None | real-stacked | mixed | heldout | UNet-defect2-w1 | 1.356e-01 | 1.379e-01 | 1.595e-04 | 3.393e-06 | nan | nan | 0.2388678938150406 | 0.06874844431877136 |
| 64 | transport_tsa | None | real-stacked | anisotropic | train | tsa1-w0.5 | 5.275e-01 | 7.078e-01 | 1.193e-05 | 1.192e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | anisotropic | train | tsa1 | 1.577e-01 | 4.745e-01 | 1.276e-05 | 1.192e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | anisotropic | train | tsa2-w0.5 | 2.922e-01 | 5.340e-01 | 1.113e-05 | 1.209e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | anisotropic | train | tsa2 | 8.251e-02 | 3.059e-01 | 1.281e-05 | 1.244e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | anisotropic | train | tsa3-w0.5 | 1.750e-01 | 4.212e-01 | 1.171e-05 | 1.336e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | anisotropic | train | tsa3 | 5.400e-02 | 2.061e-01 | 1.335e-05 | 1.285e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | anisotropic | train | tsa4-w0.5 | 1.158e-01 | 3.413e-01 | 1.290e-05 | 1.228e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | anisotropic | train | tsa4 | 3.645e-02 | 1.398e-01 | 1.239e-05 | 1.280e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | anisotropic | train | UNet-defect2-w1 | 8.609e-02 | 6.473e-02 | 1.246e-05 | 1.221e-05 | nan | nan | 0.09450490772724152 | 0.0017001429805532098 |
| 64 | transport_tsa | None | real-stacked | anisotropic | heldout | tsa1-w0.5 | 5.262e-01 | 7.087e-01 | 1.159e-05 | 1.224e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | anisotropic | heldout | tsa1 | 1.564e-01 | 4.778e-01 | 1.297e-05 | 1.224e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | anisotropic | heldout | tsa2-w0.5 | 2.907e-01 | 5.360e-01 | 1.189e-05 | 1.222e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | anisotropic | heldout | tsa2 | 8.236e-02 | 3.088e-01 | 1.280e-05 | 1.226e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | anisotropic | heldout | tsa3-w0.5 | 1.738e-01 | 4.237e-01 | 1.234e-05 | 1.257e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | anisotropic | heldout | tsa3 | 5.399e-02 | 2.081e-01 | 1.301e-05 | 1.208e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | anisotropic | heldout | tsa4-w0.5 | 1.150e-01 | 3.439e-01 | 1.277e-05 | 1.371e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | anisotropic | heldout | tsa4 | 3.645e-02 | 1.412e-01 | 1.307e-05 | 1.303e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | anisotropic | heldout | UNet-defect2-w1 | 6.990e-02 | 6.848e-02 | 1.344e-05 | 1.269e-05 | nan | nan | 0.09450490772724152 | 0.0017001429805532098 |
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa1-w0.5 | 6.207e-01 | 6.679e-01 | 4.840e-03 | 8.625e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa1 | 4.039e-01 | 4.276e-01 | 4.511e-04 | 8.625e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa2-w0.5 | 4.082e-01 | 4.780e-01 | 4.893e-04 | 1.272e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa2 | 1.719e-01 | 2.856e-01 | 9.550e-06 | 7.310e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa3-w0.5 | 2.814e-01 | 3.704e-01 | 5.743e-05 | 8.002e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa3 | 1.193e-01 | 2.383e-01 | 7.653e-06 | 7.460e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa4-w0.5 | 2.043e-01 | 3.088e-01 | 9.819e-06 | 7.568e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | tsa4 | 9.809e-02 | 2.049e-01 | 7.708e-06 | 7.519e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | train | UNet-defect2-w1 | 1.659e-01 | 1.203e-01 | 7.837e-06 | 7.538e-06 | nan | nan | 0.24183976650238037 | 0.07238715887069702 |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa1-w0.5 | 5.624e-01 | 6.766e-01 | 4.587e-03 | 6.892e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa1 | 2.608e-01 | 4.605e-01 | 4.056e-04 | 6.892e-05 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa2-w0.5 | 3.470e-01 | 5.132e-01 | 4.388e-04 | 9.665e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa2 | 1.790e-01 | 3.604e-01 | 9.315e-06 | 7.603e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa3-w0.5 | 2.425e-01 | 4.260e-01 | 5.432e-05 | 7.511e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa3 | 1.432e-01 | 3.012e-01 | 7.778e-06 | 7.362e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa4-w0.5 | 1.907e-01 | 3.735e-01 | 9.955e-06 | 8.014e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | tsa4 | 1.145e-01 | 2.544e-01 | 7.676e-06 | 7.935e-06 | nan | nan |  |  |
| 64 | transport_tsa | None | real-stacked | mixed | heldout | UNet-defect2-w1 | 1.245e-01 | 1.286e-01 | 7.823e-06 | 7.544e-06 | nan | nan | 0.24183976650238037 | 0.07238715887069702 |
