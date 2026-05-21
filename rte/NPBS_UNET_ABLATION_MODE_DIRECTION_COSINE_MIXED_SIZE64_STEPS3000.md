# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['mixed']`
- `sizes`: `[64]`
- `n_angles`: `24`
- `base_kinds`: `['transport_tsa_moment']`
- `tsa_harmonics`: `None`
- `forms`: `['coeff_weighted_moment_tsa', 'coeff_mixed_moment_tsa', 'residual_subspace_moment_tsa']`
- `moment_harmonics`: `4`
- `moment_weight_scale`: `0.5`
- `moment_transport_relaxation`: `0.4`
- `moment_synthetic_relaxation`: `0.4`
- `learn_scalar_moment`: `False`
- `defect_relaxations`: `[1.0]`
- `baseline_depths`: `[1, 2, 3, 4]`
- `baseline_relaxations`: `[0.4, 1.0]`
- `eta_scale`: `0.1`
- `width`: `24`
- `levels`: `4`
- `steps`: `3000`
- `lr`: `0.001`
- `weight_decay`: `0.0001`
- `scheduler`: `cosine`
- `scheduler_step_size`: `0`
- `scheduler_gamma`: `0.5`
- `scheduler_eta_min`: `None`
- `scheduler_min_lr_ratio`: `0.01`
- `scheduler_pct_start`: `0.1`
- `log_steps`: `250`
- `residual_mode`: `resample`
- `residuals_per_step`: `128`
- `rollout_samples_per_step`: `64`
- `n_rollout`: `16`
- `n_random`: `6`
- `random_modes`: `5`
- `aux_sources`: `5`
- `metric_weight`: `0.0`
- `physical_weight`: `1.0`
- `correction_weight`: `1e-06`
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
- `maxiter`: `120`
- `fgmres_maxiter`: `60`
- `fgmres_mode`: `real-stacked`
- `tol`: `1e-06`
- `tol_grid`: `[]`
- `seed`: `20260532`
- `device`: `cuda:0`
- `allow_tf32`: `False`
- `output_init`: `zero`
- `output_csv`: `rte/npbs_unet_ablation_mode_direction_cosine_mixed_size64_steps3000.csv`
- `trace_csv`: ``
- `output_md`: `rte/NPBS_UNET_ABLATION_MODE_DIRECTION_COSINE_MIXED_SIZE64_STEPS3000.md`

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
- `LearnedRP-TSA-MK` learns coefficient-conditioned restriction/prolongation maps around the fixed closed moment inverse. It is initialized to the identity and can preserve the scalar moment exactly.
- `ResidualSubspaceTSA-MK` uses the current post-transport residual to choose a bounded global mixing matrix on retained moments before the fixed closed inverse. This is a residual-dependent angular-subspace ablation and is nonlinear in the residual.
- By default `CoeffWeightedTSA-MK` freezes the scalar `m=0` synthetic weight to preserve the scalar diffusion moment; pass `--learn-scalar-moment` to ablate this constraint. `--moment-transport-relaxation` and `--moment-synthetic-relaxation` allow stable damped variants.
- `--rollout-space` and `--stability-space` choose whether rollout and stability penalties measure contraction in the base metric space or in the physical residual norm.
- `base-call equal` means equal counts of selected base-preconditioner applications inside one preconditioner call, excluding diagnostic metric evaluations, `A_h` applications, scalar reductions, and U-Net overhead. `UNet-defect2` is base-call equal to `dsa2`/`tsa2`; `dsa3`/`dsa4` and `tsa3`/`tsa4` are deeper classical work-budget baselines, not strict equal-cost matches.

## Results

| Size | Base | K | FGMRES | Case | Split | Method | one-step true | one-step base metric | stat final | FGMRES final | stat ttol | FGMRES ttol | loss0 | loss1 |
|---:|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | train | tsa_moment1-w0.4 | 6.948e-01 | 7.303e-01 | 4.069e-03 | 1.669e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | train | tsa_moment1 | 4.459e-01 | 4.299e-01 | 1.905e-02 | 1.680e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | train | tsa_moment2-w0.4 | 5.052e-01 | 5.537e-01 | 5.686e-04 | 7.280e-06 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | train | tsa_moment2 | 2.339e-01 | 2.860e-01 | 2.272e+03 | 9.132e-06 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | train | tsa_moment3-w0.4 | 3.821e-01 | 4.386e-01 | 8.180e-05 | 7.851e-06 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | train | tsa_moment3 | 1.543e-01 | 2.363e-01 | 1.162e+09 | 7.559e-06 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | train | tsa_moment4-w0.4 | 2.990e-01 | 3.637e-01 | 1.372e-05 | 7.605e-06 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | train | tsa_moment4 | 1.191e-01 | 2.014e-01 | 2.900e+14 | 7.708e-06 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | train | CoeffWeightedTSA-M4 | 6.479e-01 | 6.959e-01 | 4.217e-03 | 1.793e-05 | nan | nan | 0.6083250641822815 | 0.5848846435546875 |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | train | CoeffMixedTSA-M4 | 5.757e-01 | 6.415e-01 | 4.420e-03 | 3.852e-05 | nan | nan | 0.5986688733100891 | 0.5825952887535095 |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | train | ResidualSubspaceTSA-M4 | 5.832e-01 | 6.402e-01 | 8.256e-03 | 5.317e-05 | nan | nan | 0.6078401207923889 | 0.5849836468696594 |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | tsa_moment1-w0.4 | 6.476e-01 | 7.365e-01 | 2.888e-03 | 1.972e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | tsa_moment1 | 2.700e-01 | 4.604e-01 | 1.929e-02 | 1.961e-05 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | tsa_moment2-w0.4 | 4.382e-01 | 5.780e-01 | 4.406e-04 | 7.586e-06 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | tsa_moment2 | 1.879e-01 | 3.593e-01 | 4.944e+03 | 9.790e-06 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | tsa_moment3-w0.4 | 3.153e-01 | 4.813e-01 | 6.998e-05 | 7.564e-06 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | tsa_moment3 | 1.590e-01 | 3.008e-01 | 8.457e+08 | 7.383e-06 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | tsa_moment4-w0.4 | 2.440e-01 | 4.197e-01 | 1.301e-05 | 7.318e-06 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | tsa_moment4 | 1.357e-01 | 2.546e-01 | 2.993e+14 | 7.403e-06 | nan | nan |  |  |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | CoeffWeightedTSA-M4 | 6.019e-01 | 7.031e-01 | 2.963e-03 | 1.942e-05 | nan | nan | 0.6083250641822815 | 0.5848846435546875 |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | CoeffMixedTSA-M4 | 5.877e-01 | 6.969e-01 | 3.411e-03 | 4.221e-05 | nan | nan | 0.5986688733100891 | 0.5825952887535095 |
| 64 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | ResidualSubspaceTSA-M4 | 5.874e-01 | 7.091e-01 | 1.245e-01 | 1.745e-04 | nan | nan | 0.6078401207923889 | 0.5849836468696594 |
