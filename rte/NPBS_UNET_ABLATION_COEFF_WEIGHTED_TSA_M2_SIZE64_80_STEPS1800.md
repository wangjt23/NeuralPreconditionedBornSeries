# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['anisotropic', 'mixed']`
- `sizes`: `[64, 80]`
- `n_angles`: `24`
- `base_kinds`: `['transport_tsa_moment']`
- `tsa_harmonics`: `None`
- `forms`: `['coeff_weighted_moment_tsa']`
- `moment_harmonics`: `2`
- `moment_weight_scale`: `0.5`
- `defect_relaxations`: `[1.0]`
- `baseline_depths`: `[1, 2, 3, 4]`
- `baseline_relaxations`: `[1.0]`
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
- `seed`: `20260520`
- `device`: `cuda:0`
- `allow_tf32`: `False`
- `output_init`: `zero`
- `output_csv`: `rte/npbs_unet_ablation_coeff_weighted_tsa_m2_size64_80_steps1800.csv`
- `trace_csv`: `rte/npbs_unet_ablation_coeff_weighted_tsa_m2_size64_80_steps1800_traces.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_COEFF_WEIGHTED_TSA_M2_SIZE64_80_STEPS1800.md`

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
- `base-call equal` means equal counts of selected base-preconditioner applications inside one preconditioner call, excluding diagnostic metric evaluations, `A_h` applications, scalar reductions, and U-Net overhead. `UNet-defect2` is base-call equal to `dsa2`/`tsa2`; `dsa3`/`dsa4` and `tsa3`/`tsa4` are deeper classical work-budget baselines, not strict equal-cost matches.

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | one-step base metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | transport_tsa_moment | anisotropic | train | tsa_moment1 | 1.438e-01 | 2.057e-01 | 4.939e-01 | 1.329e-05 | 1.226e-05 |  |  |
| 64 | transport_tsa_moment | anisotropic | train | tsa_moment2 | 8.014e-02 | 1.172e-01 | 3.277e-01 | 1.347e-05 | 1.356e-05 |  |  |
| 64 | transport_tsa_moment | anisotropic | train | tsa_moment3 | 5.364e-02 | 7.846e-02 | 2.221e-01 | 1.312e-05 | 1.314e-05 |  |  |
| 64 | transport_tsa_moment | anisotropic | train | tsa_moment4 | 3.639e-02 | 5.322e-02 | 1.510e-01 | 1.255e-05 | 1.251e-05 |  |  |
| 64 | transport_tsa_moment | anisotropic | train | CoeffWeightedTSA-M2 | 6.413e-01 | 6.716e-01 | 7.000e-01 | 1.231e+12 | 1.246e-05 | 1.700103998184204 | 0.8480647802352905 |
| 64 | transport_tsa_moment | anisotropic | heldout | tsa_moment1 | 1.435e-01 | 2.053e-01 | 4.951e-01 | 1.298e-05 | 1.310e-05 |  |  |
| 64 | transport_tsa_moment | anisotropic | heldout | tsa_moment2 | 8.014e-02 | 1.172e-01 | 3.286e-01 | 1.326e-05 | 1.274e-05 |  |  |
| 64 | transport_tsa_moment | anisotropic | heldout | tsa_moment3 | 5.364e-02 | 7.848e-02 | 2.227e-01 | 1.291e-05 | 1.328e-05 |  |  |
| 64 | transport_tsa_moment | anisotropic | heldout | tsa_moment4 | 3.639e-02 | 5.324e-02 | 1.514e-01 | 1.290e-05 | 1.292e-05 |  |  |
| 64 | transport_tsa_moment | anisotropic | heldout | CoeffWeightedTSA-M2 | 6.399e-01 | 6.690e-01 | 7.002e-01 | 1.171e+12 | 1.277e-05 | 1.700103998184204 | 0.8480647802352905 |
| 64 | transport_tsa_moment | mixed | train | tsa_moment1 | 4.068e-01 | 3.865e-01 | 4.279e-01 | 1.045e-05 | 7.246e-06 |  |  |
| 64 | transport_tsa_moment | mixed | train | tsa_moment2 | 1.736e-01 | 1.851e-01 | 2.849e-01 | 7.541e-06 | 7.848e-06 |  |  |
| 64 | transport_tsa_moment | mixed | train | tsa_moment3 | 1.195e-01 | 1.382e-01 | 2.360e-01 | 7.291e-06 | 7.738e-06 |  |  |
| 64 | transport_tsa_moment | mixed | train | tsa_moment4 | 9.690e-02 | 1.140e-01 | 2.019e-01 | 7.343e-06 | 7.641e-06 |  |  |
| 64 | transport_tsa_moment | mixed | train | CoeffWeightedTSA-M2 | 6.801e-01 | 6.222e-01 | 4.939e-01 | 1.338e+14 | 5.414e-05 | 0.9113172888755798 | 0.8448329567909241 |
| 64 | transport_tsa_moment | mixed | heldout | tsa_moment1 | 2.607e-01 | 2.849e-01 | 4.611e-01 | 1.013e-05 | 7.425e-06 |  |  |
| 64 | transport_tsa_moment | mixed | heldout | tsa_moment2 | 1.789e-01 | 2.070e-01 | 3.605e-01 | 7.061e-06 | 7.442e-06 |  |  |
| 64 | transport_tsa_moment | mixed | heldout | tsa_moment3 | 1.436e-01 | 1.663e-01 | 3.011e-01 | 7.389e-06 | 7.213e-06 |  |  |
| 64 | transport_tsa_moment | mixed | heldout | tsa_moment4 | 1.147e-01 | 1.335e-01 | 2.541e-01 | 7.232e-06 | 7.499e-06 |  |  |
| 64 | transport_tsa_moment | mixed | heldout | CoeffWeightedTSA-M2 | 3.940e-01 | 3.958e-01 | 4.482e-01 | 4.086e+13 | 6.226e-05 | 0.9113172888755798 | 0.8448329567909241 |
| 80 | transport_tsa_moment | anisotropic | train | tsa_moment1 | 1.438e-01 | 2.057e-01 | 4.939e-01 | 1.586e-05 | 1.600e-05 |  |  |
| 80 | transport_tsa_moment | anisotropic | train | tsa_moment2 | 8.014e-02 | 1.172e-01 | 3.277e-01 | 1.597e-05 | 1.603e-05 |  |  |
| 80 | transport_tsa_moment | anisotropic | train | tsa_moment3 | 5.364e-02 | 7.846e-02 | 2.221e-01 | 1.526e-05 | 1.597e-05 |  |  |
| 80 | transport_tsa_moment | anisotropic | train | tsa_moment4 | 3.639e-02 | 5.322e-02 | 1.510e-01 | 1.594e-05 | 1.576e-05 |  |  |
| 80 | transport_tsa_moment | anisotropic | train | CoeffWeightedTSA-M2 | 6.413e-01 | 6.716e-01 | 7.000e-01 | 6.039e+12 | 1.578e-05 | 1.691990613937378 | 0.8504697680473328 |
| 80 | transport_tsa_moment | anisotropic | heldout | tsa_moment1 | 1.435e-01 | 2.053e-01 | 4.951e-01 | 1.624e-05 | 1.570e-05 |  |  |
| 80 | transport_tsa_moment | anisotropic | heldout | tsa_moment2 | 8.014e-02 | 1.172e-01 | 3.286e-01 | 1.624e-05 | 1.569e-05 |  |  |
| 80 | transport_tsa_moment | anisotropic | heldout | tsa_moment3 | 5.364e-02 | 7.848e-02 | 2.227e-01 | 1.723e-05 | 1.652e-05 |  |  |
| 80 | transport_tsa_moment | anisotropic | heldout | tsa_moment4 | 3.639e-02 | 5.324e-02 | 1.514e-01 | 1.609e-05 | 1.580e-05 |  |  |
| 80 | transport_tsa_moment | anisotropic | heldout | CoeffWeightedTSA-M2 | 6.399e-01 | 6.690e-01 | 7.002e-01 | 6.621e+12 | 1.638e-05 | 1.691990613937378 | 0.8504697680473328 |
| 80 | transport_tsa_moment | mixed | train | tsa_moment1 | 4.094e-01 | 3.898e-01 | 4.272e-01 | 1.129e-05 | 8.767e-06 |  |  |
| 80 | transport_tsa_moment | mixed | train | tsa_moment2 | 1.734e-01 | 1.843e-01 | 2.814e-01 | 9.008e-06 | 9.531e-06 |  |  |
| 80 | transport_tsa_moment | mixed | train | tsa_moment3 | 1.181e-01 | 1.365e-01 | 2.321e-01 | 9.156e-06 | 8.652e-06 |  |  |
| 80 | transport_tsa_moment | mixed | train | tsa_moment4 | 9.542e-02 | 1.124e-01 | 1.981e-01 | 9.693e-06 | 8.817e-06 |  |  |
| 80 | transport_tsa_moment | mixed | train | CoeffWeightedTSA-M2 | 6.918e-01 | 6.278e-01 | 4.960e-01 | inf | 5.837e-05 | 0.9147014021873474 | 0.8471114635467529 |
| 80 | transport_tsa_moment | mixed | heldout | tsa_moment1 | 2.604e-01 | 2.846e-01 | 4.601e-01 | 1.168e-05 | 1.005e-05 |  |  |
| 80 | transport_tsa_moment | mixed | heldout | tsa_moment2 | 1.782e-01 | 2.063e-01 | 3.589e-01 | 1.061e-05 | 9.905e-06 |  |  |
| 80 | transport_tsa_moment | mixed | heldout | tsa_moment3 | 1.429e-01 | 1.656e-01 | 2.991e-01 | 1.033e-05 | 1.043e-05 |  |  |
| 80 | transport_tsa_moment | mixed | heldout | tsa_moment4 | 1.139e-01 | 1.327e-01 | 2.517e-01 | 1.003e-05 | 1.048e-05 |  |  |
| 80 | transport_tsa_moment | mixed | heldout | CoeffWeightedTSA-M2 | 3.925e-01 | 3.935e-01 | 4.463e-01 | inf | 6.681e-05 | 0.9147014021873474 | 0.8471114635467529 |
