# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['anisotropic', 'mixed']`
- `sizes`: `[80]`
- `n_angles`: `24`
- `base_kinds`: `['transport_tsa_moment']`
- `tsa_harmonics`: `None`
- `forms`: `['coeff_weighted_moment_tsa']`
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
- `seed`: `20260524`
- `device`: `cuda:0`
- `allow_tf32`: `False`
- `output_init`: `zero`
- `output_csv`: `rte/npbs_unet_ablation_coeff_weighted_tsa_m4_damped_size80_steps1800.csv`
- `trace_csv`: `rte/npbs_unet_ablation_coeff_weighted_tsa_m4_damped_size80_steps1800_traces.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_COEFF_WEIGHTED_TSA_M4_DAMPED_SIZE80_STEPS1800.md`

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
| 80 | transport_tsa_moment | 4 | real-stacked | anisotropic | train | tsa_moment1-w0.4 | 6.564e-01 | 7.559e-01 | 1.331e-05 | 1.566e-05 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | anisotropic | train | tsa_moment1 | 5.676e-01 | 5.282e-01 | inf | 1.561e-05 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | anisotropic | train | tsa_moment2-w0.4 | 4.686e-01 | 6.068e-01 | 1.393e-05 | 1.568e-05 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | anisotropic | train | tsa_moment2 | 2.089e-01 | 3.321e-01 | nan | 3.304e-03 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | anisotropic | train | tsa_moment3-w0.4 | 3.470e-01 | 5.042e-01 | 1.473e-05 | 1.555e-05 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | anisotropic | train | tsa_moment3 | 1.009e-01 | 2.217e-01 | nan | 1.569e-05 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | anisotropic | train | tsa_moment4-w0.4 | 2.613e-01 | 4.270e-01 | 1.500e-05 | 1.609e-05 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | anisotropic | train | tsa_moment4 | 5.950e-02 | 1.502e-01 | nan | 3.198e-03 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | anisotropic | train | CoeffWeightedTSA-M4 | 6.300e-01 | 7.296e-01 | 1.362e-05 | 1.533e-05 | nan | nan | 0.8407623171806335 | 0.8150221705436707 |
| 80 | transport_tsa_moment | 4 | real-stacked | anisotropic | heldout | tsa_moment1-w0.4 | 6.559e-01 | 7.563e-01 | 1.278e-05 | 1.612e-05 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | anisotropic | heldout | tsa_moment1 | 5.669e-01 | 5.294e-01 | inf | 1.662e-05 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | anisotropic | heldout | tsa_moment2-w0.4 | 4.679e-01 | 6.075e-01 | 1.487e-05 | 1.661e-05 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | anisotropic | heldout | tsa_moment2 | 2.082e-01 | 3.328e-01 | nan | 3.334e-03 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | anisotropic | heldout | tsa_moment3-w0.4 | 3.462e-01 | 5.050e-01 | 1.473e-05 | 1.655e-05 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | anisotropic | heldout | tsa_moment3 | 1.006e-01 | 2.223e-01 | nan | 1.650e-05 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | anisotropic | heldout | tsa_moment4-w0.4 | 2.604e-01 | 4.278e-01 | 1.575e-05 | 1.684e-05 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | anisotropic | heldout | tsa_moment4 | 5.950e-02 | 1.506e-01 | nan | 3.179e-03 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | anisotropic | heldout | CoeffWeightedTSA-M4 | 6.294e-01 | 7.301e-01 | 1.321e-05 | 1.644e-05 | nan | nan | 0.8407623171806335 | 0.8150221705436707 |
| 80 | transport_tsa_moment | 4 | real-stacked | mixed | train | tsa_moment1-w0.4 | 6.953e-01 | 7.303e-01 | 4.065e-03 | 1.981e-05 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | mixed | train | tsa_moment1 | 4.503e-01 | 4.293e-01 | 6.141e+02 | 1.967e-05 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | mixed | train | tsa_moment2-w0.4 | 5.065e-01 | 5.532e-01 | 5.634e-04 | 8.819e-06 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | mixed | train | tsa_moment2 | 2.377e-01 | 2.826e-01 | 6.979e+12 | 1.826e-03 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | mixed | train | tsa_moment3-w0.4 | 3.840e-01 | 4.374e-01 | 8.051e-05 | 8.755e-06 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | mixed | train | tsa_moment3 | 1.545e-01 | 2.325e-01 | inf | 8.886e-06 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | mixed | train | tsa_moment4-w0.4 | 3.012e-01 | 3.617e-01 | 1.418e-05 | 8.708e-06 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | mixed | train | tsa_moment4 | 1.182e-01 | 1.976e-01 | inf | 1.347e-03 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | mixed | train | CoeffWeightedTSA-M4 | 6.412e-01 | 6.827e-01 | 3.900e-03 | 4.059e-05 | nan | nan | 1.2311152219772339 | 1.2148457765579224 |
| 80 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | tsa_moment1-w0.4 | 6.476e-01 | 7.361e-01 | 2.777e-03 | 2.193e-05 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | tsa_moment1 | 2.696e-01 | 4.595e-01 | 6.580e+02 | 2.192e-05 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | tsa_moment2-w0.4 | 4.381e-01 | 5.774e-01 | 4.207e-04 | 9.473e-06 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | tsa_moment2 | 1.873e-01 | 3.576e-01 | 1.120e+13 | 7.658e-04 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | tsa_moment3-w0.4 | 3.151e-01 | 4.804e-01 | 6.662e-05 | 1.053e-05 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | tsa_moment3 | 1.584e-01 | 2.987e-01 | inf | 1.013e-05 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | tsa_moment4-w0.4 | 2.437e-01 | 4.184e-01 | 1.382e-05 | 1.025e-05 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | tsa_moment4 | 1.352e-01 | 2.523e-01 | inf | 5.309e-04 | nan | nan |  |  |
| 80 | transport_tsa_moment | 4 | real-stacked | mixed | heldout | CoeffWeightedTSA-M4 | 6.008e-01 | 6.957e-01 | 2.514e-03 | 4.711e-05 | nan | nan | 1.2311152219772339 | 1.2148457765579224 |
