# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['anisotropic', 'mixed']`
- `sizes`: `[32]`
- `n_angles`: `12`
- `base_kinds`: `['transport_tsa']`
- `tsa_harmonics`: `None`
- `forms`: `['defect2']`
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
- `physical_weight`: `0.5`
- `correction_weight`: `1e-06`
- `rollout_weight`: `0.2`
- `rollout_steps`: `3`
- `rollout_decay`: `0.8`
- `stability_weight`: `0.5`
- `contraction_target`: `0.95`
- `maxiter`: `60`
- `fgmres_maxiter`: `25`
- `tol`: `1e-06`
- `seed`: `20260519`
- `device`: `cuda:0`
- `allow_tf32`: `False`
- `output_init`: `small`
- `output_csv`: `rte/npbs_unet_ablation_tsa_full_equalcost_steps900.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_TSA_FULL_EQUALCOST_STEPS900.md`

## Terminology

- `transport_tsa` is one reference transport correction followed by one reference transport-scattering synthetic correction.
- Here `tsa_harmonics=None`, so the synthetic correction keeps all angular Fourier modes.
- `tsaK` means `K` residual-correction applications of this base. Starting with `z=0` and `e_0=r`, each stage applies `z <- z + omega B_h e_j` and `e_{j+1} <- e_j - A_h(omega B_h e_j)`.
- A suffix such as `-w0.5` means relaxation `omega=0.5`; no suffix means `omega=1`.
- `UNet-defect2-w1` uses `z0=B_h r`, `z1=B_h(r-A_h z0)`, and returns `z0 + z1 + C_theta(z0,z1,coeff)`.
- `base-call equal` means equal counts of selected base-preconditioner applications inside one preconditioner call, excluding diagnostic metric evaluations, `A_h` applications, scalar reductions, and U-Net overhead. `UNet-defect2` is base-call equal to `tsa2`; `tsa3` and `tsa4` are deeper classical work-budget baselines, not strict equal-cost matches.

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | one-step base metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | transport_tsa | anisotropic | train | tsa1-w0.5 | 5.275e-01 | 5.634e-01 | 7.078e-01 | 6.895e-06 | 5.920e-06 |  |  |
| 32 | transport_tsa | anisotropic | train | tsa1 | 1.577e-01 | 2.261e-01 | 4.745e-01 | 5.646e-06 | 5.920e-06 |  |  |
| 32 | transport_tsa | anisotropic | train | tsa2-w0.5 | 2.922e-01 | 3.389e-01 | 5.340e-01 | 5.633e-06 | 5.573e-06 |  |  |
| 32 | transport_tsa | anisotropic | train | tsa2 | 8.251e-02 | 1.205e-01 | 3.059e-01 | 5.824e-06 | 5.443e-06 |  |  |
| 32 | transport_tsa | anisotropic | train | tsa3-w0.5 | 1.750e-01 | 2.211e-01 | 4.212e-01 | 5.493e-06 | 5.765e-06 |  |  |
| 32 | transport_tsa | anisotropic | train | tsa3 | 5.400e-02 | 7.888e-02 | 2.061e-01 | 6.302e-06 | 5.821e-06 |  |  |
| 32 | transport_tsa | anisotropic | train | tsa4-w0.5 | 1.158e-01 | 1.567e-01 | 3.413e-01 | 5.439e-06 | 6.542e-06 |  |  |
| 32 | transport_tsa | anisotropic | train | tsa4 | 3.645e-02 | 5.325e-02 | 1.398e-01 | 5.701e-06 | 5.472e-06 |  |  |
| 32 | transport_tsa | anisotropic | train | UNet-defect2-w1 | 1.004e-01 | 7.818e-02 | 6.656e-02 | 5.805e-06 | 6.467e-06 | 0.23324571549892426 | 0.0033545950427651405 |
| 32 | transport_tsa | anisotropic | heldout | tsa1-w0.5 | 5.262e-01 | 5.608e-01 | 7.087e-01 | 6.712e-06 | 5.946e-06 |  |  |
| 32 | transport_tsa | anisotropic | heldout | tsa1 | 1.564e-01 | 2.241e-01 | 4.778e-01 | 6.169e-06 | 5.946e-06 |  |  |
| 32 | transport_tsa | anisotropic | heldout | tsa2-w0.5 | 2.907e-01 | 3.361e-01 | 5.360e-01 | 5.270e-06 | 5.953e-06 |  |  |
| 32 | transport_tsa | anisotropic | heldout | tsa2 | 8.236e-02 | 1.203e-01 | 3.088e-01 | 6.101e-06 | 5.765e-06 |  |  |
| 32 | transport_tsa | anisotropic | heldout | tsa3-w0.5 | 1.738e-01 | 2.191e-01 | 4.237e-01 | 6.036e-06 | 6.079e-06 |  |  |
| 32 | transport_tsa | anisotropic | heldout | tsa3 | 5.399e-02 | 7.891e-02 | 2.081e-01 | 5.693e-06 | 5.686e-06 |  |  |
| 32 | transport_tsa | anisotropic | heldout | tsa4-w0.5 | 1.150e-01 | 1.555e-01 | 3.439e-01 | 6.782e-06 | 5.618e-06 |  |  |
| 32 | transport_tsa | anisotropic | heldout | tsa4 | 3.645e-02 | 5.328e-02 | 1.412e-01 | 5.661e-06 | 5.997e-06 |  |  |
| 32 | transport_tsa | anisotropic | heldout | UNet-defect2-w1 | 8.680e-02 | 7.088e-02 | 6.854e-02 | 6.082e-06 | 6.325e-06 | 0.23324571549892426 | 0.0033545950427651405 |
| 32 | transport_tsa | mixed | train | tsa1-w0.5 | 6.204e-01 | 6.377e-01 | 6.671e-01 | 4.653e-03 | 7.224e-05 |  |  |
| 32 | transport_tsa | mixed | train | tsa1 | 4.058e-01 | 3.888e-01 | 4.252e-01 | 4.249e-04 | 7.224e-05 |  |  |
| 32 | transport_tsa | mixed | train | tsa2-w0.5 | 4.083e-01 | 4.230e-01 | 4.757e-01 | 4.621e-04 | 8.465e-06 |  |  |
| 32 | transport_tsa | mixed | train | tsa2 | 1.709e-01 | 1.829e-01 | 2.792e-01 | 6.058e-06 | 3.372e-06 |  |  |
| 32 | transport_tsa | mixed | train | tsa3-w0.5 | 2.816e-01 | 2.938e-01 | 3.665e-01 | 5.210e-05 | 3.621e-06 |  |  |
| 32 | transport_tsa | mixed | train | tsa3 | 1.168e-01 | 1.363e-01 | 2.315e-01 | 3.400e-06 | 3.561e-06 |  |  |
| 32 | transport_tsa | mixed | train | tsa4-w0.5 | 2.041e-01 | 2.169e-01 | 3.037e-01 | 6.861e-06 | 3.467e-06 |  |  |
| 32 | transport_tsa | mixed | train | tsa4 | 9.576e-02 | 1.137e-01 | 1.984e-01 | 3.588e-06 | 3.585e-06 |  |  |
| 32 | transport_tsa | mixed | train | UNet-defect2-w1 | 1.810e-01 | 1.772e-01 | 1.262e-01 | 3.438e-06 | 3.331e-06 | 0.5467498302459717 | 0.14502346515655518 |
| 32 | transport_tsa | mixed | heldout | tsa1-w0.5 | 5.625e-01 | 5.736e-01 | 6.761e-01 | 4.170e-03 | 5.675e-05 |  |  |
| 32 | transport_tsa | mixed | heldout | tsa1 | 2.638e-01 | 2.889e-01 | 4.607e-01 | 3.552e-04 | 5.675e-05 |  |  |
| 32 | transport_tsa | mixed | heldout | tsa2-w0.5 | 3.476e-01 | 3.671e-01 | 5.125e-01 | 3.852e-04 | 5.656e-06 |  |  |
| 32 | transport_tsa | mixed | heldout | tsa2 | 1.811e-01 | 2.102e-01 | 3.595e-01 | 5.955e-06 | 3.584e-06 |  |  |
| 32 | transport_tsa | mixed | heldout | tsa3-w0.5 | 2.436e-01 | 2.682e-01 | 4.250e-01 | 4.583e-05 | 3.581e-06 |  |  |
| 32 | transport_tsa | mixed | heldout | tsa3 | 1.442e-01 | 1.676e-01 | 2.986e-01 | 3.406e-06 | 3.348e-06 |  |  |
| 32 | transport_tsa | mixed | heldout | tsa4-w0.5 | 1.918e-01 | 2.183e-01 | 3.720e-01 | 6.503e-06 | 3.420e-06 |  |  |
| 32 | transport_tsa | mixed | heldout | tsa4 | 1.143e-01 | 1.338e-01 | 2.507e-01 | 3.251e-06 | 3.278e-06 |  |  |
| 32 | transport_tsa | mixed | heldout | UNet-defect2-w1 | 1.482e-01 | 1.570e-01 | 1.434e-01 | 3.533e-06 | 3.438e-06 | 0.5467498302459717 | 0.14502346515655518 |
