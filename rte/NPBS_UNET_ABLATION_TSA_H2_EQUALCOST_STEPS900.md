# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['anisotropic', 'mixed']`
- `sizes`: `[32]`
- `n_angles`: `12`
- `base_kinds`: `['transport_tsa']`
- `tsa_harmonics`: `2`
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
- `output_csv`: `rte/npbs_unet_ablation_tsa_h2_equalcost_steps900.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_TSA_H2_EQUALCOST_STEPS900.md`

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | one-step base metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | transport_tsa | anisotropic | train | tsa1-w0.5 | 5.362e-01 | 5.893e-01 | 7.165e-01 | 8.021e-06 | 5.178e-06 |  |  |
| 32 | transport_tsa | anisotropic | train | tsa1 | 2.462e-01 | 3.146e-01 | 4.912e-01 | 5.858e-06 | 5.178e-06 |  |  |
| 32 | transport_tsa | anisotropic | train | tsa2-w0.5 | 3.164e-01 | 3.845e-01 | 5.485e-01 | 5.550e-06 | 5.226e-06 |  |  |
| 32 | transport_tsa | anisotropic | train | tsa2 | 1.406e-01 | 1.877e-01 | 3.224e-01 | 5.728e-06 | 5.951e-06 |  |  |
| 32 | transport_tsa | anisotropic | train | tsa3-w0.5 | 2.109e-01 | 2.757e-01 | 4.375e-01 | 5.684e-06 | 6.026e-06 |  |  |
| 32 | transport_tsa | anisotropic | train | tsa3 | 8.509e-02 | 1.161e-01 | 2.170e-01 | 5.732e-06 | 5.662e-06 |  |  |
| 32 | transport_tsa | anisotropic | train | tsa4-w0.5 | 1.552e-01 | 2.106e-01 | 3.570e-01 | 5.894e-06 | 5.793e-06 |  |  |
| 32 | transport_tsa | anisotropic | train | tsa4 | 5.287e-02 | 7.318e-02 | 1.468e-01 | 5.808e-06 | 5.830e-06 |  |  |
| 32 | transport_tsa | anisotropic | train | UNet-defect2-w1 | 1.148e-01 | 1.217e-01 | 7.729e-02 | 6.069e-06 | 5.593e-06 | 0.2211408019065857 | 0.04074271768331528 |
| 32 | transport_tsa | anisotropic | heldout | tsa1-w0.5 | 5.351e-01 | 5.867e-01 | 7.167e-01 | 8.323e-06 | 5.853e-06 |  |  |
| 32 | transport_tsa | anisotropic | heldout | tsa1 | 2.436e-01 | 3.100e-01 | 4.925e-01 | 6.711e-06 | 5.853e-06 |  |  |
| 32 | transport_tsa | anisotropic | heldout | tsa2-w0.5 | 3.144e-01 | 3.805e-01 | 5.490e-01 | 5.471e-06 | 6.324e-06 |  |  |
| 32 | transport_tsa | anisotropic | heldout | tsa2 | 1.375e-01 | 1.834e-01 | 3.233e-01 | 5.678e-06 | 5.944e-06 |  |  |
| 32 | transport_tsa | anisotropic | heldout | tsa3-w0.5 | 2.083e-01 | 2.713e-01 | 4.382e-01 | 5.462e-06 | 6.075e-06 |  |  |
| 32 | transport_tsa | anisotropic | heldout | tsa3 | 8.281e-02 | 1.130e-01 | 2.177e-01 | 6.202e-06 | 5.702e-06 |  |  |
| 32 | transport_tsa | anisotropic | heldout | tsa4-w0.5 | 1.525e-01 | 2.062e-01 | 3.577e-01 | 5.973e-06 | 5.929e-06 |  |  |
| 32 | transport_tsa | anisotropic | heldout | tsa4 | 5.131e-02 | 7.103e-02 | 1.473e-01 | 6.293e-06 | 6.456e-06 |  |  |
| 32 | transport_tsa | anisotropic | heldout | UNet-defect2-w1 | 1.197e-01 | 1.272e-01 | 7.464e-02 | 5.716e-06 | 5.764e-06 | 0.2211408019065857 | 0.04074271768331528 |
| 32 | transport_tsa | mixed | train | tsa1-w0.5 | 6.285e-01 | 6.462e-01 | 6.662e-01 | 1.065e-02 | 1.327e-03 |  |  |
| 32 | transport_tsa | mixed | train | tsa1 | 4.435e-01 | 4.181e-01 | 4.221e-01 | 2.419e-03 | 1.327e-03 |  |  |
| 32 | transport_tsa | mixed | train | tsa2-w0.5 | 4.308e-01 | 4.426e-01 | 4.753e-01 | 2.503e-03 | 2.226e-04 |  |  |
| 32 | transport_tsa | mixed | train | tsa2 | 2.413e-01 | 2.317e-01 | 2.802e-01 | 1.430e-04 | 3.155e-05 |  |  |
| 32 | transport_tsa | mixed | train | tsa3-w0.5 | 3.169e-01 | 3.222e-01 | 3.669e-01 | 6.154e-04 | 4.294e-05 |  |  |
| 32 | transport_tsa | mixed | train | tsa3 | 1.587e-01 | 1.669e-01 | 2.309e-01 | 9.408e-06 | 3.705e-06 |  |  |
| 32 | transport_tsa | mixed | train | tsa4-w0.5 | 2.458e-01 | 2.493e-01 | 3.043e-01 | 1.528e-04 | 1.000e-05 |  |  |
| 32 | transport_tsa | mixed | train | tsa4 | 1.193e-01 | 1.332e-01 | 1.972e-01 | 3.610e-06 | 3.306e-06 |  |  |
| 32 | transport_tsa | mixed | train | UNet-defect2-w1 | 2.507e-01 | 2.314e-01 | 1.320e-01 | 4.994e-05 | 2.099e-05 | 0.5817745327949524 | 0.21515589952468872 |
| 32 | transport_tsa | mixed | heldout | tsa1-w0.5 | 5.630e-01 | 5.735e-01 | 6.748e-01 | 9.595e-03 | 1.423e-03 |  |  |
| 32 | transport_tsa | mixed | heldout | tsa1 | 2.682e-01 | 2.910e-01 | 4.564e-01 | 2.347e-03 | 1.423e-03 |  |  |
| 32 | transport_tsa | mixed | heldout | tsa2-w0.5 | 3.497e-01 | 3.685e-01 | 5.102e-01 | 2.416e-03 | 2.451e-04 |  |  |
| 32 | transport_tsa | mixed | heldout | tsa2 | 1.922e-01 | 2.208e-01 | 3.535e-01 | 1.790e-04 | 3.251e-05 |  |  |
| 32 | transport_tsa | mixed | heldout | tsa3-w0.5 | 2.485e-01 | 2.732e-01 | 4.221e-01 | 6.723e-04 | 4.559e-05 |  |  |
| 32 | transport_tsa | mixed | heldout | tsa3 | 1.662e-01 | 1.910e-01 | 2.956e-01 | 1.434e-05 | 3.625e-06 |  |  |
| 32 | transport_tsa | mixed | heldout | tsa4-w0.5 | 2.008e-01 | 2.278e-01 | 3.691e-01 | 1.891e-04 | 9.279e-06 |  |  |
| 32 | transport_tsa | mixed | heldout | tsa4 | 1.427e-01 | 1.632e-01 | 2.507e-01 | 3.680e-06 | 3.424e-06 |  |  |
| 32 | transport_tsa | mixed | heldout | UNet-defect2-w1 | 1.805e-01 | 1.892e-01 | 1.461e-01 | 1.228e-05 | 1.208e-05 | 0.5817745327949524 | 0.21515589952468872 |
