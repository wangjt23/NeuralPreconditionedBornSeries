# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['anisotropic', 'mixed']`
- `sizes`: `[32]`
- `n_angles`: `12`
- `base_kinds`: `['transport_dsa']`
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
- `output_csv`: `rte/npbs_unet_ablation_dsa_equalcost_steps900.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_DSA_EQUALCOST_STEPS900.md`

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | one-step base metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | transport_dsa | anisotropic | train | dsa1-w0.5 | 7.321e-01 | 8.029e-01 | 5.088e-01 | 2.769e-03 | 2.020e-05 |  |  |
| 32 | transport_dsa | anisotropic | train | dsa1 | 6.777e-01 | 7.586e-01 | 1.129e-01 | 3.440e-05 | 2.020e-05 |  |  |
| 32 | transport_dsa | anisotropic | train | dsa2-w0.5 | 5.778e-01 | 6.754e-01 | 2.661e-01 | 4.588e-05 | 5.987e-06 |  |  |
| 32 | transport_dsa | anisotropic | train | dsa2 | 4.294e-01 | 5.232e-01 | 8.233e-02 | 5.687e-06 | 5.939e-06 |  |  |
| 32 | transport_dsa | anisotropic | train | dsa3-w0.5 | 4.657e-01 | 5.703e-01 | 1.488e-01 | 5.466e-06 | 5.532e-06 |  |  |
| 32 | transport_dsa | anisotropic | train | dsa3 | 2.708e-01 | 3.507e-01 | 5.325e-02 | 6.116e-06 | 5.585e-06 |  |  |
| 32 | transport_dsa | anisotropic | train | dsa4-w0.5 | 3.762e-01 | 4.792e-01 | 9.345e-02 | 5.786e-06 | 6.281e-06 |  |  |
| 32 | transport_dsa | anisotropic | train | dsa4 | 1.950e-01 | 2.579e-01 | 3.828e-02 | 5.668e-06 | 5.473e-06 |  |  |
| 32 | transport_dsa | anisotropic | train | UNet-defect2-w1 | 2.064e-01 | 2.126e-01 | 8.719e-02 | 5.609e-06 | 5.947e-06 | 0.3803770840167999 | 0.10656953603029251 |
| 32 | transport_dsa | anisotropic | heldout | dsa1-w0.5 | 7.318e-01 | 8.016e-01 | 5.088e-01 | 2.769e-03 | 2.135e-05 |  |  |
| 32 | transport_dsa | anisotropic | heldout | dsa1 | 6.774e-01 | 7.572e-01 | 1.126e-01 | 3.430e-05 | 2.135e-05 |  |  |
| 32 | transport_dsa | anisotropic | heldout | dsa2-w0.5 | 5.769e-01 | 6.729e-01 | 2.660e-01 | 4.588e-05 | 6.123e-06 |  |  |
| 32 | transport_dsa | anisotropic | heldout | dsa2 | 4.283e-01 | 5.205e-01 | 8.195e-02 | 5.951e-06 | 6.005e-06 |  |  |
| 32 | transport_dsa | anisotropic | heldout | dsa3-w0.5 | 4.642e-01 | 5.667e-01 | 1.485e-01 | 5.608e-06 | 6.075e-06 |  |  |
| 32 | transport_dsa | anisotropic | heldout | dsa3 | 2.691e-01 | 3.472e-01 | 5.276e-02 | 6.031e-06 | 5.799e-06 |  |  |
| 32 | transport_dsa | anisotropic | heldout | dsa4-w0.5 | 3.741e-01 | 4.749e-01 | 9.293e-02 | 6.148e-06 | 6.146e-06 |  |  |
| 32 | transport_dsa | anisotropic | heldout | dsa4 | 1.932e-01 | 2.545e-01 | 3.776e-02 | 5.755e-06 | 5.973e-06 |  |  |
| 32 | transport_dsa | anisotropic | heldout | UNet-defect2-w1 | 2.044e-01 | 2.043e-01 | 9.511e-02 | 6.258e-06 | 5.859e-06 | 0.3803770840167999 | 0.10656953603029251 |
| 32 | transport_dsa | mixed | train | dsa1-w0.5 | 8.641e-01 | 8.788e-01 | 4.891e-01 | 1.409e-02 | 4.873e-03 |  |  |
| 32 | transport_dsa | mixed | train | dsa1 | 9.638e-01 | 9.703e-01 | 1.701e-01 | 5.084e-03 | 4.873e-03 |  |  |
| 32 | transport_dsa | mixed | train | dsa2-w0.5 | 7.771e-01 | 7.930e-01 | 2.519e-01 | 5.170e-03 | 1.095e-03 |  |  |
| 32 | transport_dsa | mixed | train | dsa2 | 6.522e-01 | 6.433e-01 | 1.524e-01 | 7.067e-04 | 2.976e-04 |  |  |
| 32 | transport_dsa | mixed | train | dsa3-w0.5 | 6.624e-01 | 6.744e-01 | 1.520e-01 | 1.933e-03 | 3.310e-04 |  |  |
| 32 | transport_dsa | mixed | train | dsa3 | 3.415e-01 | 3.246e-01 | 9.792e-02 | 1.013e-04 | 1.960e-05 |  |  |
| 32 | transport_dsa | mixed | train | dsa4-w0.5 | 5.303e-01 | 5.373e-01 | 1.118e-01 | 7.296e-04 | 1.116e-04 |  |  |
| 32 | transport_dsa | mixed | train | dsa4 | 2.967e-01 | 2.951e-01 | 4.796e-02 | 1.516e-05 | 3.779e-06 |  |  |
| 32 | transport_dsa | mixed | train | UNet-defect2-w1 | 3.602e-01 | 3.397e-01 | 1.395e-01 | 9.650e-06 | 1.885e-05 | 0.48508450388908386 | 0.1648988574743271 |
| 32 | transport_dsa | mixed | heldout | dsa1-w0.5 | 7.836e-01 | 7.980e-01 | 5.414e-01 | 1.377e-02 | 4.450e-03 |  |  |
| 32 | transport_dsa | mixed | heldout | dsa1 | 8.548e-01 | 8.693e-01 | 1.728e-01 | 4.921e-03 | 4.450e-03 |  |  |
| 32 | transport_dsa | mixed | heldout | dsa2-w0.5 | 6.697e-01 | 6.903e-01 | 3.036e-01 | 5.003e-03 | 1.154e-03 |  |  |
| 32 | transport_dsa | mixed | heldout | dsa2 | 5.448e-01 | 5.734e-01 | 1.151e-01 | 6.935e-04 | 2.681e-04 |  |  |
| 32 | transport_dsa | mixed | heldout | dsa3-w0.5 | 5.573e-01 | 5.805e-01 | 1.768e-01 | 1.884e-03 | 3.211e-04 |  |  |
| 32 | transport_dsa | mixed | heldout | dsa3 | 2.778e-01 | 3.004e-01 | 8.850e-02 | 9.983e-05 | 1.791e-05 |  |  |
| 32 | transport_dsa | mixed | heldout | dsa4-w0.5 | 4.406e-01 | 4.642e-01 | 1.095e-01 | 7.158e-04 | 8.919e-05 |  |  |
| 32 | transport_dsa | mixed | heldout | dsa4 | 2.327e-01 | 2.532e-01 | 5.446e-02 | 1.486e-05 | 3.776e-06 |  |  |
| 32 | transport_dsa | mixed | heldout | UNet-defect2-w1 | 2.821e-01 | 2.775e-01 | 1.269e-01 | 1.119e-05 | 2.076e-05 | 0.48508450388908386 | 0.1648988574743271 |
