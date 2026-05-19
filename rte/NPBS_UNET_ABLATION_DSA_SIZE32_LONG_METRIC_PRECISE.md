# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['diffusive', 'heterogeneous', 'anisotropic', 'mixed']`
- `sizes`: `[32]`
- `n_angles`: `12`
- `base_kinds`: `['transport_dsa']`
- `forms`: `['residual', 'defect2']`
- `defect_relaxations`: `[0.5, 1.0]`
- `eta_scale`: `0.1`
- `width`: `16`
- `levels`: `4`
- `steps`: `300`
- `lr`: `0.002`
- `weight_decay`: `0.0001`
- `scheduler`: `step`
- `scheduler_step_size`: `120`
- `scheduler_gamma`: `0.5`
- `scheduler_eta_min`: `1e-05`
- `log_steps`: `100`
- `residual_mode`: `resample`
- `residuals_per_step`: `48`
- `rollout_samples_per_step`: `24`
- `n_rollout`: `8`
- `n_random`: `6`
- `random_modes`: `5`
- `aux_sources`: `2`
- `physical_weight`: `0.5`
- `correction_weight`: `1e-06`
- `rollout_weight`: `0.0`
- `rollout_steps`: `3`
- `rollout_decay`: `0.8`
- `stability_weight`: `0.0`
- `contraction_target`: `0.95`
- `maxiter`: `60`
- `fgmres_maxiter`: `25`
- `tol`: `1e-06`
- `seed`: `20260519`
- `device`: `cuda:0`
- `allow_tf32`: `False`
- `output_csv`: `rte/npbs_unet_ablation_dsa_size32_long_metric_precise.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_DSA_SIZE32_LONG_METRIC_PRECISE.md`

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 32 | transport_dsa | diffusive | train | dsa1 | 2.369e+00 | 2.399e+00 | 9.128e-06 | 8.688e-06 |  |  |
| 32 | transport_dsa | diffusive | train | dsa2 | 4.219e-01 | 4.251e-01 | 9.129e-06 | 8.984e-06 |  |  |
| 32 | transport_dsa | diffusive | train | UNet-residual-w1 | 1.788e-01 | 1.726e-01 | 8.934e-06 | 1.068e-05 | 0.2415972352027893 | 0.010354205034673214 |
| 32 | transport_dsa | diffusive | train | UNet-defect2-w0.5 | 1.123e-01 | 1.094e-01 | 8.915e-06 | 1.110e-05 | 0.08265478163957596 | 0.0036839558742940426 |
| 32 | transport_dsa | diffusive | train | UNet-defect2-w1 | 7.507e-02 | 7.660e-02 | 8.738e-06 | 8.134e-06 | 0.009979547001421452 | 0.0008330386481247842 |
| 32 | transport_dsa | diffusive | heldout | dsa1 | 2.133e+00 | 2.159e+00 | 7.735e-06 | 7.924e-06 |  |  |
| 32 | transport_dsa | diffusive | heldout | dsa2 | 3.824e-01 | 3.850e-01 | 7.851e-06 | 8.130e-06 |  |  |
| 32 | transport_dsa | diffusive | heldout | UNet-residual-w1 | 1.631e-01 | 1.566e-01 | 7.314e-06 | 9.147e-06 | 0.2415972352027893 | 0.010354205034673214 |
| 32 | transport_dsa | diffusive | heldout | UNet-defect2-w0.5 | 1.036e-01 | 9.992e-02 | 8.777e-06 | 7.722e-06 | 0.08265478163957596 | 0.0036839558742940426 |
| 32 | transport_dsa | diffusive | heldout | UNet-defect2-w1 | 6.803e-02 | 6.937e-02 | 7.770e-06 | 8.182e-06 | 0.009979547001421452 | 0.0008330386481247842 |
| 32 | transport_dsa | heterogeneous | train | dsa1 | 4.859e-01 | 5.279e-01 | 4.176e-06 | 4.512e-06 |  |  |
| 32 | transport_dsa | heterogeneous | train | dsa2 | 2.303e-01 | 2.907e-01 | 4.610e-06 | 4.498e-06 |  |  |
| 32 | transport_dsa | heterogeneous | train | UNet-residual-w1 | 3.646e-01 | 4.192e-01 | 4.518e-06 | 4.625e-06 | 0.25229084491729736 | 0.08705103397369385 |
| 32 | transport_dsa | heterogeneous | train | UNet-defect2-w0.5 | 2.075e-01 | 2.364e-01 | 4.530e-06 | 4.371e-06 | 0.1512298882007599 | 0.030512727797031403 |
| 32 | transport_dsa | heterogeneous | train | UNet-defect2-w1 | 1.906e-01 | 2.215e-01 | 4.601e-06 | 4.602e-06 | 0.08957049250602722 | 0.02227230928838253 |
| 32 | transport_dsa | heterogeneous | heldout | dsa1 | 3.723e-01 | 3.726e-01 | 3.626e-06 | 3.725e-06 |  |  |
| 32 | transport_dsa | heterogeneous | heldout | dsa2 | 1.503e-01 | 1.858e-01 | 3.807e-06 | 3.694e-06 |  |  |
| 32 | transport_dsa | heterogeneous | heldout | UNet-residual-w1 | 3.709e-01 | 3.298e-01 | 3.658e-06 | 3.585e-06 | 0.25229084491729736 | 0.08705103397369385 |
| 32 | transport_dsa | heterogeneous | heldout | UNet-defect2-w0.5 | 1.702e-01 | 1.672e-01 | 3.704e-06 | 3.645e-06 | 0.1512298882007599 | 0.030512727797031403 |
| 32 | transport_dsa | heterogeneous | heldout | UNet-defect2-w1 | 1.592e-01 | 1.590e-01 | 3.406e-06 | 3.502e-06 | 0.08957049250602722 | 0.02227230928838253 |
| 32 | transport_dsa | anisotropic | train | dsa1 | 6.777e-01 | 7.586e-01 | 3.440e-05 | 2.020e-05 |  |  |
| 32 | transport_dsa | anisotropic | train | dsa2 | 4.294e-01 | 5.232e-01 | 5.687e-06 | 5.939e-06 |  |  |
| 32 | transport_dsa | anisotropic | train | UNet-residual-w1 | 3.328e-01 | 3.917e-01 | 9.227e-06 | 3.180e-05 | 0.4161929488182068 | 0.11654666066169739 |
| 32 | transport_dsa | anisotropic | train | UNet-defect2-w0.5 | 2.660e-01 | 3.167e-01 | 5.678e-06 | 5.677e-06 | 0.2856840193271637 | 0.08094978332519531 |
| 32 | transport_dsa | anisotropic | train | UNet-defect2-w1 | 2.595e-01 | 3.039e-01 | 5.981e-06 | 5.746e-06 | 0.21409691870212555 | 0.05921095609664917 |
| 32 | transport_dsa | anisotropic | heldout | dsa1 | 6.774e-01 | 7.572e-01 | 3.430e-05 | 2.135e-05 |  |  |
| 32 | transport_dsa | anisotropic | heldout | dsa2 | 4.283e-01 | 5.205e-01 | 5.951e-06 | 6.005e-06 |  |  |
| 32 | transport_dsa | anisotropic | heldout | UNet-residual-w1 | 2.788e-01 | 3.227e-01 | 5.601e-06 | 2.430e-05 | 0.4161929488182068 | 0.11654666066169739 |
| 32 | transport_dsa | anisotropic | heldout | UNet-defect2-w0.5 | 2.420e-01 | 2.846e-01 | 5.797e-06 | 6.039e-06 | 0.2856840193271637 | 0.08094978332519531 |
| 32 | transport_dsa | anisotropic | heldout | UNet-defect2-w1 | 2.292e-01 | 2.644e-01 | 5.386e-06 | 6.081e-06 | 0.21409691870212555 | 0.05921095609664917 |
| 32 | transport_dsa | mixed | train | dsa1 | 9.638e-01 | 9.703e-01 | 5.084e-03 | 4.873e-03 |  |  |
| 32 | transport_dsa | mixed | train | dsa2 | 6.522e-01 | 6.433e-01 | 7.067e-04 | 2.976e-04 |  |  |
| 32 | transport_dsa | mixed | train | UNet-residual-w1 | 4.336e-01 | 4.314e-01 | 4.487e-03 | 2.345e-03 | 0.5202147960662842 | 0.1893014907836914 |
| 32 | transport_dsa | mixed | train | UNet-defect2-w0.5 | 4.154e-01 | 4.243e-01 | 1.456e-03 | 4.840e-04 | 0.34518054127693176 | 0.14077943563461304 |
| 32 | transport_dsa | mixed | train | UNet-defect2-w1 | 3.550e-01 | 3.572e-01 | 5.533e-04 | 1.719e-04 | 0.29108095169067383 | 0.11566277593374252 |
| 32 | transport_dsa | mixed | heldout | dsa1 | 8.548e-01 | 8.693e-01 | 4.921e-03 | 4.450e-03 |  |  |
| 32 | transport_dsa | mixed | heldout | dsa2 | 5.448e-01 | 5.734e-01 | 6.935e-04 | 2.681e-04 |  |  |
| 32 | transport_dsa | mixed | heldout | UNet-residual-w1 | 3.014e-01 | 3.001e-01 | 4.269e-03 | 2.645e-03 | 0.5202147960662842 | 0.1893014907836914 |
| 32 | transport_dsa | mixed | heldout | UNet-defect2-w0.5 | 2.673e-01 | 2.784e-01 | 1.450e-03 | 4.816e-04 | 0.34518054127693176 | 0.14077943563461304 |
| 32 | transport_dsa | mixed | heldout | UNet-defect2-w1 | 2.297e-01 | 2.352e-01 | 5.737e-04 | 1.439e-04 | 0.29108095169067383 | 0.11566277593374252 |
