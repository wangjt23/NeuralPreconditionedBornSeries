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
- `output_csv`: `rte/npbs_unet_ablation_dsa_size32_long_rollout_precise.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_DSA_SIZE32_LONG_ROLLOUT_PRECISE.md`

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 32 | transport_dsa | diffusive | train | dsa1 | 2.369e+00 | 2.399e+00 | 9.128e-06 | 8.688e-06 |  |  |
| 32 | transport_dsa | diffusive | train | dsa2 | 4.219e-01 | 4.251e-01 | 9.129e-06 | 8.984e-06 |  |  |
| 32 | transport_dsa | diffusive | train | UNet-residual-w1 | 1.835e-01 | 1.765e-01 | 9.720e-06 | 8.464e-06 | 0.2688005864620209 | 0.020583532750606537 |
| 32 | transport_dsa | diffusive | train | UNet-defect2-w0.5 | 1.138e-01 | 1.098e-01 | 8.490e-06 | 8.355e-06 | 0.09104878455400467 | 0.005574927199631929 |
| 32 | transport_dsa | diffusive | train | UNet-defect2-w1 | 5.885e-02 | 5.634e-02 | 8.737e-06 | 8.595e-06 | 0.011703116819262505 | 0.0007595375645905733 |
| 32 | transport_dsa | diffusive | heldout | dsa1 | 2.133e+00 | 2.159e+00 | 7.735e-06 | 7.924e-06 |  |  |
| 32 | transport_dsa | diffusive | heldout | dsa2 | 3.824e-01 | 3.850e-01 | 7.851e-06 | 8.130e-06 |  |  |
| 32 | transport_dsa | diffusive | heldout | UNet-residual-w1 | 1.664e-01 | 1.599e-01 | 9.152e-06 | 7.826e-06 | 0.2688005864620209 | 0.020583532750606537 |
| 32 | transport_dsa | diffusive | heldout | UNet-defect2-w0.5 | 1.050e-01 | 1.005e-01 | 8.039e-06 | 7.862e-06 | 0.09104878455400467 | 0.005574927199631929 |
| 32 | transport_dsa | diffusive | heldout | UNet-defect2-w1 | 5.443e-02 | 5.226e-02 | 7.827e-06 | 8.386e-06 | 0.011703116819262505 | 0.0007595375645905733 |
| 32 | transport_dsa | heterogeneous | train | dsa1 | 4.859e-01 | 5.279e-01 | 4.176e-06 | 4.512e-06 |  |  |
| 32 | transport_dsa | heterogeneous | train | dsa2 | 2.303e-01 | 2.907e-01 | 4.610e-06 | 4.498e-06 |  |  |
| 32 | transport_dsa | heterogeneous | train | UNet-residual-w1 | 3.506e-01 | 3.965e-01 | 4.305e-06 | 4.500e-06 | 0.3908056616783142 | 0.17138312757015228 |
| 32 | transport_dsa | heterogeneous | train | UNet-defect2-w0.5 | 2.247e-01 | 2.447e-01 | 4.276e-06 | 4.401e-06 | 0.25215834379196167 | 0.08053945749998093 |
| 32 | transport_dsa | heterogeneous | train | UNet-defect2-w1 | 2.163e-01 | 2.496e-01 | 4.303e-06 | 4.683e-06 | 0.1635359823703766 | 0.054936859756708145 |
| 32 | transport_dsa | heterogeneous | heldout | dsa1 | 3.723e-01 | 3.726e-01 | 3.626e-06 | 3.725e-06 |  |  |
| 32 | transport_dsa | heterogeneous | heldout | dsa2 | 1.503e-01 | 1.858e-01 | 3.807e-06 | 3.694e-06 |  |  |
| 32 | transport_dsa | heterogeneous | heldout | UNet-residual-w1 | 3.422e-01 | 3.068e-01 | 3.561e-06 | 3.428e-06 | 0.3908056616783142 | 0.17138312757015228 |
| 32 | transport_dsa | heterogeneous | heldout | UNet-defect2-w0.5 | 1.942e-01 | 1.990e-01 | 3.505e-06 | 3.298e-06 | 0.25215834379196167 | 0.08053945749998093 |
| 32 | transport_dsa | heterogeneous | heldout | UNet-defect2-w1 | 1.864e-01 | 1.912e-01 | 3.466e-06 | 3.371e-06 | 0.1635359823703766 | 0.054936859756708145 |
| 32 | transport_dsa | anisotropic | train | dsa1 | 6.777e-01 | 7.586e-01 | 3.440e-05 | 2.020e-05 |  |  |
| 32 | transport_dsa | anisotropic | train | dsa2 | 4.294e-01 | 5.232e-01 | 5.687e-06 | 5.939e-06 |  |  |
| 32 | transport_dsa | anisotropic | train | UNet-residual-w1 | 3.402e-01 | 4.034e-01 | 5.571e-06 | 1.975e-05 | 0.6279622912406921 | 0.27037641406059265 |
| 32 | transport_dsa | anisotropic | train | UNet-defect2-w0.5 | 2.525e-01 | 2.858e-01 | 5.680e-06 | 5.842e-06 | 0.45918428897857666 | 0.17618291079998016 |
| 32 | transport_dsa | anisotropic | train | UNet-defect2-w1 | 2.812e-01 | 3.272e-01 | 5.902e-06 | 5.630e-06 | 0.3595793843269348 | 0.15056923031806946 |
| 32 | transport_dsa | anisotropic | heldout | dsa1 | 6.774e-01 | 7.572e-01 | 3.430e-05 | 2.135e-05 |  |  |
| 32 | transport_dsa | anisotropic | heldout | dsa2 | 4.283e-01 | 5.205e-01 | 5.951e-06 | 6.005e-06 |  |  |
| 32 | transport_dsa | anisotropic | heldout | UNet-residual-w1 | 3.036e-01 | 3.626e-01 | 6.455e-06 | 9.631e-06 | 0.6279622912406921 | 0.27037641406059265 |
| 32 | transport_dsa | anisotropic | heldout | UNet-defect2-w0.5 | 2.250e-01 | 2.513e-01 | 5.686e-06 | 6.088e-06 | 0.45918428897857666 | 0.17618291079998016 |
| 32 | transport_dsa | anisotropic | heldout | UNet-defect2-w1 | 2.510e-01 | 2.787e-01 | 6.489e-06 | 5.863e-06 | 0.3595793843269348 | 0.15056923031806946 |
| 32 | transport_dsa | mixed | train | dsa1 | 9.638e-01 | 9.703e-01 | 5.084e-03 | 4.873e-03 |  |  |
| 32 | transport_dsa | mixed | train | dsa2 | 6.522e-01 | 6.433e-01 | 7.067e-04 | 2.976e-04 |  |  |
| 32 | transport_dsa | mixed | train | UNet-residual-w1 | 4.534e-01 | 4.587e-01 | 1.825e-03 | 2.578e-03 | 0.778205931186676 | 0.405650794506073 |
| 32 | transport_dsa | mixed | train | UNet-defect2-w0.5 | 4.471e-01 | 4.484e-01 | 8.375e-04 | 6.367e-04 | 0.5508406162261963 | 0.28524041175842285 |
| 32 | transport_dsa | mixed | train | UNet-defect2-w1 | 4.186e-01 | 4.237e-01 | 3.722e-04 | 1.879e-04 | 0.4756126403808594 | 0.23898349702358246 |
| 32 | transport_dsa | mixed | heldout | dsa1 | 8.548e-01 | 8.693e-01 | 4.921e-03 | 4.450e-03 |  |  |
| 32 | transport_dsa | mixed | heldout | dsa2 | 5.448e-01 | 5.734e-01 | 6.935e-04 | 2.681e-04 |  |  |
| 32 | transport_dsa | mixed | heldout | UNet-residual-w1 | 3.448e-01 | 3.494e-01 | 1.787e-03 | 2.403e-03 | 0.778205931186676 | 0.405650794506073 |
| 32 | transport_dsa | mixed | heldout | UNet-defect2-w0.5 | 3.453e-01 | 3.584e-01 | 7.698e-04 | 6.650e-04 | 0.5508406162261963 | 0.28524041175842285 |
| 32 | transport_dsa | mixed | heldout | UNet-defect2-w1 | 3.067e-01 | 3.177e-01 | 2.888e-04 | 1.604e-04 | 0.4756126403808594 | 0.23898349702358246 |
