# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['diffusive', 'heterogeneous', 'anisotropic', 'mixed']`
- `sizes`: `[32]`
- `n_angles`: `12`
- `base_kinds`: `['transport_dsa']`
- `forms`: `['residual', 'defect2']`
- `defect_relaxations`: `[1.0]`
- `eta_scale`: `0.1`
- `width`: `12`
- `levels`: `3`
- `steps`: `60`
- `lr`: `0.002`
- `maxiter`: `60`
- `fgmres_maxiter`: `25`
- `tol`: `1e-06`
- `seed`: `20260519`
- `output_csv`: `rte/npbs_unet_ablation_dsa_size32.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_DSA_SIZE32.md`

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 32 | transport_dsa | diffusive | train | dsa1 | 2.369e+00 | 2.399e+00 | 1.033e-05 | 8.436e-06 |  |  |
| 32 | transport_dsa | diffusive | train | dsa2 | 4.219e-01 | 4.251e-01 | 9.733e-06 | 8.980e-06 |  |  |
| 32 | transport_dsa | diffusive | train | UNet-residual-w1 | 1.135e+00 | 1.200e+00 | 8.972e-06 | 8.234e-06 | 0.33361175656318665 | 0.06839054077863693 |
| 32 | transport_dsa | diffusive | train | UNet-defect2-w1 | 1.031e-01 | 1.050e-01 | 1.023e-05 | 9.382e-06 | 0.012805855832993984 | 0.00154593912884593 |
| 32 | transport_dsa | diffusive | heldout | dsa1 | 2.133e+00 | 2.159e+00 | 7.849e-06 | 7.625e-06 |  |  |
| 32 | transport_dsa | diffusive | heldout | dsa2 | 3.824e-01 | 3.850e-01 | 8.228e-06 | 7.684e-06 |  |  |
| 32 | transport_dsa | diffusive | heldout | UNet-residual-w1 | 1.010e+00 | 1.064e+00 | 7.787e-06 | 7.843e-06 | 0.33361175656318665 | 0.06839054077863693 |
| 32 | transport_dsa | diffusive | heldout | UNet-defect2-w1 | 9.612e-02 | 9.803e-02 | 7.795e-06 | 7.524e-06 | 0.012805855832993984 | 0.00154593912884593 |
| 32 | transport_dsa | heterogeneous | train | dsa1 | 4.859e-01 | 5.279e-01 | 4.257e-06 | 4.620e-06 |  |  |
| 32 | transport_dsa | heterogeneous | train | dsa2 | 2.303e-01 | 2.907e-01 | 4.441e-06 | 4.517e-06 |  |  |
| 32 | transport_dsa | heterogeneous | train | UNet-residual-w1 | 4.257e-01 | 4.690e-01 | 4.710e-06 | 4.334e-06 | 0.20894108712673187 | 0.12450789660215378 |
| 32 | transport_dsa | heterogeneous | train | UNet-defect2-w1 | 2.150e-01 | 2.522e-01 | 4.302e-06 | 4.988e-06 | 0.05344218388199806 | 0.03153529018163681 |
| 32 | transport_dsa | heterogeneous | heldout | dsa1 | 3.723e-01 | 3.726e-01 | 3.743e-06 | 3.532e-06 |  |  |
| 32 | transport_dsa | heterogeneous | heldout | dsa2 | 1.503e-01 | 1.858e-01 | 3.733e-06 | 3.428e-06 |  |  |
| 32 | transport_dsa | heterogeneous | heldout | UNet-residual-w1 | 4.346e-01 | 4.296e-01 | 3.665e-06 | 3.448e-06 | 0.20894108712673187 | 0.12450789660215378 |
| 32 | transport_dsa | heterogeneous | heldout | UNet-defect2-w1 | 1.726e-01 | 1.934e-01 | 3.591e-06 | 3.658e-06 | 0.05344218388199806 | 0.03153529018163681 |
| 32 | transport_dsa | anisotropic | train | dsa1 | 6.777e-01 | 7.586e-01 | 3.414e-05 | 2.018e-05 |  |  |
| 32 | transport_dsa | anisotropic | train | dsa2 | 4.294e-01 | 5.232e-01 | 5.918e-06 | 5.505e-06 |  |  |
| 32 | transport_dsa | anisotropic | train | UNet-residual-w1 | 4.434e-01 | 5.319e-01 | 1.775e-05 | 6.697e-05 | 0.39027318358421326 | 0.2387682944536209 |
| 32 | transport_dsa | anisotropic | train | UNet-defect2-w1 | 3.192e-01 | 3.880e-01 | 5.796e-06 | 6.160e-06 | 0.19114117324352264 | 0.12720632553100586 |
| 32 | transport_dsa | anisotropic | heldout | dsa1 | 6.774e-01 | 7.572e-01 | 3.419e-05 | 2.166e-05 |  |  |
| 32 | transport_dsa | anisotropic | heldout | dsa2 | 4.283e-01 | 5.205e-01 | 5.517e-06 | 5.436e-06 |  |  |
| 32 | transport_dsa | anisotropic | heldout | UNet-residual-w1 | 4.857e-01 | 5.831e-01 | 2.938e-05 | 8.534e-05 | 0.39027318358421326 | 0.2387682944536209 |
| 32 | transport_dsa | anisotropic | heldout | UNet-defect2-w1 | 3.374e-01 | 4.131e-01 | 6.080e-06 | 6.044e-06 | 0.19114117324352264 | 0.12720632553100586 |
| 32 | transport_dsa | mixed | train | dsa1 | 9.638e-01 | 9.703e-01 | 5.084e-03 | 4.873e-03 |  |  |
| 32 | transport_dsa | mixed | train | dsa2 | 6.522e-01 | 6.433e-01 | 7.067e-04 | 2.974e-04 |  |  |
| 32 | transport_dsa | mixed | train | UNet-residual-w1 | 6.317e-01 | 6.386e-01 | 6.160e-03 | 6.143e-03 | 0.5100200772285461 | 0.20163850486278534 |
| 32 | transport_dsa | mixed | train | UNet-defect2-w1 | 3.782e-01 | 3.698e-01 | 8.293e-04 | 3.071e-04 | 0.24483679234981537 | 0.07783466577529907 |
| 32 | transport_dsa | mixed | heldout | dsa1 | 8.548e-01 | 8.693e-01 | 4.921e-03 | 4.450e-03 |  |  |
| 32 | transport_dsa | mixed | heldout | dsa2 | 5.448e-01 | 5.734e-01 | 6.935e-04 | 2.682e-04 |  |  |
| 32 | transport_dsa | mixed | heldout | UNet-residual-w1 | 6.455e-01 | 6.780e-01 | 5.462e-03 | 5.759e-03 | 0.5100200772285461 | 0.20163850486278534 |
| 32 | transport_dsa | mixed | heldout | UNet-defect2-w1 | 3.173e-01 | 3.191e-01 | 8.028e-04 | 3.328e-04 | 0.24483679234981537 | 0.07783466577529907 |
