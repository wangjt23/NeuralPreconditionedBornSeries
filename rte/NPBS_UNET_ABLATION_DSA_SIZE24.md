# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['diffusive', 'heterogeneous', 'anisotropic', 'mixed']`
- `sizes`: `[24]`
- `n_angles`: `12`
- `base_kinds`: `['transport_dsa']`
- `forms`: `['residual', 'defect2']`
- `defect_relaxations`: `[0.5, 1.0]`
- `eta_scale`: `0.1`
- `width`: `12`
- `levels`: `3`
- `steps`: `80`
- `lr`: `0.002`
- `maxiter`: `60`
- `fgmres_maxiter`: `25`
- `tol`: `1e-06`
- `seed`: `20260519`
- `output_csv`: `rte/npbs_unet_ablation_dsa_size24.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_DSA_SIZE24.md`

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 24 | transport_dsa | diffusive | train | dsa1 | 2.369e+00 | 2.399e+00 | 6.001e-06 | 5.993e-06 |  |  |
| 24 | transport_dsa | diffusive | train | dsa2 | 4.219e-01 | 4.251e-01 | 5.954e-06 | 6.065e-06 |  |  |
| 24 | transport_dsa | diffusive | train | UNet-residual-w1 | 4.770e-01 | 4.867e-01 | 5.934e-06 | 5.898e-06 | 0.3327040374279022 | 0.0388513021171093 |
| 24 | transport_dsa | diffusive | train | UNet-defect2-w0.5 | 2.808e-01 | 2.862e-01 | 6.143e-06 | 6.313e-06 | 0.11421135067939758 | 0.012991166673600674 |
| 24 | transport_dsa | diffusive | train | UNet-defect2-w1 | 9.105e-02 | 9.143e-02 | 6.332e-06 | 6.043e-06 | 0.012778420932590961 | 0.0014565425226464868 |
| 24 | transport_dsa | diffusive | heldout | dsa1 | 2.133e+00 | 2.159e+00 | 5.513e-06 | 5.492e-06 |  |  |
| 24 | transport_dsa | diffusive | heldout | dsa2 | 3.824e-01 | 3.850e-01 | 5.658e-06 | 5.548e-06 |  |  |
| 24 | transport_dsa | diffusive | heldout | UNet-residual-w1 | 4.334e-01 | 4.419e-01 | 5.434e-06 | 5.459e-06 | 0.3327040374279022 | 0.0388513021171093 |
| 24 | transport_dsa | diffusive | heldout | UNet-defect2-w0.5 | 2.544e-01 | 2.590e-01 | 5.269e-06 | 5.510e-06 | 0.11421135067939758 | 0.012991166673600674 |
| 24 | transport_dsa | diffusive | heldout | UNet-defect2-w1 | 8.677e-02 | 8.700e-02 | 5.448e-06 | 5.445e-06 | 0.012778420932590961 | 0.0014565425226464868 |
| 24 | transport_dsa | heterogeneous | train | dsa1 | 4.830e-01 | 5.230e-01 | 3.062e-06 | 3.226e-06 |  |  |
| 24 | transport_dsa | heterogeneous | train | dsa2 | 2.257e-01 | 2.856e-01 | 3.282e-06 | 3.320e-06 |  |  |
| 24 | transport_dsa | heterogeneous | train | UNet-residual-w1 | 3.379e-01 | 3.808e-01 | 3.180e-06 | 3.142e-06 | 0.20716334879398346 | 0.09433484077453613 |
| 24 | transport_dsa | heterogeneous | train | UNet-defect2-w0.5 | 2.440e-01 | 2.783e-01 | 2.888e-06 | 3.200e-06 | 0.10939515382051468 | 0.044352397322654724 |
| 24 | transport_dsa | heterogeneous | train | UNet-defect2-w1 | 1.814e-01 | 2.184e-01 | 3.128e-06 | 2.903e-06 | 0.052685659378767014 | 0.025625713169574738 |
| 24 | transport_dsa | heterogeneous | heldout | dsa1 | 3.681e-01 | 3.718e-01 | 2.265e-06 | 2.327e-06 |  |  |
| 24 | transport_dsa | heterogeneous | heldout | dsa2 | 1.511e-01 | 1.889e-01 | 2.407e-06 | 2.409e-06 |  |  |
| 24 | transport_dsa | heterogeneous | heldout | UNet-residual-w1 | 4.530e-01 | 4.402e-01 | 2.336e-06 | 2.551e-06 | 0.20716334879398346 | 0.09433484077453613 |
| 24 | transport_dsa | heterogeneous | heldout | UNet-defect2-w0.5 | 2.130e-01 | 2.305e-01 | 2.409e-06 | 2.434e-06 | 0.10939515382051468 | 0.044352397322654724 |
| 24 | transport_dsa | heterogeneous | heldout | UNet-defect2-w1 | 1.690e-01 | 1.964e-01 | 2.405e-06 | 2.337e-06 | 0.052685659378767014 | 0.025625713169574738 |
| 24 | transport_dsa | anisotropic | train | dsa1 | 6.777e-01 | 7.586e-01 | 3.394e-05 | 1.972e-05 |  |  |
| 24 | transport_dsa | anisotropic | train | dsa2 | 4.294e-01 | 5.232e-01 | 3.802e-06 | 4.091e-06 |  |  |
| 24 | transport_dsa | anisotropic | train | UNet-residual-w1 | 4.596e-01 | 5.488e-01 | 1.762e-02 | 2.129e-04 | 0.39039066433906555 | 0.167185977101326 |
| 24 | transport_dsa | anisotropic | train | UNet-defect2-w0.5 | 3.519e-01 | 4.129e-01 | 4.145e-06 | 3.850e-06 | 0.2599010169506073 | 0.08859845250844955 |
| 24 | transport_dsa | anisotropic | train | UNet-defect2-w1 | 3.137e-01 | 3.706e-01 | 3.874e-06 | 3.935e-06 | 0.19115625321865082 | 0.07437621802091599 |
| 24 | transport_dsa | anisotropic | heldout | dsa1 | 6.774e-01 | 7.572e-01 | 3.394e-05 | 2.080e-05 |  |  |
| 24 | transport_dsa | anisotropic | heldout | dsa2 | 4.283e-01 | 5.205e-01 | 3.893e-06 | 3.876e-06 |  |  |
| 24 | transport_dsa | anisotropic | heldout | UNet-residual-w1 | 4.917e-01 | 5.777e-01 | 1.566e-02 | 3.238e-04 | 0.39039066433906555 | 0.167185977101326 |
| 24 | transport_dsa | anisotropic | heldout | UNet-defect2-w0.5 | 3.758e-01 | 4.509e-01 | 4.020e-06 | 3.791e-06 | 0.2599010169506073 | 0.08859845250844955 |
| 24 | transport_dsa | anisotropic | heldout | UNet-defect2-w1 | 3.313e-01 | 3.936e-01 | 3.813e-06 | 3.914e-06 | 0.19115625321865082 | 0.07437621802091599 |
| 24 | transport_dsa | mixed | train | dsa1 | 9.621e-01 | 9.684e-01 | 5.104e-03 | 5.020e-03 |  |  |
| 24 | transport_dsa | mixed | train | dsa2 | 6.483e-01 | 6.391e-01 | 7.084e-04 | 2.936e-04 |  |  |
| 24 | transport_dsa | mixed | train | UNet-residual-w1 | 5.664e-01 | 5.400e-01 | 5.595e-03 | 4.457e-03 | 0.5097471475601196 | 0.17460604012012482 |
| 24 | transport_dsa | mixed | train | UNet-defect2-w0.5 | 4.735e-01 | 4.656e-01 | 1.921e-03 | 5.526e-04 | 0.3110898733139038 | 0.11341575533151627 |
| 24 | transport_dsa | mixed | train | UNet-defect2-w1 | 3.748e-01 | 3.713e-01 | 7.595e-04 | 2.737e-04 | 0.24571074545383453 | 0.08413466811180115 |
| 24 | transport_dsa | mixed | heldout | dsa1 | 8.546e-01 | 8.696e-01 | 5.074e-03 | 4.576e-03 |  |  |
| 24 | transport_dsa | mixed | heldout | dsa2 | 5.452e-01 | 5.742e-01 | 7.165e-04 | 2.755e-04 |  |  |
| 24 | transport_dsa | mixed | heldout | UNet-residual-w1 | 4.967e-01 | 5.062e-01 | 5.202e-03 | 4.140e-03 | 0.5097471475601196 | 0.17460604012012482 |
| 24 | transport_dsa | mixed | heldout | UNet-defect2-w0.5 | 4.435e-01 | 4.363e-01 | 1.964e-03 | 6.636e-04 | 0.3110898733139038 | 0.11341575533151627 |
| 24 | transport_dsa | mixed | heldout | UNet-defect2-w1 | 2.977e-01 | 3.127e-01 | 7.419e-04 | 2.917e-04 | 0.24571074545383453 | 0.08413466811180115 |
