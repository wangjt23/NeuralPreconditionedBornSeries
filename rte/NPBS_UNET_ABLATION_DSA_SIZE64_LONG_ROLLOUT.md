# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['diffusive', 'heterogeneous', 'anisotropic', 'mixed']`
- `sizes`: `[64]`
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
- `residuals_per_step`: `64`
- `rollout_samples_per_step`: `32`
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
- `output_csv`: `rte/npbs_unet_ablation_dsa_size64_long_rollout.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_DSA_SIZE64_LONG_ROLLOUT.md`

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 64 | transport_dsa | diffusive | train | dsa1 | 2.369e+00 | 2.399e+00 | 2.297e-05 | 1.980e-05 |  |  |
| 64 | transport_dsa | diffusive | train | dsa2 | 4.219e-01 | 4.251e-01 | 1.790e-05 | 2.113e-05 |  |  |
| 64 | transport_dsa | diffusive | train | UNet-residual-w1 | 4.896e-01 | 4.915e-01 | 1.942e-05 | 1.828e-05 | 0.2144247144460678 | 0.04349265620112419 |
| 64 | transport_dsa | diffusive | train | UNet-defect2-w0.5 | 1.259e-01 | 1.243e-01 | 2.280e-05 | 1.757e-05 | 0.07240085303783417 | 0.005171294789761305 |
| 64 | transport_dsa | diffusive | train | UNet-defect2-w1 | 4.945e-02 | 4.802e-02 | 1.820e-05 | 2.076e-05 | 0.009461297653615475 | 0.000568718125578016 |
| 64 | transport_dsa | diffusive | heldout | dsa1 | 2.133e+00 | 2.159e+00 | 1.751e-05 | 1.696e-05 |  |  |
| 64 | transport_dsa | diffusive | heldout | dsa2 | 3.824e-01 | 3.850e-01 | 1.696e-05 | 1.736e-05 |  |  |
| 64 | transport_dsa | diffusive | heldout | UNet-residual-w1 | 4.420e-01 | 4.419e-01 | 1.702e-05 | 1.645e-05 | 0.2144247144460678 | 0.04349265620112419 |
| 64 | transport_dsa | diffusive | heldout | UNet-defect2-w0.5 | 1.150e-01 | 1.135e-01 | 1.759e-05 | 1.788e-05 | 0.07240085303783417 | 0.005171294789761305 |
| 64 | transport_dsa | diffusive | heldout | UNet-defect2-w1 | 4.459e-02 | 4.275e-02 | 1.880e-05 | 1.758e-05 | 0.009461297653615475 | 0.000568718125578016 |
| 64 | transport_dsa | heterogeneous | train | dsa1 | 4.876e-01 | 5.290e-01 | 9.972e-06 | 1.010e-05 |  |  |
| 64 | transport_dsa | heterogeneous | train | dsa2 | 2.307e-01 | 2.914e-01 | 1.013e-05 | 9.964e-06 |  |  |
| 64 | transport_dsa | heterogeneous | train | UNet-residual-w1 | 4.381e-01 | 4.938e-01 | 9.443e-06 | 9.846e-06 | 0.33156538009643555 | 0.18781059980392456 |
| 64 | transport_dsa | heterogeneous | train | UNet-defect2-w0.5 | 2.612e-01 | 2.977e-01 | 1.031e-05 | 1.030e-05 | 0.21113058924674988 | 0.09323839098215103 |
| 64 | transport_dsa | heterogeneous | train | UNet-defect2-w1 | 2.264e-01 | 2.602e-01 | 1.004e-05 | 1.025e-05 | 0.13658709824085236 | 0.0543576180934906 |
| 64 | transport_dsa | heterogeneous | heldout | dsa1 | 3.710e-01 | 3.739e-01 | 8.489e-06 | 8.140e-06 |  |  |
| 64 | transport_dsa | heterogeneous | heldout | dsa2 | 1.516e-01 | 1.891e-01 | 8.014e-06 | 8.221e-06 |  |  |
| 64 | transport_dsa | heterogeneous | heldout | UNet-residual-w1 | 3.572e-01 | 3.429e-01 | 8.284e-06 | 7.961e-06 | 0.33156538009643555 | 0.18781059980392456 |
| 64 | transport_dsa | heterogeneous | heldout | UNet-defect2-w0.5 | 2.228e-01 | 2.385e-01 | 7.975e-06 | 8.357e-06 | 0.21113058924674988 | 0.09323839098215103 |
| 64 | transport_dsa | heterogeneous | heldout | UNet-defect2-w1 | 1.926e-01 | 2.029e-01 | 7.994e-06 | 7.988e-06 | 0.13658709824085236 | 0.0543576180934906 |
| 64 | transport_dsa | anisotropic | train | dsa1 | 6.777e-01 | 7.586e-01 | 3.583e-05 | 2.326e-05 |  |  |
| 64 | transport_dsa | anisotropic | train | dsa2 | 4.294e-01 | 5.232e-01 | 1.236e-05 | 1.353e-05 |  |  |
| 64 | transport_dsa | anisotropic | train | UNet-residual-w1 | 3.805e-01 | 4.464e-01 | 1.410e-05 | 9.791e-05 | 0.5321900248527527 | 0.27416521310806274 |
| 64 | transport_dsa | anisotropic | train | UNet-defect2-w0.5 | 2.603e-01 | 3.009e-01 | 1.313e-05 | 1.281e-05 | 0.3812812864780426 | 0.17357540130615234 |
| 64 | transport_dsa | anisotropic | train | UNet-defect2-w1 | 3.144e-01 | 3.571e-01 | 1.243e-05 | 1.264e-05 | 0.2957068085670471 | 0.1530810296535492 |
| 64 | transport_dsa | anisotropic | heldout | dsa1 | 6.774e-01 | 7.572e-01 | 3.604e-05 | 2.482e-05 |  |  |
| 64 | transport_dsa | anisotropic | heldout | dsa2 | 4.283e-01 | 5.205e-01 | 1.296e-05 | 1.281e-05 |  |  |
| 64 | transport_dsa | anisotropic | heldout | UNet-residual-w1 | 3.341e-01 | 3.898e-01 | 1.360e-05 | 4.743e-05 | 0.5321900248527527 | 0.27416521310806274 |
| 64 | transport_dsa | anisotropic | heldout | UNet-defect2-w0.5 | 2.419e-01 | 2.724e-01 | 1.244e-05 | 1.309e-05 | 0.3812812864780426 | 0.17357540130615234 |
| 64 | transport_dsa | anisotropic | heldout | UNet-defect2-w1 | 2.943e-01 | 3.282e-01 | 1.316e-05 | 1.268e-05 | 0.2957068085670471 | 0.1530810296535492 |
| 64 | transport_dsa | mixed | train | dsa1 | 9.777e-01 | 9.837e-01 | 5.235e-03 | 5.288e-03 |  |  |
| 64 | transport_dsa | mixed | train | dsa2 | 6.613e-01 | 6.522e-01 | 7.563e-04 | 3.155e-04 |  |  |
| 64 | transport_dsa | mixed | train | UNet-residual-w1 | 4.835e-01 | 4.976e-01 | 3.602e-03 | 2.154e-03 | 0.6761741638183594 | 0.4155771732330322 |
| 64 | transport_dsa | mixed | train | UNet-defect2-w0.5 | 4.512e-01 | 4.476e-01 | 8.734e-04 | 5.912e-04 | 0.47480911016464233 | 0.26657286286354065 |
| 64 | transport_dsa | mixed | train | UNet-defect2-w1 | 4.472e-01 | 4.576e-01 | 5.821e-04 | 3.093e-04 | 0.40619078278541565 | 0.25406599044799805 |
| 64 | transport_dsa | mixed | heldout | dsa1 | 8.728e-01 | 8.874e-01 | 5.201e-03 | 4.745e-03 |  |  |
| 64 | transport_dsa | mixed | heldout | dsa2 | 5.548e-01 | 5.829e-01 | 7.608e-04 | 3.031e-04 |  |  |
| 64 | transport_dsa | mixed | heldout | UNet-residual-w1 | 3.549e-01 | 3.768e-01 | 3.412e-03 | 2.543e-03 | 0.6761741638183594 | 0.4155771732330322 |
| 64 | transport_dsa | mixed | heldout | UNet-defect2-w0.5 | 3.396e-01 | 3.623e-01 | 7.945e-04 | 6.949e-04 | 0.47480911016464233 | 0.26657286286354065 |
| 64 | transport_dsa | mixed | heldout | UNet-defect2-w1 | 3.557e-01 | 3.741e-01 | 5.234e-04 | 2.831e-04 | 0.40619078278541565 | 0.25406599044799805 |
