# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['diffusive', 'heterogeneous', 'anisotropic', 'mixed']`
- `sizes`: `[24]`
- `n_angles`: `12`
- `base_kinds`: `['transport']`
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
- `output_csv`: `rte/npbs_unet_ablation_born_size24.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_BORN_SIZE24.md`

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 24 | transport | diffusive | train | transport1 | 8.606e-01 | 8.726e-01 | 1.462e-01 | 1.317e-05 |  |  |
| 24 | transport | diffusive | train | transport2 | 7.515e-01 | 7.698e-01 | 8.370e-02 | 7.313e-06 |  |  |
| 24 | transport | diffusive | train | UNet-residual-w1 | 4.053e-01 | 4.141e-01 | 1.020e-03 | 6.158e-06 | 0.6168208718299866 | 0.17734277248382568 |
| 24 | transport | diffusive | train | UNet-defect2-w0.5 | 4.147e-01 | 4.236e-01 | 1.788e-04 | 6.014e-06 | 0.5590050220489502 | 0.2484174370765686 |
| 24 | transport | diffusive | train | UNet-defect2-w1 | 5.196e-01 | 5.190e-01 | 2.414e-04 | 6.146e-06 | 0.5074710845947266 | 0.2642468512058258 |
| 24 | transport | diffusive | heldout | transport1 | 7.781e-01 | 7.884e-01 | 1.310e-01 | 1.390e-05 |  |  |
| 24 | transport | diffusive | heldout | transport2 | 6.787e-01 | 6.949e-01 | 7.501e-02 | 6.649e-06 |  |  |
| 24 | transport | diffusive | heldout | UNet-residual-w1 | 3.915e-01 | 3.902e-01 | 9.290e-04 | 6.424e-06 | 0.6168208718299866 | 0.17734277248382568 |
| 24 | transport | diffusive | heldout | UNet-defect2-w0.5 | 3.935e-01 | 3.982e-01 | 1.551e-04 | 5.312e-06 | 0.5590050220489502 | 0.2484174370765686 |
| 24 | transport | diffusive | heldout | UNet-defect2-w1 | 4.971e-01 | 4.982e-01 | 2.130e-04 | 5.347e-06 | 0.5074710845947266 | 0.2642468512058258 |
| 24 | transport | heterogeneous | train | transport1 | 4.620e-01 | 5.922e-01 | 4.741e-04 | 3.187e-06 |  |  |
| 24 | transport | heterogeneous | train | transport2 | 2.866e-01 | 4.270e-01 | 3.332e-06 | 3.124e-06 |  |  |
| 24 | transport | heterogeneous | train | UNet-residual-w1 | 3.832e-01 | 4.659e-01 | 3.027e-06 | 3.235e-06 | 0.47550129890441895 | 0.17024487257003784 |
| 24 | transport | heterogeneous | train | UNet-defect2-w0.5 | 2.992e-01 | 3.731e-01 | 3.046e-06 | 2.918e-06 | 0.38789331912994385 | 0.13782428205013275 |
| 24 | transport | heterogeneous | train | UNet-defect2-w1 | 2.329e-01 | 2.974e-01 | 3.235e-06 | 3.455e-06 | 0.3254413902759552 | 0.09566006064414978 |
| 24 | transport | heterogeneous | heldout | transport1 | 4.158e-01 | 4.990e-01 | 3.830e-04 | 2.448e-06 |  |  |
| 24 | transport | heterogeneous | heldout | transport2 | 2.294e-01 | 3.379e-01 | 2.538e-06 | 2.546e-06 |  |  |
| 24 | transport | heterogeneous | heldout | UNet-residual-w1 | 3.879e-01 | 3.876e-01 | 2.338e-06 | 2.392e-06 | 0.47550129890441895 | 0.17024487257003784 |
| 24 | transport | heterogeneous | heldout | UNet-defect2-w0.5 | 2.310e-01 | 2.710e-01 | 2.424e-06 | 2.392e-06 | 0.38789331912994385 | 0.13782428205013275 |
| 24 | transport | heterogeneous | heldout | UNet-defect2-w1 | 1.876e-01 | 2.140e-01 | 2.393e-06 | 2.439e-06 | 0.3254413902759552 | 0.09566006064414978 |
| 24 | transport | anisotropic | train | transport1 | 5.973e-01 | 7.168e-01 | 1.826e-02 | 2.039e-04 |  |  |
| 24 | transport | anisotropic | train | transport2 | 4.103e-01 | 5.378e-01 | 1.962e-03 | 3.808e-06 |  |  |
| 24 | transport | anisotropic | train | UNet-residual-w1 | 4.147e-01 | 4.800e-01 | 2.881e-04 | 5.127e-04 | 0.5132909417152405 | 0.17020034790039062 |
| 24 | transport | anisotropic | train | UNet-defect2-w0.5 | 3.470e-01 | 4.219e-01 | 5.311e-06 | 1.095e-05 | 0.40150126814842224 | 0.11144277453422546 |
| 24 | transport | anisotropic | train | UNet-defect2-w1 | 2.994e-01 | 3.690e-01 | 4.022e-06 | 5.297e-06 | 0.33011025190353394 | 0.07564349472522736 |
| 24 | transport | anisotropic | heldout | transport1 | 5.970e-01 | 7.156e-01 | 1.826e-02 | 2.075e-04 |  |  |
| 24 | transport | anisotropic | heldout | transport2 | 4.094e-01 | 5.355e-01 | 1.962e-03 | 3.792e-06 |  |  |
| 24 | transport | anisotropic | heldout | UNet-residual-w1 | 4.773e-01 | 5.364e-01 | 3.935e-04 | 5.742e-04 | 0.5132909417152405 | 0.17020034790039062 |
| 24 | transport | anisotropic | heldout | UNet-defect2-w0.5 | 3.766e-01 | 4.485e-01 | 4.017e-06 | 1.263e-05 | 0.40150126814842224 | 0.11144277453422546 |
| 24 | transport | anisotropic | heldout | UNet-defect2-w1 | 3.210e-01 | 3.890e-01 | 4.049e-06 | 4.271e-06 | 0.33011025190353394 | 0.07564349472522736 |
| 24 | transport | mixed | train | transport1 | 7.601e-01 | 7.824e-01 | 2.486e-02 | 4.349e-03 |  |  |
| 24 | transport | mixed | train | transport2 | 5.553e-01 | 5.735e-01 | 3.838e-03 | 5.368e-04 |  |  |
| 24 | transport | mixed | train | UNet-residual-w1 | 4.195e-01 | 4.154e-01 | 5.420e-03 | 6.605e-03 | 0.5115188956260681 | 0.28012433648109436 |
| 24 | transport | mixed | train | UNet-defect2-w0.5 | 3.493e-01 | 3.496e-01 | 2.532e-03 | 1.157e-03 | 0.3774542808532715 | 0.22177787125110626 |
| 24 | transport | mixed | train | UNet-defect2-w1 | 3.361e-01 | 3.258e-01 | 8.788e-04 | 3.113e-04 | 0.2941819131374359 | 0.1275530308485031 |
| 24 | transport | mixed | heldout | transport1 | 5.770e-01 | 6.042e-01 | 2.734e-02 | 2.215e-03 |  |  |
| 24 | transport | mixed | heldout | transport2 | 3.994e-01 | 4.360e-01 | 4.092e-03 | 2.583e-04 |  |  |
| 24 | transport | mixed | heldout | UNet-residual-w1 | 5.387e-01 | 5.214e-01 | 6.167e-03 | 7.013e-03 | 0.5115188956260681 | 0.28012433648109436 |
| 24 | transport | mixed | heldout | UNet-defect2-w0.5 | 3.945e-01 | 4.069e-01 | 2.508e-03 | 1.144e-03 | 0.3774542808532715 | 0.22177787125110626 |
| 24 | transport | mixed | heldout | UNet-defect2-w1 | 3.964e-01 | 4.028e-01 | 8.479e-04 | 3.944e-04 | 0.2941819131374359 | 0.1275530308485031 |
