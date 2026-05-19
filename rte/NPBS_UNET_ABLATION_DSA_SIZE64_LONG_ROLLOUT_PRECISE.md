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
- `allow_tf32`: `False`
- `output_csv`: `rte/npbs_unet_ablation_dsa_size64_long_rollout_precise.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_DSA_SIZE64_LONG_ROLLOUT_PRECISE.md`

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 64 | transport_dsa | diffusive | train | dsa1 | 2.369e+00 | 2.399e+00 | 2.297e-05 | 1.980e-05 |  |  |
| 64 | transport_dsa | diffusive | train | dsa2 | 4.219e-01 | 4.251e-01 | 1.790e-05 | 2.113e-05 |  |  |
| 64 | transport_dsa | diffusive | train | UNet-residual-w1 | 3.140e-01 | 3.080e-01 | 1.855e-05 | 1.753e-05 | 0.2144230455160141 | 0.03719763457775116 |
| 64 | transport_dsa | diffusive | train | UNet-defect2-w0.5 | 1.280e-01 | 1.259e-01 | 1.931e-05 | 2.110e-05 | 0.07239854335784912 | 0.005629262421280146 |
| 64 | transport_dsa | diffusive | train | UNet-defect2-w1 | 5.690e-02 | 5.581e-02 | 2.259e-05 | 1.773e-05 | 0.0094610545784235 | 0.0006162719801068306 |
| 64 | transport_dsa | diffusive | heldout | dsa1 | 2.133e+00 | 2.159e+00 | 1.751e-05 | 1.696e-05 |  |  |
| 64 | transport_dsa | diffusive | heldout | dsa2 | 3.824e-01 | 3.850e-01 | 1.696e-05 | 1.736e-05 |  |  |
| 64 | transport_dsa | diffusive | heldout | UNet-residual-w1 | 2.906e-01 | 2.794e-01 | 1.767e-05 | 1.658e-05 | 0.2144230455160141 | 0.03719763457775116 |
| 64 | transport_dsa | diffusive | heldout | UNet-defect2-w0.5 | 1.171e-01 | 1.148e-01 | 1.781e-05 | 1.863e-05 | 0.07239854335784912 | 0.005629262421280146 |
| 64 | transport_dsa | diffusive | heldout | UNet-defect2-w1 | 5.094e-02 | 4.962e-02 | 1.638e-05 | 1.773e-05 | 0.0094610545784235 | 0.0006162719801068306 |
| 64 | transport_dsa | heterogeneous | train | dsa1 | 4.876e-01 | 5.290e-01 | 9.972e-06 | 1.010e-05 |  |  |
| 64 | transport_dsa | heterogeneous | train | dsa2 | 2.307e-01 | 2.914e-01 | 1.013e-05 | 9.964e-06 |  |  |
| 64 | transport_dsa | heterogeneous | train | UNet-residual-w1 | 4.276e-01 | 4.847e-01 | 1.072e-05 | 9.469e-06 | 0.33156535029411316 | 0.18795764446258545 |
| 64 | transport_dsa | heterogeneous | train | UNet-defect2-w0.5 | 2.594e-01 | 2.958e-01 | 1.009e-05 | 1.071e-05 | 0.21113061904907227 | 0.09275723993778229 |
| 64 | transport_dsa | heterogeneous | train | UNet-defect2-w1 | 2.270e-01 | 2.635e-01 | 9.859e-06 | 1.123e-05 | 0.13658714294433594 | 0.05626258999109268 |
| 64 | transport_dsa | heterogeneous | heldout | dsa1 | 3.710e-01 | 3.739e-01 | 8.489e-06 | 8.140e-06 |  |  |
| 64 | transport_dsa | heterogeneous | heldout | dsa2 | 1.516e-01 | 1.891e-01 | 8.014e-06 | 8.221e-06 |  |  |
| 64 | transport_dsa | heterogeneous | heldout | UNet-residual-w1 | 3.718e-01 | 3.522e-01 | 8.170e-06 | 8.148e-06 | 0.33156535029411316 | 0.18795764446258545 |
| 64 | transport_dsa | heterogeneous | heldout | UNet-defect2-w0.5 | 2.242e-01 | 2.386e-01 | 7.585e-06 | 8.187e-06 | 0.21113061904907227 | 0.09275723993778229 |
| 64 | transport_dsa | heterogeneous | heldout | UNet-defect2-w1 | 1.943e-01 | 2.042e-01 | 8.104e-06 | 8.331e-06 | 0.13658714294433594 | 0.05626258999109268 |
| 64 | transport_dsa | anisotropic | train | dsa1 | 6.777e-01 | 7.586e-01 | 3.583e-05 | 2.326e-05 |  |  |
| 64 | transport_dsa | anisotropic | train | dsa2 | 4.294e-01 | 5.232e-01 | 1.236e-05 | 1.353e-05 |  |  |
| 64 | transport_dsa | anisotropic | train | UNet-residual-w1 | 3.627e-01 | 4.270e-01 | 1.238e-05 | 3.850e-05 | 0.5325866341590881 | 0.2748972475528717 |
| 64 | transport_dsa | anisotropic | train | UNet-defect2-w0.5 | 2.728e-01 | 3.115e-01 | 1.237e-05 | 1.279e-05 | 0.3816557824611664 | 0.17538008093833923 |
| 64 | transport_dsa | anisotropic | train | UNet-defect2-w1 | 3.133e-01 | 3.562e-01 | 1.234e-05 | 1.287e-05 | 0.29611366987228394 | 0.1544380784034729 |
| 64 | transport_dsa | anisotropic | heldout | dsa1 | 6.774e-01 | 7.572e-01 | 3.604e-05 | 2.482e-05 |  |  |
| 64 | transport_dsa | anisotropic | heldout | dsa2 | 4.283e-01 | 5.205e-01 | 1.296e-05 | 1.281e-05 |  |  |
| 64 | transport_dsa | anisotropic | heldout | UNet-residual-w1 | 3.156e-01 | 3.764e-01 | 1.423e-05 | 2.463e-05 | 0.5325866341590881 | 0.2748972475528717 |
| 64 | transport_dsa | anisotropic | heldout | UNet-defect2-w0.5 | 2.563e-01 | 2.881e-01 | 1.304e-05 | 1.380e-05 | 0.3816557824611664 | 0.17538008093833923 |
| 64 | transport_dsa | anisotropic | heldout | UNet-defect2-w1 | 2.922e-01 | 3.268e-01 | 1.316e-05 | 1.263e-05 | 0.29611366987228394 | 0.1544380784034729 |
| 64 | transport_dsa | mixed | train | dsa1 | 9.777e-01 | 9.837e-01 | 5.235e-03 | 5.288e-03 |  |  |
| 64 | transport_dsa | mixed | train | dsa2 | 6.613e-01 | 6.522e-01 | 7.563e-04 | 3.155e-04 |  |  |
| 64 | transport_dsa | mixed | train | UNet-residual-w1 | 4.861e-01 | 5.008e-01 | 3.942e-03 | 2.331e-03 | 0.6762080788612366 | 0.41694140434265137 |
| 64 | transport_dsa | mixed | train | UNet-defect2-w0.5 | 4.579e-01 | 4.527e-01 | 8.985e-04 | 6.088e-04 | 0.47482776641845703 | 0.26889270544052124 |
| 64 | transport_dsa | mixed | train | UNet-defect2-w1 | 3.957e-01 | 3.894e-01 | 5.664e-04 | 3.213e-04 | 0.40621647238731384 | 0.25483807921409607 |
| 64 | transport_dsa | mixed | heldout | dsa1 | 8.728e-01 | 8.874e-01 | 5.201e-03 | 4.745e-03 |  |  |
| 64 | transport_dsa | mixed | heldout | dsa2 | 5.548e-01 | 5.829e-01 | 7.608e-04 | 3.031e-04 |  |  |
| 64 | transport_dsa | mixed | heldout | UNet-residual-w1 | 3.581e-01 | 3.797e-01 | 3.715e-03 | 2.752e-03 | 0.6762080788612366 | 0.41694140434265137 |
| 64 | transport_dsa | mixed | heldout | UNet-defect2-w0.5 | 3.395e-01 | 3.603e-01 | 8.138e-04 | 6.979e-04 | 0.47482776641845703 | 0.26889270544052124 |
| 64 | transport_dsa | mixed | heldout | UNet-defect2-w1 | 3.301e-01 | 3.456e-01 | 5.090e-04 | 3.023e-04 | 0.40621647238731384 | 0.25483807921409607 |
