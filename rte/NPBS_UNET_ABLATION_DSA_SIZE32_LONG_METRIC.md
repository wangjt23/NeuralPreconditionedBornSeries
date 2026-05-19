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
- `output_csv`: `rte/npbs_unet_ablation_dsa_size32_long_metric.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_DSA_SIZE32_LONG_METRIC.md`

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 32 | transport_dsa | diffusive | train | dsa1 | 2.369e+00 | 2.399e+00 | 2.319e-03 | 2.362e-03 |  |  |
| 32 | transport_dsa | diffusive | train | dsa2 | 4.222e-01 | 4.255e-01 | 2.623e-03 | 2.086e-03 |  |  |
| 32 | transport_dsa | diffusive | train | UNet-residual-w1 | 1.875e-01 | 1.793e-01 | 2.178e-03 | 2.225e-03 | 0.23650158941745758 | 0.01280039269477129 |
| 32 | transport_dsa | diffusive | train | UNet-defect2-w0.5 | 1.107e-01 | 1.064e-01 | 1.656e-03 | 1.966e-03 | 0.08100391179323196 | 0.004481464624404907 |
| 32 | transport_dsa | diffusive | train | UNet-defect2-w1 | 5.483e-02 | 5.298e-02 | 2.346e-03 | 2.218e-03 | 0.009403194300830364 | 0.000606189074460417 |
| 32 | transport_dsa | diffusive | heldout | dsa1 | 2.133e+00 | 2.159e+00 | 2.213e-03 | 1.636e-03 |  |  |
| 32 | transport_dsa | diffusive | heldout | dsa2 | 3.826e-01 | 3.851e-01 | 2.126e-03 | 2.153e-03 |  |  |
| 32 | transport_dsa | diffusive | heldout | UNet-residual-w1 | 1.714e-01 | 1.631e-01 | 1.886e-03 | 1.502e-03 | 0.23650158941745758 | 0.01280039269477129 |
| 32 | transport_dsa | diffusive | heldout | UNet-defect2-w0.5 | 1.014e-01 | 9.700e-02 | 1.883e-03 | 2.464e-03 | 0.08100391179323196 | 0.004481464624404907 |
| 32 | transport_dsa | diffusive | heldout | UNet-defect2-w1 | 4.957e-02 | 4.809e-02 | 2.052e-03 | 1.741e-03 | 0.009403194300830364 | 0.000606189074460417 |
| 32 | transport_dsa | heterogeneous | train | dsa1 | 4.859e-01 | 5.279e-01 | 3.502e-05 | 5.902e-05 |  |  |
| 32 | transport_dsa | heterogeneous | train | dsa2 | 2.303e-01 | 2.907e-01 | 2.718e-05 | 5.125e-05 |  |  |
| 32 | transport_dsa | heterogeneous | train | UNet-residual-w1 | 3.706e-01 | 4.260e-01 | 4.487e-05 | 4.721e-05 | 0.2522982954978943 | 0.0893610492348671 |
| 32 | transport_dsa | heterogeneous | train | UNet-defect2-w0.5 | 2.120e-01 | 2.404e-01 | 2.959e-05 | 5.053e-05 | 0.15123434364795685 | 0.03122648037970066 |
| 32 | transport_dsa | heterogeneous | train | UNet-defect2-w1 | 1.862e-01 | 2.164e-01 | 2.341e-05 | 4.287e-05 | 0.08957258611917496 | 0.020981015637516975 |
| 32 | transport_dsa | heterogeneous | heldout | dsa1 | 3.723e-01 | 3.726e-01 | 3.790e-05 | 4.738e-05 |  |  |
| 32 | transport_dsa | heterogeneous | heldout | dsa2 | 1.503e-01 | 1.858e-01 | 4.735e-05 | 4.544e-05 |  |  |
| 32 | transport_dsa | heterogeneous | heldout | UNet-residual-w1 | 3.729e-01 | 3.298e-01 | 3.886e-05 | 5.001e-05 | 0.2522982954978943 | 0.0893610492348671 |
| 32 | transport_dsa | heterogeneous | heldout | UNet-defect2-w0.5 | 1.751e-01 | 1.725e-01 | 4.000e-05 | 6.049e-05 | 0.15123434364795685 | 0.03122648037970066 |
| 32 | transport_dsa | heterogeneous | heldout | UNet-defect2-w1 | 1.567e-01 | 1.549e-01 | 4.352e-05 | 5.403e-05 | 0.08957258611917496 | 0.020981015637516975 |
| 32 | transport_dsa | anisotropic | train | dsa1 | 6.774e-01 | 7.583e-01 | 9.418e-04 | 9.359e-04 |  |  |
| 32 | transport_dsa | anisotropic | train | dsa2 | 4.291e-01 | 5.228e-01 | 7.248e-04 | 7.945e-04 |  |  |
| 32 | transport_dsa | anisotropic | train | UNet-residual-w1 | 3.406e-01 | 3.991e-01 | 9.932e-04 | 9.131e-04 | 0.4157106280326843 | 0.11944670975208282 |
| 32 | transport_dsa | anisotropic | train | UNet-defect2-w0.5 | 2.763e-01 | 3.282e-01 | 9.107e-04 | 8.822e-04 | 0.28529730439186096 | 0.082647904753685 |
| 32 | transport_dsa | anisotropic | train | UNet-defect2-w1 | 2.744e-01 | 3.244e-01 | 8.598e-04 | 8.149e-04 | 0.21372711658477783 | 0.057136598974466324 |
| 32 | transport_dsa | anisotropic | heldout | dsa1 | 6.771e-01 | 7.569e-01 | 1.077e-03 | 9.333e-04 |  |  |
| 32 | transport_dsa | anisotropic | heldout | dsa2 | 4.280e-01 | 5.201e-01 | 1.069e-03 | 8.183e-04 |  |  |
| 32 | transport_dsa | anisotropic | heldout | UNet-residual-w1 | 2.888e-01 | 3.334e-01 | 1.119e-03 | 8.628e-04 | 0.4157106280326843 | 0.11944670975208282 |
| 32 | transport_dsa | anisotropic | heldout | UNet-defect2-w0.5 | 2.463e-01 | 2.917e-01 | 1.078e-03 | 8.053e-04 | 0.28529730439186096 | 0.082647904753685 |
| 32 | transport_dsa | anisotropic | heldout | UNet-defect2-w1 | 2.369e-01 | 2.752e-01 | 8.107e-04 | 7.987e-04 | 0.21372711658477783 | 0.057136598974466324 |
| 32 | transport_dsa | mixed | train | dsa1 | 9.638e-01 | 9.703e-01 | 5.135e-03 | 4.929e-03 |  |  |
| 32 | transport_dsa | mixed | train | dsa2 | 6.522e-01 | 6.433e-01 | 1.040e-03 | 8.054e-04 |  |  |
| 32 | transport_dsa | mixed | train | UNet-residual-w1 | 4.342e-01 | 4.314e-01 | 4.534e-03 | 2.434e-03 | 0.5201749205589294 | 0.18931058049201965 |
| 32 | transport_dsa | mixed | train | UNet-defect2-w0.5 | 4.176e-01 | 4.287e-01 | 1.537e-03 | 8.566e-04 | 0.34515783190727234 | 0.14095017313957214 |
| 32 | transport_dsa | mixed | train | UNet-defect2-w1 | 3.548e-01 | 3.571e-01 | 8.898e-04 | 7.005e-04 | 0.2910665273666382 | 0.11561612039804459 |
| 32 | transport_dsa | mixed | heldout | dsa1 | 8.547e-01 | 8.691e-01 | 4.990e-03 | 4.553e-03 |  |  |
| 32 | transport_dsa | mixed | heldout | dsa2 | 5.445e-01 | 5.730e-01 | 1.265e-03 | 8.291e-04 |  |  |
| 32 | transport_dsa | mixed | heldout | UNet-residual-w1 | 3.016e-01 | 3.006e-01 | 4.282e-03 | 2.787e-03 | 0.5201749205589294 | 0.18931058049201965 |
| 32 | transport_dsa | mixed | heldout | UNet-defect2-w0.5 | 2.694e-01 | 2.812e-01 | 1.522e-03 | 9.482e-04 | 0.34515783190727234 | 0.14095017313957214 |
| 32 | transport_dsa | mixed | heldout | UNet-defect2-w1 | 2.290e-01 | 2.347e-01 | 1.155e-03 | 7.558e-04 | 0.2910665273666382 | 0.11561612039804459 |
