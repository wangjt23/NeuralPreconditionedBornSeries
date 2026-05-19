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
- `output_csv`: `rte/npbs_unet_ablation_dsa_size32_long_rollout.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_DSA_SIZE32_LONG_ROLLOUT.md`

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 32 | transport_dsa | diffusive | train | dsa1 | 2.369e+00 | 2.399e+00 | 2.319e-03 | 2.362e-03 |  |  |
| 32 | transport_dsa | diffusive | train | dsa2 | 4.222e-01 | 4.255e-01 | 2.623e-03 | 2.086e-03 |  |  |
| 32 | transport_dsa | diffusive | train | UNet-residual-w1 | 1.840e-01 | 1.761e-01 | 1.910e-03 | 2.160e-03 | 0.26321470737457275 | 0.017156420275568962 |
| 32 | transport_dsa | diffusive | train | UNet-defect2-w0.5 | 1.136e-01 | 1.084e-01 | 2.328e-03 | 1.971e-03 | 0.08926121890544891 | 0.006939346436411142 |
| 32 | transport_dsa | diffusive | train | UNet-defect2-w1 | 4.648e-02 | 4.446e-02 | 1.715e-03 | 1.921e-03 | 0.011083461344242096 | 0.0007058645715005696 |
| 32 | transport_dsa | diffusive | heldout | dsa1 | 2.133e+00 | 2.159e+00 | 2.213e-03 | 1.636e-03 |  |  |
| 32 | transport_dsa | diffusive | heldout | dsa2 | 3.826e-01 | 3.851e-01 | 2.126e-03 | 2.153e-03 |  |  |
| 32 | transport_dsa | diffusive | heldout | UNet-residual-w1 | 1.672e-01 | 1.598e-01 | 2.087e-03 | 1.895e-03 | 0.26321470737457275 | 0.017156420275568962 |
| 32 | transport_dsa | diffusive | heldout | UNet-defect2-w0.5 | 1.033e-01 | 9.804e-02 | 1.816e-03 | 2.060e-03 | 0.08926121890544891 | 0.006939346436411142 |
| 32 | transport_dsa | diffusive | heldout | UNet-defect2-w1 | 4.266e-02 | 4.061e-02 | 2.198e-03 | 2.004e-03 | 0.011083461344242096 | 0.0007058645715005696 |
| 32 | transport_dsa | heterogeneous | train | dsa1 | 4.859e-01 | 5.279e-01 | 3.502e-05 | 5.902e-05 |  |  |
| 32 | transport_dsa | heterogeneous | train | dsa2 | 2.303e-01 | 2.907e-01 | 2.718e-05 | 5.125e-05 |  |  |
| 32 | transport_dsa | heterogeneous | train | UNet-residual-w1 | 3.459e-01 | 3.901e-01 | 4.535e-05 | 4.549e-05 | 0.39081305265426636 | 0.16978242993354797 |
| 32 | transport_dsa | heterogeneous | train | UNet-defect2-w0.5 | 2.258e-01 | 2.458e-01 | 3.896e-05 | 4.478e-05 | 0.2521626949310303 | 0.08081719279289246 |
| 32 | transport_dsa | heterogeneous | train | UNet-defect2-w1 | 2.152e-01 | 2.469e-01 | 2.542e-05 | 4.662e-05 | 0.1635380983352661 | 0.05440843850374222 |
| 32 | transport_dsa | heterogeneous | heldout | dsa1 | 3.723e-01 | 3.726e-01 | 3.790e-05 | 4.738e-05 |  |  |
| 32 | transport_dsa | heterogeneous | heldout | dsa2 | 1.503e-01 | 1.858e-01 | 4.735e-05 | 4.544e-05 |  |  |
| 32 | transport_dsa | heterogeneous | heldout | UNet-residual-w1 | 3.409e-01 | 3.060e-01 | 4.848e-05 | 5.640e-05 | 0.39081305265426636 | 0.16978242993354797 |
| 32 | transport_dsa | heterogeneous | heldout | UNet-defect2-w0.5 | 1.953e-01 | 2.002e-01 | 4.659e-05 | 5.691e-05 | 0.2521626949310303 | 0.08081719279289246 |
| 32 | transport_dsa | heterogeneous | heldout | UNet-defect2-w1 | 1.860e-01 | 1.907e-01 | 4.627e-05 | 4.422e-05 | 0.1635380983352661 | 0.05440843850374222 |
| 32 | transport_dsa | anisotropic | train | dsa1 | 6.774e-01 | 7.583e-01 | 9.418e-04 | 9.359e-04 |  |  |
| 32 | transport_dsa | anisotropic | train | dsa2 | 4.291e-01 | 5.228e-01 | 7.248e-04 | 7.945e-04 |  |  |
| 32 | transport_dsa | anisotropic | train | UNet-residual-w1 | 3.313e-01 | 3.862e-01 | 1.116e-03 | 8.619e-04 | 0.6273083686828613 | 0.2545050382614136 |
| 32 | transport_dsa | anisotropic | train | UNet-defect2-w0.5 | 2.679e-01 | 2.935e-01 | 9.694e-04 | 8.138e-04 | 0.4586339592933655 | 0.16657501459121704 |
| 32 | transport_dsa | anisotropic | train | UNet-defect2-w1 | 2.842e-01 | 3.282e-01 | 9.811e-04 | 7.938e-04 | 0.35902607440948486 | 0.1491403728723526 |
| 32 | transport_dsa | anisotropic | heldout | dsa1 | 6.771e-01 | 7.569e-01 | 1.077e-03 | 9.333e-04 |  |  |
| 32 | transport_dsa | anisotropic | heldout | dsa2 | 4.280e-01 | 5.201e-01 | 1.069e-03 | 8.183e-04 |  |  |
| 32 | transport_dsa | anisotropic | heldout | UNet-residual-w1 | 2.921e-01 | 3.465e-01 | 1.014e-03 | 7.658e-04 | 0.6273083686828613 | 0.2545050382614136 |
| 32 | transport_dsa | anisotropic | heldout | UNet-defect2-w0.5 | 2.364e-01 | 2.529e-01 | 8.474e-04 | 7.927e-04 | 0.4586339592933655 | 0.16657501459121704 |
| 32 | transport_dsa | anisotropic | heldout | UNet-defect2-w1 | 2.516e-01 | 2.775e-01 | 1.060e-03 | 7.544e-04 | 0.35902607440948486 | 0.1491403728723526 |
| 32 | transport_dsa | mixed | train | dsa1 | 9.638e-01 | 9.703e-01 | 5.135e-03 | 4.929e-03 |  |  |
| 32 | transport_dsa | mixed | train | dsa2 | 6.522e-01 | 6.433e-01 | 1.040e-03 | 8.054e-04 |  |  |
| 32 | transport_dsa | mixed | train | UNet-residual-w1 | 4.639e-01 | 4.726e-01 | 2.083e-03 | 2.678e-03 | 0.778164267539978 | 0.4051649570465088 |
| 32 | transport_dsa | mixed | train | UNet-defect2-w0.5 | 4.443e-01 | 4.425e-01 | 9.951e-04 | 8.076e-04 | 0.55082768201828 | 0.27881160378456116 |
| 32 | transport_dsa | mixed | train | UNet-defect2-w1 | 4.056e-01 | 4.043e-01 | 6.474e-04 | 7.579e-04 | 0.4755999445915222 | 0.23955699801445007 |
| 32 | transport_dsa | mixed | heldout | dsa1 | 8.547e-01 | 8.691e-01 | 4.990e-03 | 4.553e-03 |  |  |
| 32 | transport_dsa | mixed | heldout | dsa2 | 5.445e-01 | 5.730e-01 | 1.265e-03 | 8.291e-04 |  |  |
| 32 | transport_dsa | mixed | heldout | UNet-residual-w1 | 3.677e-01 | 3.808e-01 | 2.271e-03 | 2.668e-03 | 0.778164267539978 | 0.4051649570465088 |
| 32 | transport_dsa | mixed | heldout | UNet-defect2-w0.5 | 3.550e-01 | 3.707e-01 | 1.086e-03 | 9.466e-04 | 0.55082768201828 | 0.27881160378456116 |
| 32 | transport_dsa | mixed | heldout | UNet-defect2-w1 | 3.318e-01 | 3.518e-01 | 1.010e-03 | 7.231e-04 | 0.4755999445915222 | 0.23955699801445007 |
