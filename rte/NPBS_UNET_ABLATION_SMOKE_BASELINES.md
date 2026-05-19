# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['anisotropic']`
- `sizes`: `[16]`
- `n_angles`: `8`
- `base_kinds`: `['transport_tsa']`
- `tsa_harmonics`: `2`
- `forms`: `['defect2']`
- `defect_relaxations`: `[1.0]`
- `baseline_depths`: `[1, 2, 3]`
- `baseline_relaxations`: `[0.5, 1.0]`
- `eta_scale`: `0.1`
- `width`: `8`
- `levels`: `2`
- `steps`: `2`
- `lr`: `0.002`
- `weight_decay`: `0.0`
- `scheduler`: `none`
- `scheduler_step_size`: `0`
- `scheduler_gamma`: `0.5`
- `scheduler_eta_min`: `1e-05`
- `log_steps`: `0`
- `residual_mode`: `resample`
- `residuals_per_step`: `8`
- `rollout_samples_per_step`: `4`
- `n_rollout`: `2`
- `n_random`: `6`
- `random_modes`: `5`
- `aux_sources`: `1`
- `physical_weight`: `0.5`
- `correction_weight`: `1e-06`
- `rollout_weight`: `0.1`
- `rollout_steps`: `2`
- `rollout_decay`: `1.0`
- `stability_weight`: `0.1`
- `contraction_target`: `0.98`
- `maxiter`: `10`
- `fgmres_maxiter`: `5`
- `tol`: `1e-05`
- `seed`: `20260519`
- `device`: `cuda:0`
- `allow_tf32`: `False`
- `output_init`: `zero`
- `output_csv`: `rte/npbs_unet_ablation_smoke_baselines.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_SMOKE_BASELINES.md`

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | one-step base metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | transport_tsa | anisotropic | train | tsa1-w0.5 | 5.512e-01 | 6.181e-01 | 7.183e-01 | 5.524e-02 | 1.460e-02 |  |  |
| 16 | transport_tsa | anisotropic | train | tsa1 | 2.819e-01 | 3.613e-01 | 4.729e-01 | 6.801e-03 | 1.460e-02 |  |  |
| 16 | transport_tsa | anisotropic | train | tsa2-w0.5 | 3.408e-01 | 4.237e-01 | 5.431e-01 | 9.334e-03 | 6.555e-03 |  |  |
| 16 | transport_tsa | anisotropic | train | tsa2 | 1.719e-01 | 2.211e-01 | 2.992e-01 | 4.381e-04 | 1.782e-03 |  |  |
| 16 | transport_tsa | anisotropic | train | tsa3-w0.5 | 2.393e-01 | 3.155e-01 | 4.257e-01 | 2.055e-03 | 2.727e-03 |  |  |
| 16 | transport_tsa | anisotropic | train | tsa3 | 1.088e-01 | 1.417e-01 | 1.985e-01 | 5.365e-05 | 3.278e-04 |  |  |
| 16 | transport_tsa | anisotropic | train | UNet-defect2-w1 | 1.718e-01 | 2.210e-01 | 2.986e-01 | 4.390e-04 | 1.785e-03 | 0.13366003334522247 | 0.13376933336257935 |
| 16 | transport_tsa | anisotropic | heldout | tsa1-w0.5 | 5.492e-01 | 6.144e-01 | 7.180e-01 | 5.311e-02 | 1.363e-02 |  |  |
| 16 | transport_tsa | anisotropic | heldout | tsa1 | 2.778e-01 | 3.548e-01 | 4.735e-01 | 6.311e-03 | 1.363e-02 |  |  |
| 16 | transport_tsa | anisotropic | heldout | tsa2-w0.5 | 3.374e-01 | 4.183e-01 | 5.430e-01 | 8.770e-03 | 6.045e-03 |  |  |
| 16 | transport_tsa | anisotropic | heldout | tsa2 | 1.673e-01 | 2.149e-01 | 2.999e-01 | 3.885e-04 | 1.614e-03 |  |  |
| 16 | transport_tsa | anisotropic | heldout | tsa3-w0.5 | 2.352e-01 | 3.095e-01 | 4.258e-01 | 1.871e-03 | 2.514e-03 |  |  |
| 16 | transport_tsa | anisotropic | heldout | tsa3 | 1.051e-01 | 1.369e-01 | 1.990e-01 | 4.778e-05 | 3.053e-04 |  |  |
| 16 | transport_tsa | anisotropic | heldout | UNet-defect2-w1 | 1.672e-01 | 2.148e-01 | 2.993e-01 | 3.899e-04 | 1.619e-03 | 0.13366003334522247 | 0.13376933336257935 |
