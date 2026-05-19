# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['anisotropic', 'mixed']`
- `sizes`: `[32]`
- `n_angles`: `12`
- `base_kinds`: `['transport_dsa']`
- `forms`: `['defect2']`
- `defect_relaxations`: `[1.0]`
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
- `log_steps`: `0`
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
- `output_init`: `small`
- `output_csv`: `rte/npbs_unet_ablation_q7q8_small_steps300.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_Q7Q8_SMALL_STEPS300.md`

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | one-step base metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | transport_dsa | anisotropic | train | dsa1 | 6.777e-01 | 7.586e-01 | 1.129e-01 | 3.440e-05 | 2.020e-05 |  |  |
| 32 | transport_dsa | anisotropic | train | dsa2 | 4.294e-01 | 5.232e-01 | 8.233e-02 | 5.687e-06 | 5.939e-06 |  |  |
| 32 | transport_dsa | anisotropic | train | UNet-defect2-w1 | 3.078e-01 | 3.394e-01 | 1.728e-01 | 5.601e-06 | 6.229e-06 | 0.3594030737876892 | 0.154933899641037 |
| 32 | transport_dsa | anisotropic | heldout | dsa1 | 6.774e-01 | 7.572e-01 | 1.126e-01 | 3.430e-05 | 2.135e-05 |  |  |
| 32 | transport_dsa | anisotropic | heldout | dsa2 | 4.283e-01 | 5.205e-01 | 8.195e-02 | 5.951e-06 | 6.005e-06 |  |  |
| 32 | transport_dsa | anisotropic | heldout | UNet-defect2-w1 | 2.918e-01 | 3.187e-01 | 1.743e-01 | 6.040e-06 | 5.880e-06 | 0.3594030737876892 | 0.154933899641037 |
| 32 | transport_dsa | mixed | train | dsa1 | 9.638e-01 | 9.703e-01 | 1.701e-01 | 5.084e-03 | 4.873e-03 |  |  |
| 32 | transport_dsa | mixed | train | dsa2 | 6.522e-01 | 6.433e-01 | 1.524e-01 | 7.067e-04 | 2.976e-04 |  |  |
| 32 | transport_dsa | mixed | train | UNet-defect2-w1 | 3.874e-01 | 3.840e-01 | 1.561e-01 | 2.858e-04 | 1.681e-04 | 0.46982529759407043 | 0.2332673817873001 |
| 32 | transport_dsa | mixed | heldout | dsa1 | 8.548e-01 | 8.693e-01 | 1.728e-01 | 4.921e-03 | 4.450e-03 |  |  |
| 32 | transport_dsa | mixed | heldout | dsa2 | 5.448e-01 | 5.734e-01 | 1.151e-01 | 6.935e-04 | 2.681e-04 |  |  |
| 32 | transport_dsa | mixed | heldout | UNet-defect2-w1 | 2.739e-01 | 2.838e-01 | 1.474e-01 | 2.449e-04 | 1.396e-04 | 0.46982529759407043 | 0.2332673817873001 |
