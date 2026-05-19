# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['diffusive']`
- `sizes`: `[32]`
- `n_angles`: `12`
- `base_kinds`: `['transport_dsa']`
- `forms`: `['residual']`
- `defect_relaxations`: `[1.0]`
- `eta_scale`: `0.1`
- `width`: `16`
- `levels`: `3`
- `steps`: `1`
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
- `rollout_weight`: `0.0`
- `rollout_steps`: `1`
- `rollout_decay`: `1.0`
- `stability_weight`: `0.0`
- `contraction_target`: `0.98`
- `maxiter`: `60`
- `fgmres_maxiter`: `25`
- `tol`: `1e-06`
- `seed`: `20260519`
- `device`: `cuda:0`
- `allow_tf32`: `False`
- `output_csv`: `rte/npbs_unet_ablation_tf32check.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_TF32CHECK.md`

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 32 | transport_dsa | diffusive | train | dsa1 | 2.369e+00 | 2.399e+00 | 9.128e-06 | 8.688e-06 |  |  |
| 32 | transport_dsa | diffusive | train | dsa2 | 4.219e-01 | 4.251e-01 | 9.129e-06 | 8.984e-06 |  |  |
| 32 | transport_dsa | diffusive | train | UNet-residual-w1 | 2.365e+00 | 2.394e+00 | 9.999e-06 | 8.157e-06 | 0.6738430261611938 | 0.6738430261611938 |
| 32 | transport_dsa | diffusive | heldout | dsa1 | 2.133e+00 | 2.159e+00 | 7.735e-06 | 7.924e-06 |  |  |
| 32 | transport_dsa | diffusive | heldout | dsa2 | 3.824e-01 | 3.850e-01 | 7.851e-06 | 8.130e-06 |  |  |
| 32 | transport_dsa | diffusive | heldout | UNet-residual-w1 | 2.129e+00 | 2.155e+00 | 7.944e-06 | 7.927e-06 | 0.6738430261611938 | 0.6738430261611938 |
