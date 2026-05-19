# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['anisotropic']`
- `sizes`: `[16]`
- `n_angles`: `8`
- `base_kinds`: `['transport_dsa']`
- `forms`: `['residual', 'defect2']`
- `defect_relaxations`: `[1.0]`
- `eta_scale`: `0.1`
- `width`: `16`
- `levels`: `3`
- `steps`: `2`
- `lr`: `0.002`
- `weight_decay`: `0.0`
- `scheduler`: `step`
- `scheduler_step_size`: `1`
- `scheduler_gamma`: `0.5`
- `scheduler_eta_min`: `1e-05`
- `log_steps`: `0`
- `residual_mode`: `resample`
- `residuals_per_step`: `8`
- `rollout_samples_per_step`: `4`
- `n_rollout`: `3`
- `n_random`: `4`
- `random_modes`: `5`
- `aux_sources`: `1`
- `physical_weight`: `0.5`
- `correction_weight`: `1e-06`
- `rollout_weight`: `0.2`
- `rollout_steps`: `1`
- `rollout_decay`: `1.0`
- `stability_weight`: `0.2`
- `contraction_target`: `0.98`
- `maxiter`: `10`
- `fgmres_maxiter`: `8`
- `tol`: `1e-06`
- `seed`: `20260519`
- `device`: `cpu`
- `allow_tf32`: `False`
- `output_init`: `zero`
- `output_csv`: `rte/npbs_unet_ablation_smoke_fixes.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_SMOKE_FIXES.md`

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | one-step base metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | transport_dsa | anisotropic | train | dsa1 | 6.972e-01 | 7.882e-01 | 1.176e-01 | 1.005e-01 | 2.859e-02 |  |  |
| 16 | transport_dsa | anisotropic | train | dsa2 | 4.768e-01 | 5.928e-01 | 9.101e-02 | 2.454e-02 | 4.024e-03 |  |  |
| 16 | transport_dsa | anisotropic | train | UNet-residual-w1 | 6.960e-01 | 7.871e-01 | 1.175e-01 | 1.004e-01 | 2.855e-02 | 0.4090668559074402 | 0.4751761257648468 |
| 16 | transport_dsa | anisotropic | train | UNet-defect2-w1 | 4.762e-01 | 5.924e-01 | 9.092e-02 | 2.459e-02 | 4.008e-03 | 0.26251426339149475 | 0.27417242527008057 |
| 16 | transport_dsa | anisotropic | heldout | dsa1 | 6.972e-01 | 7.867e-01 | 1.170e-01 | 9.855e-02 | 2.976e-02 |  |  |
| 16 | transport_dsa | anisotropic | heldout | dsa2 | 4.754e-01 | 5.891e-01 | 9.035e-02 | 2.424e-02 | 3.958e-03 |  |  |
| 16 | transport_dsa | anisotropic | heldout | UNet-residual-w1 | 6.960e-01 | 7.857e-01 | 1.169e-01 | 9.845e-02 | 2.967e-02 | 0.4090668559074402 | 0.4751761257648468 |
| 16 | transport_dsa | anisotropic | heldout | UNet-defect2-w1 | 4.747e-01 | 5.887e-01 | 9.025e-02 | 2.430e-02 | 3.945e-03 | 0.26251426339149475 | 0.27417242527008057 |
