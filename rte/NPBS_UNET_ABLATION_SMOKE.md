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
- `steps`: `4`
- `lr`: `0.002`
- `weight_decay`: `0.0`
- `scheduler`: `step`
- `scheduler_step_size`: `2`
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
- `rollout_steps`: `2`
- `rollout_decay`: `1.0`
- `stability_weight`: `0.2`
- `contraction_target`: `0.98`
- `maxiter`: `10`
- `fgmres_maxiter`: `8`
- `tol`: `1e-06`
- `seed`: `20260519`
- `device`: `cpu`
- `output_csv`: `rte/npbs_unet_ablation_smoke.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_SMOKE.md`

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 16 | transport_dsa | anisotropic | train | dsa1 | 6.972e-01 | 7.882e-01 | 1.005e-01 | 2.859e-02 |  |  |
| 16 | transport_dsa | anisotropic | train | dsa2 | 4.768e-01 | 5.928e-01 | 2.454e-02 | 4.024e-03 |  |  |
| 16 | transport_dsa | anisotropic | train | UNet-residual-w1 | 6.933e-01 | 7.848e-01 | 1.004e-01 | 2.881e-02 | 0.5016041994094849 | 0.5758128762245178 |
| 16 | transport_dsa | anisotropic | train | UNet-defect2-w1 | 4.735e-01 | 5.900e-01 | 2.443e-02 | 3.980e-03 | 0.2866957187652588 | 0.3425058126449585 |
| 16 | transport_dsa | anisotropic | heldout | dsa1 | 6.972e-01 | 7.867e-01 | 9.855e-02 | 2.976e-02 |  |  |
| 16 | transport_dsa | anisotropic | heldout | dsa2 | 4.754e-01 | 5.891e-01 | 2.424e-02 | 3.958e-03 |  |  |
| 16 | transport_dsa | anisotropic | heldout | UNet-residual-w1 | 6.936e-01 | 7.837e-01 | 9.835e-02 | 2.961e-02 | 0.5016041994094849 | 0.5758128762245178 |
| 16 | transport_dsa | anisotropic | heldout | UNet-defect2-w1 | 4.717e-01 | 5.858e-01 | 2.413e-02 | 3.911e-03 | 0.2866957187652588 | 0.3425058126449585 |
