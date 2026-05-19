# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['anisotropic', 'mixed']`
- `sizes`: `[64]`
- `n_angles`: `24`
- `base_kinds`: `['transport_dsa']`
- `tsa_harmonics`: `None`
- `forms`: `['defect2']`
- `defect_relaxations`: `[1.0]`
- `baseline_depths`: `[2, 4]`
- `baseline_relaxations`: `[1.0]`
- `eta_scale`: `0.1`
- `width`: `24`
- `levels`: `4`
- `steps`: `600`
- `lr`: `0.002`
- `weight_decay`: `0.0001`
- `scheduler`: `step`
- `scheduler_step_size`: `200`
- `scheduler_gamma`: `0.5`
- `scheduler_eta_min`: `1e-05`
- `log_steps`: `0`
- `residual_mode`: `resample`
- `residuals_per_step`: `96`
- `rollout_samples_per_step`: `48`
- `n_rollout`: `12`
- `n_random`: `6`
- `random_modes`: `5`
- `aux_sources`: `4`
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
- `output_csv`: `rte/npbs_unet_ablation_dsa_size64_heavy_steps600.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_DSA_SIZE64_HEAVY_STEPS600.md`

## Terminology

- `transport_dsa` is one reference transport correction followed by one scalar diffusion synthetic correction.
- `dsaK` means `K` residual-correction applications of this base. Starting with `z=0` and `e_0=r`, each stage applies `z <- z + omega B_h e_j` and `e_{j+1} <- e_j - A_h(omega B_h e_j)`.
- `UNet-defect2-w1` uses `z0=B_h r`, `z1=B_h(r-A_h z0)`, and returns `z0 + z1 + C_theta(z0,z1,coeff)`.
- `base-call equal` means equal counts of selected base-preconditioner applications inside one preconditioner call, excluding diagnostic metric evaluations, `A_h` applications, scalar reductions, and U-Net overhead. `UNet-defect2` is base-call equal to `dsa2`; `dsa4` is a deeper classical work-budget baseline, not a strict equal-cost match.

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | one-step base metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | transport_dsa | anisotropic | train | dsa2 | 3.849e-01 | 4.566e-01 | 7.303e-02 | 1.264e-05 | 1.265e-05 |  |  |
| 64 | transport_dsa | anisotropic | train | dsa4 | 1.416e-01 | 1.794e-01 | 2.668e-02 | 1.293e-05 | 1.322e-05 |  |  |
| 64 | transport_dsa | anisotropic | train | UNet-defect2-w1 | 2.866e-01 | 3.343e-01 | 9.136e-02 | 1.288e-05 | 1.303e-05 | 0.3242250382900238 | 0.05446021631360054 |
| 64 | transport_dsa | anisotropic | heldout | dsa2 | 3.845e-01 | 4.554e-01 | 7.287e-02 | 1.321e-05 | 1.244e-05 |  |  |
| 64 | transport_dsa | anisotropic | heldout | dsa4 | 1.413e-01 | 1.786e-01 | 2.655e-02 | 1.321e-05 | 1.304e-05 |  |  |
| 64 | transport_dsa | anisotropic | heldout | UNet-defect2-w1 | 2.862e-01 | 3.314e-01 | 9.131e-02 | 1.295e-05 | 1.397e-05 | 0.3242250382900238 | 0.05446021631360054 |
| 64 | transport_dsa | mixed | train | dsa2 | 6.601e-01 | 6.489e-01 | 1.499e-01 | 5.228e-04 | 3.640e-04 |  |  |
| 64 | transport_dsa | mixed | train | dsa4 | 3.022e-01 | 2.991e-01 | 4.630e-02 | 1.358e-05 | 8.731e-06 |  |  |
| 64 | transport_dsa | mixed | train | UNet-defect2-w1 | 3.883e-01 | 3.754e-01 | 1.982e-01 | 1.393e-04 | 1.667e-04 | 0.49541664123535156 | 0.2634320855140686 |
| 64 | transport_dsa | mixed | heldout | dsa2 | 5.507e-01 | 5.781e-01 | 1.123e-01 | 5.175e-04 | 4.169e-04 |  |  |
| 64 | transport_dsa | mixed | heldout | dsa4 | 2.312e-01 | 2.500e-01 | 5.174e-02 | 1.385e-05 | 8.410e-06 |  |  |
| 64 | transport_dsa | mixed | heldout | UNet-defect2-w1 | 2.763e-01 | 2.857e-01 | 1.596e-01 | 3.068e-05 | 8.597e-05 | 0.49541664123535156 | 0.2634320855140686 |
