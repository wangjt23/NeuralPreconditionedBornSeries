# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['mixed']`
- `sizes`: `[64, 80]`
- `n_angles`: `24`
- `base_kinds`: `['transport_dsa']`
- `tsa_harmonics`: `None`
- `forms`: `['defect2']`
- `defect_relaxations`: `[1.0]`
- `baseline_depths`: `[2, 4]`
- `baseline_relaxations`: `[1.0]`
- `eta_scale`: `0.1`
- `width`: `8`
- `levels`: `2`
- `steps`: `1000`
- `lr`: `0.002`
- `weight_decay`: `0.0001`
- `scheduler`: `step`
- `scheduler_step_size`: `350`
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
- `teacher_depth`: `4`
- `teacher_relaxation`: `1.0`
- `teacher_kind`: `transport_dsa`
- `teacher_weight`: `0.3`
- `maxiter`: `80`
- `fgmres_maxiter`: `40`
- `tol`: `1e-06`
- `tol_grid`: `[0.0001, 1e-05]`
- `seed`: `20260519`
- `device`: `cuda:0`
- `allow_tf32`: `False`
- `output_init`: `small`
- `output_csv`: `rte/npbs_unet_ablation_dsa_size64plus_cheap_teacher1000.csv`
- `trace_csv`: `rte/npbs_unet_ablation_dsa_size64plus_cheap_teacher1000_traces.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_DSA_SIZE64PLUS_CHEAP_TEACHER1000.md`

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | one-step base metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | transport_dsa | mixed | train | dsa2 | 6.601e-01 | 6.489e-01 | 1.499e-01 | 1.460e-04 | 1.347e-05 |  |  |
| 64 | transport_dsa | mixed | train | dsa4 | 3.022e-01 | 2.991e-01 | 4.630e-02 | 7.120e-06 | 7.408e-06 |  |  |
| 64 | transport_dsa | mixed | train | UNet-defect2-w1 | 4.228e-01 | 4.002e-01 | 2.714e-01 | 6.337e-05 | 1.305e-05 | 0.536705493927002 | 0.40092918276786804 |
| 64 | transport_dsa | mixed | heldout | dsa2 | 5.507e-01 | 5.781e-01 | 1.123e-01 | 1.467e-04 | 1.184e-05 |  |  |
| 64 | transport_dsa | mixed | heldout | dsa4 | 2.312e-01 | 2.500e-01 | 5.174e-02 | 7.735e-06 | 7.607e-06 |  |  |
| 64 | transport_dsa | mixed | heldout | UNet-defect2-w1 | 3.412e-01 | 3.298e-01 | 2.230e-01 | 5.703e-05 | 1.344e-05 | 0.536705493927002 | 0.40092918276786804 |
| 80 | transport_dsa | mixed | train | dsa2 | 6.608e-01 | 6.498e-01 | 1.525e-01 | 1.428e-04 | 1.415e-05 |  |  |
| 80 | transport_dsa | mixed | train | dsa4 | 3.021e-01 | 2.988e-01 | 4.705e-02 | 9.004e-06 | 8.905e-06 |  |  |
| 80 | transport_dsa | mixed | train | UNet-defect2-w1 | 4.171e-01 | 4.000e-01 | 2.851e-01 | 8.558e-05 | 7.984e-05 | 0.5353420972824097 | 0.39562326669692993 |
| 80 | transport_dsa | mixed | heldout | dsa2 | 5.486e-01 | 5.759e-01 | 1.141e-01 | 1.408e-04 | 1.305e-05 |  |  |
| 80 | transport_dsa | mixed | heldout | dsa4 | 2.296e-01 | 2.484e-01 | 5.266e-02 | 1.037e-05 | 9.943e-06 |  |  |
| 80 | transport_dsa | mixed | heldout | UNet-defect2-w1 | 3.363e-01 | 3.249e-01 | 2.373e-01 | 6.559e-05 | 7.396e-05 | 0.5353420972824097 | 0.39562326669692993 |
