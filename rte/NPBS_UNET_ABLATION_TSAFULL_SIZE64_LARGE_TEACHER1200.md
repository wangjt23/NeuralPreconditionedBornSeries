# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['mixed']`
- `sizes`: `[64]`
- `n_angles`: `24`
- `base_kinds`: `['transport_tsa']`
- `tsa_harmonics`: `None`
- `forms`: `['defect2']`
- `defect_relaxations`: `[1.0]`
- `baseline_depths`: `[2, 4]`
- `baseline_relaxations`: `[1.0]`
- `eta_scale`: `0.1`
- `width`: `40`
- `levels`: `5`
- `steps`: `1200`
- `lr`: `0.002`
- `weight_decay`: `0.0001`
- `scheduler`: `step`
- `scheduler_step_size`: `400`
- `scheduler_gamma`: `0.5`
- `scheduler_eta_min`: `1e-05`
- `log_steps`: `0`
- `residual_mode`: `resample`
- `residuals_per_step`: `128`
- `rollout_samples_per_step`: `64`
- `n_rollout`: `14`
- `n_random`: `6`
- `random_modes`: `5`
- `aux_sources`: `5`
- `physical_weight`: `0.5`
- `correction_weight`: `1e-06`
- `rollout_weight`: `0.2`
- `rollout_steps`: `3`
- `rollout_decay`: `0.8`
- `stability_weight`: `0.5`
- `contraction_target`: `0.95`
- `teacher_depth`: `4`
- `teacher_relaxation`: `1.0`
- `teacher_kind`: `transport_tsa`
- `teacher_weight`: `0.25`
- `maxiter`: `80`
- `fgmres_maxiter`: `40`
- `tol`: `1e-06`
- `tol_grid`: `[0.0001, 1e-05]`
- `seed`: `20260519`
- `device`: `cuda:0`
- `allow_tf32`: `False`
- `output_init`: `small`
- `output_csv`: `rte/npbs_unet_ablation_tsafull_size64_large_teacher1200.csv`
- `trace_csv`: `rte/npbs_unet_ablation_tsafull_size64_large_teacher1200_traces.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_TSAFULL_SIZE64_LARGE_TEACHER1200.md`

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | one-step base metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | transport_tsa | mixed | train | tsa2 | 1.736e-01 | 1.851e-01 | 2.849e-01 | 7.233e-06 | 7.694e-06 |  |  |
| 64 | transport_tsa | mixed | train | tsa4 | 9.690e-02 | 1.140e-01 | 2.019e-01 | 7.565e-06 | 7.535e-06 |  |  |
| 64 | transport_tsa | mixed | train | UNet-defect2-w1 | 2.070e-01 | 1.955e-01 | 1.862e-01 | 7.368e-06 | 7.756e-06 | 0.5812024474143982 | 0.2322045862674713 |
| 64 | transport_tsa | mixed | heldout | tsa2 | 1.789e-01 | 2.070e-01 | 3.605e-01 | 7.429e-06 | 7.356e-06 |  |  |
| 64 | transport_tsa | mixed | heldout | tsa4 | 1.147e-01 | 1.335e-01 | 2.541e-01 | 7.415e-06 | 7.456e-06 |  |  |
| 64 | transport_tsa | mixed | heldout | UNet-defect2-w1 | 1.741e-01 | 1.796e-01 | 1.999e-01 | 7.551e-06 | 7.355e-06 | 0.5812024474143982 | 0.2322045862674713 |
