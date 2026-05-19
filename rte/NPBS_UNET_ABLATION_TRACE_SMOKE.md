# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['mixed']`
- `sizes`: `[16]`
- `n_angles`: `8`
- `base_kinds`: `['transport_dsa']`
- `tsa_harmonics`: `None`
- `forms`: `['defect2']`
- `defect_relaxations`: `[1.0]`
- `baseline_depths`: `[2, 4]`
- `baseline_relaxations`: `[1.0]`
- `eta_scale`: `0.1`
- `width`: `8`
- `levels`: `2`
- `steps`: `3`
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
- `teacher_depth`: `4`
- `teacher_relaxation`: `1.0`
- `teacher_kind`: ``
- `teacher_weight`: `0.2`
- `maxiter`: `8`
- `fgmres_maxiter`: `6`
- `tol`: `1e-05`
- `tol_grid`: `[0.001, 0.0001]`
- `seed`: `20260519`
- `device`: `cuda:0`
- `allow_tf32`: `False`
- `output_init`: `zero`
- `output_csv`: `rte/npbs_unet_ablation_trace_smoke.csv`
- `trace_csv`: `rte/npbs_unet_ablation_trace_smoke_traces.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_TRACE_SMOKE.md`

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | one-step base metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | transport_dsa | mixed | train | dsa2 | 6.461e-01 | 6.443e-01 | 1.432e-01 | 3.099e-02 | 4.071e-02 |  |  |
| 16 | transport_dsa | mixed | train | dsa4 | 2.957e-01 | 2.991e-01 | 4.710e-02 | 1.643e-02 | 1.448e-02 |  |  |
| 16 | transport_dsa | mixed | train | UNet-defect2-w1 | 6.458e-01 | 6.441e-01 | 1.431e-01 | 3.097e-02 | 4.070e-02 | 0.3156575560569763 | 0.3149875998497009 |
| 16 | transport_dsa | mixed | heldout | dsa2 | 5.652e-01 | 5.974e-01 | 1.121e-01 | 3.480e-02 | 4.200e-02 |  |  |
| 16 | transport_dsa | mixed | heldout | dsa4 | 2.525e-01 | 2.763e-01 | 5.410e-02 | 1.808e-02 | 1.406e-02 |  |  |
| 16 | transport_dsa | mixed | heldout | UNet-defect2-w1 | 5.652e-01 | 5.975e-01 | 1.121e-01 | 3.476e-02 | 4.194e-02 | 0.3156575560569763 | 0.3149875998497009 |
