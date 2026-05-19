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
- `steps`: `1800`
- `lr`: `0.002`
- `weight_decay`: `0.0001`
- `scheduler`: `step`
- `scheduler_step_size`: `600`
- `scheduler_gamma`: `0.5`
- `scheduler_eta_min`: `1e-05`
- `log_steps`: `0`
- `residual_mode`: `resample`
- `residuals_per_step`: `128`
- `rollout_samples_per_step`: `64`
- `n_rollout`: `16`
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
- `teacher_weight`: `0.3`
- `maxiter`: `120`
- `fgmres_maxiter`: `60`
- `tol`: `1e-06`
- `tol_grid`: `[0.0001, 1e-05]`
- `seed`: `20260519`
- `device`: `cuda:0`
- `allow_tf32`: `False`
- `output_init`: `small`
- `output_csv`: `rte/npbs_unet_ablation_tsafull_size64_large_teacher1800.csv`
- `trace_csv`: `rte/npbs_unet_ablation_tsafull_size64_large_teacher1800_traces.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_TSAFULL_SIZE64_LARGE_TEACHER1800.md`

## Terminology

- `transport` is the shifted/reference transport inverse `G_eta`.
- `transport_dsa` applies one reference transport correction followed by one scalar diffusion synthetic correction.
- `transport_tsa` applies one reference transport correction followed by one reference transport-scattering synthetic correction. `tsa_harmonics=None` keeps all angular Fourier modes; an integer such as `2` keeps only the lowest angular harmonics.
- `dsaK` and `tsaK` mean `K` residual-correction applications of the selected base preconditioner. Starting with `z=0` and residual `e_0=r`, each stage applies `z <- z + omega B_h e_j` and `e_{j+1} <- e_j - A_h(omega B_h e_j)`. A suffix such as `-w0.5` uses relaxation `omega=0.5`; no suffix means `omega=1`.
- `UNet-defect2-w1` uses `z0=B_h r`, `z1=B_h(r-A_h z0)`, and returns `z0 + z1 + C_theta(z0,z1,coeff)`. It costs two base applications plus the U-Net evaluation per preconditioner call.
- `base-call equal` means equal counts of selected base-preconditioner applications inside one preconditioner call, excluding diagnostic metric evaluations, `A_h` applications, scalar reductions, and U-Net overhead. `UNet-defect2` is base-call equal to `dsa2`/`tsa2`; `dsa3`/`dsa4` and `tsa3`/`tsa4` are deeper classical work-budget baselines, not strict equal-cost matches.

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | one-step base metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | transport_tsa | mixed | train | tsa2 | 1.736e-01 | 1.851e-01 | 2.849e-01 | 7.541e-06 | 8.139e-06 |  |  |
| 64 | transport_tsa | mixed | train | tsa4 | 9.690e-02 | 1.140e-01 | 2.019e-01 | 7.343e-06 | 7.633e-06 |  |  |
| 64 | transport_tsa | mixed | train | UNet-defect2-w1 | 2.200e-01 | 2.244e-01 | 2.177e-01 | 7.248e-06 | 7.535e-06 | 0.5977877974510193 | 0.29409098625183105 |
| 64 | transport_tsa | mixed | heldout | tsa2 | 1.789e-01 | 2.070e-01 | 3.605e-01 | 7.061e-06 | 7.277e-06 |  |  |
| 64 | transport_tsa | mixed | heldout | tsa4 | 1.147e-01 | 1.335e-01 | 2.541e-01 | 7.232e-06 | 7.322e-06 |  |  |
| 64 | transport_tsa | mixed | heldout | UNet-defect2-w1 | 1.923e-01 | 2.074e-01 | 2.243e-01 | 7.328e-06 | 7.278e-06 | 0.5977877974510193 | 0.29409098625183105 |
