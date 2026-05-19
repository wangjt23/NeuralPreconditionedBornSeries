# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['mixed']`
- `sizes`: `[64]`
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
- `steps`: `2200`
- `lr`: `0.001`
- `weight_decay`: `0.0001`
- `scheduler`: `step`
- `scheduler_step_size`: `800`
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
- `stability_weight`: `0.8`
- `contraction_target`: `0.95`
- `teacher_depth`: `4`
- `teacher_relaxation`: `1.0`
- `teacher_kind`: `transport_dsa`
- `teacher_weight`: `0.2`
- `maxiter`: `120`
- `fgmres_maxiter`: `60`
- `tol`: `1e-06`
- `tol_grid`: `[0.0001, 1e-05]`
- `seed`: `20260521`
- `device`: `cuda:0`
- `allow_tf32`: `False`
- `output_init`: `small`
- `output_csv`: `rte/npbs_unet_ablation_dsa_size64_cheap_teacher2200_stable.csv`
- `trace_csv`: `rte/npbs_unet_ablation_dsa_size64_cheap_teacher2200_stable_traces.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_DSA_SIZE64_CHEAP_TEACHER2200_STABLE.md`

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
| 64 | transport_dsa | mixed | train | dsa2 | 6.601e-01 | 6.489e-01 | 1.499e-01 | 1.374e-05 | 7.564e-06 |  |  |
| 64 | transport_dsa | mixed | train | dsa4 | 3.022e-01 | 2.991e-01 | 4.630e-02 | 7.763e-06 | 7.528e-06 |  |  |
| 64 | transport_dsa | mixed | train | UNet-defect2-w1 | 4.300e-01 | 4.169e-01 | 3.052e-01 | 9.427e-04 | 7.756e-06 | 0.5821987986564636 | 0.44702303409576416 |
| 64 | transport_dsa | mixed | heldout | dsa2 | 5.507e-01 | 5.781e-01 | 1.123e-01 | 1.421e-05 | 8.060e-06 |  |  |
| 64 | transport_dsa | mixed | heldout | dsa4 | 2.312e-01 | 2.500e-01 | 5.174e-02 | 7.428e-06 | 7.296e-06 |  |  |
| 64 | transport_dsa | mixed | heldout | UNet-defect2-w1 | 3.766e-01 | 3.762e-01 | 2.574e-01 | 7.116e-04 | 7.462e-06 | 0.5821987986564636 | 0.44702303409576416 |
