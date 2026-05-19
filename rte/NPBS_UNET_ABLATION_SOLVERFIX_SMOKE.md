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
- `steps`: `2`
- `lr`: `0.001`
- `weight_decay`: `0.0001`
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
- `rollout_weight`: `0.1`
- `rollout_steps`: `2`
- `rollout_decay`: `1.0`
- `stability_weight`: `0.1`
- `contraction_target`: `0.95`
- `teacher_depth`: `4`
- `teacher_relaxation`: `1.0`
- `teacher_kind`: `transport_dsa`
- `teacher_weight`: `0.2`
- `maxiter`: `8`
- `fgmres_maxiter`: `6`
- `tol`: `1e-05`
- `tol_grid`: `[]`
- `seed`: `20260519`
- `device`: `cuda:0`
- `allow_tf32`: `False`
- `output_init`: `small`
- `output_csv`: `rte/npbs_unet_ablation_solverfix_smoke.csv`
- `trace_csv`: `rte/npbs_unet_ablation_solverfix_smoke_traces.csv`
- `output_md`: `rte/NPBS_UNET_ABLATION_SOLVERFIX_SMOKE.md`

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
| 16 | transport_dsa | mixed | train | dsa2 | 6.461e-01 | 6.443e-01 | 1.432e-01 | 3.099e-02 | 4.071e-02 |  |  |
| 16 | transport_dsa | mixed | train | dsa4 | 2.957e-01 | 2.991e-01 | 4.710e-02 | 1.643e-02 | 1.448e-02 |  |  |
| 16 | transport_dsa | mixed | train | UNet-defect2-w1 | 6.461e-01 | 6.442e-01 | 1.432e-01 | 3.098e-02 | 4.071e-02 | 0.3858325481414795 | 0.3895171582698822 |
| 16 | transport_dsa | mixed | heldout | dsa2 | 5.652e-01 | 5.974e-01 | 1.121e-01 | 3.480e-02 | 4.200e-02 |  |  |
| 16 | transport_dsa | mixed | heldout | dsa4 | 2.525e-01 | 2.763e-01 | 5.410e-02 | 1.808e-02 | 1.406e-02 |  |  |
| 16 | transport_dsa | mixed | heldout | UNet-defect2-w1 | 5.652e-01 | 5.974e-01 | 1.121e-01 | 3.480e-02 | 4.200e-02 | 0.3858325481414795 | 0.3895171582698822 |
