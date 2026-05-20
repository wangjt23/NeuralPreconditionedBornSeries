# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['anisotropic']`
- `sizes`: `[12]`
- `n_angles`: `12`
- `base_kinds`: `['transport_tsa_moment']`
- `tsa_harmonics`: `2`
- `forms`: `['weighted_moment_tsa', 'moment_defect2']`
- `moment_harmonics`: `2`
- `moment_weight_scale`: `0.5`
- `defect_relaxations`: `[1.0]`
- `baseline_depths`: `[1, 2]`
- `baseline_relaxations`: `[1.0]`
- `eta_scale`: `0.1`
- `width`: `8`
- `levels`: `2`
- `steps`: `5`
- `lr`: `0.002`
- `weight_decay`: `0.0`
- `scheduler`: `none`
- `scheduler_step_size`: `0`
- `scheduler_gamma`: `0.5`
- `scheduler_eta_min`: `1e-05`
- `log_steps`: `0`
- `residual_mode`: `pool`
- `residuals_per_step`: `16`
- `rollout_samples_per_step`: `8`
- `n_rollout`: `6`
- `n_random`: `6`
- `random_modes`: `5`
- `aux_sources`: `0`
- `physical_weight`: `0.5`
- `correction_weight`: `1e-06`
- `rollout_weight`: `0.0`
- `rollout_steps`: `1`
- `rollout_decay`: `1.0`
- `stability_weight`: `0.0`
- `contraction_target`: `0.98`
- `teacher_depth`: `0`
- `teacher_relaxation`: `1.0`
- `teacher_kind`: ``
- `teacher_weight`: `0.0`
- `maxiter`: `10`
- `fgmres_maxiter`: `6`
- `fgmres_mode`: `real-stacked`
- `tol`: `1e-05`
- `tol_grid`: `[]`
- `seed`: `20260519`
- `device`: `cpu`
- `allow_tf32`: `False`
- `output_init`: `zero`
- `output_csv`: `rte/moment_learned_smoke.csv`
- `trace_csv`: ``
- `output_md`: `rte/MOMENT_LEARNED_SMOKE.md`

## Terminology

- `transport` is the shifted/reference transport inverse `G_eta`.
- `transport_dsa` applies one reference transport correction followed by one scalar diffusion synthetic correction.
- `transport_tsa` applies one reference transport correction followed by one reference transport-scattering synthetic correction. `tsa_harmonics=None` keeps all angular Fourier modes; an integer such as `2` keeps only the lowest angular harmonics.
- `transport_tsa_moment` applies one reference transport correction followed by a closed Galerkin angular Fourier synthetic correction `Q_K (Q_K^* A_ref Q_K)^{-1} Q_K^*` on the post-transport residual. This differs from `transport_tsa` with `tsa_harmonics=K`, which only projects the RHS before applying the full angular reference inverse.
- `dsaK` and `tsaK` mean `K` residual-correction applications of the selected base preconditioner. Starting with `z=0` and residual `e_0=r`, each stage applies `z <- z + omega B_h e_j` and `e_{j+1} <- e_j - A_h(omega B_h e_j)`. A suffix such as `-w0.5` uses relaxation `omega=0.5`; no suffix means `omega=1`.
- `UNet-defect2-w1` uses `z0=B_h r`, `z1=B_h(r-A_h z0)`, and returns `z0 + z1 + C_theta(z0,z1,coeff)`. It costs two base applications plus the U-Net evaluation per preconditioner call.
- `MomentUNet-K` uses the same defect2 structure but restricts the learned correction to angular Fourier modes `0,+/-1,...,+/-K` before prolonging back to ordinates.
- `WeightedTSA-MK` is a tiny learned structural model initialized as closed TSA-MK; it learns bounded real weights for each absolute angular mode in the synthetic correction.
- `base-call equal` means equal counts of selected base-preconditioner applications inside one preconditioner call, excluding diagnostic metric evaluations, `A_h` applications, scalar reductions, and U-Net overhead. `UNet-defect2` is base-call equal to `dsa2`/`tsa2`; `dsa3`/`dsa4` and `tsa3`/`tsa4` are deeper classical work-budget baselines, not strict equal-cost matches.

## Results

| Size | Base | Case | Split | Method | one-step true | one-step Born metric | one-step base metric | stat final | FGMRES final | loss0 | loss1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 12 | transport_tsa_moment | anisotropic | train | tsa_moment1 | 1.219e+00 | 1.069e+00 | 7.632e-01 | 2.609e+02 | 8.949e-02 |  |  |
| 12 | transport_tsa_moment | anisotropic | train | tsa_moment2 | 1.209e+00 | 8.383e-01 | 7.455e-01 | 2.742e+06 | 5.267e-01 |  |  |
| 12 | transport_tsa_moment | anisotropic | train | WeightedTSA-M2 | 1.213e+00 | 1.066e+00 | 7.606e-01 | 2.461e+02 | 8.920e-02 | 3.0654752254486084 | 3.038142204284668 |
| 12 | transport_tsa_moment | anisotropic | train | MomentUNet-K2-w1 | 1.204e+00 | 8.356e-01 | 7.423e-01 | 2.661e+06 | 5.270e-01 | 12.073328971862793 | 12.022597312927246 |
| 12 | transport_tsa_moment | anisotropic | heldout | tsa_moment1 | 1.219e+00 | 1.066e+00 | 7.686e-01 | 3.111e+02 | 9.093e-02 |  |  |
| 12 | transport_tsa_moment | anisotropic | heldout | tsa_moment2 | 1.225e+00 | 8.407e-01 | 7.539e-01 | 3.602e+06 | 5.264e-01 |  |  |
| 12 | transport_tsa_moment | anisotropic | heldout | WeightedTSA-M2 | 1.214e+00 | 1.062e+00 | 7.659e-01 | 2.937e+02 | 9.064e-02 | 3.0654752254486084 | 3.038142204284668 |
| 12 | transport_tsa_moment | anisotropic | heldout | MomentUNet-K2-w1 | 1.221e+00 | 8.387e-01 | 7.514e-01 | 3.527e+06 | 5.267e-01 | 12.073328971862793 | 12.022597312927246 |
