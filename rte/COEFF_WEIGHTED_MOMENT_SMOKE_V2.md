# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['anisotropic']`
- `sizes`: `[12]`
- `n_angles`: `12`
- `base_kinds`: `['transport_tsa_moment']`
- `tsa_harmonics`: `None`
- `forms`: `['coeff_weighted_moment_tsa']`
- `moment_harmonics`: `2`
- `moment_weight_scale`: `0.5`
- `moment_transport_relaxation`: `0.4`
- `moment_synthetic_relaxation`: `0.4`
- `learn_scalar_moment`: `False`
- `defect_relaxations`: `[1.0]`
- `baseline_depths`: `[1]`
- `baseline_relaxations`: `[0.4, 1.0]`
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
- `maxiter`: `8`
- `fgmres_maxiter`: `5`
- `fgmres_mode`: `real-stacked`
- `tol`: `1e-05`
- `tol_grid`: `[]`
- `seed`: `20260519`
- `device`: `cpu`
- `allow_tf32`: `False`
- `output_init`: `zero`
- `output_csv`: `rte/coeff_weighted_moment_smoke_v2.csv`
- `trace_csv`: ``
- `output_md`: `rte/COEFF_WEIGHTED_MOMENT_SMOKE_V2.md`

## Terminology

- `transport` is the shifted/reference transport inverse `G_eta`.
- `transport_dsa` applies one reference transport correction followed by one scalar diffusion synthetic correction.
- `transport_tsa` applies one reference transport correction followed by one reference transport-scattering synthetic correction. `tsa_harmonics=None` keeps all angular Fourier modes; an integer such as `2` keeps only the lowest angular harmonics.
- `transport_tsa_moment` applies one reference transport correction followed by a closed Galerkin angular Fourier synthetic correction `Q_K (Q_K^* A_ref Q_K)^{-1} Q_K^*` on the post-transport residual. This differs from `transport_tsa` with `tsa_harmonics=K`, which only projects the RHS before applying the full angular reference inverse.
- `dsaK` and `tsaK` mean `K` residual-correction applications of the selected base preconditioner. Starting with `z=0` and residual `e_0=r`, each stage applies `z <- z + omega B_h e_j` and `e_{j+1} <- e_j - A_h(omega B_h e_j)`. A suffix such as `-w0.5` uses relaxation `omega=0.5`; no suffix means `omega=1`.
- `UNet-defect2-w1` uses `z0=B_h r`, `z1=B_h(r-A_h z0)`, and returns `z0 + z1 + C_theta(z0,z1,coeff)`. It costs two base applications plus the U-Net evaluation per preconditioner call.
- `MomentUNet-K` uses the same defect2 structure but restricts the learned correction to angular Fourier modes `0,+/-1,...,+/-K` before prolonging back to ordinates.
- `WeightedTSA-MK` is a tiny learned structural model initialized as closed TSA-MK; it learns bounded real weights for each absolute angular mode in the synthetic correction.
- `CoeffWeightedTSA-MK` keeps the closed TSA-MK solve fixed, then uses a coefficient-conditioned nonlinear network to learn bounded physical-space weights for each absolute angular mode. It is nonlinear in coefficients but linear in residuals for fixed coefficients.
- By default `CoeffWeightedTSA-MK` freezes the scalar `m=0` synthetic weight to preserve the scalar diffusion moment; pass `--learn-scalar-moment` to ablate this constraint. `--moment-transport-relaxation` and `--moment-synthetic-relaxation` allow stable damped variants.
- `base-call equal` means equal counts of selected base-preconditioner applications inside one preconditioner call, excluding diagnostic metric evaluations, `A_h` applications, scalar reductions, and U-Net overhead. `UNet-defect2` is base-call equal to `dsa2`/`tsa2`; `dsa3`/`dsa4` and `tsa3`/`tsa4` are deeper classical work-budget baselines, not strict equal-cost matches.

## Results

| Size | Base | K | FGMRES | Case | Split | Method | one-step true | one-step base metric | stat final | FGMRES final | stat ttol | FGMRES ttol | loss0 | loss1 |
|---:|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | transport_tsa_moment | 2 | real-stacked | anisotropic | train | tsa_moment1-w0.4 | 7.921e-01 | 6.887e-01 | 1.839e-01 | 1.437e-01 | nan | nan |  |  |
| 12 | transport_tsa_moment | 2 | real-stacked | anisotropic | train | tsa_moment1 | 1.219e+00 | 7.632e-01 | 5.084e+01 | 1.437e-01 | nan | nan |  |  |
| 12 | transport_tsa_moment | 2 | real-stacked | anisotropic | train | CoeffWeightedTSA-M2 | 7.907e-01 | 6.888e-01 | 1.838e-01 | 1.434e-01 | nan | nan | 0.3573806583881378 | 0.3568441569805145 |
| 12 | transport_tsa_moment | 2 | real-stacked | anisotropic | heldout | tsa_moment1-w0.4 | 7.912e-01 | 6.888e-01 | 1.782e-01 | 1.426e-01 | nan | nan |  |  |
| 12 | transport_tsa_moment | 2 | real-stacked | anisotropic | heldout | tsa_moment1 | 1.219e+00 | 7.686e-01 | 5.783e+01 | 1.426e-01 | nan | nan |  |  |
| 12 | transport_tsa_moment | 2 | real-stacked | anisotropic | heldout | CoeffWeightedTSA-M2 | 7.898e-01 | 6.888e-01 | 1.782e-01 | 1.422e-01 | nan | nan | 0.3573806583881378 | 0.3568441569805145 |
