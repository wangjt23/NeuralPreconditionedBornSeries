# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['anisotropic']`
- `sizes`: `[12]`
- `n_angles`: `12`
- `base_kinds`: `['transport_tsa_moment']`
- `tsa_harmonics`: `None`
- `forms`: `['learned_rp_moment_tsa']`
- `moment_harmonics`: `4`
- `moment_weight_scale`: `0.5`
- `moment_transport_relaxation`: `0.4`
- `moment_synthetic_relaxation`: `0.4`
- `learn_scalar_moment`: `False`
- `defect_relaxations`: `[1.0]`
- `baseline_depths`: `[1]`
- `baseline_relaxations`: `[0.4]`
- `eta_scale`: `0.1`
- `width`: `8`
- `levels`: `2`
- `steps`: `2`
- `lr`: `0.001`
- `weight_decay`: `0.0`
- `scheduler`: `none`
- `scheduler_step_size`: `0`
- `scheduler_gamma`: `0.5`
- `scheduler_eta_min`: `1e-05`
- `log_steps`: `0`
- `residual_mode`: `resample`
- `residuals_per_step`: `16`
- `rollout_samples_per_step`: `8`
- `n_rollout`: `4`
- `n_random`: `6`
- `random_modes`: `5`
- `aux_sources`: `2`
- `metric_weight`: `0.0`
- `physical_weight`: `1.0`
- `correction_weight`: `0.0`
- `rollout_weight`: `0.2`
- `rollout_steps`: `2`
- `rollout_decay`: `0.8`
- `rollout_space`: `physical`
- `stability_weight`: `0.8`
- `contraction_target`: `0.95`
- `stability_space`: `physical`
- `teacher_depth`: `0`
- `teacher_relaxation`: `1.0`
- `teacher_kind`: ``
- `teacher_weight`: `0.0`
- `maxiter`: `4`
- `fgmres_maxiter`: `3`
- `fgmres_mode`: `real-stacked`
- `tol`: `1e-05`
- `tol_grid`: `[]`
- `seed`: `20260519`
- `device`: `cpu`
- `allow_tf32`: `False`
- `output_init`: `zero`
- `output_csv`: `rte/tmp_learned_rp_physical_rollout_smoke.csv`
- `trace_csv`: ``
- `output_md`: `rte/TMP_LEARNED_RP_PHYSICAL_ROLLOUT_SMOKE.md`

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
- `CoeffMixedTSA-MK` keeps the closed TSA-MK solve fixed, then uses a coefficient-conditioned local linear mixing matrix on retained moments. It is initialized to the identity map and can preserve the scalar moment exactly.
- `LearnedRP-TSA-MK` learns coefficient-conditioned restriction/prolongation maps around the fixed closed moment inverse. It is initialized to the identity and can preserve the scalar moment exactly.
- `ResidualSubspaceTSA-MK` uses the current post-transport residual to choose a bounded global mixing matrix on retained moments before the fixed closed inverse. This is a residual-dependent angular-subspace ablation and is nonlinear in the residual.
- By default `CoeffWeightedTSA-MK` freezes the scalar `m=0` synthetic weight to preserve the scalar diffusion moment; pass `--learn-scalar-moment` to ablate this constraint. `--moment-transport-relaxation` and `--moment-synthetic-relaxation` allow stable damped variants.
- `--rollout-space` and `--stability-space` choose whether rollout and stability penalties measure contraction in the base metric space or in the physical residual norm.
- `base-call equal` means equal counts of selected base-preconditioner applications inside one preconditioner call, excluding diagnostic metric evaluations, `A_h` applications, scalar reductions, and U-Net overhead. `UNet-defect2` is base-call equal to `dsa2`/`tsa2`; `dsa3`/`dsa4` and `tsa3`/`tsa4` are deeper classical work-budget baselines, not strict equal-cost matches.

## Results

| Size | Base | K | FGMRES | Case | Split | Method | one-step true | one-step base metric | stat final | FGMRES final | stat ttol | FGMRES ttol | loss0 | loss1 |
|---:|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | transport_tsa_moment | 4 | real-stacked | anisotropic | train | tsa_moment1-w0.4 | 6.864e-01 | 7.461e-01 | 3.390e-01 | 1.842e-01 | nan | nan |  |  |
| 12 | transport_tsa_moment | 4 | real-stacked | anisotropic | train | LearnedRP-TSA-M4 | 6.858e-01 | 7.460e-01 | 3.381e-01 | 1.839e-01 | nan | nan | 0.5123437643051147 | 0.5213743448257446 |
| 12 | transport_tsa_moment | 4 | real-stacked | anisotropic | heldout | tsa_moment1-w0.4 | 6.861e-01 | 7.455e-01 | 3.344e-01 | 1.822e-01 | nan | nan |  |  |
| 12 | transport_tsa_moment | 4 | real-stacked | anisotropic | heldout | LearnedRP-TSA-M4 | 6.855e-01 | 7.453e-01 | 3.337e-01 | 1.820e-01 | nan | nan | 0.5123437643051147 | 0.5213743448257446 |
