# NPBS U-Net Preconditioner Ablation

This ablation uses the repository's matrix-free periodic RTE NPBS operator, not the ATFPS discretization.

## Settings

- `cases`: `['mixed']`
- `sizes`: `[12]`
- `n_angles`: `8`
- `base_kinds`: `['transport_tsa_moment']`
- `tsa_harmonics`: `None`
- `forms`: `['learned_rp_moment_tsa']`
- `moment_harmonics`: `2`
- `moment_weight_scale`: `0.5`
- `moment_transport_relaxation`: `0.4`
- `moment_synthetic_relaxation`: `0.4`
- `learn_scalar_moment`: `False`
- `defect_relaxations`: `[1.0]`
- `baseline_depths`: `[1, 2]`
- `baseline_relaxations`: `[1.0]`
- `eta_scale`: `0.1`
- `width`: `16`
- `levels`: `3`
- `steps`: `2`
- `lr`: `0.001`
- `weight_decay`: `0.0`
- `scheduler`: `cosine`
- `scheduler_step_size`: `0`
- `scheduler_gamma`: `0.5`
- `scheduler_eta_min`: `None`
- `scheduler_min_lr_ratio`: `0.01`
- `scheduler_pct_start`: `0.1`
- `log_steps`: `0`
- `residual_mode`: `resample`
- `residuals_per_step`: `4`
- `rollout_samples_per_step`: `2`
- `n_rollout`: `2`
- `n_random`: `6`
- `random_modes`: `5`
- `aux_sources`: `1`
- `metric_weight`: `0.0`
- `physical_weight`: `1.0`
- `correction_weight`: `1e-06`
- `rollout_weight`: `0.0`
- `rollout_steps`: `1`
- `rollout_decay`: `1.0`
- `rollout_space`: `metric`
- `stability_weight`: `0.0`
- `contraction_target`: `0.98`
- `stability_space`: `metric`
- `teacher_depth`: `0`
- `teacher_relaxation`: `1.0`
- `teacher_kind`: ``
- `teacher_weight`: `0.0`
- `maxiter`: `4`
- `fgmres_maxiter`: `4`
- `fgmres_mode`: `real-stacked`
- `tol`: `0.0001`
- `tol_grid`: `[]`
- `seed`: `20260531`
- `device`: `cpu`
- `allow_tf32`: `False`
- `output_init`: `zero`
- `output_csv`: `rte/tmp_ablation_scheduler_smoke.csv`
- `trace_csv`: ``
- `output_md`: `rte/tmp_ablation_scheduler_smoke.md`

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
| 12 | transport_tsa_moment | 2 | real-stacked | mixed | train | tsa_moment1 | 5.941e-01 | 4.545e-01 | 1.558e-01 | 1.469e-01 | nan | nan |  |  |
| 12 | transport_tsa_moment | 2 | real-stacked | mixed | train | tsa_moment2 | 3.233e-01 | 3.173e-01 | 7.813e-02 | 7.235e-02 | nan | nan |  |  |
| 12 | transport_tsa_moment | 2 | real-stacked | mixed | train | LearnedRP-TSA-M2 | 7.115e-01 | 7.270e-01 | 3.631e-01 | 1.470e-01 | nan | nan | 0.4973815977573395 | 0.4469551146030426 |
| 12 | transport_tsa_moment | 2 | real-stacked | mixed | heldout | tsa_moment1 | 3.872e-01 | 4.757e-01 | 1.824e-01 | 1.661e-01 | nan | nan |  |  |
| 12 | transport_tsa_moment | 2 | real-stacked | mixed | heldout | tsa_moment2 | 2.431e-01 | 3.667e-01 | 8.971e-02 | 7.571e-02 | nan | nan |  |  |
| 12 | transport_tsa_moment | 2 | real-stacked | mixed | heldout | LearnedRP-TSA-M2 | 6.592e-01 | 7.313e-01 | 2.794e-01 | 1.661e-01 | nan | nan | 0.4973815977573395 | 0.4469551146030426 |
