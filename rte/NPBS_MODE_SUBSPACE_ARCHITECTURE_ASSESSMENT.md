# NPBS RTE Mode/Subspace Architecture Assessment

This note summarizes the current evidence for learned angular mode/subspace
architectures in the RTE NPBS experiments.  The main question is whether mode
coupling, learned restriction/prolongation, or residual-dependent angular
subspaces are promising enough to push beyond diagonal coefficient-conditioned
moment weights.

## Architecture Classes

- `CoeffWeightedTSA-MK`: coefficient-conditioned diagonal weights on fixed
  retained Fourier moment corrections.  This is the control model.  It cannot
  rotate modes, couple modes, or choose a residual-dependent subspace.
- `CoeffMixedTSA-MK`: coefficient-conditioned local mixing matrix on retained
  moment coefficients after the closed TSA-MK solve.  It can couple modes, but
  it remains inside the fixed retained Fourier span.
- `LearnedRP-TSA-MK`: coefficient-conditioned learned restriction and
  prolongation around the closed TSA-MK inverse.  It is the best current
  structured candidate because it changes the low-order RHS and output maps,
  not just the post-solve weights.
- `ResidualSubspaceTSA-MK`: residual-dependent global mixing matrix on retained
  moment RHS before the closed inverse.  It is the direct residual-dependent
  subspace ablation, but it is nonlinear in the residual and currently less
  stable.
- `UNet-defect2`: generic residual-dependent defect correction.  It is not a
  clean learned angular-subspace model, but it is important evidence that raw
  physical residual loss can train a useful nonlinear correction when the base
  TSA operator has enough angular information.

## Mechanism Of The Promising Models

All structured moment models start from the same two-stage synthetic
acceleration template.  Given a residual `r`, first apply the shifted/reference
transport inverse

`z_t = G_eta r`

and form the post-transport residual

`s = r - A_h z_t`.

The classical closed moment TSA correction then restricts `s` to the angular
Fourier space `Q_K = span{0, +/-1, ..., +/-K}`, solves the closed low-order
Galerkin problem, and prolongs:

`z_m = Q_K (Q_K^* A_ref Q_K)^(-1) Q_K^* s`.

The damped stable initialization used in the hard mixed experiments is

`B_0 r = 0.4 z_t + 0.4 z_m`.

The learned variants modify this moment correction in increasingly expressive
ways:

- `CoeffWeightedTSA-MK` predicts spatially varying scalar weights
  `w_|m|(x)` from coefficient features such as normalized `sigma_t`, albedo,
  gradients, optical thickness, absorption ratio, and transport ratio.  It
  multiplies each retained moment coefficient by its weight after the closed
  moment solve.  It is linear in the residual for fixed coefficients.  Because
  it only changes amplitudes, it cannot create a new angular direction.
- `CoeffMixedTSA-MK` predicts a local matrix `M_theta(x)` on retained moment
  coefficients after the closed moment solve.  This permits local mode
  coupling, for example allowing `m=1` and `m=2` information to mix before
  prolongation.  It is still confined to the original `Q_K` span and is still
  residual-linear for fixed coefficients.
- `LearnedRP-TSA-MK` predicts two local matrices: one before the closed inverse
  and one after it.  Schematically,
  `Q_K^* s -> R_theta(x) Q_K^* s`, solve the fixed closed moment inverse, then
  apply `P_theta(x)` to the output moments.  This is more expressive than
  post-solve mixing because it changes the effective low-order RHS seen by the
  inverse and the lifted output.  It is currently the best structured model.
- `ResidualSubspaceTSA-MK` uses the current residual moments to predict a
  global bounded mixing matrix before the moment inverse.  This is nonlinear
  in the residual, so it is closer to an adaptive angular-subspace selector.
  The same residual dependence is also what makes it easiest to destabilize.
- `UNet-defect2` is different.  It forms two classical defect corrections,
  `z0 = B_h r` and `z1 = B_h (r - A_h z0)`, then returns
  `z0 + z1 + C_theta(z0, z1, coeff)`.  This model does not explicitly preserve
  a moment subspace, but it is residual dependent and can learn a nonlinear
  deep correction when `B_h` is full-angular TSA.

Parameter counts in the current representative settings are:

| Model | Setting | Trainable Parameters |
|---|---:|---:|
| `CoeffWeightedTSA-M4` | width 24, 4 blocks, 24 angles | `22,709` |
| `CoeffMixedTSA-M4` | width 24, 4 blocks, 24 angles | `24,609` |
| `LearnedRP-TSA-M4` | width 24, 4 blocks, 24 angles | `26,634` |
| `ResidualSubspaceTSA-M4` | width 24, 4 blocks, 24 angles | `8,001` |
| `UNet-defect2` | width 16, 4 levels, 12 angles | `103,280` |

The important distinction is not only parameter count.  The moment models are
structurally constrained to the retained Fourier moment space, while
`UNet-defect2` has a much less constrained residual-dependent correction.

## Metrics And Loss Definitions

Let `B_theta` be the learned preconditioner/correction map, `A_h` the discrete
RTE operator, and `r` a residual sample.  One preconditioner application gives

`z_theta = B_theta r`, `r_plus = r - A_h z_theta`.

The reported one-step metrics are:

- `one_step_true`:
  `rho_phys(r) = ||r_plus||_2 / ||r||_2`.
- `one_step_born_metric`:
  `rho_born(r) = ||G_eta r_plus||_2 / ||G_eta r||_2`, where `G_eta` is the
  shifted/reference transport inverse.
- `one_step_base_metric`:
  `rho_base(r) = ||B_h r_plus||_2 / ||B_h r||_2`, where `B_h` is the selected
  base preconditioner such as `transport_tsa` or `transport_tsa_moment`.

The physical training loss used in the physical-only runs is

`L_phys(theta) = mean_i rho_phys(r_i)^2`

over the residual samples in the current training batch.  Therefore the
physical loss is the mean squared one-step true residual ratio on the training
residual batch.

It is not numerically identical to the reported `one_step_true` column unless
the batch contains exactly the same residual used for evaluation and no other
loss terms are active.  In the ablation runner, `one_step_true` is evaluated on
the train source residual or heldout source residual from the zero initial
state.  The training loss is evaluated on sampled residual batches containing
rollout residuals and random probe residuals, often from auxiliary sources.
Also, the loss is squared and averaged, while `one_step_true` is an unsquared
single-ratio diagnostic.

The general ablation objective is

`metric_weight * mean rho_base^2 + physical_weight * mean rho_phys^2`

plus optional correction, rollout, stability, and teacher terms.  Thus only
runs with `metric_weight=0`, `physical_weight=1`, `correction_weight=0`,
`rollout_weight=0`, `stability_weight=0`, and `teacher_weight=0` are pure
physical one-step residual training.

## Evidence From Existing Size-64 Runs

### Fixed Moment-Space Direction Tests

`npbs_unet_ablation_moment_structure_physical_size64_steps1800.csv` and
`npbs_unet_ablation_mode_direction_cosine_mixed_size64_steps3000.csv` compare
the structured moment models with raw one-step physical residual losses.

For the physical-loss size-64 moment-structure run, the one-step residual on
the training source and on the heldout source was:

| Case | Method | Train One-Step | Heldout One-Step | Heldout Stat | Heldout FGMRES | Loss Change |
|---|---:|---:|---:|---:|---:|---:|
| anisotropic | `tsa_moment1-w0.4` | `0.6564` | `0.6559` | `1.11e-5` | `1.27e-5` | |
| anisotropic | `tsa_moment4-w0.4` | `0.2613` | `0.2604` | `1.22e-5` | `1.25e-5` | |
| anisotropic | `CoeffWeightedTSA-M4` | `0.6255` | `0.6249` | `1.04e-5` | `1.29e-5` | `+0.5%` |
| anisotropic | `CoeffMixedTSA-M4` | `0.6034` | `0.6066` | `1.06e-5` | `1.24e-5` | `+0.6%` |
| anisotropic | `LearnedRP-TSA-M4` | `0.5693` | `0.5753` | `1.06e-5` | `1.30e-5` | `-6.9%` |
| anisotropic | `ResidualSubspaceTSA-M4` | `0.5839` | `0.5988` | `9.78e-6` | `1.25e-5` | `-8.8%` |
| mixed | `tsa_moment1-w0.4` | `0.6948` | `0.6476` | `2.89e-3` | `1.97e-5` | |
| mixed | `tsa_moment4-w0.4` | `0.2990` | `0.2440` | `1.30e-5` | `7.32e-6` | |
| mixed | `CoeffWeightedTSA-M4` | `0.6479` | `0.6024` | `2.96e-3` | `1.93e-5` | `-2.3%` |
| mixed | `CoeffMixedTSA-M4` | `0.5720` | `0.5910` | `3.41e-3` | `4.29e-5` | `-3.8%` |
| mixed | `LearnedRP-TSA-M4` | `0.5186` | `0.5000` | `3.42e-3` | `3.21e-4` | `-2.9%` |
| mixed | `ResidualSubspaceTSA-M4` | `0.5865` | `0.5879` | `1.32e-1` | `1.23e-4` | `-2.6%` |

This table is the most important structured-model evidence.  The learned
models do improve the training-source one-step residual.  For the mixed case,
`LearnedRP-TSA-M4` improves train one-step from `0.6948` for one damped moment
step to `0.5186`, and heldout one-step to `0.5000`.  But it is still far from
the deeper stable classical `tsa_moment4-w0.4` one-step values
`0.2990`/`0.2440`, and its heldout FGMRES is much worse.

| Case | Method | Loss Change | Heldout One-Step | Heldout FGMRES | Interpretation |
|---|---:|---:|---:|---:|---|
| anisotropic | `CoeffWeightedTSA-M4` | `+0.5%` to `-3.0%` | `0.625` | `~1.2e-5` | Mostly saturated diagonal control. |
| anisotropic | `CoeffMixedTSA-M4` | `+0.6%` to `-3.5%` | `0.607` | `~1.2e-5` | Mode coupling helps one-step mildly. |
| anisotropic | `LearnedRP-TSA-M4` | `-6.9%` | `0.575` | `1.30e-5` | Best stable structured candidate. |
| anisotropic | `ResidualSubspaceTSA-M4` | `-8.8%` | `0.599` | `1.25e-5` | Strong loss movement, stable here. |
| mixed | `CoeffWeightedTSA-M4` | `-0.1%` to `-3.9%` | `0.600-0.602` | `~2e-5` | Small improvement only. |
| mixed | `CoeffMixedTSA-M4` | `-1.0%` to `-3.8%` | `0.588-0.591` | `4.2e-5` | Better one-step, worse Krylov. |
| mixed | `LearnedRP-TSA-M4` | `-2.6%` to `-3.9%` | `0.500-0.506` | `2.6e-4` to `3.2e-4` | Largest one-step gain, unstable for FGMRES. |
| mixed | `ResidualSubspaceTSA-M4` | `-2.6%` to `-3.8%` | `0.587` | `1.2e-4` to `1.7e-4` | Loss moves but solver instability is severe. |

The fixed classical baselines on the same mixed size-64 moment problem are much
stronger in one-step residual:

- `tsa_moment1-w0.4`: `0.648`
- `tsa_moment2-w0.4`: `0.438`
- `tsa_moment3-w0.4`: `0.315`
- `tsa_moment4-w0.4`: `0.244`
- undamped `tsa_moment4`: `0.136` one-step, but stationary iteration explodes.

This is the core bottleneck: the learned structured moment models improve over
one damped moment step, but they do not yet approximate the deeper stable
classical correction.

### Learned R/P With Rollout And Stability Losses

The longer `LearnedRP-TSA-M4` runs with rollout/stability terms did not fix the
mixed-case Krylov issue.  The training-source one-step residual remained good
relative to `CoeffWeightedTSA`, but not competitive with deeper classical
moment TSA:

| Run | Case | Train One-Step | Heldout One-Step | Heldout Stat | Heldout FGMRES | Loss Change |
|---|---|---:|---:|---:|---:|---:|
| rollout/stability | anisotropic | `0.5891` | `0.5920` | `1.04e-5` | `1.27e-5` | `-1.4%` |
| physical+rollout/stability | anisotropic | `0.5768` | `0.5826` | `1.06e-5` | `1.27e-5` | `-4.1%` |
| rollout/stability | mixed | `0.5165` | `0.5059` | `2.76e-3` | `2.58e-4` | `-3.9%` |
| physical+rollout/stability | mixed | `0.5120` | `0.5056` | `3.13e-3` | `2.85e-4` | `-2.6%` |

This means the current rollout/stability penalties are not enough.  They
penalize some local contraction behavior, but they do not force the learned
operator to have the right spectrum as a preconditioner inside FGMRES.

### Generic Defect Network Evidence

The strongest positive evidence for learning is from the full TSA U-Net defect
runs, especially `npbs_unet_ablation_tsa_full_physical_steps900.csv`.
This run is pure physical-loss training:

- `metric_weight = 0`
- `physical_weight = 1`
- `correction_weight = 0`
- `rollout_weight = 0`
- `stability_weight = 0`
- `teacher_weight = 0`

The base preconditioner is full angular `transport_tsa` with
`tsa_harmonics=None`; `UNet-defect2` costs two base applications per call.

On mixed size 64 with full angular `transport_tsa`:

- `tsa2`: train/heldout one-step `0.1719`/`0.1790`, FGMRES `7.60e-6`
- `tsa4`: train/heldout one-step `0.0981`/`0.1145`, FGMRES `7.94e-6`
- `UNet-defect2-w1`: loss `0.2418 -> 0.0724` (`-70.1%`), train/heldout
  one-step `0.1659`/`0.1245`, stationary residual `7.82e-6`, FGMRES `7.54e-6`

On anisotropic size 64:

- `tsa2`: train/heldout one-step `0.0825`/`0.0824`
- `tsa4`: train/heldout one-step `0.0365`/`0.0365`
- `UNet-defect2-w1`: loss `0.0945 -> 0.00170` (`-98.2%`), train/heldout
  one-step `0.0861`/`0.0699`, FGMRES `1.27e-5`

The full physical-loss-trained `UNet-defect2` table is:

| Size | Case | Split | Loss Change | `one_step_true` | `one_step_base_metric` | `one_step_born_metric` | Stat Final | FGMRES Final |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 32 | anisotropic | train | `0.0945 -> 0.000946` | `0.0686` | `0.0604` | `0.0575` | `5.53e-6` | `5.38e-6` |
| 32 | anisotropic | heldout | `0.0945 -> 0.000946` | `0.0571` | `0.0639` | `0.0520` | `5.87e-6` | `5.97e-6` |
| 32 | mixed | train | `0.2389 -> 0.0687` | `0.1572` | `0.1166` | `0.1474` | `1.39e-4` | `3.50e-6` |
| 32 | mixed | heldout | `0.2389 -> 0.0687` | `0.1356` | `0.1379` | `0.1455` | `1.59e-4` | `3.39e-6` |
| 64 | anisotropic | train | `0.0945 -> 0.00170` | `0.0861` | `0.0647` | `0.0668` | `1.25e-5` | `1.22e-5` |
| 64 | anisotropic | heldout | `0.0945 -> 0.00170` | `0.0699` | `0.0685` | `0.0595` | `1.34e-5` | `1.27e-5` |
| 64 | mixed | train | `0.2418 -> 0.0724` | `0.1659` | `0.1203` | `0.1536` | `7.84e-6` | `7.54e-6` |
| 64 | mixed | heldout | `0.2418 -> 0.0724` | `0.1245` | `0.1286` | `0.1330` | `7.82e-6` | `7.54e-6` |

The loss values should not be read as the square of the displayed train-source
`one_step_true`.  For example, mixed size 64 has final physical loss `0.0724`,
whose square root is about `0.269`.  The displayed train-source one-step value
is `0.1659` because the loss was averaged over a broader sampled residual set,
not only the train source residual.

This is important: a learned residual-dependent map can nearly match deeper
classical TSA at two base-call cost when the base operator carries the full
angular space.  The failure is not simply "neural networks cannot help RTE."
The failure is that the current structured moment models are too constrained
or their losses do not enforce the stable deep correction in the hard mixed
regime.

## Small CPU Push Experiment

After wiring the bottleneck harness to expose `CoeffMixedTSA` and
`ResidualSubspaceTSA`, a small mixed size-16 CPU test was run with
`raw_rollout_teacher`, two rollout steps, stability penalty, and a damped
four-step TSA-moment teacher.  This was only a laptop-scale diagnostic.

| Method | Loss Change | Raw Ratio Change | Teacher Ratio Change | Heldout One-Step | Heldout FGMRES |
|---|---:|---:|---:|---:|---:|
| `CoeffMixedTSA-M4` | `0.8667 -> 0.8392` | `0.7076 -> 0.6917` | `0.6271 -> 0.6127` | `0.5968` | `2.81e-3` |
| `LearnedRP-TSA-M4` | `0.8668 -> 0.8188` | `0.7076 -> 0.6773` | `0.6270 -> 0.5996` | `0.5397` | `5.47e-3` |
| `ResidualSubspaceTSA-M4` | `0.8665 -> 0.8454` | `0.7075 -> 0.6949` | `0.6268 -> 0.6138` | `0.6063` | `3.56e-3` |

The best training movement and one-step movement again comes from learned R/P.
However, checkpointing by heldout FGMRES selected the initial model for all
three candidates, confirming the same stability/objective mismatch seen at
size 64.

## Performance Interpretation

The current experiments separate three effects:

1. Diagonal coefficient conditioning is too weak.  `CoeffWeightedTSA-M4`
   barely moves the one-step residual in the hard mixed case.  Increasing its
   width did not materially change that pattern in the recorded runs.
2. Mode coupling and learned R/P help the one-step residual.  On mixed size 64,
   `CoeffMixedTSA-M4` improves train one-step from `0.6479` to `0.5720`
   relative to the diagonal model, and `LearnedRP-TSA-M4` improves further to
   `0.5186`.  This is real evidence that learning mode directions is useful.
3. One-step residual improvement is not enough.  The same `LearnedRP-TSA-M4`
   mixed run gives heldout one-step `0.5000`, but FGMRES worsens to
   `3.21e-4` compared with `7.32e-6` for `tsa_moment4-w0.4`.  Therefore the
   learned operator is reducing the norm of the first residual while damaging
   the spectrum/non-normal behavior seen by Krylov iteration.

The train versus heldout one-step numbers are similar.  That argues against a
simple generalization failure.  The dominant failure is operator quality: the
learned map is not yet a stable approximation to a deeper synthetic
acceleration operator.

## Current Ranking

1. `LearnedRP-TSA-M4` is the most promising structured moment candidate.  It
   has the clearest and most repeated one-step improvement signal.  The next
   issue is stability and solver alignment, not raw trainability.
2. `ResidualSubspaceTSA-M4` is high-upside but currently unsafe.  It decreases
   loss, especially on anisotropic cases, but the mixed case shows large
   stationary and FGMRES degradation.
3. `CoeffMixedTSA-M4` is a useful low-risk ablation for mode coupling.  It
   improves over diagonal weights but appears too weak by itself.
4. `CoeffWeightedTSA-M4` should remain a control, not the main direction.
5. Full-angular `UNet-defect2` is the proof-of-concept that residual-dependent
   learned corrections can work.  The promising next architecture should borrow
   its residual conditioning but constrain it through moment/subspace structure.

## Recommended Next Designs

The current `LearnedRP-TSA-M4` still learns inside a fixed retained Fourier
span.  To truly learn better modes, the next architecture should use a larger
candidate angular space and learn a low-dimensional subspace inside it:

1. Choose candidate harmonics `Kcand > Ksolve`, for example `Ksolve=4`,
   `Kcand=6` or `8`.
2. Encode coefficients and residual summaries.
3. Predict a bounded, near-identity mixing map from retained solve coordinates
   into candidate coordinates.
4. Preserve the scalar moment exactly and enforce conjugate symmetry where
   possible.
5. Train with raw one-step physical residual plus rollout stability and a
   teacher term to stable damped `tsa_moment4-w0.4` or full `tsa4`.
6. Select checkpoints by heldout FGMRES or heldout stationary residual, not by
   training loss.

The most useful immediate experiments are:

- `LearnedRP-TSA-M4`: raw physical residual only versus raw rollout/stability
  versus raw rollout plus damped deep teacher, with checkpoint selection.
- `ResidualSubspaceTSA-M4`: repeat with stronger damping, smaller
  `moment_weight_scale`, and checkpointing by heldout FGMRES.
- New candidate-space learned R/P: compare against `tsa_moment4-w0.4`,
  `tsa_moment6-w0.4`, and full-angular `UNet-defect2`.

The success criterion should be strict: a candidate must lower heldout one-step
error without making stationary iteration or FGMRES worse.  Large training-loss
decrease alone is not enough.
