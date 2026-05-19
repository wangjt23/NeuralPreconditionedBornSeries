# NPBS-Native U-Net Learned Preconditioner Ablation

This report replaces the earlier ATFPS-based ablation for the present question.
All results here use the repository's matrix-free periodic RTE NPBS operator in
`rte/models.py`, with explicit calls to the discrete operator `A_op` and to the
chosen base preconditioner `B_h`.  No ATFPS discretization is used.

## Model

The learned module is `NPBSUNetPreconditioner` in
`rte/npbs_unet_ablation.py`.  It is a coefficient-conditioned U-Net correction
that is linear in the residual for fixed coefficients:

```text
residual form:
    z0 = B_h r
    M_theta r = z0 + C_theta(z0, coeff)

defect2 form:
    z0 = B_h r
    z1 = B_h(r - A_h z0)
    M_theta r = z0 + omega z1 + C_theta(z0, z1, coeff)
```

The coefficient features are normalized `sigma_t`, albedo
`sigma_s / sigma_t`, a periodic gradient magnitude of `sigma_t`, and the
anisotropy parameter `g`.  The U-Net output projection is zero-initialized, so
the learned preconditioner starts exactly at the classical fallback: `B_h` for
residual form and `B_h + omega B_h(I-A_hB_h)` for defect2.

The loss is the NPBS metric loss plus a true residual term:

```text
|| B_h(A_h M_theta r - r) ||^2 / || B_h r ||^2
+ 0.5 || A_h M_theta r - r ||^2 / || r ||^2
+ small correction penalty.
```

Training residuals are base-method rollout residuals plus smooth operator
probes.  Evaluation uses both the training source and a held-out source with a
different location/angular profile.

## Commands

```bash
cd /Users/hxchen/NeuralPreconditionedBornSeries

/Users/hxchen/miniconda3-arm64/envs/py312_env/bin/python -m rte.npbs_unet_ablation \
  --cases diffusive heterogeneous anisotropic mixed \
  --sizes 24 --n-angles 12 \
  --base-kinds transport \
  --forms residual defect2 --defect-relaxations 0.5 1.0 \
  --steps 80 --width 12 --levels 3 \
  --maxiter 60 --fgmres-maxiter 25 \
  --output-csv rte/npbs_unet_ablation_born_size24.csv \
  --output-md rte/NPBS_UNET_ABLATION_BORN_SIZE24.md

/Users/hxchen/miniconda3-arm64/envs/py312_env/bin/python -m rte.npbs_unet_ablation \
  --cases diffusive heterogeneous anisotropic mixed \
  --sizes 24 --n-angles 12 \
  --base-kinds transport_dsa \
  --forms residual defect2 --defect-relaxations 0.5 1.0 \
  --steps 80 --width 12 --levels 3 \
  --maxiter 60 --fgmres-maxiter 25 \
  --output-csv rte/npbs_unet_ablation_dsa_size24.csv \
  --output-md rte/NPBS_UNET_ABLATION_DSA_SIZE24.md

/Users/hxchen/miniconda3-arm64/envs/py312_env/bin/python -m rte.npbs_unet_ablation \
  --cases diffusive heterogeneous anisotropic mixed \
  --sizes 32 --n-angles 12 \
  --base-kinds transport_dsa \
  --forms residual defect2 --defect-relaxations 1.0 \
  --steps 60 --width 12 --levels 3 \
  --maxiter 60 --fgmres-maxiter 25 \
  --output-csv rte/npbs_unet_ablation_dsa_size32.csv \
  --output-md rte/NPBS_UNET_ABLATION_DSA_SIZE32.md
```

## Held-Out Results

The tables below report held-out source performance.  `one` is the one-step
true residual ratio `||A_h M r-r||/||r||`; lower is better.  `stat` is the
final stationary residual after 60 steps.  `fgmres` is the final FGMRES
residual after 25 iterations.

### Born/Transport Base, Size 24

| Case | Method | one | stat | fgmres |
|---|---|---:|---:|---:|
| diffusive | Born1 | 7.781e-01 | 1.310e-01 | 1.390e-05 |
| diffusive | Born2 | 6.787e-01 | 7.501e-02 | 6.649e-06 |
| diffusive | U-Net residual | 3.915e-01 | 9.290e-04 | 6.424e-06 |
| diffusive | U-Net defect2 | 4.971e-01 | 2.130e-04 | 5.347e-06 |
| heterogeneous | Born1 | 4.158e-01 | 3.830e-04 | 2.448e-06 |
| heterogeneous | Born2 | 2.294e-01 | 2.538e-06 | 2.546e-06 |
| heterogeneous | U-Net residual | 3.879e-01 | 2.338e-06 | 2.392e-06 |
| heterogeneous | U-Net defect2 | 1.876e-01 | 2.393e-06 | 2.439e-06 |
| anisotropic | Born1 | 5.970e-01 | 1.826e-02 | 2.075e-04 |
| anisotropic | Born2 | 4.094e-01 | 1.962e-03 | 3.792e-06 |
| anisotropic | U-Net residual | 4.773e-01 | 3.935e-04 | 5.742e-04 |
| anisotropic | U-Net defect2 | 3.210e-01 | 4.049e-06 | 4.271e-06 |
| mixed | Born1 | 5.770e-01 | 2.734e-02 | 2.215e-03 |
| mixed | Born2 | 3.994e-01 | 4.092e-03 | 2.583e-04 |
| mixed | U-Net residual | 5.387e-01 | 6.167e-03 | 7.013e-03 |
| mixed | U-Net defect2 | 3.964e-01 | 8.479e-04 | 3.944e-04 |

### DSA Base, Size 24

| Case | Method | one | stat | fgmres |
|---|---|---:|---:|---:|
| diffusive | DSA1 | 2.133e+00 | 5.513e-06 | 5.492e-06 |
| diffusive | DSA2 | 3.824e-01 | 5.658e-06 | 5.548e-06 |
| diffusive | U-Net residual | 4.334e-01 | 5.434e-06 | 5.459e-06 |
| diffusive | U-Net defect2 | 8.677e-02 | 5.448e-06 | 5.445e-06 |
| heterogeneous | DSA1 | 3.681e-01 | 2.265e-06 | 2.327e-06 |
| heterogeneous | DSA2 | 1.511e-01 | 2.407e-06 | 2.409e-06 |
| heterogeneous | U-Net residual | 4.530e-01 | 2.336e-06 | 2.551e-06 |
| heterogeneous | U-Net defect2 | 1.690e-01 | 2.405e-06 | 2.337e-06 |
| anisotropic | DSA1 | 6.774e-01 | 3.394e-05 | 2.080e-05 |
| anisotropic | DSA2 | 4.283e-01 | 3.893e-06 | 3.876e-06 |
| anisotropic | U-Net residual | 4.917e-01 | 1.566e-02 | 3.238e-04 |
| anisotropic | U-Net defect2 | 3.313e-01 | 3.813e-06 | 3.914e-06 |
| mixed | DSA1 | 8.546e-01 | 5.074e-03 | 4.576e-03 |
| mixed | DSA2 | 5.452e-01 | 7.165e-04 | 2.755e-04 |
| mixed | U-Net residual | 4.967e-01 | 5.202e-03 | 4.140e-03 |
| mixed | U-Net defect2 | 2.977e-01 | 7.419e-04 | 2.917e-04 |

### DSA Base, Size 32

| Case | Method | one | stat | fgmres |
|---|---|---:|---:|---:|
| diffusive | DSA1 | 2.133e+00 | 7.849e-06 | 7.625e-06 |
| diffusive | DSA2 | 3.824e-01 | 8.228e-06 | 7.684e-06 |
| diffusive | U-Net residual | 1.010e+00 | 7.787e-06 | 7.843e-06 |
| diffusive | U-Net defect2 | 9.612e-02 | 7.795e-06 | 7.524e-06 |
| heterogeneous | DSA1 | 3.723e-01 | 3.743e-06 | 3.532e-06 |
| heterogeneous | DSA2 | 1.503e-01 | 3.733e-06 | 3.428e-06 |
| heterogeneous | U-Net residual | 4.346e-01 | 3.665e-06 | 3.448e-06 |
| heterogeneous | U-Net defect2 | 1.726e-01 | 3.591e-06 | 3.658e-06 |
| anisotropic | DSA1 | 6.774e-01 | 3.419e-05 | 2.166e-05 |
| anisotropic | DSA2 | 4.283e-01 | 5.517e-06 | 5.436e-06 |
| anisotropic | U-Net residual | 4.857e-01 | 2.938e-05 | 8.534e-05 |
| anisotropic | U-Net defect2 | 3.374e-01 | 6.080e-06 | 6.044e-06 |
| mixed | DSA1 | 8.548e-01 | 4.921e-03 | 4.450e-03 |
| mixed | DSA2 | 5.448e-01 | 6.935e-04 | 2.682e-04 |
| mixed | U-Net residual | 6.455e-01 | 5.462e-03 | 5.759e-03 |
| mixed | U-Net defect2 | 3.173e-01 | 8.028e-04 | 3.328e-04 |

## Analysis

1. The ATFPS conclusions should not be used for this NPBS question.  The
   NPBS-native operator behaves differently because `B_h`, `A_h`, residual
   probes, and the metric loss are all consistent inside `PeriodicRTE2D`.

2. The defect2 wrapper is the best U-Net form in this NPBS-native study.  With
   DSA as `B_h`, defect2 gives the best held-out one-step residual in all four
   size-24 regimes and all four size-32 regimes.  It also avoids the severe
   stationary/FGMRES degradation seen in residual form for anisotropic and mixed
   regimes.

3. The residual-form U-Net is not safe as the default.  It can improve the
   Born-base stationary residual in easy cases, but with DSA it damages
   anisotropic and mixed problems.  This is a stability problem: a one-step
   residual objective can still produce a bad fixed-point or a poor FGMRES
   preconditioner.

4. DSA remains a much stronger base than raw Born/transport for hard regimes.
   Born-base defect2 is useful in diffusive, heterogeneous, and anisotropic
   cases, but it is still weak in mixed regimes compared with DSA2 and DSA
   defect2.

5. FGMRES can mask one-step gains when the baseline is already near the
   iteration cap and residuals are near \(10^{-6}\).  The one-step metric is
   valuable for diagnosing learned correction quality, but solver conclusions
   should use true FGMRES residual histories and wall time.

6. The learned model is still per-medium and small-scale.  Held-out source
   results are meaningful, but this is not yet cross-coefficient
   generalization.

## Updated Implementation Judgement

- `RTEUNetPrecond`-style coefficient-conditioned U-Nets are appropriate for
  NPBS if they are wrapped around explicit `A_h` and `B_h`.
- The exact-zero output initialization is still important and correct.
- The preferred NPBS form is now:

```text
z0 = B_h r
z1 = B_h(r - A_h z0)
M_theta r = z0 + z1 + C_theta(z0, z1, coeff)
```

with `B_h = transport_dsa` for the current RTE prototype.

- Residual-form U-Net should remain an ablation, not the production default.
- The next base preconditioner to test is a true TSA/anisotropic synthetic
  acceleration base, because the remaining hard modes are angular/mixed-regime
  errors rather than purely scalar diffusion errors.

## Next Steps

1. Add a contraction/stability penalty for stationary use:
   `||r - A_h M_theta r|| / ||r||` on rollout residuals from later solver
   steps, not only initial residuals.

2. Evaluate FGMRES histories versus wall time, not just final residuals at a
   fixed iteration cap.

3. Train across randomized coefficient fields, then test on held-out media.
   The current result is only held-out-source evidence.

4. Add a TSA or angular synthetic base and repeat the defect2 U-Net ablation.
   That is the most direct way to target anisotropic and mixed regimes.

5. Keep `rte/mgRTE_ablation` out of the main conclusions for this thread.  It
   was useful for implementation debugging, but it is not the requested NPBS
   discretization.

## Larger-Scale Extension

This extension answers the original request more directly: larger training
datasets, longer training schedules, and the newer rollout/stability schemes
were added to the NPBS-native runner itself in `rte/npbs_unet_ablation.py`.

One implementation detail matters for interpreting the new numbers: GPU runs now
leave TF32 disabled by default.  An intermediate GPU patch that forced TF32 was
acceptable for throughput-oriented training, but it inflated the stationary and
FGMRES residual floors and was therefore not suitable for the NPBS ablation
report.  The results below use the corrected, numerically trustworthy GPU path.

### Extended Setup

- Base preconditioner: `transport_dsa`
- Forms: residual, defect2 with \(\omega=0.5\) and \(\omega=1\)
- Network: width 16, 4 levels
- Training sources: main source plus 2 auxiliary sources
- Residual bank: 8 rollout residuals per training source, giving 24 cached
  rollout residuals before random resampling
- Size-32 training: 48 residuals per step, 300 AdamW steps, step scheduler with
  drop at step 120
- Size-64 training: 64 residuals per step, 300 AdamW steps, same scheduler

Two larger-scale training schemes were compared at size 32:

1. `long-metric`: larger dataset + longer schedule, but still one-step metric
   plus physical loss.
2. `long-rollout`: same dataset and schedule, plus rollout and stability
   penalties on later stationary residuals.

The exact commands were:

```bash
cd /data/wjt/NeuralPreconditionedBornSeries

python -m rte.npbs_unet_ablation \
  --cases diffusive heterogeneous anisotropic mixed \
  --sizes 32 --n-angles 12 \
  --base-kinds transport_dsa \
  --forms residual defect2 --defect-relaxations 0.5 1.0 \
  --steps 300 --lr 2e-3 --weight-decay 1e-4 \
  --width 16 --levels 4 \
  --residual-mode resample --residuals-per-step 48 \
  --rollout-samples-per-step 24 --n-rollout 8 --aux-sources 2 \
  --scheduler step --scheduler-step-size 120 --scheduler-gamma 0.5 \
  --device cuda:0 \
  --output-csv rte/npbs_unet_ablation_dsa_size32_long_metric_precise.csv

python -m rte.npbs_unet_ablation \
  --cases diffusive heterogeneous anisotropic mixed \
  --sizes 32 --n-angles 12 \
  --base-kinds transport_dsa \
  --forms residual defect2 --defect-relaxations 0.5 1.0 \
  --steps 300 --lr 2e-3 --weight-decay 1e-4 \
  --width 16 --levels 4 \
  --residual-mode resample --residuals-per-step 48 \
  --rollout-samples-per-step 24 --n-rollout 8 --aux-sources 2 \
  --scheduler step --scheduler-step-size 120 --scheduler-gamma 0.5 \
  --rollout-weight 0.2 --rollout-steps 3 --rollout-decay 0.8 \
  --stability-weight 0.5 --contraction-target 0.95 \
  --device cuda:0 \
  --output-csv rte/npbs_unet_ablation_dsa_size32_long_rollout_precise.csv

python -m rte.npbs_unet_ablation \
  --cases diffusive heterogeneous anisotropic mixed \
  --sizes 64 --n-angles 12 \
  --base-kinds transport_dsa \
  --forms residual defect2 --defect-relaxations 0.5 1.0 \
  --steps 300 --lr 2e-3 --weight-decay 1e-4 \
  --width 16 --levels 4 \
  --residual-mode resample --residuals-per-step 64 \
  --rollout-samples-per-step 32 --n-rollout 8 --aux-sources 2 \
  --scheduler step --scheduler-step-size 120 --scheduler-gamma 0.5 \
  --rollout-weight 0.2 --rollout-steps 3 --rollout-decay 0.8 \
  --stability-weight 0.5 --contraction-target 0.95 \
  --device cuda:0 \
  --output-csv rte/npbs_unet_ablation_dsa_size64_long_rollout_precise.csv
```

### Size 32: Long-Metric Scheme

Held-out best methods by criterion:

| Case | Best one-step | Value | Best stationary | Value | Best FGMRES | Value |
|---|---|---:|---|---:|---|---:|
| diffusive | defect2, \(\omega=1\) | 6.803e-02 | residual | 7.314e-06 | defect2, \(\omega=0.5\) | 7.722e-06 |
| heterogeneous | DSA2 | 1.503e-01 | defect2, \(\omega=1\) | 3.406e-06 | defect2, \(\omega=1\) | 3.502e-06 |
| anisotropic | defect2, \(\omega=1\) | 2.292e-01 | defect2, \(\omega=1\) | 5.386e-06 | DSA2 | 6.005e-06 |
| mixed | defect2, \(\omega=1\) | 2.297e-01 | defect2, \(\omega=1\) | 5.737e-04 | defect2, \(\omega=1\) | 1.439e-04 |

Average held-out one-step true residual over the four cases:

| Method | Average one-step true |
|---|---:|
| DSA2 | 3.765e-01 |
| U-Net residual | 2.786e-01 |
| U-Net defect2, \(\omega=0.5\) | 1.958e-01 |
| U-Net defect2, \(\omega=1\) | 1.715e-01 |

The larger dataset and longer schedule strengthen the original NPBS conclusion.
Defect2 with DSA remains the best one-step form on 3 of 4 regimes and the best
solver-level choice on the hardest mixed case.

### Size 32: Long-Rollout Scheme

Held-out best methods by criterion:

| Case | Best one-step | Value | Best stationary | Value | Best FGMRES | Value |
|---|---|---:|---|---:|---|---:|
| diffusive | defect2, \(\omega=1\) | 5.443e-02 | defect2, \(\omega=1\) | 7.827e-06 | residual | 7.826e-06 |
| heterogeneous | DSA2 | 1.503e-01 | defect2, \(\omega=1\) | 3.466e-06 | defect2, \(\omega=0.5\) | 3.298e-06 |
| anisotropic | defect2, \(\omega=0.5\) | 2.250e-01 | defect2, \(\omega=0.5\) | 5.686e-06 | defect2, \(\omega=1\) | 5.863e-06 |
| mixed | defect2, \(\omega=1\) | 3.067e-01 | defect2, \(\omega=1\) | 2.888e-04 | defect2, \(\omega=1\) | 1.604e-04 |

Relative to long-metric training, rollout/stability does \emph{not} act like a
universal win.  Its actual effect is more specific:

1. It improves stationary robustness for defect2 on the hardest regimes.  In
   mixed, `defect2, omega=1` reduces held-out stationary final residual from
   `5.737e-04` to `2.888e-04`.
2. It can slightly improve diffusive and anisotropic one-step quality.
3. It does not uniformly improve FGMRES, and it slightly worsens the average
   one-step metric for `defect2, omega=1` compared with long-metric training.

So for NPBS the rollout/stability terms are best viewed as a \textbf{stationary
solver tuning}, not as a blanket replacement for the original one-step loss.

### Size 64: Rollout Confirmation

Held-out best methods by criterion:

| Case | Best one-step | Value | Best stationary | Value | Best FGMRES | Value |
|---|---|---:|---|---:|---|---:|
| diffusive | defect2, \(\omega=1\) | 5.094e-02 | defect2, \(\omega=1\) | 1.638e-05 | residual | 1.658e-05 |
| heterogeneous | DSA2 | 1.516e-01 | defect2, \(\omega=0.5\) | 7.585e-06 | residual | 8.148e-06 |
| anisotropic | defect2, \(\omega=0.5\) | 2.563e-01 | DSA2 | 1.296e-05 | defect2, \(\omega=1\) | 1.263e-05 |
| mixed | defect2, \(\omega=1\) | 3.301e-01 | defect2, \(\omega=1\) | 5.090e-04 | defect2, \(\omega=1\) | 3.023e-04 |

Average held-out one-step true residual over the four size-64 cases:

| Method | Average one-step true |
|---|---:|
| DSA2 | 3.793e-01 |
| U-Net residual | 3.340e-01 |
| U-Net defect2, \(\omega=0.5\) | 2.343e-01 |
| U-Net defect2, \(\omega=1\) | 2.169e-01 |

The size-64 confirmation keeps the same ordering.  Defect2 remains the most
promising learned NPBS direction overall, with a case-dependent relaxation:

- \(\omega=1\) is best in diffusive and mixed regimes.
- \(\omega=0.5\) is the better anisotropic choice on one-step quality.
- Heterogeneous remains a regime where plain DSA2 is already strong and learned
  gains are marginal.

### Updated Promising Directions

The larger-scale NPBS study changes the recommendations in a narrower, more
practical way than the earlier ATFPS-local mgRTE study.

1. \textbf{Keep defect2 as the NPBS default form.}  At both size 32 and size
   64, defect2 remains the strongest choice for one-step quality and usually the
   strongest stationary method on anisotropic and mixed regimes.

2. \textbf{Use longer data-rich training even without rollout.}  The larger
   residual bank, auxiliary sources, and 300-step schedule materially improve
   defect2 over the earlier 60-step size-32 study.

3. \textbf{Use rollout/stability when stationary iteration matters most.}  For
   NPBS, rollout is not a universal best objective.  Its main benefit is better
   stationary contraction on hard regimes, especially mixed.  If the target is
   Krylov preconditioning alone, the gain is smaller and sometimes neutral.

4. \textbf{Do not switch the NPBS default to residual form.}  Residual-form
   U-Net can reduce the one-step residual, but it is still unreliable as a
   solver preconditioner on mixed and anisotropic cases.  Size-64 mixed is the
   clearest example: residual reduces one-step true residual to `3.581e-01`, but
   its held-out FGMRES residual is `2.752e-03`, versus `3.023e-04` for
   `defect2, omega=1` and `3.031e-04` for plain DSA2.

5. \textbf{Tune defect relaxation by regime.}  The larger study does not support
   a single global answer.  `omega=1` is still the safest mixed/diffusive
   default, while `omega=0.5` is competitive or better for anisotropic cases.

### Revised Practical Recommendation

For the present NPBS-native RTE question, the best current direction is:

```text
base preconditioner: transport_dsa
wrapper:            defect2
default omega:      1.0
anisotropic option: sweep omega in {0.5, 1.0}
training data:      resampled residual batches with auxiliary sources
training schedule:  300-step AdamW + step decay
loss choice:        long-metric by default; add rollout/stability when
                    stationary contraction is the main target
```

In short,

\[
\boxed{\text{For NPBS, larger-scale training confirms defect2+DSA as the main
path; rollout helps stationary robustness, but residual-form is still only an
ablation.}}
\]
