# Quantum-PDF-Harness Audit: Helmholtz and RTE Learned Preconditioners

Date: 2026-08-04

This is a strict code-and-mathematics audit of the current project. The
external `quantum-pde-harness` is used for source hashing, claim-status
validation, and append-only evidence recording. It is not a numerical solver
and it does not replace the native RTE/Helmholtz tests. Numerical observations
below were rerun with the repository's PyTorch environment.

## Executive verdict

The previously identified issues are partly correct, but they are not all
equally important.

The RTE learned-preconditioner failure is not explained by a fundamental bug
in `A_h`, the periodic Fourier transport operator, or the FGMRES evaluator.
Those pieces pass independent algebraic and dense-matrix checks. The dominant
failure is a mismatch between the learned map, the training objective, and the
operator that matters inside a Krylov method:

1. The learned correction is trained to reduce an average one-step residual on
   a sampled residual distribution. FGMRES needs a stable action of the
   preconditioned operator over the whole evolving Krylov subspace, especially
   for the non-normal mixed RTE operator.
2. The moment models can only act in the fixed angular space
   (Q_K=operatorname{span}{0,\pm1,\ldots,\pm K}). Learned weights, mode
   mixing, and learned restriction/prolongation inside (Q_K) cannot represent
   a correction that requires higher angular modes.
3. The full-angular U-Net has a severe channel bottleneck in the representative
   settings: a defect2 input has (4n_\mathrm{angles}) real channels, but the
   residual path is lifted to `width` channels. At 12 angles and width 16 this
   compresses 48 real channels to 16 before the correction is formed.
4. The RTE operator is periodic, but only the ordinary convolutions in the
   learned correction use circular padding. Transposed convolutions and the
   resize padding use zero-boundary behavior. The learned correction therefore
   does not preserve the spatial symmetry of the operator.
5. The zero-output initialization is mathematically useful because it preserves
   the classical starting preconditioner exactly, but it gives a first-step
   gradient to only the final projection. This slows the first phase of
   optimization. In a small control, changing to small initialization did not
   materially fix the solver result, so this is a real optimization issue but
   not the main scientific bottleneck.

The strongest next direction is a structure-preserving, teacher-compressed
angular preconditioner: preserve the torus discretization, expose a candidate
angular space larger than the solve space, learn bounded restriction,
prolongation, and mode mixing around a stable classical teacher, and select
models by held-out FGMRES/time-to-tolerance rather than one-step loss alone.

## 1. Mathematical objects and exact code paths

### 1.1 Periodic RTE prototype

The current RTE code uses a periodic discrete-ordinates system with
\(\psi\in\mathbb C^{n_\Omega\times N\times N}\):

\[
 A_h\psi = S_h\psi+\sigma_t\psi-\sigma_s P_h\psi,
\]

where `streaming_op` applies the Fourier symbol
\(i(\omega_x k_x+\omega_y k_y)\), `sigma_t` is physical-space
multiplication, and `scattering_op` is the quadrature-weighted angular phase
matrix. These definitions are in
`rte/models.py:17-39,173-216`.

The shifted reference operator and the Born splitting are

\[
 L_{\eta,h}=S_h+\sigma_0 I,\qquad
 V_h=L_{\eta,h}-A_h,
 \qquad I-G_{\eta,h}V_h=G_{\eta,h}A_h.
\]

The code implements `G` by spatial FFT division and the identity is numerically
verified below. DSA and TSA are synthetic corrections applied after a reference
transport correction. `transport_tsa` uses a full angular reference inverse
when no harmonic truncation is requested; `transport_tsa_moment` uses the
Galerkin inverse on the retained angular Fourier space. See
`rte/models.py:302-434`.

The RTE benchmark generator uses four fixed media: diffusive, heterogeneous,
anisotropic, and mixed. For each run, one medium is constructed by
`rte/run_hard_case_experiments.py:52-108`; the learned model is then trained
from scratch for that medium. Thus current RTE experiments are held-out-source
experiments, not coefficient-field generalization experiments.

### 1.2 Helmholtz NPBS path

The Helmholtz training loop loads a batch of slowness fields, calls
`model.setup(sos=sos)` for each coefficient batch, and draws a fresh random
complex right-hand side for every batch. This is visible in
`helmholtz/train2d.py:169-244`. Representative OpenFWI configurations use
3,000 training media, batch size 30, and 200--400 epochs, for example
`helmholtz/configs/2d/openfwi/CurveFaultB/ppw12/unet_bs_nopreloss.yaml`.

The Helmholtz learned preconditioners have a scalar complex field, not 12--24
angular channels. Their learned solver paths are explicitly linear in the
right-hand side for fixed coefficients, while coefficient encoders are
nonlinear. The multilevel architectures use coefficient features at each
level; see `helmholtz/models/model2d.py:595-823`.

The three Helmholtz loss modes are not interchangeable:

- loss 1 uses the raw physical residual
  \(\|r-A_h M_\theta r\|/\|r\|\);
- loss 2 feeds a Green-transformed right-hand side but records the physical
  residual;
- loss 3 trains in the Green metric
  \(\|G_h(r-A_hM_\theta r)\|/\|G_hr\|\) and records the physical residual.

This is a much broader coefficient-conditioned operator-learning problem than
the current per-medium RTE tests.

## 2. Exact implementation audit

### 2.1 Checks that pass

The following were independently rerun in float32/float64 on small grids.

| Check | Result |
|---|---:|
| (L_{\eta,h}-V_h=A_h), size 8, 4 angles, float32 | relative error (6.5\times10^{-8}) |
| Dense matrix representation versus `A_op`, size 4, 4 angles, float64 | relative error (2.6\times10^{-16}) |
| Direct dense solve residual, size 4, 4 angles | (2.2\times10^{-15}) |
| Complex FGMRES against the dense solution | final residual (1.9\times10^{-12}) |
| Real-stacked FGMRES against the dense solution | final residual (1.0\times10^{-11}) |
| Exact zero-init defect2 identity | relative error 0 |
| Fixed-coefficient real linearity of the U-Net map | relative error (8.8\times10^{-7}) |
| Classical DSA periodic shift test | relative error (2.0\times10^{-7}) |

Therefore the main negative RTE result should not be attributed to an invalid
periodic `A_op` or an obviously broken FGMRES implementation.

The non-normality is real: for the small mixed matrix the diagnostic
\(\|A_h^*A_h-A_hA_h^*\|/\|A_h\|^2\) was approximately `0.069`. This is not a
proof of the observed solver degradation, but it explains why a one-step norm
can be a weak proxy for Krylov behavior.

### 2.2 Periodic symmetry defect in the learned U-Net

`LinearGatedUNet` uses `circular_pad_conv` for ordinary convolutions, but
`self.up` is a standard `ConvTranspose2d` and `resize_like` uses ordinary
zero-padding. These operations are in
`rte/npbs_unet_ablation.py:177-305`.

For a uniform periodic medium, the classical correction commutes with a torus
shift. With a small nonzero U-Net output, the total correction is dominated by
the classical base, so the total shift error looks small. After subtracting
the classical defect2 base, the learned correction itself had the following
relative shift defects on a size-16, four-angle uniform medium:

| U-Net levels | relative shift defect of learned correction |
|---:|---:|
| 2 | 0.157 |
| 3 | 0.0226 |
| 4 | 0.0677 |

This is an implementation mismatch, not merely a modeling preference. It can
be fixed by replacing transposed convolution with periodic upsampling followed
by circular convolution, and by replacing zero resize padding with a periodic
restriction/prolongation. Because stride-2 subsampling is not equivariant to
every unit shift, exact symmetry requires a phase-aware or spectral multilevel
restriction, not only changing the padding mode.

### 2.3 Width and angular-channel bottleneck

For a defect2 full-angular U-Net, the input to `LinearGatedUNet` has
`4*n_angles` real channels and the output has `2*n_angles` real channels; see
`rte/npbs_unet_ablation.py:641-655,703-713`. The first operation is a bias-free
1x1 lift to `width`. In the common 12-angle, width-16 setting this is a 48 to
16 channel compression. In the 24-angle, width-24 setting it is a 96 to 24
compression.

This does not prove a global rank bound for the full spatial operator, because
spatial convolutions mix locations. It does prove that the local channel
representation cannot preserve all angular input directions before spatial
mixing. This is qualitatively different from Helmholtz, where the input is
only two real channels and width 32 is expansive.

An independent CPU control used the same mixed size-16, 12-angle case, DSA
base, raw one-step loss, 80 steps where recorded, and real-stacked FGMRES with
12 iterations. The results were:

| width | parameters | sampled loss change | held-out one-step | held-out FGMRES final |
|---:|---:|---:|---:|---:|
| 8 | 18,528 | 0.2160 -> 0.1026 | 0.3340 | 0.00750 |
| 16 | 71,872 | 0.2161 -> 0.0982 | 0.3288 | 0.00767 |
| 32 | 283,008 | 0.2161 -> 0.0732 | 0.3334 | 0.00826 |

The larger model learns the sampled objective better but does not improve the
held-out one-step residual and is worse at the capped FGMRES residual. This is
evidence against width alone being the primary bottleneck. It supports the
stronger conclusion that the representation and objective are misaligned.

### 2.4 Initialization gradient starvation

The zero-output option is correct as an identity-preserving initialization:
the model starts exactly at `z0 + defect_relaxation*z1`. However, because the
final projection is zero, the first backward pass gives nonzero gradients only
to the final projection among 38 parameter tensors in a width-8 size-8 test.
With `output_init="small"`, all 38 tensors received nonzero gradients.

The same small control showed nearly identical 80-step results for zero and
small initialization (held-out one-step `0.3288` versus `0.3290`; FGMRES
`0.00767` versus `0.00757`). Thus zero initialization is a secondary
optimization bottleneck, not the explanation for the full RTE gap.

## 3. What the reported losses actually measure

There are two related but different RTE runners.

### 3.1 Native U-Net ablation loss

In `rte/npbs_unet_ablation.py:1777-1850`, with

\[
 z_\theta=M_\theta r,\qquad r^+=r-A_hz_\theta,
\]

the physical ratio is

\[
 \rho_{\rm phys}(r)=\frac{\|r^+\|_2}{\|r\|_2}.
\]

The general loss is

\[
 \mathcal L = w_m\mathbb E\rho_{B_h}^2
 +w_p\mathbb E\rho_{\rm phys}^2
 +w_c\mathbb E\rho_c^2
 +\mathcal L_{\rm rollout}
 +\mathcal L_{\rm stability}
 +w_T\mathbb E\frac{\|M_\theta r-T_h r\|_2^2}{\|T_h r\|_2^2}.
\]

The pure physical runs set (w_m=0), (w_p=1), and all other optional terms
to zero. In that special case the loss is the mean squared one-step physical
residual ratio on the sampled training batch.

### 3.2 Bottleneck harness loss

`rte/npbs_bottleneck_harness.py:110-130,192-292` defines named modes. Its
`raw` mode is the mean squared physical ratio plus the small default correction
penalty. `metric_raw_rollout` adds the base-metric ratio and rollout terms.
`raw_rollout_teacher` adds teacher imitation. The teacher term is not a raw
residual term; it is an operator-action imitation term.

### 3.3 Why a large loss drop need not improve one-step or FGMRES

The recorded training loss is evaluated on residuals sampled from base
rollouts, smooth random probes, and sometimes auxiliary sources. The reported
`one_step_true` is evaluated on one source residual, either the training source
or a different held-out source, and is an unsquared ratio. Therefore the two
numbers are not algebraically equal even for pure physical training.

More importantly, minimizing

\[
 \mathbb E_{r\sim\mu}\| (I-A_hM_\theta)r\|^2
\]

only controls the action of the error operator on the sampled distribution
\(\mu\). FGMRES depends on the sequence of vectors generated by the
preconditioned operator and on its non-normal pseudospectral behavior. A model
can reduce the average sampled norm while increasing the error operator on an
unseen Krylov direction. The mixed-case non-normality and the observed
one-step/FGMRES disagreement are consistent with this mechanism.

The small independent loss control supports this interpretation:

| loss | final sampled loss | held-out one-step | held-out FGMRES final |
|---|---:|---:|---:|
| raw | 0.0982 | 0.3288 | 0.00767 |
| metric + raw + rollout | 0.3193 | 0.3111 | 0.00793 |
| raw + rollout + depth-4 teacher | 0.2402 | 0.3280 | 0.00818 |

These are small, single-seed controls, so they do not establish universal
rankings. They do establish that a lower recorded training objective is not
evidence by itself that the preconditioner is better.

## 4. Why Helmholtz succeeds more readily

The difference is structural and experimental, not just that one network is
larger.

### Helmholtz advantages

- The unknown is a scalar complex field: two real channels, so the learned
  channel lift is expansive rather than angularly compressive.
- Training changes the coefficient field every batch across thousands of media.
  Coefficient conditioning is therefore trained as a family map.
- The residual is expressed in the same Green/Born coordinates used by the
  solver. Loss type 3 directly measures the Green-preconditioned residual.
- The learned architectures use multilevel coefficient gates, Fourier/global
  operators, or implicit Green-like layers that match Helmholtz propagation.
- Boundary handling is aligned with the PBL/zero-exterior Helmholtz setup rather
  than a periodic torus.

### RTE disadvantages

- Current learned experiments mostly train one fixed medium at a time, so the
  coefficient encoder cannot demonstrate coefficient generalization.
- The RTE state has many angular directions and has streaming/scattering
  coupling. Low-order moment restrictions discard information that may be
  needed in mixed thin/thick and forward-peaked regimes.
- Classical TSA4 is a strong teacher. A two-application learned defect2 map is
  asked to match a deeper correction at lower base-call cost and with extra
  neural overhead.
- The physical one-step objective is not a spectral or Krylov objective.
- The learned periodic U-Net is not fully periodic at its multilevel decoder.
- Strong classical baselines, not raw transport, are the relevant comparison;
  the remaining margin is consequently much smaller than in the Helmholtz
  Born baseline.

## 5. Audit of the mode/R/P hypotheses

The earlier claims about mode coupling, learned restriction/prolongation, and
residual-dependent angular subspaces are directionally right but need sharper
scope.

- `CoeffWeightedTSA-M4` only rescales fixed (Q_4) modes. It cannot rotate or
  create modes.
- `CoeffMixedTSA-M4` learns a coefficient-dependent matrix after the fixed
  moment solve. It couples modes only inside (Q_4).
- `LearnedRP-TSA-M4` learns coefficient-dependent pre- and post-solve matrices,
  but both still act inside (Q_4). It is more expressive than post-solve
  mixing, but it is not a learned angular basis over the full ordinate space.
- `ResidualSubspaceTSA-M4` uses a residual-dependent global matrix inside (Q_4).
  Its current selector is global in space and therefore cannot choose a
  spatially varying angular subspace.
- `MomentNPBSUNetPreconditioner` restricts the learned U-Net input to moments
  before learning. It cannot recover discarded high angular information.

The evidence is coherent: learned R/P and residual-subspace models reduce their
training/one-step objectives modestly, but on mixed cases they can degrade
FGMRES sharply. The models are changing the low-order correction, but they are
not approximating the deeper stable operator in the directions that matter to
Krylov.

## 6. Causal ranking of bottlenecks

### Primary: objective/operator mismatch

The loss is a sampled average of one-step action. It does not constrain the
preconditioned spectrum, non-normal growth, or the full residual distribution
encountered by FGMRES. Teacher imitation and rollout terms tried so far are
not enough because their residual banks remain narrow and their stability
penalty is a local contraction penalty, not a Krylov-subspace criterion.

### Primary: angular representation mismatch

The moment models cannot leave (Q_K), and full-angular models compress many
angular channels into a small width. This is the clearest mathematical reason
that RTE is harder than scalar Helmholtz.

### Secondary but real: periodic decoder mismatch

The learned correction breaks a symmetry that the target operator has. This can
inject artificial boundary modes and is particularly dangerous for a learned
preconditioner used repeatedly.

### Secondary: per-medium training protocol

The current experiments cannot test the main benefit promised by a
coefficient-conditioned network: amortizing across a coefficient family. This
is a scientific-design bottleneck and makes the coefficient encoder's role
unidentifiable.

### Secondary: initialization and parameter tuning

Zero-output gradient starvation and the choice of `eta_scale`, relaxation, and
base depth affect optimization. The small controls show they are not the main
cause of the RTE gap, but they should be handled in the next design.

## 7. Recommended next experiment matrix

### P0: fix the discretization symmetry

Implement periodic restriction/prolongation and periodic upsampling in the
U-Net. Verify, on uniform media, the relative shift defect of the learned
correction is below a prescribed threshold before training. Keep ordinary
zero-init as a stability control and add small-init as an optimization control.

### P1: learn a larger angular candidate space

Let (Q_{K_{\rm solve}}\subset Q_{K_{\rm cand}}), with
\(K_{\rm cand}>K_{\rm solve}\). Preserve the scalar mode exactly. Learn
bounded coefficient-dependent maps

\[
 R_\theta(x):Q_{K_{\rm cand}}\to Q_{K_{\rm solve}},\qquad
 P_\theta(x):Q_{K_{\rm solve}}\to Q_{K_{\rm cand}},
\]

around a stable classical inverse, with identity/adjoint consistency penalties
and a fallback gate. This is the first version of learned mode/subspace
selection that can actually introduce information outside the original solve
space.

### P2: train against a stable teacher and the solver metric

Use a depth-4 damped TSA/DSA teacher and train the student to imitate the
teacher correction, not just its residual. Add residuals from random source
locations, angular profiles, smooth random fields, later teacher rollouts, and
randomized coefficient fields. Select checkpoints by held-out FGMRES residual
and time-to-tolerance. Report the one-step physical residual separately.

### P3: make coefficient conditioning identifiable

Generate a small coefficient ensemble per batch, not one fixed medium. Split
by coefficient field, source, and resolution. The first decisive cross-medium
test should compare fixed-medium training, cross-medium training, and a
nonlearned per-medium tuned baseline.

### P4: remove the local channel bottleneck

Use an angular Fourier front end or grouped per-mode lift so that low and high
angular modes are not compressed into width 16. A practical small model is a
transport-aware block that preserves (m=0), couples nearby (m)'s with a
bounded matrix, and uses spatial U-Net blocks on each mode group. Measure
parameter count and wall time, not only loss.

## 8. Readiness assessment

The current project is a strong research hypothesis and a useful negative
result, but the RTE learned method is not yet a paper-level solver improvement.
The Helmholtz learned result is credible as an operator-learning direction,
subject to its own coefficient/data and wall-time audit. The RTE direction is
promising as a structure-preserving compression/adaptation problem, not as
evidence that an unconstrained U-Net automatically beats TSA.

The next decisive falsifier is small and unambiguous:

1. repair periodic multilevel transfer;
2. use (K_{\rm cand}>K_{\rm solve}) with scalar preservation;
3. train on a coefficient ensemble using teacher imitation plus held-out
   FGMRES checkpointing;
4. compare against TSA2 at equal base calls and TSA4 at equal wall-time;
5. reject the learned direction if one-step, FGMRES, and time-to-tolerance do
   not improve together on held-out media.

