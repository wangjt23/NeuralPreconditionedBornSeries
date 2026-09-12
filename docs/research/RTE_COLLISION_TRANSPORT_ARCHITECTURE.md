# A Transport-Structured Learned Preconditioner

## Decision

The new candidate is a **coefficient-conditioned collision/transport factorization**, implemented in
[`rte/collision_transport.py`](../../rte/collision_transport.py). It is not a larger generic U-Net and does not
ask a network to rediscover the streaming operator. It keeps all angular modes, exactly applies the
local collision spectrum, and learns bounded corrections to a sequence of collision and streaming solves.
The coefficient network is evaluated once per medium at inference.

This is an additional experimental family. It does not replace the existing NPBS, DSA/TSA or U-Net
implementations. The accompanying [numerical tables and figures](RTE_COLLISION_TRANSPORT_RESULTS.md)
report positive and negative controls. Historical results remain unchanged.

## 1. Which Structure Comes From RTE?

### Discretization and norm

The implemented problem is steady transport on the periodic square `[0,1)^2`, with `N x N` spatial
points, spacing `h=1/N`, and `L` equally spaced directions on the circle:

\[
 \theta_l=2\pi l/L,\qquad \Omega_l=(\cos\theta_l,\sin\theta_l),\qquad w_l=1/L.
\]

The field is complex-valued, of shape `(batch,L,N,N)`. The physical matrix is

\[
 A=S+C,\quad S=\Omega\cdot\nabla_h,\quad C(x)=\sigma_t(x)I-\sigma_s(x)P,
 \qquad \|v\|_h^2={h^2\over L}\sum_{l,i,j}|v_{lij}|^2.
\]

Spatial differentiation uses `k=2*pi*fftfreq(N,d=h)`. Thus `dx` enters the operator, not just the loss.
The constant norm factor cancels in relative residuals. The angular phase matrix `P` is the
quadrature-weighted, row-normalized circular Poisson kernel with parameter `g`.

This is the repository's pseudo-spectral, periodic **S1** transport prototype. There are no upwind
sweeps, vacuum/inflow boundaries, S2 quadrature, or positivity claims. The architecture is specialized
to its circulant phase matrix; an arbitrary non-circulant scattering kernel needs a different collision solver.

### Exact angular damping and known coupling

Let `F_theta` be the unitary angular DFT. Circulant scattering gives

\[
 C(x)=F_\theta^*\operatorname{diag}_m\lambda_m(x)F_\theta,
 \qquad \lambda_m(x)=\sigma_t(x)-\sigma_s(x)p_m.
\]

The code computes `p_m` by FFT of the **actual discretized kernel**, not by substituting the continuum
formula `g^|m|`. This matters for coarse angular resolution and strong forward scattering.
The scalar mode has `p_0=1`, so `lambda_0=sigma_a=sigma_t-sigma_s`; nearly conservative scattering
makes it weakly damped. For a forward-peaked kernel, several angular modes can be weakly damped.

Streaming itself already specifies angular mode coupling. In angular Fourier coordinates,

\[
 (S\widehat\psi)_m={1\over2}(\partial_x-i\partial_y)\widehat\psi_{m-1}
                  +{1\over2}(\partial_x+i\partial_y)\widehat\psi_{m+1},
\]

with indices interpreted cyclically for the finite circular quadrature. In ordinate/spatial Fourier
coordinates, it instead has the diagonal symbol `i Omega_l . k`. These two coordinate systems give
cheap complementary inverses. A free dense learned angular mixer is not needed to represent this known coupling.

This is the difference from `CoeffWeightedTSA-M4`: that model rescales a *truncated reference moment
correction*. Here the full residual repeatedly passes through exact heterogeneous collision and
full-angle streaming factors. The learned gates are still diagonal in angular modes; **this is not
learned restriction/prolongation or a residual-dependent learned angular subspace**.

## 2. Classical Starting Point And Proof Boundary

**Proposition 1 (finite-dimensional special case, proved below).** Suppose `P` is real symmetric,
nonnegative and stochastic, `sigma_s>=0`, and `min_x(sigma_t-sigma_s)>0`. On the periodic FFT grid,
`S*=-S`, `C*=C>0`, and `A` is invertible. For fixed `alpha>0`, define

\[
 B_\alpha=2\alpha(\alpha I+S)^{-1}(\alpha I+C)^{-1}.
\]

The solution-error iteration `e_new=(I-B_alpha A)e` has spectral radius bounded by

\[
 \rho(I-B_\alpha A)\le q(\alpha)
 =\max_{\lambda\in\operatorname{spec}(C)}
   \left|{\alpha-\lambda\over\alpha+\lambda}\right|<1.
\]

*Proof.* Symmetric stochastic `P` has spectral norm at most one, so
`C(x)>=(sigma_t-sigma_s)I`. The streaming symbol is purely imaginary. Consequently
`Re <v,Av>_h >= min(sigma_a)||v||_h^2`, implying invertibility. Direct multiplication gives

\[
 (\alpha I+S)(I-B_\alpha A)(\alpha I+S)^{-1}
 = (\alpha I-C)(\alpha I+C)^{-1}
   (\alpha I-S)(\alpha I+S)^{-1}.
\]

The first factor is Hermitian with norm `q(alpha)` and the second is unitary. Similarity preserves
spectral radius, and spectral radius is bounded by the operator norm. This proves the assertion.
For spectral endpoints `lambda_min,lambda_max`, minimizing this endpoint bound gives
`alpha0=sqrt(lambda_min*lambda_max)`. It need not minimize actual GMRES iterations. End of proof.

This is the classical Hermitian/skew-Hermitian splitting (HSS) construction, not a new convergence
theorem. The primary reference is Bai, Golub and Ng (2003), *Hermitian and skew-Hermitian splitting
methods for non-Hermitian positive definite linear systems*, SIAM J. Matrix Anal. Appl. 24(3), 603-626,
[DOI 10.1137/S0895479801395458](https://hub.hku.hk/handle/10722/276739).
The proof above supplies the precise normalization and finite-grid instantiation used here.

**Do not infer more than this proposition states.** It bounds spectral radius, not monotone physical
residual decay. `I-A B_alpha` is similar to `I-B_alpha A` through invertible `A`, but their Euclidean norms
need not agree. Products with different shifts, learned gains, and macro augmentation do not automatically
inherit the fixed-shift bound. Strictly zero absorption is excluded. This is not an asymptotic-preserving
diffusion-limit theorem.

## 3. Learned Architecture

Set `z=0`. For each stage `j=1,...,d`, use the exact physical residual `r_j=r-Az` and update

\[
 z\leftarrow z+2\alpha_j(\alpha_j I+S)^{-1}
       F_\theta^*\operatorname{diag}_m
       \left({g_{jm}(x)\over\alpha_j+\lambda_m(x)}\right)F_\theta r_j.
\]

The implementation obtains every residual after the first by applying the actual repository `A_op`.
No equation, boundary condition, or discrete phase matrix is changed.

### Coefficient network

The six spatial input channels are centered `log(sigma_t)`, albedo, `log(sigma_a/sigma_t)`,
`log(1+h*sigma_t)`, a periodic 5-by-5 mean of `log(sigma_t)` minus its global mean, and `g`.
Thus both collision rates and cell optical thickness are visible. The local gain head is a width-12
`3x3 circular convolution -> SiLU -> 1x1 convolution -> SiLU -> 1x1 convolution`.
It outputs one map per stage and absolute angular mode.

An independent `8 -> 12 -> d` MLP uses log minimum/maximum collision eigenvalues, log mean total
and absorption coefficients, mean/std albedo, mean log optical thickness, and `g` to choose shifts.
The parameterization is

\[
 \alpha_j=f\sqrt{\lambda_{\min}\lambda_{\max}}
       \exp(\log(8)\tanh a_j),\qquad
 g_{jm}(x)=\exp(0.7\tanh b_{j,|m|}(x)).
\]

The fixed scale `f` is selected by classical validation before training. The code also supports fixed
per-stage multipliers for nonlearned sequence controls. Opposite modes share gains; the scalar gain
is fixed at `g_j0=1` by default. Zero output-head initialization exactly recovers the selected classical
factorization. Positive bounded gains prevent division through a learned negative collision rate, but
**do not prove stability of the complete preconditioner**.

At 12 angles there are 1,132 trainable parameters for two stages and 1,340 for four. The shift-only
two-stage control has 134. The matched original width-16/three-level defect U-Net has 71,872.
Small parameter count is intentional: the physical factors perform the expensive representational work.

### Why this could improve the algorithm

1. Heterogeneous extinction and scattering are handled locally by `C(x)`, rather than only through a
   spatially averaged transport reference. The learned gains can compensate for the error of the
   alternating factorization near interfaces and across optical regimes.
2. All angular modes survive. Forward-peaked scattering is not forced into a fixed low-order closure.
3. The streaming inverse propagates along every discrete direction across the entire periodic domain.
   It is not replaced by a small isotropic convolution stencil.
4. For fixed coefficients the residual-to-correction map is **complex-linear**. Coefficient dependence
   remains nonlinear. There are no residual-dependent gates or real/imaginary asymmetries to learn.
5. At inference the nonlinear CNN/MLP fields are cached. Each application then uses FFTs, pointwise
   products and `d-1` physical operator applications, not a residual U-Net.

These are architectural mechanisms, not proofs of empirical superiority. In particular, we do not
claim a U-Net is incapable of learning these actions, or that every weak mode needs a learned basis.

### Cost and caching

Each factor stage uses one forward/inverse angular FFT pair and one forward/inverse spatial FFT pair.
It stores mode gains in `O(d L N^2)` memory and does not build a spatial-grid-sized dense angular inverse.
The shared `A_op` still uses the repository's dense angular phase action; no `O(L log L)` complexity
claim is made for the whole current solver. Full-angle TSA caches an `L x L` inverse per spatial frequency.

`prepare()` builds no-gradient coefficient fields. Cached fields are invalidated by coefficient,
parameter, dtype/device, or relevant grid changes. Training/gradient-enabled execution rebuilds them,
so no detached inference cache is accidentally used for learning. Reconstruct the operator when changing
physical spacing; changing `dx` without updating its `k` arrays is not a valid grid change.

For fairness, [`PreparedDefectUNet`](../../rte/prepared_defect_unet.py) freezes and caches the old U-Net's
coefficient encoder and all gate values too. Its residual convolution path is unchanged. Numerical
equivalence is tested. This snapshot must be rebuilt after coefficient/network changes. It is deliberately
limited to the standard defect2 gate family, not the later router/ROM/scale-aware variants.

## 4. Training And Experimental Controls

[`run_collision_transport_experiment.py`](../../rte/run_collision_transport_experiment.py) trains on one
medium at a time. There are nine train, three validation and three test source centers, each split
containing beam, dipole and isotropic profiles. The exact source generator is shared with the preceding
audit. Sources are not arbitrary angular wave packets or new materials.

For each source the fixed bank contains four consecutive one-stage TSA residuals, including the
initial source. A further two smooth random `A e` probes per source are added. Hence the train bank has
54 samples and each validation/test bank has 18; each split has the same 2:1 trajectory/probe mixture.
The runner preserves subgroup losses for each trajectory stage and the smooth-action probes.

The sole training objective is

\[
 L_{\rm phys}(\vartheta)={1\over n}\sum_{i=1}^n
 {\|r_i-A M_\vartheta r_i\|_h^2\over\|r_i\|_h^2}.
\]

There is **no preconditioned metric, teacher loss, rollout penalty, or correction regularizer** in this
experiment. `sqrt(L_phys)` is RMS one-step residual, not mean one-step residual. A source-only one-step
ratio is a different statistic from the bank loss. A low bank loss weights the sampled residual
covariance; it does not control the operator norm or every Krylov direction.

Training uses 300 AdamW updates, batch 16 sampled with replacement, learning rate 0.003, weight decay
0.0001, cosine learning-rate decay to 1% of the initial value, and gradient clipping at norm 1.
This is about 88.9 nominal bank passes, **not 300 epochs**. All compared learned variants receive the
same residual samples and update budget for a seed. Seeds 1, 2, 3 vary source centers, probes and initialization.

Every 50 updates, including update zero, full train/validation losses and validation-source GMRES are
evaluated. Checkpoints minimize the lexicographic tuple `(failures, mean iterations, mean final residual)`.
Only after selection are test losses and test solves evaluated. A selected update-zero checkpoint is
reported as such, not as a trained solver improvement. Last-update losses are also recorded separately.

Controls include TSA2 and TSA4 (two/four **full-angle TSA** defect corrections, not two/four harmonics),
HSS2/HSS4 with equal shifts selected from `f in {0.5,1,2,4}`, all 16 two-stage shift sequences, and a
two-sweep coordinate search over four-stage shifts. The four-stage search is stronger than equal-shift
tuning but not a global optimization. Shift-only learning, global angular gains, and spatial angular
gains distinguish scalar tuning from learned local structure. Global gains use the same CNN but spatially
average its positive gain maps before applying them; they are separately trained.

All methods use the same real-stacked FGMRES, restart 20, at most 30 iterations, and true relative
residual tolerance `1e-5`. Although the new map is complex-linear, it receives no special complex-Krylov
advantage over the old real-linear U-Net. Solve times are medians of three warmed repetitions per source.
Setup/warmup and training are recorded separately; shared operator/bank construction is excluded.
Base calls and factor stages are reported separately: **one HSS stage is not one TSA call**, so these
counts are not interchangeable equal-cost units. Laptop timing and small three-seed variation are not GPU benchmarks.

## 5. Rejected Coarse-Moment Shortcut

The optional `macro_harmonics=1` control appends `Q (Q* A_ref Q)^(-1) Q* (r-Az)`, where `Q` retains
angular modes `0,+1,-1`, is the identity spatially, and `A_ref=S+mean_x(C)`. Its small inverse is formed
per spatial Fourier frequency. There is no artificial absorption shift in this projected reference.
The implementation is independently checked against a dense projected reference solve.

In the mixed medium this augmentation fails at the 30-iteration cap and can leave a residual around
0.1, whereas unaugmented HSS converges. Correct algebra alone does not make an averaged coarse model
appropriate for a heterogeneous thin/thick interface. This is a **negative experiment**, not a recommended
default or production DSA. Residual minimization could guard an individual step, but would not by itself
prove an improved Krylov solver. A future macro correction needs discrete transport consistency or an
explicit Schur-complement error test before being promoted.

## 6. What The New Experiments Establish

The primary size-16 experiment is `collision_transport_mixed_final`; its four-stage models use the
same sources, banks and training budget as each other. Means below cover three seeds and three test
sources per seed. Every listed method reaches the `1e-5` true residual tolerance on all nine solves.

| Selected method | Train physical loss | Validation loss | Test loss | GMRES iterations | Solve ms |
|---|---:|---:|---:|---:|---:|
| TSA4 | 0.08875 | 0.09618 | 0.09304 | 10.67 | 21.84 |
| HSS4, sequence-tuned | 0.18952 | 0.18840 | 0.19614 | 9.89 | 12.42 |
| Learned global mode gains, depth 4 | 0.02833 | 0.02850 | 0.03050 | 7.78 | 10.18 |
| Learned spatial mode gains, depth 4 | 0.01659 | 0.01712 | 0.01943 | 7.00 | 9.60 |
| Prepared original defect U-Net | 0.09526 | 0.09841 | 0.09286 | 17.44 | 36.31 |

The full last-update train loss for spatial mode learning decreases from `0.18952` to `0.01611`,
about 91.5%. Validation chooses updates 150, 250 and 150 rather than always choosing update 300.
The selected model is approximately 23% faster than the measured sequence-tuned HSS4 control and
56% faster than TSA4. These are **warmed per-solve CPU timings**, not end-to-end savings including learning.

The ablations refine the explanation:

- **Depth matters.** Two-stage learned gains improve on two-stage HSS, but lose to classical four-stage
  HSS. Four-stage learning improves beyond that stronger baseline.
- **Shift learning is insufficient under this objective.** Four-stage shift-only learning reduces
  train loss from `0.18952` to `0.04378`, yet all three validation-selected checkpoints remain at update
  zero. A large decrease in physical bank loss still does not imply a better Krylov map.
- **Most of the learned gain is angular relaxation, not necessarily spatial localization.** Separately
  trained global mode gains already reach 7.78 iterations. Local maps reduce that to 7.00 and lower
  test loss by about 36%. This supports a useful but incremental role for local coefficient dependence;
  it does not establish that a sophisticated learned R/P is necessary.
- **A source-only first step is not the same test as the bank loss.** Spatial mode learning has mean
  source first-step ratio `0.16046`, versus `0.15842` for TSA4. It nevertheless converges in fewer Krylov
  iterations. Reporting only either the bank loss or the source first step would miss this distinction.
- **The old U-Net is not compared with deliberately uncached coefficients.** Its prepared and original
  outputs agree, and the prepared version is timed separately. The trained residual convolution path
  remains more expensive in this matched-budget comparison. This is not an exhaustive optimal-width,
  learning-rate, or training-budget comparison of every historical U-Net variant.

At size 32, retraining the four-stage spatial model gives train/validation/test losses
`0.01544 / 0.01666 / 0.01821`, still 7.00 mean iterations. In the reevaluation with stronger controls,
its mean solve takes `20.24 ms`, compared with `29.58 ms` for sequence-tuned HSS4 and `50.94 ms` for TSA4.
This is corroboration at a second resolution of the same medium, **not generalization to a new material**.

The homogeneous anisotropic control gives an important opposite result: two-stage mode learning
reduces the last train loss to about `0.0192`, but all selected checkpoints remain at initialization.
TSA4 takes 3 iterations, HSS4 takes 4, and the two-stage learned family takes 7. The current evidence
therefore favors targeting heterogeneous mixed regimes, not universally replacing classical transport solvers.

### Amortization and provenance

Spatial-mode training plus its periodic validation takes about `10.30 s` per seed in the final size-16
experiment. Dividing this by the measured per-solve saving suggests roughly 3,600 right-hand sides
to amortize that cost relative to the fastest tested classical HSS4 timing, or roughly 840 relative to
TSA4. These rough estimates omit both methods' tuning/setup costs and uncertainty; training one network
for a single right-hand side is not justified by these timings. Across-medium reuse has not been demonstrated.

The generated results document reports seed variation, selected/last losses, failure counts, subgroup
CSVs and artifact hashes. `collision_transport_mixed` and `collision_transport_mixed_controls` are the
initial two-stage runs; the latter only reevaluates saved checkpoints with stronger/cached controls.
`collision_transport_mixed_depth4` is the initial depth-four run. `collision_transport_mixed_final`
adds the separately trained global-gain/shift controls. `collision_transport_mixed_size32_controls`
reevaluates `collision_transport_mixed_size32` with the four-stage sequence control. These repeated
evaluations reuse checkpoints/sources and **must not be counted as independent replications**.

## 7. Verification And Limits

Unit tests independently check collision FFT diagonalization, a dense HSS layer, its classical spectral
radius bound, separate-stage composition, complex linearity, gradients through trainable fields, scalar
protection, opposite-mode tying, batch broadcasting, the macro inverse, cache reuse/invalidation,
training gradients after preparation, global-gain ablation, bank normalization/composition, and exact
prepared-U-Net equivalence. The existing RTE regression suite is also run.

The complete RTE test suite reports **47 passed and one CUDA-only skip** (48 tests collected).
This does not certify an arbitrary learned map's convergence. The experiments are per-medium learning
on a small periodic grid, not coefficient-heldout validation.
The size-32 test **retrains** at that resolution; it is not zero-shot grid transfer. The homogeneous
anisotropic control is important: its reference operator is already strong, and two-stage mode learning
does not improve the selected solver. Claims of a universal replacement for DSA/TSA would be unsupported.
CUDA, physical S2 angular transport, near-voids, zero-absorption limits, and inflow boundaries remain untested.

## 8. Next Decisions

- Prioritize four-stage local collision/transport learning on heterogeneous mixed regimes if it survives
  the global-gain and independently tuned shift controls. Do not infer that merely adding width helps.
- Retain the best classical solver in homogeneous/highly diffusive regimes. The natural eventual router
  should make a **per-medium** selection with validation and account for preparation cost; it is not implemented here.
- Before scaling the dataset, test angular refinement, independently generated material interfaces,
  absorption approaching zero, and residual/operator amplification probes. Report both source and Krylov-state losses.
- A learnable macro branch should approximate the heterogeneous macro/micro Schur complement, with a
  transport-consistent coarse solve. A free low-moment reference correction already failed the simplest control.
- For nonperiodic problems replace the spatial FFT transport resolvent with a boundary-compatible sweep/solve;
  for S2 use the actual quadrature-weighted scattering eigenspace or a tested spherical-harmonic approximation.
  Neither replacement is a drop-in consequence of the current tests.

## Reproduction

From the repository root, using the environment containing PyTorch:

```bash
python -m unittest discover -s rte/tests -v
python -m rte.run_collision_transport_experiment --depth 4 \
  --models learned_shifts learned_global_modes learned_modes legacy_raw \
  --seeds 1 2 3 --output rte/results/new_collision_transport_mixed
python -m rte.run_collision_transport_experiment --depth 4 --size 32 \
  --models learned_modes --seeds 1 2 3 --output rte/results/new_collision_transport_size32
python -m rte.run_collision_transport_experiment --case anisotropic --macro-control \
  --seeds 1 2 3 --output rte/results/new_collision_transport_anisotropic
```

Output directories must be new. Checkpoints are local `.pt` files; numerical CSVs, settings and figures
are retained as research artifacts. `--evaluate-from` can add classical/cache controls to trusted local
checkpoints without retraining and rejects mismatched training/problem settings. The source `A_op` and
original benchmark scripts are unchanged by this new architectural experiment.
