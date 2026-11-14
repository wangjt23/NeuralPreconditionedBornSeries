# Neural transport operator: architecture and experimental contract

Completed pilot update: [matched results and next decision](RTE_NEURAL_TRANSPORT_OPERATOR_RESULTS_2026_09_12.md).
NTO-HSS4 is the retained candidate; the initial two-stage speed hypothesis did
not pass. The sections below preserve the design rationale and experimental
contract, rather than silently rewriting the original hypothesis.

## Objective and scope

Build an AI-enhanced preconditioned solver that reaches the same true physical
residual tolerance faster than validation-tuned classical controls. The target
is **wall time**, including a separate accounting of preparation, not lowest
training loss or the largest neural network. Initial scope remains the periodic
Fourier RTE on N32, L12, g=0.65; success here will not establish superiority on
other discretizations or boundary conditions.

The new model is implemented in
[`neural_transport_operator.py`](../../rte/neural_transport_operator.py).
It learns material-dependent, full-angle operator factors. It is substantially
more expressive than diagonal CT gains, while keeping the expensive nonlinear
material encoder outside the repeated residual-application path.

## Architecture: NTO-HSS2

Let A=S+C be the unchanged physical RTE matrix. A depth-d preconditioner forms

```text
z = 0
for j = 1,...,d:
    e = r - A z                # first e is r; no unnecessary first A call
    z = z + K_j(coefficients) e
return z
```

The initial main candidate has d=2. Each learned block is

\[
K_j = P_j(x)\,\mathcal F_x^{-1}
 \operatorname{diag}_{\ell}\!\left[
 g_{j\ell}(k)\frac{2\alpha_j}{\alpha_j+i\Omega_\ell\cdot k}
 \right]\mathcal F_x\,(\alpha_j I+C(x))^{-1}R_j(x).
\]

The inverse local collision matrix comes from the actual discrete circular
phase kernel, not a continuum phase approximation. Every angular and spatial
frequency is retained. The physical factors carry units/scaling; the learned
matrices and spectral gains are dimensionless.

### What AI learns

1. **Two full angular maps per stage.** R_j(x) changes which angular combinations
   enter transport and P_j(x) recombines its output. Each is an L-by-L matrix at
   every spatial point, not a fixed angular-moment truncation or a diagonal
   modal gain. Both maps depend nonlinearly on material geometry.
2. **Complex frequency responses.** A coordinate-conditioned MLP predicts an
   amplitude and phase modifier for every stage, spatial frequency and ordinate.
   It can change nonzero-frequency transport response, including frequencies
   absent from the old smooth-error probe support. This is richer than choosing
   one scalar shift per stage.
3. **Stage shifts.** A material-summary MLP predicts positive shifts within a
   factor eight of sqrt(lambda_min(C) lambda_max(C)).

The eight-channel material input includes centered log extinction, albedo,
log absorption fraction, cell optical thickness, a local average, g, and signed
spatial differences of log extinction. The width-24 encoder has a circular 3x3
stem and three residual context convolutions with dilations 1, 2 and 4. Its
receptive field is 17x17. A 1x1 head outputs the angular matrices. Global material
statistics supplement the local encoder in the shift and spectral networks.

The spectral MLP is 12 -> 32 -> 32 -> (2dL), with SiLU hidden activations. Its
input consists of eight material summaries and four normalized frequency/scale
coordinates. The model has **35,186 trainable parameters at d=2,L=12,width=24**.
Angle-count changes require rebuilding/retraining its heads. Spatial-size
changes are structurally possible but transfer has not been established.

### Constraints and initialization

For uniform angular quadrature define Q=I-11^T/L. Each learned angular map is

\[
 R=I+Q\Delta Q,\qquad \Delta=(0.5/L)\tanh(H_R),
\]

and similarly for P. Thus R1=1 and 1^T R=1^T: the neural angular transformation
preserves a scalar angular input and the local angular mean. The perturbation
has operator norm at most 0.5, so these individual maps remain full rank. This
is **not** a convergence proof for the composed preconditioner or a positivity
guarantee for the transport solution.

Spectral modulation is exp(0.7 tanh(a)+i 0.7 tanh(b)); it is exactly one at
spatial frequency zero. All output heads start at zero. Therefore NTO-HSS2
initially equals two exact classical HSS corrections, up to rounding. Learning
can introduce signed angular mixing and complex phase changes; it does not
inherit the fixed-HSS convergence theorem.

For fixed coefficients the residual map is complex-linear and sends zero to
zero. Nonlinearity is in the mapping from materials to operators. A residual
activation or router is not necessary to make this an AI-generated inverse
approximation; residual linearity also makes it easier to test and deploy within
the existing FGMRES solver.

### Deployment cost

Preparation runs the encoder and compiles

```text
input_matrix_j(x) = inverse(alpha_j I + C(x)) @ R_j(x)
output_matrix_j(x) = P_j(x)
stream_filter_j(k,l) = learned_gain_j(k,l) * 2 alpha_j/(alpha_j+i Omega_l.k)
```

Each subsequent stage executes two local angular matrix-vector products and
one spatial FFT/inverse-FFT pair. Angular FFTs are absorbed into the prepared
input matrices. A two-stage application needs one internal A evaluation; CT4
needs three. No material CNN/MLP or matrix inversion runs on a valid cache hit.
Cache keys include setup generation, tensor versions/device/dtype and parameter
state; input tensors remain referenced until refresh to prevent allocator reuse.

This trades more local angular work/storage, O(d L^2 N^2), for fewer stages and
FFT launches. It is particularly a small-to-moderate-L hypothesis. A low-rank
factorization for large L would need a separate accuracy/cost ablation.

The speed condition is explicit:

\[
T_{\rm setup,AI}+n_{\rm AI}(T_A+T_{M,AI}+T_{\rm Krylov})
< T_{\rm setup,classical}+n_{\rm classical}(T_A+T_{M,classical}+T_{\rm Krylov}).
\]

No parameter-count or iteration-count comparison substitutes for measuring it.

## Historical ideas retained and not assumed

- Retain physical defect updates and full angular coverage. The truncated
  moment/subspace models had a real range restriction; this model does not.
- Reuse scale-aware material features and multiscale coefficient conditioning.
  Their historical improvements were conditional, often on a single medium,
  not proof of coefficient-heldout solver speed.
- Keep nonlinear inference work in preparation, following the effective CT
  deployment pattern. The old residual U-Net's repeated convolution overhead
  was a concrete obstacle to time-to-tolerance gains.
- Learn angular input/output maps, but do not reuse the failed low-order averaged
  macro inverse or assume old learned R/P trials were successful Krylov solvers.
- Do not restore an expensive residual router by default. Better one-step
  routing metrics did not establish an adequate runtime benefit.

There is also an implemented averaged-reference kernel arm. Its zero-head N32
pilot failed 4/16 validation-selection solves; after 20 updates the objective
fell but failures increased to 6/16, leaving step zero selected. This short run
does not establish a capacity failure. It motivated testing the heterogeneous
collision/HSS anchor while preserving the reference-arm artifacts.

## Training contract

[`train_neural_transport.py`](../../rte/train_neural_transport.py) generates
fresh coefficient/material-source sets for seed 401: 256 training and 64
validation media, with one stored source per medium. **No test split is built.**
These are architecture-development data, not a confirmation set.

The pilot horizon is 1,600 optimizer updates, batch 16: 100 complete passes
over 256 media. The NTO and CT controls use identical data, residual-sampler
seeds, optimizer horizon, learning rate and objective. Different architectures
have different parameter counts and initialization; this is not a capacity-
matched comparison. Checkpoints retain optimizer, scheduler, RNG and the current
permutation/offset, so an update-boundary stop can resume unchanged.

Residuals are 20% full-spectrum A e probes. The remaining 80% uses the previous
TSA-trajectory/smooth-A e mixture in its 2:1 ratio. Fresh probes are drawn per
batch. The loss is

\[
L=\mathbb E\frac{\|E_\theta r\|^2}{\|r\|^2}
 +0.25\,\mathbb E\frac{\|E_\theta(E_\theta r)\|^2}{\|r\|^2},
\qquad E_\theta=I-AM_\theta.
\]

The second term backpropagates through both applications and uses the original
residual norm. This is physical one-step plus two-step rollout training, **not**
the previous pure one-step objective. Its usefulness must be checked separately
by a zero-rollout-weight ablation; it is not a stability certificate.

Validation every 200 updates selects by failures, mean iterations, then mean
true residual on the first 16 validation sources. Frozen checkpoints are paired
against classical solvers on the remaining 48 media. Those 48 still contribute
to validation loss and architecture decisions: they are not untouched test data.

## Evaluation and decision gates

[`evaluate_neural_transport.py`](../../rte/evaluate_neural_transport.py) tunes
HSS/TSA/DSA/transport-DSA/full-reference candidates on the selection subset and
randomizes method order per remaining validation medium. It retains first/warm
solves, failures, setup, A calls, and memory observations. The comparison uses
the same physical operator, real-stacked FGMRES, tolerance 1e-5, and cap 30.

Primary criterion: no worse failed-media count than the strongest tuned
classical competitor and a reproducible reduction in paired time-to-tolerance.
A 20% median saving with no p95 regression is the promotion target, not an
achieved result. Preparation-inclusive single-RHS and amortized repeated-RHS
costs must both be reported. The old CT4 checkpoint is a useful deployment
reference but has a much larger/different training budget, so matched CT2/CT4
training controls are required for causal architecture claims.

An unrelated job currently occupies the GPU. Training/correctness checks can
proceed, but any GPU timing collected under contention is labeled as such.
CPU one-thread timing provides a useful separate measurement, not a substitute
for uncontended-GPU confirmation.

If the two-stage model misses the speed gate, inspect setup/application cost
and residual directions before increasing training. Test depth four and the
no-mixing/no-spectral ablations one factor at a time. After freezing the design,
run independent training seeds and generate fresh confirmation test seeds. Do
not recycle the old audited test media as fresh evidence.

## Literature relationship

Learning a material-to-operator construction draws on the idea of predicting
solver operators across a PDE family in
[Greenfeld et al., Learning to Optimize Multigrid PDE Solvers](https://arxiv.org/abs/1902.10248).
This implementation is not a multigrid method and does not inherit its diffusion
results. Learning Fourier-space kernels is inspired by
[Li et al., Fourier Neural Operator](https://arxiv.org/abs/2010.08895), but this
model keeps residual linearity and all frequencies, and uses the exact physical
residual in a classical solver. Neither paper establishes RTE speedup for the
present architecture; that remains the experiment's goal.
