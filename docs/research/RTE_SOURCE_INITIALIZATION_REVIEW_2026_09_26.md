# RTE bigger-picture review and source-conditioned learning, September 26

## Decision after the completed experiments

The most promising next target is **one learned correction direction after a
short classical prefix**, inside continuing FGMRES. It should learn the
remaining difficult error, run only once on cases that still need work, and
retain the original residual certificate. Training/validation tail datasets
have been prepared. No tail-correction network has yet been trained.

This campaign first tested one-shot, source-conditioned neural initialization
at N64/N128/N256. Neither a global/local Fourier predictor nor its fine-grid
characteristic decoder produced a useful held-out speedup. Those are actual
trained negative results, not just oracle tests. The correction-direction
decision below follows the measured error decomposition and fit control;
it is not a claim that the next network already works.

The previous [recursive-response results](RTE_RECURSIVE_RESPONSE_RESULTS_2026_09_19.md)
remain valid negative evidence about the representations tested. A general
inverse cache already occupies about 127 MiB at N32 and loses its online
advantage there. Fixed coarse responses and low-rank repairs did not fix
deployment cost. Meanwhile a tuned classical solver needs only a few outer
steps in the moderate regime. Repeating an expensive network inside those
steps leaves little overhead budget.

Learning an entire inverse is stronger than the workload requires. A source-
conditioned field has O(L N^2) outputs, runs once, and can amortize training
across independently sampled materials and sources. The potentially useful
learned information is the **global, material-dependent source response**:
long paths, interfaces and direction-dependent cancellation. Neither a local
coefficient head nor a fixed radial low-frequency subspace captures that
information reliably in the previous trials. This is a hypothesis, not a
demonstrated speed advantage.

Neural warm starts are not a new general idea:
[NOWS](https://arxiv.org/abs/2511.02481) already studies learned initialization
for Krylov solvers. [HINTS](https://arxiv.org/abs/2208.13273) motivates combining
learned and classical error removal, but its spectral intuition should not
be assumed to identify the hard modes of this anisotropic transport problem.
The RTE-specific question here is which response to predict, in what metric,
and whether its online cost beats genuinely tuned classical controls.

## Accuracy and workload contract

- Original periodic two-dimensional Fourier RTE, twelve angles, g=.65;
  the Nyquist convention and physical discretization are unchanged.
- Stiff scaling epsilon=.1, performed on the stored FP32 material. Outer
  FP64 arithmetic copies that exact geometry/material/phase, not a regenerated
  higher-precision problem.
- Every final solve independently checks ||q-A psi|| / ||q|| <= 1e-5.
  With a nonzero initial guess the correction solve uses tolerance
  1e-5 * ||q|| / ||q-A psi0||. Using 1e-5 relative to the smaller correction
  RHS would impose a different and unnecessarily strict stopping criterion.
- Network output is not an accepted solution by itself. Full-angle FGMRES
  performs refinement, capped at 100 steps, restart 25. All failures remain
  in the results.
- New independent train/validation/test material and source draws; material
  rotations and translations are independent of the sources. Nested N64/N128
  coefficient fields agree exactly at common grid points. The two resolutions
  share test media and are not independent statistical samples.
- This is still a synthetic mixed-medium family, fixed angle count and fixed
  stiffness. It is not evidence for arbitrary geometry, inflow boundaries,
  strongly out-of-distribution sources, or general RTE performance.

## Diagnostic headroom: fine-solution information, not a deployable solver

The source-initialization oracle uses the original four-medium seed-1919
development draw. Accurate fine solutions supply selected Fourier
coefficients. Teacher solves are not available to an online solver; their
time is excluded only to measure an optimistic information target.

Mean iterations and mean per-medium median synchronized GPU time:

| Grid / diagnostic run | Initial guess | Iterations | Warm ms |
|---|---|---:|---:|
| N64, first run | Zero | 9.75 | 153.84 |
| N64, first run | Exact fine solution, radial band32 | 7.50 | 119.30 |
| N64, first run | Exact fine solution, directional 1024 / angle | 7.00 | 111.44 |
| N128, first run | Zero | 9.75 | 153.89 |
| N128, first run | Exact fine solution, radial band32 | 7.50 | 119.08 |
| N128, first run | Exact fine solution, directional 1024 / angle | 7.25 | 116.29 |
| N128, wider-band run | Zero | 9.75 | 165.32 |
| N128, wider-band run | Exact fine solution, radial band64 | 7.00 | 119.16 |
| N128, wider-band run | Exact fine solution, directional 4096 / angle | 5.50 | 96.26 |
| N128, wider-band run | Actual classical coarse64 solve + prolongation | 7.75 | 644.81 |

Directional selection ranks (Omega_l dot k)^2 + .01 |k|^2 separately per
angle. These are cross-characteristic modes, not simply the smallest |k|.
The radial and directional controls above have only approximately equal
coefficient budgets because strict radial restriction drops coarse Nyquist
lines. The result suggests preserving fine directional content in the
prediction; it does not establish that those coefficients are learnable or
that a learned approximation will achieve the oracle times.

The actual rediscretized coarse solves include their solve and prolongation
cost. They are too expensive in these controls. A globally balanced constant
guess saves no iterations. Fine-solution MSE alone is also insufficient:
small solution error can coexist with an initial physical residual exceeding
the original RHS norm.

## Tested initialization architecture and training

`GlobalSourcePredictor` combines a cheap homogeneous full-angle transport
response with a multiscale global/local network. Inputs include material,
absorption, source, reference solution, periodic material gradients and scale.
Two downsampling levels feed four Fourier mixing blocks; a fine-grid decoder
with skip connections emits all twelve complex angular fields. The reference
response is retained as an additive skip. Width is 32, Fourier mode parameter
6; the final parameter count is recorded by the run.

An exact scalar correction enforces integrated source/collision balance
using the actual stored phase matrix, not an assumed exact normalization.
No fine inverse, dense teacher matrix or test solution enters inference.
The current decoder is not yet a characteristic-aligned architecture; the
directional oracle motivates that follow-up if the general global/local
pilot cannot realize the headroom.

Training has 128 accurate N64 solution labels, tolerance 1e-9. Failed TSA8
teacher solves fall back to HSS16 shift2 rather than dropping difficult media.
There are sixteen validation media; eight select checkpoints by failures,
then TSA8 outer iterations, then true residual. The first 1,000-update pilot
uses 400 supervised updates followed by three-step differentiable FGMRES
training. It improves validation from 9.375 to 9.0 iterations only.

Its timing sweep was stopped before test-set evaluation. The selected
checkpoint and exact teacher bank are reused for a precommitted 1,500-update
continuation: six-step differentiable FGMRES, learning rate 2e-4, and loss
log(1 + terminal_relative_residual_squared / 1e-10), plus a small solution
error regularizer. This aims directly at reaching the actual tolerance after
six steps. The optimizer/scheduler/RNG reset is deliberate, not an exact
training restart. The new test seed offset is 4,000.

The final comparison freezes classical and neural refinement choices on
validation before generating twenty-four test media. Classical tuning
includes TSA depths 2/4/8/16/32, full-reference depths 4/8/16/32, and five HSS
depth/shift combinations, both with zero and balanced analytical initial
guesses. Neural refinement is selected among the three best classical base
families plus TSA8. Test timings use three shuffled repetitions after warmup;
initialization, reference application, balance and final residual checking
are included. Preconditioner setup is recorded separately: setup plus warm
solve is an estimate, not a measured cold-start end-to-end solve.

## Completed global/local predictor: no useful held-out speedup

The base predictor has 666,680 real-equivalent trainable parameters. Its
six-step continuation takes 379.14 seconds including checkpoint validation;
update 1,500 is selected. There are 864 measured held-out solves, but only
twenty-four independent test media, shared across N64/N128.

Validation chooses balanced-reference-initialized TSA8 as the classical
method, and the neural predictor followed by TSA16 as the neural method.
These choices are frozen before generating test data. Main test result:

| Grid | Classical warm ms | Neural warm ms | Neural time saving | 95% paired-media bootstrap interval |
|---|---:|---:|---:|---:|
| N64 | 153.94 | 158.99 | -3.28% | [-15.36%, +5.64%] |
| N128 | 153.64 | 160.93 | -4.75% | [-17.66%, +4.64%] |

Every method meets the physical tolerance on all twenty-four cases.
Negative saving means the neural method is slower. The uncertainty estimate
uses 10,000 paired resamples of media, not timing repetitions. It does not
replace independent training/data replications.

With TSA8 held fixed, the neural initialization saves 0.78% against the
balanced analytical initializer at N64 and -0.04% at N128. All corresponding
intervals include zero. It reduces mean outer iterations from 9.42 to 9.00
at N64 and from 9.42 to 9.08 at N128, but initialization costs about 6.1 ms
versus 0.55 ms analytically. This is not a meaningful learned speedup.

Thus neither more general global mixing nor the longer-horizon objective
alone clears the deployment gate. The directional oracle remains an
information target; its 42% optimistic time reduction is not achieved by
this trained network. See `source_initialization_report_20260926/` for
all methods, failures, timing repetitions, paired comparisons and hashes.

### Completed, precommitted structural continuation

Before inspecting the first held-out results, the validation plateau at
8.875 iterations motivates one further bounded architecture upgrade:
`CharacteristicSourcePredictor`. The nonlinear global/local network predicts
three sets of complex angular forcing fields. A fine-grid spectral decoder
applies rate / (rate + i Omega dot k), with positive, trainable rate scales
initially .25, 1 and 4 times mean sigma_t. These filters have modulus <= 1
and retain all cross-characteristic frequencies. A learned channel projection
combines their responses. This is not a radial frequency truncation.

The known directional filters themselves are classical and inexpensive.
The learned part is the material- and illumination-dependent forcing they
carry, conditioned on the global neural representation. A zero-initialized
new output path preserves the base predictor exactly at upgrade, as tested.
This is a physical inductive bias, not evidence that AI is needed for a
homogeneous transport inverse.

The precommitted continuation uses the six-step pilot's validation-selected
checkpoint, identical train/validation data, 1,000 additional updates at
2e-4, six-step tolerance-aware loss and checkpoints every 200 updates. The
fresh test-seed offset is 5,000; all classical selection is repeated before
generating the test set. This variant also tests N256, without retraining;
the same twenty-four media are shared across N64/N128/N256. GPU training
starts only after the preceding timing
sweep ends. Because this also adds training, any difference is not a
matched-budget causal proof about the decoder alone.

The directional model has 670,811 real-equivalent parameters. Its 1,000-update
continuation takes 260.18 seconds including checkpoint validation, and
selects update 800. Validation chooses TSA8 with a balanced analytical guess
for the classical method and TSA8 with the network for the neural method.
The fresh test cohort has twenty-four independent media shared across three
resolutions, with 648 measured solves in total.

| Grid | Classical warm ms | Directional neural warm ms | Neural time saving | 95% paired-media interval |
|---|---:|---:|---:|---:|
| N64 | 164.92 | 164.98 | -0.04% | [-2.87%, +2.32%] |
| N128 | 152.95 | 155.40 | -1.60% | [-4.31%, +0.64%] |
| N256 | 177.69 | 181.26 | -2.01% | [-4.71%, +0.33%] |

All solves meet tolerance. Mean iterations change from 9.33 to 8.92 at N64,
9.38 to 9.13 at N128, and 9.38 to 9.21 at N256. The network's roughly
7.1-7.7 ms initialization cost erases these small reductions. No significant
AI benefit is established. Compare methods within a grid/run; different
measurement windows are not evidence that N128 is intrinsically cheaper
than N64. The two neural campaigns use different held-out cohorts, so their
tables are not a paired architecture ablation.

## What the error audit changes

Separate CPU diagnostics use eight validation cases and sixteen training
cases. These are numerical controls, not performance benchmarks. Validation
sources are reconstructed on CPU and may differ in final rounding bits from
the GPU timing sources; training diagnostics reuse the exact saved GPU RHS.

On the eight validation cases, the directional model has mean relative
solution error 8.05%, versus 8.90% for the balanced analytical guess. About
99.80% of its error energy lies within the directional 1024-coefficient-per-
angle band. Removing the complementary frequencies does not help:

| Validation initializer / substitution | Mean TSA8 iterations |
|---|---:|
| Zero | 9.625 |
| Balanced analytical reference | 9.375 |
| Neural prediction | 8.750 |
| Neural prediction, directional band1024 only | 8.875 |
| Oracle: replace only the neural directional band with exact values | 6.625 |
| Oracle: replace only its complement with exact values | 8.875 |
| Oracle: exact directional band1024 only | 6.375 |

Thus this is not primarily a noisy high-frequency-output problem. The
material-dependent response inside the useful band is inaccurate. The
sixteen training cases also retain 6.47% mean solution error, with only a
0.44-iteration reduction against the balanced reference. A held-out
generalization gap alone does not explain the failure.

A four-training-case fit control uses the same architecture and direct
relative solution loss for 1,000 CPU updates. It reduces mean solution error
from 8.093% to 0.2667%, but iterations only from 4.75 to 4.25. This is a
capacity/metric diagnostic, not an AI speed claim: memorizing four solutions
is not a meaningful solver benchmark. On those same four cases, exact
directional-band information needs 3.25 steps, whereas exact radial band32
information needs 4.25, despite comparable total solution error. Even a
thirtyfold reduction in solution error does not identify the useful error
geometry by itself.

## Next architecture target: one-shot Krylov-tail correction

The next model should predict **the remaining error after four TSA8 FGMRES
steps starting from the balanced analytical guess**, not the full flux.
The prepared training bank has 128 cases: sixteen already converge in the
prefix and should bypass AI, while 112 remain eligible. Across all 128,
the median remaining solution-error norm is only 2.15e-5 of the full-solution
norm, while the median physical residual is 7.81e-4 of the original RHS.
This scale separation explains why full-flux MSE is a weak guide to the
remaining solver work.

The saved target is e_star = psi_star - psi_prefix, with RHS
r_prefix = q - A psi_prefix. Its physical equation A e_star = r_prefix is
checked against the original RHS norm; the maximum discrepancy is 9.91e-10.
The bank stores exact FP64 tail RHS/labels, FP32 neural inputs, the cheap
homogeneous reference response, original material/source identities and
eligibility. Fine solutions are offline labels, never inference inputs.
A separate sixteen-case validation-tail bank has thirteen eligible cases.
Test cohorts remain excluded from these banks.

Concrete implementation order:

1. Add a single optional neural search direction to FGMRES after the prefix,
   retaining its existing basis and least-squares problem. Do not restart
   and discard useful classical work. Skip AI if already converged.
2. Reuse the global/material encoder and characteristic path initially to
   isolate the change of target. Condition on the actual signed/complex tail
   residual and its reference response. Normalize by residual/reference
   norms, not the positive-source full-flux scale. No per-iteration network
   or dense inverse is needed.
3. Train for the residual reduction actually obtained by adding that search
   direction, with normalized tail-teacher supervision as a regularizer.
   If Q spans the current **search images A Z**, set
   w_perp = (I-Q Q*) A z_theta using the real-stacked inner product. The
   one-direction optimal squared residual ratio is
   1 - <r,w_perp>_R^2 / (||r||^2 ||w_perp||^2). Handle a zero candidate safely.
   This is scale-invariant and targets useful physical response, rather than
   uniform six-step success on cases with very different difficulty.
4. Test the injection with exact tail teachers first: it must close the solve
   in one extra search direction. Do not confuse the search-image span with
   the entire Arnoldi span, which also contains the current residual. A
   perfect teacher can cause happy breakdown and must not be discarded.
5. Train only if that actual online oracle clears the cost gate. Then freeze
   choices on validation and use a fresh held-out cohort, including N256,
   against the same analytical initializer/classical depth controls.

Adding a direction cannot worsen the current least-squares residual in exact
arithmetic, because the old search space is retained. This is not a guarantee
of fewer future iterations or faster total time. The learned information is
the coherent, material-dependent response to the difficult residual; the
cheap homogeneous transport and scalar coefficients remain classical.
Only add a larger geometry-conditioned attention encoder or multiple learned
directions if the isolated target test supports it.

### Interpretation limits and next gate

The primary tuning is at N64 and transfers unchanged to larger grids; it is
not a separate exhaustive classical tuning at every resolution. A positive
large-grid claim would require a resolution-matched validation depth audit.
The current classical set is stronger than the old fixed-depth reference,
but is not a production heterogeneous DSA/AMG implementation or a fully
GPU-graph-optimized transport solver. Timings come from a shared workstation
with no competing GPU compute process at the resource checks; short CPU
software checks and single-core, low-priority CPU error/fit diagnostics also
occur during the campaign. This is not an isolated hardware benchmark.

The acceptance gate is a repeatable total-time reduction at the same true
residual, across independent media, with inference and relevant setup costs
counted. Lower training loss, fewer outer steps at a different depth, or
oracle-only headroom do not meet that gate. The directional pilot failed;
extending the same full-flux network budget is not the next priority. The
tail-target experiment isolates whether an actual difficult-residual input,
scale-invariant direction objective and conditional invocation change this.

## Evidence locations

- `rte/workstation_runs/source_initialization_oracle_fixed_20260926/`:
  completed N64/N128 eight-method oracle control.
- `rte/workstation_runs/source_initialization_band64_20260926/`:
  completed N128 wider-band oracle and actual coarse-solve control.
- `rte/workstation_runs/source_predictor_pilot_20260926/`: completed first
  training budget, exact teachers, checkpoints, partial validation timing,
  and explicit `STOP_STATE.md`.
- `rte/workstation_runs/source_predictor_terminal6_20260926/`: longer-horizon
  continuation, independent test protocol and final measurements.
- `rte/workstation_runs/source_predictor_characteristic_20260926/`: completed
  directional training and N64/N128/N256 held-out measurements.
- `rte/workstation_runs/source_prediction_error_20260926/` and
  `source_prediction_train_error_20260926/`: validation/training error audits.
- `rte/workstation_runs/source_fit_control_20260926/` and
  `source_fit_error_20260926/`: four-training-case capacity/metric controls,
  explicitly not validation-selected deployable weights.
- `rte/workstation_runs/source_tail_bank_20260926/` and
  `source_tail_validation_20260926/`: prepared next-target datasets only.
- `rte/workstation_runs/source_initialization_report_20260926/` and
  `source_characteristic_report_20260926/`: completed summaries/provenance.
- `rte/workstation_runs/source_predictor_smoke_20260926/`: end-to-end software
  smoke test only; not a research comparison.
- `rte/workstation_runs/source_initialization_oracle_20260926/`: failed initial
  API smoke attempt, preserved. No numerical results from it are used.
