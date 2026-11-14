# Independent RTE scientific audit — 2026-10-02

Scope: read-only review of the current FWI source snapshot from
`/data/wjt/NeuralPreconditionedBornSeries`, branch `main`, recorded HEAD
`028caea3f35d100225f656e2b1e910c1366aaafe`, including its later uncommitted
September research changes. Local audit sources are under `remote_reference/`.
RTE means radiative transfer equation. No model, solver, original test, or
training code was changed by this audit. Numerical tables below are documented
saved results, not an independent execution of the experiments.

The initial assessment below preceded the controlled study. The dated
root-execution follow-up at the end records its completed validation result and
the subsequently executed error-bound diagnostics; earlier proposals are
preserved as the decision record.

## Evidence that survives the audit

The actual benchmark is a periodic two-dimensional Fourier discrete-ordinate
problem, with twelve angular directions, fixed circular phase parameter
`g=0.65`, and stiff scaling `epsilon=0.1`. It does not establish results for
inflow boundaries, spherical angular quadrature, arbitrary sources, or other
stiffness/phase regimes (`rte/models.py:17,82,89,189`).

The latest tail benchmark has a materially sound comparison protocol. Classical
and neural choices are frozen on sixteen validation media at each resolution
before fresh test generation (`rte/benchmark_tail_solver.py:195,205,207`).
Repeated timings are reduced to case medians, and confidence intervals resample
paired media rather than timing repetitions (`:46,156`). The 32 seed-8626 media
are shared across N64/N128/N256; they are 32 independent physical draws.

Documented selected-neural slowdowns are 4.78%, 5.84%, and 10.05%, with paired
intervals wholly favoring the classical control. All 1,440 measured held-out
solves pass the original-RHS residual tolerance. A restricted TSA8 comparison
would obscure this loss; the tuned N256 winner is TSA16 with zero initialization
(`docs/research/RTE_KRYLOV_TAIL_RESULTS_2026_09_26.md:122,143,148,154,163`).
This is negative evidence about the tested pipelines in this family, not a
general impossibility result for learning.

The deployed solver checks the final residual independently against the
original RHS and preserves the old search space during injection
(`rte/augmented_fgmres.py:45,99,137`). Nonzero initialization adjusts the
correction tolerance correctly (`rte/source_initialization.py:21,26`). FP64
outer arithmetic copies the actual FP32 phase, directions, wavenumbers and
material rather than regenerating a different operator
(`rte/mixed_precision_transport.py:8`).

## Actionable limits

1. **Causal evidence:** the earlier immediate-loss and lookahead pilots change
   seed, batch size and schedule as well as objective. They do not isolate a
   loss effect. N64 training improves N64 work modestly but its benefit largely
   disappears under refinement. The N128 paired bank is prepared, not trained;
   it shares the same 128 training and 16 validation physical draws
   (`RTE_KRYLOV_TAIL_RESULTS_2026_09_26.md:205,227,231,247`).
2. **Rejection semantics:** `rte/tail_lookahead.py:41–55` masks a zero/dependent
   proposal into zero image/frontier vectors, then applies the baseline to
   zero. Its conservative masked loss stays unchanged. Deployment instead
   rejects the proposal and resumes the existing classical frontier
   (`rte/augmented_fgmres.py:70,93`). Therefore deployed-trace equivalence holds
   on accepted, nondegenerate injection/continuation paths; it does not hold on
   rejected paths. The current zero-proposal test explicitly codifies masked
   semantics (`rte/tests/test_tail_lookahead.py:43`). Exact fallback generally
   needs the original last Arnoldi vector: `span(Q,r)` determines the span,
   not its ordered frontier. Keep these semantics unchanged in the controlled
   resolution study and qualify the claim explicitly.
3. **Frontier-probe edge case:** `rte/probe_tail_frontier.py:27–31` queues the
   saved residual before acceptance. A finite nonzero proposal rejected for
   dependence can still alter the next classical input. The regression test
   covers only rejected zero (`rte/tests/test_tail_frontier.py:7`). This affects
   interpretation only if such rejections occur in the saved probe; it does
   not affect the default solver or invalidate all frontier observations.
4. **Classical trajectory mismatch:** all saved tail contexts and lookahead
   training are TSA8-balanced (`rte/audit_tail_injection.py:32,51`;
   `rte/train_tail_lookahead.py:48,90`). The strongest N256 comparison is
   TSA16-zero. A resolution-only study can diagnose data transfer while leaving
   this separate alignment question open.
5. **Symmetry and claims:** CT's coefficient-conditioned residual map is
   complex-linear; learned gains do not inherit fixed-HSS convergence
   (`rte/collision_transport.py:3,7`). Its angular FFT approximation uses the
   first rounded phase row (`:60`). Tail networks are different, nonlinear
   residual maps: biased untied real/imaginary convolutions, GELU and GroupNorm
   explain phase defects (`rte/global_source_predictor.py:20,31,40,65`;
   `rte/characteristic_source_predictor.py:66`). Real-stacked FGMRES remains
   mathematically valid. Symmetry alone implies neither contraction nor speed.
6. **Accuracy:** final physical residuals are certified numerically, while
   held-out solution-error bounds are not reported. No test teachers are used,
   appropriately. The separate certificate design below can add a sufficient
   discrete bound using the actual rounded phase.
7. **Cost and reproducibility:** online reference initialization, inference,
   projection and final checks are timed; physical/preconditioner setup is
   recorded separately (`rte/benchmark_tail_solver.py:112,143,186`). Resident
   model loading/process startup, offline teachers, training and repeated-RHS
   amortization are not established deployment costs. The same validation
   cohort has guided multiple pilots; it is now development data. Seed 9626
   remains reserved behind the failed runtime gate
   (`rte/benchmark_tail_lookahead.py:54,68,74`). Original negative runs and cache
   failures must remain in the evidence chain. The CT pointer-reuse bug has
   dedicated setup-generation/storage-retention regressions
   (`rte/collision_transport.py:65,69`;
   `rte/tests/test_collision_transport.py:201,218`).

## Ranked next directions

1. **Controlled paired-resolution continuation — first.** Reuse the prepared
   N64/N128 bank, unchanged cheap architecture and unchanged continuation
   semantics. Compare N64-only with paired N64/N128 training using the same
   initial checkpoint, seed, updates, sample presentations, optimizer and
   phase augmentation. Select using both validation grids; retain all easy
   cases in full solves. Charge different offline wall time explicitly and
   gate against frozen tuned controls before using reserved fresh test data.
   This has the strongest direct evidence and lowest incremental data cost;
   its contribution would be a controlled explanation of resolution transfer,
   not a novel generic learning architecture.
2. **Complex-linear useful-image diagnostic — next.** On existing contexts,
   hold material, residual `r` and old image basis `Q` fixed. For four quarter
   turns compute `z_phi = exp(-i phi) N(exp(i phi) r, exp(i phi) B0 r)` and
   `w_phi = (I-P_Q) A z_phi`. Measure real-line angles, projected residuals and
   two-step continuation. Raw field defects may lie inside old `AZ` or be
   removable real scaling, so the recorded 0.1766 defect alone does not justify
   redesign. A four-phase average enforces quarter-turn equivariance, but not
   additivity/full phase symmetry, and would cost four forwards. Only pursue a
   factored material-conditioned complex-linear residual path if this useful
   direction test supports the hypothesis.
3. **Align the learned continuation with the tuned classical trajectory.**
   Separately prepare a modest training-only control behind TSA16-zero and
   compare against TSA8-balanced at matched effort. Add heterogeneous DSA/AMG
   or parity-based classical controls when assessing broader solver claims;
   low-fill ILU failures do not reject stronger factorization, but the existing
   high-fill/LU setup and memory costs make a large sparse campaign a poor
   immediate substitute for this bounded test.

No novelty or production-solver superiority is claimed by this audit. The
certificate design is in `rte_error_certificate_design.md`; the root executor
subsequently ran its validation diagnostic, as recorded below.

## Root-execution follow-up — 2026-10-02

The root executor completed the controlled FWI study after this independent
pre-study audit. N64-only and alternating N64/N128 continuations used the same
initial weights, ID/phase plan, updates and material presentations; both selected
on all sixteen validation media at both training grids. Original solver,
network, loss and stored evidence were preserved. The completed comparison is
validation evidence on reused development media, not a fresh held-out claim.

Retrieved execution receipts show mixed-resolution savings against the frozen classical methods of
-0.1083% at N64, -0.6784% at N128, and -6.9643% at N256. The N256 paired interval
is [-10.3769%, -2.8457%]. Savings against the matched N64-only arm are +0.2636%,
+0.8298%, and +1.5793%; those intervals include zero. All 480 measured validation
solves pass. The predeclared zero-failure/5%-saving gate fails, so test seed 9626
remains unused. This bounded comparison does not establish that paired-grid
training repairs the runtime problem, and it does not justify an automatic
larger continuation of the same experiment.

The root executor also completed the separate coercivity diagnostic: all 48
frozen-classical validation solves pass, the minimum recorded discrete margin
is 0.01270603, and the maximum computed relative solution-error bound is
0.0002368861. These are FP64 evaluations of the stated discrete algebraic bound;
they are not interval-certified arithmetic or continuous/physical positivity
certificates. Diagnostic costs remain separate from benchmark solve times.

Execution receipts are retained under
`receipts/rte/workstation_runs/resume_audit_20261002/`. The next scientific
priority is the useful-image symmetry diagnostic, followed by alignment with
the tuned classical trajectory; the controlled resolution question above has
now received its bounded negative runtime answer.

Independent compact-receipt inspection confirmed both completed 1,000-update
arms, 4,000 presentations per arm, identical consumed original IDs/phases, and
selected step 800 for both. The 480 raw measured validation rows and all 48
bound rows contain zero failures; no test files were generated. The calculated
relative-bound formula agrees exactly with each saved record. Preservation
receipts report zero mismatches across 155 original source files and all
supplied training inputs. No serious reporting inconsistency was found.

Supporting receipts: `paired_training/settings.json`, both arm
`sampling_consumed.csv` files, `runtime_gate/validation_comparisons.json`,
`runtime_gate/validation_summary.csv`, `solution_bounds/cases.json`,
`solution_bounds/summary.json`, and `preservation_verified.json` under that
receipt directory. Training occupied 349.58 seconds including preparation,
selection and checkpoint work; this is separate from the warm runtime comparison.
