# RTE training priority: completed results and handoff

4 October 2026. All physics, training, checkpoint inspection and solver work ran on FWI. The Mac bridge rendered returned scalar evidence only.

## Decision

The earlier blanket recommendation to stop tail training was premature. A matched 1000-update experiment on TSA16 deployment residuals improved the learned direction, approximately halved its two-step development residual ratio, and reduced mean classical calls from 4.5000 to 4.3750 at all three grids. More updates on the old weak-prefix objective produced only a small solver-work improvement. Neither experiment established a speed advantage over the strongest previously frozen, measured classical controls.

All twelve benchmark candidate labels fail the predeclared gate: at least 5% mean time savings at BOTH 128 and 256, zero convergence failures, and a positive lower paired interval. The reserved seed 9626 was never generated or accessed. There is no fresh-test, independent-training-seed, OOD, continuum-error or global training-saturation conclusion.

The next priority is a bounded, tolerance-aware objective on the aligned residual distribution: optimize the expensive classical calls needed to reach the original-RHS tolerance. Hold architecture, matched data, update budget and scheduler fixed; compare against the present two-step loss. This is a hypothesis for the next study, not a demonstrated remedy. This batch is complete and no further training was launched.

## Exact workspace and preservation

SSH alias FWI authenticated as wjt on FWI-Standard-PC-i440FX-PIIX-1996. Access used existing credentials, strict host-key checking, bounded timeout and disabled forwarding.

Repository: `/data/wjt/NeuralPreconditionedBornSeries`; branch `main`; HEAD `028caea3f35d100225f656e2b1e910c1366aaafe`.

Experiment: `/data/wjt/NeuralPreconditionedBornSeries/rte/workstation_runs/training_priority_20261004`.

RTE means radiative transfer equation in this repository. This study uses the steady periodic two-dimensional Fourier-discretized problem, 12 angular directions, g=0.65 and epsilon=0.1, with the existing mixed medium/source family. It is distinct from the separately paused OL_preconditioner project.

All 133 pre-existing files in the scoped preservation manifest remained byte-identical. HEAD, branch and the original nine tracked dirty files remained unchanged; no original implementation was edited. New experiment code, logs, banks and checkpoints are isolated under the experiment directory. No existing jobs were cancelled, no Git push/merge occurred, and no packages, access settings or system configuration were changed.

## Training protocol and provenance

All arms start from model weights in `resume_audit_20261002/paired_training/mixed64_128/best.pt`, selected at update 800. Optimizer, scheduler and sampling RNG are reset explicitly; this is weights-only continuation, not an exact optimizer-state resume. TailResponseNetwork has 670811 parameters, width 32, four blocks and six modes. Batch size is four, alternating grids 64/128, with AdamW weight decay 1e-5, gradient clipping at one and cosine decay to 5% of the initial learning rate. The objective is mean log squared residual ratios after 0/1/2 classical steps plus 0.05 normalized teacher-direction loss.

Training uses existing seed-3626 media; validation and timing reuse seed-4626 development media. Original accurate teacher targets are retained and reused. The alignment study creates prefix residual states, not new accurate teacher solves. Nested spatial grids share physical media and do not constitute independent validation cohorts.

| Arm | Additional updates | Unique original train IDs | Selected update | Optimizer seconds |
|---|---:|---:|---:|---:|
| Continuation | 3000 | 112 | 3000 | 371.844 |
| Lower LR control | 1000 | 112 | 1000 | 121.445 |
| Matched TSA8 | 1000 | 56 | 1000 | 125.597 |
| Matched TSA16 | 1000 | 56 | 1000 | 181.751 |

Continuation and lower-LR control use 112 eligible physical IDs per grid, shared seed 41004 and identical first 1000 sample IDs/phases. Initial rates are 1.5e-4 and 5e-5, both with scheduler horizon 3000. Rate comparisons are valid at matched update 1000; comparing 3000 with 1000 confounds rate and budget.

The matched alignment arms use the same 56 original physical IDs eligible in both prefix banks at both grids, identical sample IDs/phases, seed 41005, initial rate 1.5e-4 and horizon 1000. Both are warm-started from the same original checkpoint. The contrast is TSA8-balanced versus TSA16-zero prefix states: prefix depth AND initialization change together. It does not isolate depth alone. Cross-study comparisons also change the eligible cohort (112 versus 56).

Full solver validation uses all 16 development media at both training grids every 250 updates. Selection is lexicographic: failures, mean classical calls plus 0.5 mean injections, then mean original-RHS residual. Fixed phase-zero panels use the first 16 eligible training IDs and all eligible development IDs: 13 for TSA8 and six for TSA16 per grid. Reference and exact-teacher diagnostics are retained. Diagnostic residual ratios are normalized to each prefix residual, not the original source; they are squared and cannot be compared directly with the 1e-5 solver tolerance.

All 6000 planned optimizer updates completed with finite logs. All observed gradients were clipped. Selected checkpoints are the final endpoints in every arm, with identical selected/final model weights confirmed by tensor-content hashes; duplicate endpoint benchmark labels measure timing variability, not model differences. Checkpoint file hashes and model hashes are retained separately.

## What the learning curves support

Weak-prefix continuation reduced the validation work proxy from 7.8750 to 7.8125 by update 1000 and remained there through 3000. Loss continued to improve, while development two-step improvement was small and six-step diagnostics were flat or slightly worse. Relative weight drift diminished across successive 1000-update windows (0.1286, 0.0727, 0.0204). This supports a task-level plateau for this objective and cohort; it does not prove adequate training for every objective or global convergence.

Matched TSA16 training reduced its own work proxy from 4.6875 to 4.5625 at update 250, then remained there through 1000. Its development two-step squared ratio improved from 0.0010048 to 0.0004743 at grid 64 and from 0.0009655 to 0.0004977 at grid 128. Teacher-direction geometry also improved. Learning continued after the discrete work plateau. The matched TSA8 arm made much smaller changes. These fixed panels have different eligible cohorts and normalization; their absolute values are not directly comparable across prefixes.

![Completed scalar evidence](training_priority_evidence.png)

Figure: upper right is normalized within each arm's own fixed panel; its panels contain different eligible media. Lower-left intervals are conditional, exploratory paired media intervals on reused development data. Lower right holds TSA16 deployment fixed.

## Final synchronized timing

The final joint sweep contains 1920 measured solves plus 640 discarded warmups; the earlier completed sweep contributes another 912 measured solves plus 304 warmups. For every method/case/grid, one warmup precedes three synchronized, shuffled repeats. Case medians are taken before averaging across 16 physical media. Timing includes reference initialization, neural inference, projection, physical solver work and the final original-RHS residual check. Physical/base setup is measured separately; the constructor prepares its classical inverse eagerly. Model loading is excluded. Setup plus warm time is an estimate, not measured cold-start latency. There was no periodic GPU telemetry or exclusive GPU lock; start/end resource snapshots were retained.

| Grid | Method | Warm solve ms | Setup ms | Setup + warm ms | Classical calls |
|---:|---|---:|---:|---:|---:|
| 64 | Frozen classical control | 133.564 | 24.455 | 158.019 | 7.8125 |
| 64 | Frozen neural / TSA16 | 140.078 | 20.536 | 160.614 | 4.5000 |
| 64 | Extended weak-prefix | 131.586 | 24.455 | 156.041 | 7.3750 |
| 64 | Matched TSA8-trained / TSA16 | 140.210 | 20.536 | 160.746 | 4.5000 |
| 64 | Matched TSA16-trained / TSA16 | 134.911 | 20.536 | 155.447 | 4.3750 |
| 128 | Frozen classical control | 144.985 | 7.975 | 152.961 | 7.8125 |
| 128 | Frozen neural / TSA16 | 151.028 | 7.894 | 158.922 | 4.5000 |
| 128 | Extended weak-prefix | 144.567 | 7.975 | 152.542 | 7.4375 |
| 128 | Matched TSA8-trained / TSA16 | 152.587 | 7.894 | 160.481 | 4.5000 |
| 128 | Matched TSA16-trained / TSA16 | 147.071 | 7.894 | 154.965 | 4.3750 |
| 256 | Frozen classical control | 145.957 | 10.460 | 156.418 | 4.5000 |
| 256 | Frozen neural / TSA16 | 152.659 | 10.460 | 163.119 | 4.5000 |
| 256 | Extended weak-prefix | 154.763 | 10.538 | 165.302 | 7.5000 |
| 256 | Matched TSA8-trained / TSA16 | 152.120 | 10.460 | 162.580 | 4.5000 |
| 256 | Matched TSA16-trained / TSA16 | 147.534 | 10.460 | 157.994 | 4.3750 |

The classical comparison is reference_tail at grids 64/128 and tsa16_zero at grid 256, frozen from previous measurements. These are the strongest previously frozen measured controls in this project protocol, not an exhaustive comparison against all radiative-transfer preconditioners.

Positive saving means faster than that grid's frozen classical control.

| Grid | Candidate | Mean saving | Conditional 95% interval |
|---:|---|---:|---:|
| 64 | Extended weak-prefix | +1.481% | [-2.113, +4.719]% |
| 64 | Aligned TSA16 | -1.009% | [-8.069, +4.848]% |
| 128 | Extended weak-prefix | +0.289% | [-3.726, +3.736]% |
| 128 | Aligned TSA16 | -1.438% | [-7.417, +4.420]% |
| 256 | Extended weak-prefix | -6.033% | [-9.807, -1.679]% |
| 256 | Aligned TSA16 | -1.080% | [-4.284, +2.093]% |

At grid 256 the aligned model is 1.08% slower than classical (interval crosses zero); the extended weak-prefix model is 6.03% slower. Alignment improves the same TSA16 neural pipeline relative to frozen neural weights by 3.69%, 2.62% and 3.36% at 64/128/256, respectively. Against the matched TSA8-trained model deployed behind the same TSA16 prefix, savings are 3.78%, 3.62% and 3.01%. These exploratory intervals are unadjusted; the 128 interval crosses zero. Identical final/selected weights differ by about 1–2 ms between duplicate labels, illustrating timing noise. The deterministic reduction of mean classical calls by 0.125 is the stronger evidence that alignment changes solver behavior.

The 10000-resample paired bootstrap resamples physical media after timing repeats are reduced by median. It quantifies conditional cohort variability and does not account for checkpoint-selection reuse, training-seed variation or multiple comparisons. Independent NumPy recomputation verified all 123 published comparison intervals to maximum error 1.78e-15 percentage points. No candidate passed the gate, including the separately timed final endpoint labels.

## Accuracy, mathematical assumptions and failures

All 2832 measured solves converged at original-source relative residual below 1e-5, with zero failures and maximum residual 9.9850116e-6. The saved 48 extended-model fields have source tensors bit-identical to reconstructed original right-hand sides. Independently reevaluating their residuals reproduces the same maximum. At 64/128, 32 fields were compared with existing accurate teachers for the same original sources; maximum relative difference was 6.4454e-7. No new grid-256 teacher was acquired. Aligned fields were not saved for this extra teacher comparison, although their original-RHS solver acceptance was checked during every measured solve.

For the actual discrete phase action P, let mu be the largest eigenvalue of its Hermitian part and m=min(sigma_t-sigma_s*mu). Uniform angular weights, nonnegative scattering and positive m make periodic Fourier streaming skew-adjoint and give the exact-arithmetic estimate ||error|| <= ||residual||/m. With delta=||residual||/m, the relative solution-error bound is delta/(||computed solution||-delta) when the denominator is positive. All 48 saved cases pass the evaluated assumptions; maximum evaluated relative bound is 2.3689e-4. Ordinary FP64 evaluation is not a directed-rounding enclosure, interval certificate, continuum discretization-error bound or positivity guarantee.

Four existing gradient/physical-trace/batch-independence tests passed (3.69 seconds), and a two-update smoke run completed before full training. An initial unittest invocation collected zero tests and was replaced with pytest. The first alignment attempt saved the grid-64 bank, then failed in our new CSV writer because an ineligible row omitted a column; the retry supplies the column consistently and reuses the bank after verifying its parent hash. No aligned training had started at the failure. The failed receipt and original attempt remain preserved. A temporary Mac tool transport failure recovered; remote jobs continued. There is no remaining access blocker.

## Costs and reproducibility

FWI hardware: one RTX A6000, 49140 MiB; Python 3.12.4, PyTorch 2.5.1/CUDA 12.4. GPU jobs ran serially. Actual optimizer time totals 800.637 seconds. Validation, fixed panels, checkpoint I/O and bank preparation add cost: internal phase-one wall time 650.226 seconds, alignment retry 454.879 seconds, original failed preparation 25.269 seconds, final timing 380.690 seconds, and saved-field accuracy audit 7.374 seconds. The earlier timing wrapper took 198.193 seconds. These phases exclude human audit and Mac rendering; they are not a single elapsed batch duration. Peak GPU reserved memory was 8.645 GiB for continuation and 4.916 GiB for alignment, below the 14 GiB guard. There are 126 retained tensor/checkpoint files totaling 3542487710 bytes, including preserved duplicated banks and saved solution fields. They remain on FWI.

Reproduction entry points under the remote experiment: `code/train_adequacy.py`, `code/train_alignment.py`, `code/benchmark_adequacy.py` and `code/audit_saved_accuracy.py`. Run from the repository root with its existing Python environment. Read `PROTOCOL.json`, `PROTOCOL_ALIGNMENT.json`, `FINAL_ENDPOINT_CONTROLS.json` and each phase's `settings.json` for exact launch settings and paths before rerunning. Fresh output directories are required; do not overwrite these results. `training/*` and `alignment_retry/*` retain sampling ledgers, fixed panels, update histories, full solver validation, receipts and checkpoint hashes. `benchmark_joint/*` retains raw repeats, case medians, summaries and comparisons. `accuracy/*` retains original-source checks. `ANALYSIS.json`, `ALL_CANDIDATE_GATES.json` and `bootstrap_verification.json` contain independent checks.

## Remaining evidence needed

Prioritize an aligned, tolerance-aware objective with a matched two-step control before increasing model size or launching longer campaigns. Include independent initializations, clipping/learning-rate diagnostics, equal budget and identical original physical sample IDs; preserve unsuccessful runs. Require the same timing gate with all setup/inference costs and original-RHS acceptance before opening a fresh test cohort. Only then evaluate held-out media, source families, scattering/anisotropy changes, angular resolution and boundaries. One warm start, one sampling seed and repeatedly reused 16 development media cannot support generalization or global saturation claims.
