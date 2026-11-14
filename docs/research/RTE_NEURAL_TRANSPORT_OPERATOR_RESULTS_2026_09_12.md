# Neural transport operator: first matched pilots

## Decision

Continue with **NTO-HSS4**, the four-stage, 51,236-parameter neural transport
operator. Its nonlinear material encoder generates full angular mixing matrices,
complex frequency responses and stage shifts, all cached before Krylov solves.
The two-stage candidate is not competitive enough in wall time.

The first GPU pilot gives a real, scoped positive result: NTO4 reaches true
residual tolerance 1e-5 on all 48 comparison media, with median warmed solve
36.54 ms versus 48.04 ms for tuned HSS and 47.09 ms for TSA. That is roughly
24% / 22% less time using the ratios of aggregate medians. Preparation-inclusive
median first solves also improve in this run. This is **one training seed on
development validation data**, not a fresh confirmatory test or a universal win.

The [architecture document](RTE_NEURAL_TRANSPORT_OPERATOR_DESIGN_2026_09_12.md)
gives equations, constraints, costs and the training contract.

## Matched training

All four main arms were trained from their analytic zero-head initialization
on the same fresh N32/L12/g=0.65 ensemble: 256 training and 64 validation media,
seed 401, batch 16, 1,600 updates, 25,600 material presentations. All used AdamW
with lr 0.001, the same cosine horizon, clipping, residual draws, and physical
one-step plus 0.25 two-step rollout objective. Full-spectrum A e probes occupy
20% of samples; the remainder uses the old TSA/smooth-probe mixture.

| Arm | Parameters | Selected update | Selection mean iterations | Final validation objective |
|---|---:|---:|---:|---:|
| NTO2 | 35,186 | 1600 | 12.1875 | 0.068658 |
| CT2 | 1,132 | 800 | 12.8125 | 0.10212 |
| NTO4 | 51,236 | 800 | 5.1250 | 0.0050837 |
| CT4 | 1,340 | 1200 | 6.1250 | 0.010745 |

The objective column refers to the final weights, not necessarily the selected
weights. For example, NTO4's selected update-800 objective is 0.0055042. There
were no selection-source failures for these four arms. The same learning rate
is a controlled comparison, not independent optimal hyperparameter tuning for
each architecture. The old 128,000-update CT checkpoint is not used as the
matched control.

The averaged-reference NTO2 arm is separately preserved at 20 updates. It
started with 4/16 selection failures and had 6/16 after that short run; step
zero remained selected. This justified exploring the HSS anchor, not declaring
the reference arm incapable of learning.

## Primary GPU comparison: no competing process observed

Artifacts:
[`nto_matched_final_gpu_eval_20260912`](../../rte/workstation_runs/nto_matched_final_gpu_eval_20260912/summary.csv).
Device: RTX A6000, Torch 2.5.1/CUDA 12.4. Classical settings were tuned on the
first 16 validation media; frozen models were compared on the remaining 48.
There were three warmed repeats plus one first solve per method/medium, with
randomized method order. All nine methods converged on every comparison medium.
The 49 process snapshots contain no other compute process apart from the
desktop integration service. This is observed telemetry, not exclusive-device
locking or a claim that clock/host variability was eliminated.

| Method | Mean iterations | Median warmed solve ms | Median preparation + first solve ms |
|---|---:|---:|---:|
| **NTO4** | **5.8333** | **36.54** | **42.38** |
| Matched CT4 | 6.7083 | 36.37 | 40.62 |
| Tuned HSS | 8.4583 | 48.04 | 49.44 |
| Tuned TSA | 6.2292 | 47.09 | 49.88 |

These timings take a median within each medium, then a median over media;
the evaluator's `summary.csv` instead also provides percentiles pooled across
all warm repetitions, which can differ slightly. Paired median baseline/NTO4
warm speedups are **1.299 for HSS** and **1.430 for TSA**. They are not the ratios
of the table's aggregate medians. NTO4 is faster on 46/48 HSS pairs and 47/48 TSA
pairs. Its pooled warm p95 is 53.18 ms, versus 73.93 / 95.46 ms for HSS/TSA.

NTO4 uses 30.17 mean physical A calls, versus 34.54 for CT4, 43.29 for HSS and
57.06 for TSA. Counts include nested physical residual work. One HSS/NTO stage
is not the same work as one TSA stage.

NTO4 is **not yet a decisive wall-time improvement over the smaller CT4**:
their warmed aggregate medians are essentially tied, and NTO4 has higher neural
preparation cost. Its lower iteration count demonstrates useful additional
operator expressiveness; it does not by itself establish better deployment cost
than every existing learned alternative.

![Matched pilot](../../rte/workstation_runs/nto_architecture_report_20260912/comparison.png)

## Stronger classical representation control

[`compiled_hss.py`](../../rte/compiled_hss.py) gives classical HSS the same
cached local collision-matrix representation, omitting neural maps and their
overhead. Its action is checked against the original angular-FFT HSS. Twelve
depth/shift candidates were retuned on the 16 selection media; depth four and
factor one remained selected.

The five-repeat
[`strong-control rerun`](../../rte/workstation_runs/nto4_strong_controls_gpu_eval_20260912/summary.csv)
completed with no numerical failures. However, another project's training
process, PID 3406162, first appears in telemetry at medium 55 and remains
through 63. The **full rerun is not an uncontended timing confirmation**.

Using only the telemetry-defined prefix, indices 16–54 (39 paired media),
NTO4 beats compiled HSS on 38/39 warm pairs, with median paired speedup **1.312**
and preparation-plus-first paired speedup **1.160**. The prefix was defined by
process appearance, not by favorable solver outcomes. It is supplementary
evidence on a subset, not a new independent sample or a replacement for a full
uncontended rerun. Mean first-solve-plus-preparation cost in this prefix is
51.09 ms for NTO4 versus 50.01 ms for compiled HSS, reflecting startup/outlier
costs; do not claim superiority for every aggregation or cold-use scenario.

## What the learned components contribute

Frozen-weight interventions on the 16 selection media, CPU float32:

| Intervention | Mean iterations | Mean source physical loss |
|---|---:|---:|
| Full trained NTO4 | 5.1250 | 0.004946 |
| Disable angular mixing | 5.9375 | 0.013323 |
| Disable spectral modulation | 7.0625 | 0.019189 |
| Replace learned shifts with analytic shifts | 7.6250 | 0.048163 |
| Entire analytic HSS anchor | 7.3125 | 0.095023 |

All converge. The trained components interact: disabling shifts while retaining
other trained factors can be worse than removing all learning. These are
leave-component-out deployment interventions, **not retrained ablations**, and
their effects are not additive causal contributions.

Evidence:
[`component probe`](../../rte/workstation_runs/nto4_component_probe_20260912/summary.csv).

## Implementation and negative results

Cached real local matrices can apply to real/imaginary residual parts as two
real RHS columns, halving their storage relative to complex storage. Tests
check equivalence to the complex training path. This is not an automatic runtime
optimization: the isolated four-medium microbenchmark measured complex versus
real-two-RHS medians of 3.961 / 4.031 ms on GPU and 8.015 / 8.717 ms on CPU.
GPU relative action discrepancy was at most 2.09e-7. The recorded solver runs
use the real-two-RHS cache; its memory saving must not be presented as a measured
kernel speed gain.

NTO2's final GPU median is about 62.96 ms and mean iterations 13.604, versus
CT2's 61.80 ms and 14.042. NTO2 also loses to classical HSS/TSA in its CPU
comparison. Its richer action does not compensate for too few correction stages.
Do not promote the original two-stage speed hypothesis.

The early update-400 CPU timing files are exploratory only: they predate the
real-cache implementation and their exact update-400 checkpoint was overwritten
on continuation. Its original hash is retained, but the checkpoint itself was
not preserved. The final evaluations save checkpoint snapshots inside their
own directories; newer training runs also retain selected-step snapshots. Final
claims above use the fully retained frozen checkpoints.

## Proof and measurement boundaries

- The 48 comparison media are development validation data. They contribute to
  bank loss and architecture decisions; there is no untouched confirmation test.
- Only one training seed has been studied. Paired bootstrap intervals condition
  on these fixed weights, not uncertainty over training runs.
- Preparation-plus-first timing assumes an initialized process, resident model
  weights and an already constructed physical operator. It is not process-cold
  latency; model startup, loading, transfers and training need separate budgets.
- Scalar-preserving angular maps, bounded factors and residual linearity do not
  prove convergence of the complete learned solver. All acceptance decisions
  use independently recomputed true physical residuals in the existing FGMRES.
- CPU/FP32 least-squares behavior can vary near tolerance; retain per-repeat
  outcomes. No tolerance relaxation or removal of failed cases was used.

## Next research step

Freeze NTO-HSS4/update-800 as the candidate. First complete the full uncontended
compiled-HSS comparison. Then train seeds 402/403 with the same protocol and
run an untouched confirmation ensemble after fixing the remaining choices.
Retrain no-mixing/no-spectral and zero-rollout controls to test the component
interpretation. Only then scale data/updates and vary resolution, angles or g.
NTO4 is a promising more-expressive AI preconditioner, not a finished universal
replacement for classical solvers.

## Reproduction and integrity

```bash
python -m rte.train_neural_transport --model nto --depth 4 --base hss --steps 1600 --seed 401 --output rte/workstation_runs/nto4_reproduction
python -m rte.train_neural_transport --model ct --depth 4 --steps 1600 --seed 401 --output rte/workstation_runs/ct4_reproduction
python -m rte.evaluate_neural_transport --checkpoints rte/workstation_runs/nto4_reproduction/best.pt rte/workstation_runs/ct4_reproduction/best.pt --device cuda:0 --compiled-hss --repeats 5 --output rte/workstation_runs/nto4_comparison_reproduction
python -m unittest discover -s rte/tests -q
```

The final suite passes **79 tests**; the original handoff's 178 artifacts remain
unchanged. New results, checkpoints, actual train/validation tensor snapshots,
plots and hashes are retained under
[`nto_architecture_report_20260912`](../../rte/workstation_runs/nto_architecture_report_20260912/summary.json).
The telemetry-prefix analysis is separate under `nto_compiled_prefix_report_20260912`.
No RTE training or benchmark from this task remains active. No unrelated job was
interrupted, and no historical artifact was deleted or overwritten by these new
experiment directories.
