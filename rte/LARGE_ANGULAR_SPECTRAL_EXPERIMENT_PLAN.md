# Large RTE Experiment Protocol: Corrected 2026-09-12

Read the [evidence audit](../docs/research/PROJECT_AUDIT_2026_09_12.md) first.
The old version of this plan incorrectly counted 216 residual entries as
coefficient environments and called two minibatch updates an epoch. Its
4,096-sample, 2,000-epoch, batch-64 setting presented 256,000 samples: **62.5
effective passes**, not 2,000. It also had a regime-subset indexing bug.
Those configurations are not evidence that thousands of full epochs failed.

The repaired runner uses complete shuffled passes by default, including the
last partial batch. `--steps-per-epoch K` is an explicit cap, not a new meaning
of epoch. History records actual updates, sample presentations and effective
passes. No large GPU experiment described here has been run in this audit.

## Scientific controls

These are **coefficient-ensemble** trials, not fixed-medium residual-bank
trials. Source and material fields vary independently; `g=.65` remains fixed
because an operator instance has a single angular phase matrix. Data schema
version 3 includes the independent material channel phase and global regime
IDs. Old caches are rejected. The regime labels vary attenuation, absorption
and geometry; with common `g=.65`, even the `diffusive` ensemble label does not
mean the original isotropic `make_case('diffusive')` benchmark.

Default `--plan priority` schedules:

| Name | Distinct media/source samples | Full passes | Updates | Model/objective |
|---|---:|---:|---:|---|
| `ensemble_small_angular_teacher` | 256 | 32,000 | 128,000 | Angular, physical + rollout + action teacher |
| `ensemble_large_angular_teacher` | 4,096 | 2,000 | 128,000 | Same angular/objective |
| `ensemble_large_old_unet_teacher` | 4,096 | 2,000 | 128,000 | Legacy with extended coefficient features and scale-aware gates |
| `ensemble_large_angular_raw` | 4,096 | 2,000 | 128,000 | Angular, physical + shared tiny correction regularizer |

All use batch 64 and 8,192,000 presentations. The small dataset is an exact
prefix of the same 4,096-sample pool; evaluation and teacher-ramp schedules
align in optimizer updates. Different architectures are not equal parameter
counts or equal wall time. The priority plan enforces at least 1,000 full
passes for large arms. Use `--plan single` for cheap smoke/pilot runs.

Equal updates isolate an optimization budget, not statistical sample
complexity: small and large arms have different reuse rates. Compare several
seeds. A single selected step-zero checkpoint is not a negative capacity
theorem. Run a short pilot before authorizing the full schedule.
This changes the joint medium/source pool: one source is stored per medium,
so it does not separately identify coefficient diversity, source diversity and
optimization effort. Crossed medium/source counts and multiple update budgets
are additional controls, not conclusions of this plan.

## Exact objective

Let `e(r)=r-A M(r)`, `rho(r)=||e(r)||/||r||`, and `T4` be four
`transport_dsa` corrections. The `solver_teacher` objective is

```text
mean rho(r)^2
+ 0.2 sum_{j=1}^2 0.8^(j-1) mean rho(r_j)^2
+ lambda(t) mean ||A(M(r)-T4(r))||^2 / ||r||^2
+ 1e-6 mean ||M(r)||^2 / ||B(r)||^2,
r_1 = e(r), r_{j+1} = e(r_j).
```

`lambda(t)` ramps to .25 over the first 12,800 updates at default settings;
each rollout ratio is normalized by its current input residual, not the
original source. `teacher_action_weight=1`. The raw arm disables teacher and
rollout, but retains the shared `1e-6` correction penalty: it is explicitly
**not mathematically pure residual-only loss**. The laptop runner has a truly
pure raw-loss control. Do not call the correction penalty physical loss.
The raw-versus-teacher comparison removes two terms simultaneously. Add a
matched single-run `--loss-mode raw_rollout --teacher-weight 0` arm (retaining
`--rollout-weight .2`) to identify the action teacher's marginal effect.

Each training batch samples a stage from a depth-four classical trajectory;
50% are replaced by `A` applied to smooth random solution probes with spatial
Fourier cutoff 5. Variable coefficients can introduce higher frequencies in
these residuals. Sources and residual stages are not interchangeable samples.

## Measurement and checkpointing

- A reproducible fixed-probe subset of training and validation media reports
  the full objective, mean physical ratio, mean squared physical ratio, source
  one-step residual, FGMRES residual/success/iterations, and solve time.
- `--eval-count` controls this subset (default 8); it is not the entire dataset.
  Validation dataset size defaults to 32. Set both to 64 or larger for serious
  validation. Per-case results retain regime labels.
- Selection uses validation FGMRES, including the zero-initialized classical
  fallback. A separate 32-medium test split is evaluated only after selection,
  against actual parameter-free DSA2 and DSA4, without executing a zero network
  as the classical timing baseline. Set `--test-size 128` for serious runs.
- Time includes solver work and synchronized CUDA execution, but excludes
  coefficient setup and training. Base calls and preconditioner calls are
  separate. This runner does not instrument every A application; the laptop
  residual-subspace runner does. Report the relevant distinction.
- `last.pt` stores model/optimizer/scheduler, sampler and CPU/CUDA RNG state,
  epoch, actual work counters, history and selected weights. Resume is at epoch
  boundaries with the **same horizon, data fingerprint and runtime**. `best.pt`
  is inference-only. Old checkpoint versions are rejected.
- An explicitly requested unavailable CUDA device fails. Saved metadata records
  actual hardware, Torch/CUDA versions, deterministic flags and TF32 settings.
  CPU resume equivalence is tested; the CUDA regression is skipped locally.

JSON configurations, datasets and checkpoints are ignored by Git. Preserve
each run directory externally before moving results between machines. A CSV
without its configuration, checkpoint and data fingerprint is not sufficient
to reconstruct a trained model.

## Commands

Inspect without launching:

```bash
python -m rte.run_large_angular_spectral_experiments --plan priority
```

CPU smoke with evaluation rows separated by ordinary training rows:

```bash
python -m rte.run_large_angular_spectral_experiments --execute \
  --name audit_smoke --device cpu --dataset-size 5 --batch-size 3 \
  --size 8 --n-angles 4 --levels 2 --epochs 3 --eval-every 2 \
  --eval-size 2 --test-size 2 --eval-count 2 --fgmres-maxiter 2 \
  --output-root /tmp/rte_audit_smoke
```

Full remote plan, explicitly expensive:

```bash
python -m rte.run_large_angular_spectral_experiments --plan priority --execute \
  --device cuda:0 --eval-size 64 --eval-count 64 --test-size 128 \
  --output-root rte/large_angular_spectral_runs/seed20260804
```

Use separate output roots and `--seed` for repetitions. Existing runs are not
silently overwritten. Resume one interrupted run using the exact original
single-run flags and `--resume path/to/last.pt`; increasing `--epochs` is a
different schedule and is rejected by exact-resume validation.

## Decisions after the pilot

Accept a scaling direction only when paired untouched-test results improve
time to a common true-residual tolerance, or show a sufficiently strong and
consistent trend to justify further work. Report training and heldout losses
even when solver quality worsens. If bank loss improves but slow residual
witnesses and Krylov performance do not, prioritize solver-trajectory/action
targets and a better correction range, not more epochs alone. If train improves
but coefficient-heldout loss does not, investigate conditioning and diversity.
None of these outcomes alone demonstrates the cause of all historical failures.
Once pilot test results influence a design decision, they are exploratory.
Reserve a fresh confirmatory test seed/case set after those decisions; reusing
the same `seed + 200003` cases is not an untouched test.
