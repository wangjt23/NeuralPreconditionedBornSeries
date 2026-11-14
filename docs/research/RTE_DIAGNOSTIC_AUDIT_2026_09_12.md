# RTE workstation diagnostic audit and resumed investigations

Date: 2026-09-12. This updates the workstation handoff and subsequent CT
interpretations; it preserves the original artifacts. Scope: shared CT ensemble,
periodic Fourier RTE, N16/N32, L12, g=0.65. Historical U-Net results were not all
rerun, and conclusions about that architecture must not be silently transferred
to CT.

## Outcome

The apparent severe N32 generalization failure was largely an **inference cache
bug**, not evidence that training on 4,096 materials failed. Correcting the cache
changes CT from 27 failed test media to 2/256. The two remaining float32 GPU
misses are close to tolerance and disappear in the tested float64 controls.
CT remains competitive locally; neither universal solver superiority nor a
single data/expressivity bottleneck has been established.

The prescribed N32 training finished: 1,000 epochs, 128,000 updates, 4,096,000
material presentations. Epoch 200 remains selected. No further training or
architecture change was made for this audit. All audit experiments listed here
are complete; no RTE experiment from this continuation is left running.

![Audit evidence](../../rte/workstation_runs/ct_diagnostic_report_20260912/diagnostic_audit.png)

## 1. Cache fault: reproduction, correction, and affected evidence

The old CT cache signature included coefficient storage pointers, tensor version
counters, shapes and dtypes. CUDA can recycle a freed tensor's address. After
intervening calls to `rte.setup()` by classical methods, a different medium could
therefore have the same signature. CT retained derived filters but not the
coefficient tensors whose storage identified them.

A pre-fix GPU replay of the first 45 test media with intervening method setups
observed seven same-signature/different-collision-spectrum events. Case 18
reproduced the original source loss 1069.0099 and final residual about 0.9802.
Fresh preparation gave source loss about 0.007661, six iterations and residual
3.43e-6. That short reproduction was terminal evidence, not a saved standalone
run. The complete old-versus-corrected comparisons are persisted below.

The fix increments `PeriodicRTE2D.setup_generation` on every setup, includes it
in CT and classical transport signatures, and retains CT's cache-input tensor
references until refresh. This prevents pointer-reuse false hits, including
direct tensor replacements. Regression tests cover intervening setups, storage
lifetime and a deterministic same-pointer/version invalidation case.

The full N32 diagnostic compared cached and uncached CT on every test medium:
maximum relative correction difference **6.04e-7**. Twenty-eight source-loss rows
changed materially, including 25 severe failed solves and three previously
successful solves. Original `ct_n32_data4096_seed101/results.csv` is preserved
but its learned-CT evaluation is superseded by
[`ct_n32_corrected_gpu_eval_20260912`](../../rte/workstation_runs/ct_n32_corrected_gpu_eval_20260912/summary.csv).
Do not use the original apparent losses up to 96,913.6 as neural predictions.

The N16 audit found four corrupted original training-run source-action rows:
seed 2 indices 21/24 and seed 3 indices 23/31. In contrast, all **96 separate
paired-evaluator source losses** reproduce on fresh CPU evaluation within
1.79e-7. The nine N16 selected bank losses reproduce within 1.52e-9. The seed-1
pilot was not affected by these four rows. Use the dedicated evaluator files,
not the seed-2/3 training runner's original solver tables.

Training uses the differentiable uncached path. This bug does not explain the
gradient optimization. Selected and last N32 validation bank losses and the
selected checkpoint's 32-medium solver score reproduce. Intermediate epoch
weights were not retained individually, so this is not an independent replay
of every historical selection decision.

Evidence: [`cases.csv`](../../rte/workstation_runs/ct_n32_diagnostic_audit_20260912/cases.csv),
[`N16 source checks`](../../rte/workstation_runs/ct_data_identity_audit_v2_20260912/n16_source_rechecks.csv),
[`N16 bank checks`](../../rte/workstation_runs/ct_data_identity_audit_v2_20260912/n16_bank_rechecks.csv).

## 2. Corrected N32 results and arithmetic sensitivity

Same epoch-200 weights, same 256 test media, same validation-tuned classical
settings, real-stacked FGMRES, tolerance 1e-5 and limit 30. Each medium/method
has one first solve and three warm repeats; repeated solves are not independent
test media. All first/warm convergence outcomes agree for this run.

| Method | Failed media / 256 | Mean iterations | Mean physical A calls | Observed successful median ms |
|---|---:|---:|---:|---:|
| CT4 | 2 | 6.1914 | 31.9648 | 36.98 |
| HSS | 2 | 8.2813 | 42.4141 | 48.96 |
| TSA | 1 | 6.1445 | 56.3086 | 56.44 |
| Full reference | 1 | 9.6797 | 49.4063 | 54.61 |
| DSA | 2 | 11.9883 | 108.9219 | 113.32 |
| Transport DSA | 1 | 12.1133 | 110.0547 | 115.24 |

**These N32 times are shared-GPU observations, not a clean speed ranking.**
During evaluation, `nvidia-smi` showed another project's training process on
the same RTX A6000. Its exact contention interval was not recorded. It was not
interrupted. The [execution note](../../rte/workstation_runs/ct_n32_corrected_gpu_eval_20260912/execution_notes.json)
retains the observed process IDs and device. Random method order does not remove
this limitation. The loaded tuning CSV used 32 validation media; the evaluator's
unused default `validation-count=8` did not retune it.

CT's remaining failed indices are 54 and 152: GPU float32 final residuals
1.029e-5 and 1.135e-5. Promoting the same stored coefficients, source, phase,
directions and wavenumbers to float64 on CPU gives CT residuals **3.737e-6 in
8 iterations** and **8.604e-6 in 7 iterations**. HSS/TSA also converge there in
float64. CPU float32 behaves differently near the threshold: CT takes 21
iterations on these cases, while HSS/TSA still miss case 152. These controls
support arithmetic/device sensitivity, not a proof of a fixed error floor or
uniform convergence. Neural parameter inference remains float32 in this control.

The CPU/CUDA least-squares backends also differ (`gelsy` versus `gels` by
default); this is a possible contributor, not an isolated causal result.
[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.linalg.lstsq.html).
Raw evidence: [`precision.csv`](../../rte/workstation_runs/ct_n32_diagnostic_audit_20260912/precision.csv).

## 3. Which prior conclusions survive?

- **Local N16 advantage survives the numerical audit.** Dedicated GPU evaluations
  have no failures over 96 media. CT is faster than HSS on 90/96 pairs and TSA
  on 90/96; median paired baseline/CT ratios are 1.255 and 1.495. These are
  specific implementations, discretizations and observed runs, not production
  solver or cross-hardware dominance.
- **The old pooled confidence calculation was too strong for training-seed
  generalization.** The replacement resamples trained-model seeds, then paired
  media within each seed. Exploratory 95% intervals for mean CT-minus-control
  time are [-11.99, -8.15] ms for HSS and [-19.54, -14.87] ms for TSA. There are
  only three models, including a reused CPU-trained pilot and two GPU-trained
  models. Dataset and initialization both change with seed. Per-seed intervals
  are conditional on the fixed model. Failures are retained; speed differences
  condition on both methods succeeding.
- **Different split hashes alone did not prove disjointness.** The new audit
  checks all 5,440 material coefficient pairs across four runs and finds no exact
  tensor duplicates. This does not prove independent distributions or distinct
  geometry up to symmetry. Each medium still has one stored source.
- **Lower bank loss still need not improve the solver.** N32 selected losses are
  train 0.00834121, validation 0.00869797, test 0.00824920. Last validation loss is
  lower, 0.00840628. On 96 validation media outside the solver-selection subset,
  selected and last CT average 6.125 and 6.1458 GPU iterations; both converge on
  all 96. This is a small mismatch, not evidence of catastrophic overfitting.
- **Missed difficult directions are a hypothesis for CT, not an established
  cause.** The earlier U-Net audit has its own supporting witnesses. CT needs
  separate controls; its invalid N32 deployment results cannot support that
  diagnosis. Increasing resolution, material count and update budget together
  cannot identify a data-size effect.

Evidence: [`audited paired analysis`](../../rte/workstation_runs/ct_n16_replications_audited_v2_20260912/paired_summary.csv),
[`identity audit`](../../rte/workstation_runs/ct_data_identity_audit_v2_20260912/summary.json),
[`expanded validation`](../../rte/workstation_runs/ct_n32_diagnostic_audit_20260912/summary.csv).

## 4. Investigations resumed beyond the cache fix

### Validation residual coverage and stronger HSS

On the first four validation media, 30 power steps on the real representation
of E*E, E=I-AM, find selected-CT one-step ratios 0.303–0.648. Its witnesses have
97.55–100% spatial energy outside the |kx|,|ky|<=5 square. This identifies useful
probe directions, but ratios below one are **lower bounds**, not contraction
certificates. CT's twelve stationary steps reduce its own witnesses to
1.08e-9–3.16e-6. HSS has one witness above one (1.277), yet twelve steps reduce
that witness to 2.82e-6: one-step transient amplification is not divergence.
Cross-method comparisons on different own-witness inputs are not matched-input
rankings. The saved cross-evaluations use identical Arnoldi preconditioner-input
vectors for all methods; those vectors are explicitly not labeled current
physical residuals.

The validation-only HSS experiment tested nine predefined four-stage shift
sequences on 32 media. Constant factors (1,1,1,1) won the failure/iteration/final-
residual criterion: 8.25 mean CPU iterations versus 8.281–11.969 for alternatives.
On the other 96 validation media it averages 8.594 versus CT's 6.115, with no
failures. This small search found no improvement; it does not exhaust HSS tuning.
CPU values should not be interchanged with the GPU table above.

Evidence: [`validation witnesses`](../../rte/workstation_runs/ct_n32_validation_witnesses_20260912/norm_witnesses.csv),
[`matched Arnoldi inputs`](../../rte/workstation_runs/ct_n32_validation_witnesses_20260912/krylov_inputs.csv),
[`HSS sequences`](../../rte/workstation_runs/ct_n32_hss_sequences_20260912/summary.csv).

### Sparse controls: the previous ILU rejection was too broad

The six original low-fill ILU configurations failed on the separate mixed
fixture, but that is not a general ILU failure. A direct N16 check finds finite
complex64 triangular-solve entries as large as 1.12e27 whose working-precision
norm overflows. The first-action relative residual computed in float64 is
1.22e28. Invalid arithmetic precedes the downstream least-squares error; the
backend's error text alone is not evidence of a PyTorch implementation defect.
Changing precision alone does not rescue the corresponding poor fill-4 factor.

With complex128 factorization, drop tolerance 1e-4 and fill factor 64, **all
three sources converge in three iterations**, with residuals 6.95e-7–8.89e-7.
Factorization takes 7.45 s and L+U has 6,080,208 nonzeros. Smaller tested fills
and two alternative orderings still fail. SciPy explicitly advises increasing
fill and reducing drop tolerance to improve ILU approximation.
[SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spilu.html).

Exact complex64 LU also succeeded for all three sources at N16 and N32, in two
iterations. Recorded factor times were 3.54 s / 213.78 s, with 7.23M / 121.48M
factor nonzeros. The N32 run completed under a 16 GiB address-space limit. These
are CPU, fixed-fixture controls; preparation amortization and same-medium neural
comparisons are still required before a performance conclusion.

Evidence: [`ILU stability controls`](../../rte/workstation_runs/sparse_stability_audit_n16_20260912/results.csv),
[`N16 original sweep`](../../rte/workstation_runs/sparse_n16_seed1_final_20260912/results.csv),
[`N32 original sweep`](../../rte/workstation_runs/sparse_n32_seed1_limit16g_20260912/results.csv).

## 5. Measurement and reproducibility corrections

The evaluator's legacy `cold` label means first solve after method preparation
in an initialized process. It excludes checkpoint loading, operator construction,
source transfer and CUDA/process startup. CT construction/weight loading is
outside preparation, while classical construction is inside. Thus it is not
process-cold deployment latency. CPU RSS before/after is not peak memory, and
CUDA allocator peaks include live shared objects and the source-action probe.

Legacy time-to-tolerance percentiles condition on success. Schema 2 also reports
failed-media counts and all-attempt p95 with failed attempts censored as infinity;
unreached percentiles are marked rather than silently dropped. Reanalysis no
longer attributes learned bank losses to classical method rows.

All **69 RTE tests passed**. Restart tests retain exact model/optimizer/scheduler
state comparisons, but allow 1e-7 absolute variation in CPU least-squares final
residual history; do not call all numerical histories bitwise reproducible.
The handoff's 178 files (12,362,390 bytes) and 21 archived moves verify unchanged.
The old global inventory is stale after new work; no artifacts were reorganized
or deleted. New scripts compile and `git diff --check` passes. No commit or push
was made; unrelated pre-existing working-tree changes were preserved.

## 6. Next experiment order

1. **Clean performance confirmation:** rerun the corrected N32 evaluator when
   the GPU is uncontended, preserving weights, test cases and tuning. Capture
   process telemetry and isolate process-cold setup separately. The current
   convergence evidence does not require another training run.
2. **Precision control:** compare float64 scalar reductions/orthogonalization
   against full float64 and the existing float32 FGMRES on validation. Keep
   tolerance 1e-5; do not hide failures by relaxing it. Recheck true residuals
   independently before changing the default solver.
3. **Causal training control:** fix N32, architecture, optimizer update horizon
   and validation; cross nested 256/4096 material sets at matched 128,000 updates,
   then vary sources per fixed material. Reserve new confirmation test seeds
   only after design freeze. Current test media are now diagnostic data.
4. **Coverage control:** compare the existing physical loss with a fixed-share
   high-frequency probe arm at matched presentations/updates and independent
   validation witnesses. Then separately test solver-residual replay. Do not
   conflate Arnoldi inputs with physical residuals or change capacity alongside
   the residual distribution.
5. **Classical amortization:** compare viable high-fill ILU and LU on the same
   ensemble media and multiple RHS, including setup and memory. The small HSS
   sequence search is complete; broader tuning remains a separate control.

### Reproduction entry points

Run from the repository root; output directories must be new. Original training
artifacts remain under `rte/workstation_runs/ct_n32_data4096_seed101`.

```bash
python -m unittest discover -s rte/tests -q
python scripts/prepare_rte_handoff.py --verify
python -m rte.audit_ct_diagnostics --run rte/workstation_runs/ct_n32_data4096_seed101 --output rte/workstation_runs/ct_diagnostic_repeat
python -m rte.audit_ct_data_identity --output rte/workstation_runs/ct_identity_repeat
python -m rte.probe_ct_validation --run rte/workstation_runs/ct_n32_data4096_seed101 --output rte/workstation_runs/ct_witness_repeat
python -m rte.audit_hss_sequences --run rte/workstation_runs/ct_n32_data4096_seed101 --output rte/workstation_runs/ct_hss_repeat
prlimit --as=17179869184 -- python -m rte.audit_sparse_stability --output rte/workstation_runs/sparse_stability_repeat
python -m rte.evaluate_collision_transport --checkpoint rte/workstation_runs/ct_n32_data4096_seed101/best.pt --tuning rte/workstation_runs/ct_n32_data4096_seed101/classical_tuning.csv --repeats 3 --device cuda:0 --output rte/workstation_runs/ct_n32_clean_repeat
python -m rte.report_ct_diagnostic_audit --output rte/workstation_runs/ct_report_repeat
```

The report generator reads the explicitly named saved audit directories; it
does not rerun experiments. Its [artifact manifest](../../rte/workstation_runs/ct_diagnostic_report_20260912/artifact_hashes.csv)
hashes evidence, checkpoints, report and code. It is separate from the immutable
handoff manifest.
