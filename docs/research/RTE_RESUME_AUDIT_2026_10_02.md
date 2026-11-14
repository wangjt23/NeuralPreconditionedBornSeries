# RTE project resumed on FWI — audit and controlled negative result

Completed 2026-10-02. This handoff reports actual FWI execution, distinguishes it from earlier saved evidence, and preserves the unsuccessful learned-solver result. The project is accessible and the bounded continuation study is complete. A learned runtime advantage remains unestablished.

## Exact project and access

- SSH alias: `FWI`, authenticated as `wjt`; observed host: `FWI-Standard-PC-i440FX-PIIX-1996`; login directory `/home/wjt`.
- Repository and real path: `/data/wjt/NeuralPreconditionedBornSeries`; RTE workspace: `/data/wjt/NeuralPreconditionedBornSeries/rte`.
- Origin: `git@github.com:wangjt23/NeuralPreconditionedBornSeries.git`; branch `main`; HEAD `028caea3f35d100225f656e2b1e910c1366aaafe`.
- The initial checkout had nine modified tracked files and substantial untracked September research work. HEAD alone is insufficient to reproduce it. Initial Git status, a source snapshot, and SHA256s for 155 existing RTE Python/research files were retained. All 155 files and the seven study inputs were verified unchanged after execution. No original solver, dataset, checkpoint, or handoff was overwritten.
- New work is isolated in `rte/workstation_runs/resume_audit_20261002/`. This report is also saved as `docs/research/RTE_RESUME_AUDIT_2026_10_02.md`. No commit or push was made.
- Connections used existing credentials with `BatchMode=yes`, `ClearAllForwardings=yes`, `StrictHostKeyChecking=yes`, `ConnectTimeout=10`, and `ConnectionAttempts=1`. No trust acceptance, port forwarding, SSH/proxy configuration, credential change, or private-key access was performed.
- The separately paused OL_preconditioner project was not resumed. Older RTE directories were inspected only enough to distinguish the current workspace.

No applicable ancestor/subtree `AGENTS.md` or project `.agents/skills` was found in scoped checks. The existing `INVESTIGATION_WORKFLOW.md` informed the study: solver behavior, actual splits, controls, completed receipts, and retained negative results.

## Problem and evidence limits

RTE means **radiative transfer equation**. The current problem is a steady, periodic, two-dimensional Fourier pseudospectral transport system on the unit square, with twelve equispaced S1 angular ordinates, normalized circular phase at `g=0.65`, and recent stiff scaling `epsilon=0.1`:

`A = Omega · grad + sigma_t(x) I - sigma_s(x) P`.

The source family combines heterogeneous materials and varied spatial/angular source parameters. The available experiments do not establish performance for inflow boundaries, different angular counts, strongly changed scattering/stiffness, new coefficient geometries, or continuous-RTE discretization accuracy. Nested resolutions represent the same physical draws, not independent additional media.

The existing September 26 held-out study used 32 seed-8626 media across N64/N128/N256. Reaggregation of its saved case-median CSVs confirmed neural runtime losses of 4.78%, 5.84%, and 10.05% against frozen tuned classical controls, with no measured solve failures. These are earlier saved results, not a new rerun. Its exact source CSV hashes are in `prior_results_reaggregation.json`.

## Audit findings

1. **Reliable solver acceptance.** Deployed FGMRES preserves the existing search space and independently checks the original-RHS residual in FP64. Nonzero initial guesses adjust the correction tolerance. Promotion uses the actual stored FP32 phase, directions, coefficients and wavenumbers rather than silently regenerating a different operator. Nonlinear neural proposals are compatible with flexible, real-stacked Krylov projection; this does not supply a convergence theorem for stationary neural iteration.
2. **Strong immediate controls.** Frozen validation-selected winners are homogeneous-reference tail at N64/N128 and TSA16-zero at N256. Tuned HSS/TSA alternatives are relevant. A comparison only with TSA8, scalar diffusion, or low-fill ILU would not justify superiority over the available classical portfolio.
3. **Training/deployment rejection semantics.** `rte/tail_lookahead.py` masks zero/dependent proposals to zero before continuation; deployment rejects them and continues the existing classical frontier. Equivalence is restricted to accepted, nondegenerate paths. `rte/probe_tail_frontier.py` queues a residual before acceptance, so a finite rejected dependent proposal can change its next probe input. These latent edge cases were left unchanged to avoid confounding the controlled resolution ablation. No injection rejection occurred in the new measured runtime cohort; this does not prove the edge cases absent in general.
4. **Trajectory mismatch.** Existing saved contexts and the new controlled loss use TSA8-balanced trajectories. The strongest N256 control is TSA16-zero. Resolution transfer and alignment to the deployed classical trajectory are distinct hypotheses.
5. **Symmetry and mathematical claims.** The coefficient-conditioned CT action is complex-linear, while the tail network is a nonlinear biased real/imaginary architecture. A raw phase-equivariance defect need not change a useful projected Krylov direction. Positive learned HSS shifts/gains do not inherit the fixed-shift HSS contraction theorem. Actual rounded phase entries are not exactly symmetric or normalized.
6. **Split integrity and leakage.** Training parents use seed 3626 (128 media); validation uses seed 4626 (16); the older held-out study uses seed 8626 (32). Validation has guided multiple pilots and must now be called development data. Original-ID mappings, nested coefficients and source parameters were checked exactly. Only 112 training media require the saved tail context; all sixteen validation media, including easy cases, participate in full-solver selection and runtime evaluation. No test teachers were generated or used. Reserved seed 9626 remains unused after the failed gate.
7. **Cost claims.** Warm solve time includes reference initialization, neural inference, operator actions, Krylov projections, and independent final residual checks. Material/preconditioner setup is recorded separately. Cold startup/model loading, historical teacher acquisition, total tuning, and repeated-source amortization are not established deployment costs. They cannot be inferred from iteration savings.

Detailed source locations and qualifications are in `independent_audit.md` and `rte_error_certificate_design.md`.

## Direction ranking and relevant primary work

The most justified first direction was a **controlled resolution-data ablation**, because an N128 paired bank already existed and prior N64 benefits weakened under refinement. It could isolate resolution coverage at modest incremental cost without claiming a new architecture. That direction has now been pursued and did not clear the runtime gate.

The next priority is a **useful-image complex-linearity and directional-tail diagnostic** on saved contexts. Rotate the residual and its classical reference through four phases, undo the phase of the proposed correction, and measure `(I-P_AZ) A z`: real-line angles, projected residual reduction, and actual two-step continuation. Defects confined to the old image space or removable scaling do not justify a redesign. Stratify by physical Fourier bands and ordinate streaming ratio. Compare any proposed directional enrichment with a simple non-neural enrichment or snapshot ROM before training a new architecture.

Third is **trajectory alignment behind the tuned classical action**: a separately controlled modest TSA16-zero training-context study versus TSA8-balanced. Broader novelty claims require credible angular/multilevel or heterogeneous synthetic-acceleration controls, not simply another neural pilot.

Primary-source comparisons are in `literature_audit.md`. Source-linearity and compatible coarse corrections are established in [Melchers–Dolean–Abdelmalik](https://arxiv.org/html/2605.19867v3); differentiable Krylov training in [Dimola–Coclite–Zunino](https://arxiv.org/html/2507.15452v1); transport correction/trajectory learning in [Peng](https://arxiv.org/html/2410.08735v3) and [Tang–Peng](https://arxiv.org/html/2509.05001v1). [Filtered MgNet](https://arxiv.org/html/2604.23265v1) supplies direct neural RTE prior art with limited reported comparison scope. [Fan–An–Ying](https://arxiv.org/abs/1802.03061) is particularly close to the homogeneous FFT reference. [Angular multigrid](https://arxiv.org/abs/2010.04559), [AIRG](https://arxiv.org/abs/2301.05521), [consistent DSA](https://arxiv.org/abs/1810.11082), [heterogeneous/void DSA](https://arxiv.org/abs/2001.09196), and [angular subspace correction](https://arxiv.org/abs/2102.09038) establish important classical mechanisms. Their DG/upwind/inflow discretizations and proofs are not automatically transplantable to this periodic FFT/S1 system. No priority or generic novelty claim is made here.

## Executed controlled study

Both arms start from the identical previous tail checkpoint, model `TailResponseNetwork` (width 32, four blocks, six modes; 670,811 parameters). Both use seed 27002, 1,000 updates, batch 4, AdamW with learning rate 0.0003 and decay 1e-5, cosine decay to 5%, the same physical-ID and complex-phase schedule, and two actual TSA8 lookahead steps plus 0.05 teacher-direction loss.

- `n64_only`: 4,000 N64 presentations.
- `mixed64_128`: 2,000 N64 and 2,000 N128 presentations, alternating updates.
- Matched updates/presentations, not a claim of matched compute cost. The same 112 eligible physical training IDs appear on both grids.
- Selection at steps 0/200/400/600/800/1000 uses all sixteen media at each of N64 and N128. Lexicographic score: failures, mean classical calls plus half injection calls, then mean physical residual. Both select step 800; pooled work score 7.875 in both. The small final-residual tiebreak difference is not a runtime win.
- Complete `best.pt` and `last.pt` preserve optimizer, scheduler, sampling cursor and RNG state. An incomplete-run resume guard checks protocol and input/source hashes. Restart equivalence was reviewed, not tested bit-for-bit.

Runtime evaluation used the same sixteen development media at all three nested grids, five methods, one warmup, two synchronized shuffled measured repetitions, then per-medium medians: **480 measured solves**, all passing original-RHS relative tolerance `1e-5`, with max 100 directions/restart 25. The frozen classical choices were inherited before this study. The predeclared gate required zero failures and at least 5% mean warm-runtime saving at both N128 and N256 before generating seed 9626 test data.

| Grid | Frozen classical | Classical ms | N64-only ms | Mixed ms | Mixed saving vs classical, 95% interval |
|---|---|---:|---:|---:|---|
| N64 | reference_tail | 133.818 | 134.317 | 133.963 | -0.108% [-3.072%, 2.673%] |
| N128 | reference_tail | 133.211 | 135.237 | 134.114 | -0.678% [-3.539%, 1.952%] |
| N256 | tsa16_zero | 145.741 | 158.392 | 155.891 | -6.964% [-10.377%, -2.846%] |

Positive saving favors learning. Mixed versus N64-only savings were 0.264%, 0.830%, 1.579%; all three paired 95% intervals include zero. Intervals use 10,000 paired-media bootstrap draws after reducing timing repeats. There is one paired training seed and sixteen reused development media; these are conditional exploratory intervals, not confirmatory generalization evidence.

**Gate failed; `test_generated=false`.** The mixed run modestly reduced refinement loss relative to the N64-only arm, but the difference is uncertain and it remains slower than the strongest classical control. The selected checkpoints, full histories, all cases and failures/negative outcomes are retained. The result does not support a larger training campaign without a new diagnostic.

### Actual resource and cost receipt

Initial hardware: RTX A6000 with 49,140 MiB, 36 CPUs, 184 GiB RAM (about 175 GiB available); GPU 0% utilization/37 MiB used. No user research jobs were running. After completion the only GPU process was the existing desktop integration process using 8 MiB. No job was stopped or duplicated.

All actual tests, training and solves used `/data1/software/miniconda3/bin/python`: Python 3.12.4, Torch 2.5.1, CUDA 12.4, NumPy 1.26.4, SciPy 1.13.1. An initial different existing environment lacked pytest; its failure receipt was retained, and no package was installed.

| Arm | Training s | Selection validation s | Checkpoint writes s |
|---|---:|---:|---:|
| N64 only | 118.798 | 29.016 | 2.036 |
| Mixed N64/N128 | 119.353 | 27.366 | 2.965 |

Whole paired-run wall time was 349.579 s, including 20.351 s input hashing/loading/identity checks and shared preparation overhead. Peak allocated GPU memory 8,262,390,784 bytes and peak reserved 9,005,170,688 bytes include both resident cached training banks (~5.96 GB); these are not deployment memory measurements. No new teacher bank was acquired; the existing banks total roughly 3.4 GiB on disk. Historical teacher/training/tuning costs remain a total-cost gap.

Mean recorded material/preconditioner setup costs: 24.500 ms at N64, 7.690 ms at N128, and 10.390 ms classical/10.317 ms mixed at N256. These are separate finite-run setup measurements, including first-use effects; they do not establish optimized cold-start or reusable deployment setup. Adding setup still yields a loss for mixed at every grid. There is no demonstrated positive online saving from which to claim a favorable offline-training break-even count.

## Solution-error theorem and numerical check

For the actual stored phase `P`, define `mu=lambda_max((P+P*)/2)` and `m=min_x(sigma_t-sigma_s*mu)`. With real nonnegative scattering, uniform angular weights and skew-Hermitian Fourier streaming, `m>0` gives `Re<v,Av> >= m||v||^2`, hence `||A^-1|| <= 1/m`. For `delta=||q-A u_hat||/m` and `||u_hat||>delta`, the relative solution-error bound is `delta/(||u_hat||-delta)`.

The implemented manufactured test passed; measured streaming skew defect was 1.40e-17. The rounded phase had `mu=1.0000000393`, so assuming exact phase normalization would be inappropriate. On **48 separate frozen-classical development solves** (sixteen media at N64/N128/N256), all had positive margins and available bounds: minimum margin 0.01270603136, maximum relative physical residual 9.9850e-6, maximum computed relative solution-error bound 0.0002368861 (0.02369%). This demonstrates why residual tolerance is not the same as field-error tolerance.

These are FP64 evaluations of an exact-arithmetic sufficient discrete bound, without interval rounding or residual-roundoff enclosure. The new diagnostic was outside runtime benchmark windows and did not use test teachers. It has not certified the neural benchmark fields, continuous-RTE accuracy, or positivity of intensities.

## Verification and provenance

- Original FWI suite: `144 passed, 66 subtests passed in 35.59 s`.
- New protocol tests on FWI: `6 passed in 3.42 s`; they check original eligible-ID mappings, nested materials/source identity, ID/phase schedule, matched presentations, full two-grid selection, and nonfinite guards.
- Actual two-update, two-grid CUDA smoke completed before the full study.
- Full paired study completed at 1,000 updates per arm; runtime gate completed; manufactured and 48-case coercivity diagnostics completed.
- Original 155 source files and all seven input files retain their recorded SHA256s; branch/HEAD unchanged. The JSON receipts are authoritative for complete hashes and case records.

Selected full-checkpoint SHA256s:

| Arm | `paired_training/<arm>/best.pt` SHA256 | Compact deployment export SHA256 |
|---|---|---|
| n64_only | b92f29ed8fd96ad0bd87a911d37546276f54d086ebe6be5ab13c2c47a1b9a997 | 835fcba4739e73897b2257485eead9f476d0ca513bb19b5a0a8a5cb04a87b914 |
| mixed64_128 | 5b1f192ac4d4a2e50b3391f36ac42f60a0685d19af5fe99ff8f6933eee73d82d | 62215b8f70ecb72c1b30b13cbfc203521effb2bdb7da044ef80c1349ede2ebe9 |

Shared physical-ID/phase schedule SHA256: `ce92fcc981576c6bfb3a849e11cef2e579988bad17f85b981a9882d48711cfeb`. Warm parent SHA256: `135aa962f6ebc1a08b1f17c61d85cafed990da68665e1bf12cfedb097ed0775f`. Classical selection SHA256: `67cf2328721db970470643fdae52c7df7db7a191a2db2f56731ffb5c63fb6a49`.

Remote artifacts under the new run root: `code/`, `paired_training/`, `runtime_gate/`, `solution_bounds/`, `deployment/`, environment/test/preservation JSON/logs, and `validation_runtime.png/.svg`. Full optimizer checkpoints and data remain on FWI. The local compact evidence and private Library bundle contain two deployment weight exports (~2.7 MB each), code, reports, CSV/JSON/log receipts, and the original source snapshot; they deliberately omit multi-GB banks and full optimizer states. A file SHA256 manifest supports bundle verification.

## Reproduction on FWI

Use the recorded existing environment and repository root. Supply **new output directories**; completed runs refuse overwriting. These commands reproduce the study against the preserved existing inputs, without changing the original sources:

```bash
cd /data/wjt/NeuralPreconditionedBornSeries
RTE_PY=/data1/software/miniconda3/bin/python
RTE_CODE=rte/workstation_runs/resume_audit_20261002/code
PYTHONPATH=. "$RTE_PY" -m pytest -q rte/tests
PYTHONPATH=.:"$RTE_CODE" "$RTE_PY" -m pytest -q "$RTE_CODE/test_tail_multiresolution.py"
PYTHONPATH=. "$RTE_PY" "$RTE_CODE/train_tail_multiresolution.py" --output rte/workstation_runs/resume_audit_rerun_NEW/paired_training --steps 1000 --batch 4 --eval-every 200 --lr 0.0003 --seed 27002 --max-wall-seconds 900
PYTHONPATH=. "$RTE_PY" "$RTE_CODE/benchmark_tail_multiresolution.py" --training rte/workstation_runs/resume_audit_rerun_NEW/paired_training --output rte/workstation_runs/resume_audit_rerun_NEW/runtime_gate --repeats 2
PYTHONPATH=. "$RTE_PY" "$RTE_CODE/audit_coercivity.py" --manufactured-test
PYTHONPATH=. "$RTE_PY" "$RTE_CODE/audit_coercivity.py" --device cuda:0 --output rte/workstation_runs/resume_audit_rerun_NEW/solution_bounds
```

Check current user jobs/resources before any rerun. The 900 s trainer limit is soft at completed-step boundaries and does not interrupt an in-flight GPU operation or validation sweep. Only explicit `--resume` may continue an incomplete hash-identical run. The benchmark can generate reserved seed 9626 only after its gate passes; this completed study never reached that branch.

## Outstanding gaps and next decision

No access or execution blocker remains. Learned speedup, broader robustness, interval numerical certification, complete historical acquisition/tuning costs, cold-start/repeated-source deployment economics, and bitwise restart reproducibility remain unproved. Rejection-path semantics need a separate focused regression/repair if they are to be claimed equivalent. The next scientifically decisive action is the cheap useful-image diagnostic, followed only if supported by a small controlled complex-linear/directional or trajectory-aligned correction study. Preserve seed 9626 and the classical winner until a frozen development gate warrants fresh testing.
