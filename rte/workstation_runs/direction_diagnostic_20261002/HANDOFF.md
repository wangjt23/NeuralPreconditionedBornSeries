# FWI RTE continuation: useful directions, acceptance, and actual cost

Completed 2026-10-02. **Stop the tested fixed nonlinear one-shot correction and its four-forward phase average as speed mechanisms.** Preserve the classical portfolio, the learned checkpoints, this negative result, and the isolated acceptance-aware continuation repair. The diagnostic found useful learned directions behind the weaker trajectory, but insufficient saved classical work to repay inference. It supplies no evidence for funding more training of this mechanism, nor an impossibility result for learned RTE preconditioning generally.

This is the completed follow-up to the earlier audit and paired-resolution study. It contains no new training, teacher acquisition, held-out evaluation, package installation, or system change. The reserved seed 9626 remains unused.

## Exact host, repository, and preservation

Access succeeded with the existing SSH alias `FWI`, authenticated user `wjt`, observed host `FWI-Standard-PC-i440FX-PIIX-1996`, login directory `/home/wjt`. Repository and real path are `/data/wjt/NeuralPreconditionedBornSeries`; RTE workspace is its `rte/` directory. Origin is `git@github.com:wangjt23/NeuralPreconditionedBornSeries.git`; branch `main`, HEAD `028caea3f35d100225f656e2b1e910c1366aaafe`.

The original checkout was dirty: nine tracked modifications plus extensive untracked research. HEAD alone cannot reproduce it. The original 155 RTE Python/research files, seven study inputs, two selected checkpoints, frozen classical selection, and three validation parents were hashed and preserved. New files are isolated at:

`/data/wjt/NeuralPreconditionedBornSeries/rte/workstation_runs/direction_diagnostic_20261002/`

This handoff is saved there as `HANDOFF.md` and as the new dated document `docs/research/RTE_DIRECTION_DIAGNOSTIC_2026_10_02.md`. Earlier September and October reports and results remain intact. No Git commit, push, job cancellation, persistent access change, or OL_preconditioner work occurred. Scoped instruction checks found no applicable `AGENTS.md` or project `.agents/skills`; the existing `INVESTIGATION_WORKFLOW.md` informed the study.

SSH retained host-key checking and bounded connection settings: `BatchMode=yes`, `ClearAllForwardings=yes`, `StrictHostKeyChecking=yes`, `ConnectTimeout=10`, `ConnectionAttempts=1`. No forwarding, proxy, credential configuration, key-content access, or host-key acceptance occurred. The Mac task served as the authorized bridge. All RTE experiment execution occurred on FWI; the Mac also performed receipt analysis and small CPU verification.

## Scientific scope and frozen protocol

RTE means **radiative transfer equation**. Here it is the steady periodic 2D unit-square Fourier pseudospectral system

`A = Omega · grad + sigma_t(x) I - sigma_s(x) P`,

with twelve equispaced S1 ordinates, circular phase parameter `g=0.65`, and `epsilon=0.1`. Conclusions are limited to this discrete setting and coefficient/source family. There is no new evidence for inflow boundaries, different angles, changed scattering/stiffness, unseen materials, or continuous discretization accuracy.

The same sixteen seed-4626 **development** media were reused on N64/N128/N256; nested grids are not additional independent physical draws. Exact identities of all seven saved material/source fields were checked against the preceding development tensors. Prior pilots and selection have already used this cohort, so it is not fresh validation. Training seed 3626 and older test seed 8626 were not used for new evaluation; seed 9626 was neither generated nor accessed.

Two four-call prefix trajectories were captured: TSA8-balanced and TSA16-zero. There were 96 contexts (16 media × three grids × two prefixes), of which 58 still required work: TSA8 13/13/14 and TSA16 6/6/6. Every ineligible context had already converged. All sixteen media, including easy cases, remain in actual full-solve timings.

The mixed N64/N128 checkpoint from the preceding paired study was frozen for full-solver interventions. Both paired checkpoints were compared in the direction diagnostic. Frozen classical winners are homogeneous-reference tail at N64/N128 and TSA16-zero at N256. The pre-execution development gate required zero failures, at least 5% mean warm solve-time saving against frozen classical at **both N128 and N256**, and paired 95% interval lower endpoints above zero. No candidate passed. This diagnostic has no branch that generates test data.

## Actual end-to-end result

The original uninstrumented solver measured 1,488 solves: sixteen media, three shuffled timing repetitions after warmup, and ten methods on N64/N128 plus eleven on N256. All succeeded against the original physical right-hand side; the maximum relative true residual was `9.985011571756997e-6`, below the `1e-5` tolerance. Medians of the three repeats were reduced per medium before mean/paired bootstrap analysis. All 54 comparisons and their 10,000-resample intervals were independently reproduced exactly from raw tables.

Warm solve latency includes reference initialization, neural inference, physical actions, projection/least-squares, and independent final true-residual checks. Physical/base setup and model reading/loading are reported separately. Positive savings favor the candidate.

| Method | N64 mean ms | N128 mean ms | N256 mean ms |
|---|---:|---:|---:|
| Frozen classical | 139.315 | 135.830 | 143.695 |
| Mixed after TSA8 | 138.484 | 135.307 | 153.730 |
| Mixed after TSA16 | 148.247 | 142.441 | 149.938 |
| Four-phase average after TSA8 | 156.639 | 151.033 | 170.059 |

| Candidate saving versus frozen classical | N64 %, 95% interval | N128 | N256 |
|---|---:|---:|---:|
| Mixed after TSA8 | 0.597 [-2.659, 3.977] | 0.385 [-2.667, 2.956] | -6.984 [-10.313, -2.896] |
| Mixed after TSA16 | -6.411 [-13.187, -1.280] | -4.867 [-10.836, -0.247] | -4.345 [-8.493, -0.416] |
| Four-phase average after TSA8 | -12.435 [-17.772, -8.203] | -11.193 [-17.054, -6.688] | -18.347 [-22.292, -14.287] |

The mean classical-call counts explain why modest finite-horizon improvements do not establish speed. TSA8-reference uses 7.8125/7.8125/7.875 calls; mixed TSA8 uses 7.4375/7.5/7.5625; four-phase TSA8 uses 7.3125/7.375/7.4375. TSA16-plain uses 4.5625/4.5/4.5; mixed TSA16 uses 4.5/4.5/4.5. Behind the strongest trajectory, the learned intervention saves no mean classical calls on N128 or N256.

Recorded mean physical/base setup is approximately 7.4–10.2 ms in these comparisons. Adding it does not reverse the decision. Resident checkpoint loading took 0.2065 s for N64-only and 0.1819 s for mixed. Historical teacher generation, tuning, and full cold deployment/amortization remain unmeasured; there is no positive online saving from which to claim favorable training break-even.

![Development timing comparisons](direction_runtime.png)

The figure's left panel shows actual speed versus frozen classical; the right isolates direction benefit with controls that pay the same inference cost. Intervals condition on one paired training seed and sixteen repeatedly reused development media. They are exploratory, not confirmatory held-out or OOD evidence.

## Separating direction quality, projection, and overhead

The meaningful new image is `w=(I-P_AZ) A z`, with projection onto the **real** span of existing physical search images. We used actual physical actions, real inner products, two projection passes, and retained the original ordered Arnoldi frontier. Images that are merely different inside old `AZ`, or differ only by nonzero real scaling, should not motivate a new architecture.

The mixed proposals retain about 98–99% of their physical-image norm after this projection. Their weakness is not mostly wasted norm inside the old search space. They improve upon the homogeneous-reference direction behind TSA8, but are weak compared with the next classical-frontier image.

| Mean immediate squared residual ratio | TSA8 N64/N128/N256 | TSA16 N64/N128/N256 |
|---|---|---|
| Homogeneous-reference proposal | 0.624 / 0.610 / 0.602 | 0.725 / 0.722 / 0.725 |
| Mixed single-forward proposal | 0.485 / 0.487 / 0.510 | 0.668 / 0.707 / 0.754 |
| Mixed four-phase average | 0.461 / 0.447 / 0.452 | 0.635 / 0.660 / 0.697 |
| Next classical-frontier proposal | 0.0877 / 0.0883 / 0.0827 | 0.0348 / 0.0345 / 0.0345 |

The classical-frontier proposal costs another preconditioner application and is not a free oracle. After each proposed image plus two additional classical calls, mixed TSA8 squared ratios are 0.002501/0.002413/0.002482 versus reference 0.005736/0.005613/0.005122. Four-phase averaging further improves mixed to 0.002249/0.002239/0.002292. These finite-horizon quantities use different proposal-construction costs; full timed solves decide whether quality pays.

Field-preserving controls discard one or four neural outputs and keep the same reference correction. All 192 paired control checks had exactly equal fields, iteration counts, and failure status. Single-forward inference added about 3.6–4.7% to TSA8-reference timings; four forwards added about 16.2–18.5%. Mixed beats the one-forward TSA8 control by 5.02/3.84/3.88%, showing a useful learned effect at equal paid inference. Four-phase averaging likewise beats the four-forward reference control by 5.10/5.91/4.85%. Both still lose or remain uncertain against inference-free classical controls. Behind TSA16, mixed versus one-forward reference-cost saving is -1.52/1.15/0.02%, with every interval including zero.

Separate synchronized profiles are diagnostic and perturb execution. Their callback reference/model sub-times overlap the enclosing callback category; do not sum them twice or substitute these profiles for benchmark latency. The original profiles retained the first two media, which include already converged TSA16 paths. Supplementary profiles selected the first two **eligible** IDs per prefix solely from saved eligibility flags: TSA8 [0,1], TSA16 [0,3] on each grid, sixty profiled solves total, all successful. Warmed mixed single calls in these profiles take roughly 7.3–8.0 ms; four-phase calls roughly 30 ms. Sequential startup/outlier costs in the first cost-only profiles are retained and explicitly qualified in `verification_note.md`. This small ordered profile cohort is not an independent inference-overhead estimate.

## What phase sensitivity does and does not establish

Rotating residual and reference by `i`, undoing the phase of the proposal, and projecting its image still changes the useful real line. Mean quarter-turn angles are 23.29°/27.03°/32.38° behind TSA8 and 47.31°/50.69°/55.81° behind TSA16. These angles ignore either sign of real rescaling. Thus the defect survives projection and changes the proposed search space.

For fixed material, the Fourier streaming, real coefficient multiplications, fixed angular matrix, and invertible homogeneous reference are complex-linear in exact arithmetic. Their fixed classical compositions inherit this property. Thirty-six quarter-turn checks on eligible materials found relative defects at most `8.90e-15` for physical FP64 action, `2.56e-6` for the FP32 TSA preconditioner, and `3.68e-8` for the FP32 homogeneous reference. These examples check implementation behavior, not global proofs.

The group average `Fbar(r,t)=(1/4) sum_gamma gamma^-1 F(gamma*r,gamma*t)`, with `gamma in {1,i,-1,-i}`, is quarter-turn equivariant in exact arithmetic: multiplying both inputs by a group element permutes the sum. It does not impose arbitrary-phase equivariance, additivity, or complex linearity. Factor-two positive homogeneity was exact on the saved probes, while additivity defects remained. Saved fields named `phase0_*` concern the original single-forward model even when stored in the average record.

The averaged model improves geometric quality but loses actual runtime at all grids. The data do not establish that complex-linearity enforcement would remove the gap: no constrained architecture was trained. There is no warrant for a new complex-linear training campaign from these diagnostics alone.

## Smallest justified correctness repair and theorem

The original offline `rte/tail_lookahead.py` masks a rejected proposal to zero, appends it as the next frontier, and subsequently applies the classical preconditioner to zero. The deployed augmented solver rejects the proposal and continues from its original ordered Arnoldi frontier. The old retained image span plus residual can reconstruct the Arnoldi **span** at an eligible non-breakdown prefix, but cannot reconstruct its ordered next vector. Saved banks without that frontier are insufficient for correct rejection continuation.

The isolated no-gradient helper `code/continuation_acceptance.py` receives physical action, preconditioner, proposal, prefix residual, image basis, and original frontier. Rejected proposals preserve residual, image space, and frontier. Accepted images enlarge the real search space; a perfect correction inside the Arnoldi span but outside old `AZ` is accepted as a happy breakdown, without calling `B(0)`. Per-sample masks, logical calls, and explicit independent least-squares checks are included. It is an offline diagnostic repair, not a replacement differentiable training loss or deployment change. Original sources and trained models remain unchanged.

Let `Q` be real-orthonormal, `r` real-orthogonal to old image space, `A` real-linear, and `w=(I-P_Q)Az` nonzero. The expanded-space residual obeys

`||r_plus||² = ||r||² - <w,r>_R² / ||w||² <= ||r||²`,

where `<x,y>_R=Re(x* y)`. Nonzero real proposal scaling preserves the immediate residual; later sign-equivalent trajectories also require an odd/real-linear preconditioner and compatible numerical branches. Joint unit-phase rotation of `(r,z,Q,V)` preserves all real inner products and least-squares coefficients when `A,B` are complex-linear, so the complete state rotates and residual ratios/acceptance agree in exact arithmetic. Proposal-only rotation does not supply this guarantee. Immediate monotonicity alone does not promise fewer later calls, convergence under arbitrary budgets, or faster solves.

The explicit rejected-zero counterexample has `A=diag(1,2,3,4)`, `Q=e1`, `r=(0,1,1,0)`, and frontier `(2,-1,-1,0)/sqrt(6)`. Original masked squared ratios are `[1,1,1]`; corrected ratios are `[1,1/26,approximately 0]`, matching an actual deployed rejection trace. Fifteen focused tests plus 61 subtests passed on FWI in 3.96 s, covering rejection, rank loss, mixed batches, independent real least squares, scale/phase identities, monotonicity, happy breakdown, instrumentation parity, and prefixes. The preceding original 144-test suite plus 66 subtests had already passed and was not needlessly repeated.

There were **zero actual learned rejections** in the full measured cohort. Fifty-four deliberate no-op rejections belong only to the inference-cost control. All nonzero direction probes were accepted. Correcting rejection semantics is valid but cannot explain measured learned slowdown here. The separate `probe_tail_frontier.py` acceptance-order edge case remains documented and untouched.

## Solution-error bound: exact assumptions and computed limits

The earlier audit evaluated a discrete coercivity estimate on 48 **classical** development fields, not new neural fields. Uniform angular quadrature and the ordinary spatial/angular discrete inner product are required. Define the angular scattering action `P_ij=stored_phase_matrix_ij * weights_j`; the stored phase matrix alone excludes quadrature. Write `H=(P+P*)/2`, `mu=lambda_max(H)` for this actual rounded, weighted action; assume real coefficients, pointwise `sigma_s>=0`, exactly skew-Hermitian Fourier streaming, and

`m=min_x [sigma_t(x)-sigma_s(x)*mu] > 0`.

Then `Re<v,Av> >= m||v||²`, hence `||A^-1|| <= 1/m`. For the actual original-RHS residual `q-A qhat`, `delta=||q-A qhat||/m` bounds absolute discrete solution error. If `||qhat||>delta`, relative error is at most `delta/(||qhat||-delta)`. The phase asymmetry and normalization errors were accounted for through the actual Hermitian-part eigenvalue, rather than assuming exact symmetry/stochasticity.

The computed maximum relative bound in the prior 48 classical solves was about `2.3689e-4` (0.02369%), with positive margins. Ordinary FP64 eigenvalues, minima, norms, and residual calculations have no directed rounding or interval enclosure. They are **evaluated discrete estimates, not rigorous interval-certified bounds**. The theorem is exact-arithmetic and applies only under its assumptions; the computed values do not bound continuous discretization error, establish positive intensity, or certify all neural fields. A true residual threshold alone is not a universal solution-error guarantee.

## Resources, reproducibility, and receipt map

Actual FWI numerical runtime was `/data1/software/miniconda3/bin/python`: Python 3.12.4, Torch 2.5.1, CUDA 12.4, NumPy 1.26.4, SciPy 1.13.1. Hardware was RTX A6000 (49,140 MiB), 36 CPUs, 184 GiB RAM. Initial job checks found no running user research jobs; the existing desktop GPU process used 8 MiB. Main diagnostic process exited 0 in 504.449 s (internal 501.115 s). Supplement exited 0 in 21.295 s (internal 16.625 s). No training or new teacher acquisition occurred. Main final in-process GPU occupancy was 7,016 MiB, supplement 3,210 MiB: these are end-of-run occupancy, not peak or deployment measurements. Final preservation receipt records the post-process state.

Important checkpoint hashes:

- N64-only selected full state: `b92f29ed8fd96ad0bd87a911d37546276f54d086ebe6be5ab13c2c47a1b9a997`.
- Mixed selected full state: `5b1f192ac4d4a2e50b3391f36ac42f60a0685d19af5fe99ff8f6933eee73d82d`.
- Frozen classical selection: `67cf2328721db970470643fdae52c7df7db7a191a2db2f56731ffb5c63fb6a49`.
- Full diagnostic code: `d7c32f871a900d9d57fb193aaffb40d5456188f82ac3a0ce6ee182b319a9a2c1`.
- Eligible-profile code: `5728afe325e6d02bb12fc4ce4789c1f955e2da9c2ece5c0ad05356db84ac15e9`.

The run directory contains `results/runtime_n*.csv`, `case_medians_n*.csv`, `runtime_summary.csv`, `runtime_comparisons.json`, `prefixes_n*.json`, `direction_metrics_n*.json`, `direction_summary.json`, `decision.json`, and settings/source/input hashes. `eligible_profiles/` retains supplemental profiles, identities and covariance. Logs, process receipts, original-source hashes, test outcomes, the engineering smoke and its archived earlier source, and its explicit pre-full-run revision receipt are preserved. The smoke was not a scientific advancement decision. No tuning used full-run results.

To rerun focused tests from the **existing full FWI checkout**:

```bash
cd /data/wjt/NeuralPreconditionedBornSeries
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:rte/workstation_runs/direction_diagnostic_20261002/code \
  /data1/software/miniconda3/bin/python -m unittest discover \
  -s rte/workstation_runs/direction_diagnostic_20261002/code -p 'test_*.py'
```

To reproduce the completed protocol, first check current jobs and preserved hashes, then use a **fresh** output directory; the script refuses an existing directory:

```bash
cd /data/wjt/NeuralPreconditionedBornSeries
PYTHONPATH=.:rte/workstation_runs/direction_diagnostic_20261002/code \
  /data1/software/miniconda3/bin/python \
  rte/workstation_runs/direction_diagnostic_20261002/code/diagnose_tail_directions.py \
  --output rte/workstation_runs/NEW_DIAGNOSTIC_DIRECTORY \
  --sizes 64 128 256 --count 16 --repeats 3 --profile-count 2 --max-wall-seconds 900
```

This command depends on the retained original full checkout and prior run inputs; the compact source snapshot in the bundle lacks an unrelated GMRES import dependency. The bundle preserves code, all new small receipts/tables, selected inference exports, proof, prior handoff, and a SHA256 manifest, rather than claiming to replicate the multi-gigabyte training banks or optimizer/RNG states. Complete states remain on FWI in `resume_audit_20261002/paired_training/`.

The earlier verified private Library audit bundle remains available as [RTE audit and paired-resolution evidence](https://chatgpt.com/api/library/files/libfile_1371c45e4a0481918a426e50a9e8869e/download). Primary-source comparisons are retained in that bundle's `literature_audit.md`; this follow-up introduces no novelty claim or untested transfer of classical convergence theorems.

## Decision and outstanding research gaps

Retain the strongest frozen classical methods and the acceptance-aware diagnostic helper. Stop additional epochs, TSA16 retraining, or four-forward phase averaging of the tested model as a speed strategy: trajectory-aligned geometry and equal-cost comparisons do not establish a convincing benefit. No paid compute or larger campaign is justified by these results.

A future learning proposal would need a materially cheaper or reusable material-dependent operator and decisive evidence that it avoids at least one expensive classical call behind the strongest trajectory while passing the same full-solve accuracy/cost controls. That is an evidence requirement, not a currently supported architecture recommendation. Remaining gaps are fresh held-out/OOD evidence, repeated training seeds, broader boundary/angular regimes, complete cold/offline/amortized costs, and rigorous rounding enclosures for numerical certificates. There is no access blocker; these are limits of the scientific evidence. The negative result and original work are retained for a future change of mechanism.
