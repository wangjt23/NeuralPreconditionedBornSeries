# Independent full16 classical graph audit — 2026-10-02

The authoritative [v2 receipts](../../receipts/headroom_v2/verified_remote/v2/full16/settings.json) confirm a reproducible **warm implementation advantage on all 16 existing development media** for the frozen traditional recipes. This is exact-original classical CUDA-graph replay, not a learned improvement, an algorithmic FFT reduction, or a held-out generalization result. The nested grids do not turn 16 independent media into 48 independent media. This audit used local code/receipt arithmetic only, without CUDA execution.

## Identity, guards, and accuracy

The 48 unique entry comparisons all pass explicit `torch.equal`, field/image finiteness, and original physical-image gates: 24 complex64 graph rows and 24 original eager complex128 fallbacks. All **12 guard bundles** pass output ownership, distinct output pointers, same-pointer/same-shape but changed-stride material transpose rejection, changed-depth rejection, missing-cache rejection, and restored finite/bitwise action checks. The frozen helper holds strong operator/tensor references and checks identity, mutation version, shape, stride, and storage offset before replay or fallback; it rejects changed state rather than rebuilding caches. It remains a serial shared-buffer wrapper. All entry and guard checks precede full solves. The depth-16 algorithm retains its original **126 FFT/IFFT calls**.

The [full-solver receipt](../../receipts/headroom_v2/verified_remote/v2/full16/solver_cases.json) contains **288 unique rows: 192 timed and 96 retained warmups**. Every row passes direct field/residual-vector finiteness, finite positive coercivity margin, finite error bound and timing, independent physical residual, solver status, and paired iteration/classical-call equality. All paired full fields have an explicit recorded `torch.equal=true`, with relative field difference zero. This strengthens the earlier two-media screen, which retained only zero solution-difference norms.

There are zero solver failures. The maximum independent physical relative residual is **9.9850115718e-6**; minimum evaluated coercivity margin **0.0127060313575**; maximum evaluated relative solution-error bound **2.3688605765e-4**, below the frozen 1e-3 gate. Bounds use the original discrete FP64 physical operator and rounded weighted-phase coercivity computations; they are not interval-certified or continuous-RTE error estimates.

Policy, runner, and helper hashes independently match the retrieved files. The runner and collection receipts record preservation of 155 original source files, all three development-parent hashes, and the frozen v1 stage; no training or seed9626/test access. These are verified execution receipts, not an independent remote reread by this auditor.

## Warm comparison

Each medium/arm has two timed repeats in reversed paired order after one recorded warmup. Take its within-medium median, then the mean across 16 media. The unchanged recipes are `reference_tail` with TSA8 at N64/N128 and `tsa16_zero` at N256.

| Grid | Original mean case median, ms | Graph mean case median, ms | Mean warm saving | Minimum case saving |
|---|---:|---:|---:|---:|
| 64 | 134.205 | 33.328 | 75.166% | 72.865% |
| 128 | 136.580 | 40.847 | 70.093% | 67.515% |
| 256 | 148.651 | 96.806 | 34.877% | 34.211% |

No individual medium/grid has a warm regression. The frozen exploratory ≥5% warm gate at both N128/N256 therefore passes. The receipts support replacing the slower eager backend with this expanded traditional control in future warm comparisons; they do not support a universal cold-cost winner claim.

## Conditional setup-plus-Q cost

Recomputed costs match every stored per-case Q table. For each fixed material, original cost is `Sc + Q To`, graph cost `Sc + C + Q Tg`: Sc is shared physical/only-required-base setup; C includes the graph's three side-stream warmups and capture; To/Tg are that case's warm medians. Mean Sc is **7.832/8.454/11.208 ms**, mean C **120.481/122.534/186.229 ms**, at N64/N128/N256. The table averages costs over media before computing the percentage; it does not average individual percentages.

| Grid | Q | Mean original estimate, s | Mean graph estimate, s | Estimated saving | Cases with ≥5% saving |
|---|---:|---:|---:|---:|---:|
| 64 | 1 | 0.142038 | 0.161641 | −13.802% | 3/16 |
| 64 | 4 | 0.544654 | 0.261625 | 51.965% | 16/16 |
| 64 | 16 | 2.155118 | 0.661561 | 69.303% | 16/16 |
| 64 | 64 | 8.596976 | 2.261305 | 73.697% | 16/16 |
| 128 | 1 | 0.145034 | 0.171835 | −18.479% | 3/16 |
| 128 | 4 | 0.554773 | 0.294377 | 46.937% | 16/16 |
| 128 | 16 | 2.193729 | 0.784543 | 64.237% | 16/16 |
| 128 | 64 | 8.749554 | 2.745207 | 68.625% | 16/16 |
| 256 | 1 | 0.159859 | 0.294244 | −84.065% | 0/16 |
| 256 | 4 | 0.605812 | 0.584663 | 3.491% | 6/16 |
| 256 | 16 | 2.389622 | 1.746339 | 26.920% | 16/16 |
| 256 | 64 | 9.524863 | 6.393042 | 32.880% | 16/16 |

At Q=1, only 5/16 N64 and 4/16 N128 media have nonnegative savings; none at N256. At N256/Q=4, only 8/16 are nonnegative, and case savings range **−33.006% to +20.041%**. Individual break-even Q ranges are 1–4 at N64/N128 and 2–9 at N256; ≥5% total-saving Q ranges are 1–4 and 2–10, respectively. At the reported Q=16/64, all tested cases exceed 5%.

These are **conditional estimates**, not measured distinct-RHS amortization or process-cold latency. They assume unchanged material/grid/settings/dtype/batch, valid serial graph reuse, and future distinct-source solve latency approximated by repeated timings of the current saved RHS. Source-dependent iteration counts can change. Shared startup/import/global cold effects are not measured. A workload-aware comparator should retain the eager path where capture does not pay off, rather than charging every single-RHS case for a graph.

The launcher records exit 0 without timeout, process wall **218.324 s** within 300 s; script wall is 216.263 s. Process-global peak allocated/reserved memory is **1.072/2.244 GiB**, including captures, guards, and cohort verification, not isolated method peaks. The angular-negative results remain separate: this successful traditional backend provides their stronger baseline and does not reopen the angular q5/learning gate.
