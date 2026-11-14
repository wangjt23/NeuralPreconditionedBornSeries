# Independent CUDA-graph classical cost audit — 2026-10-02

The frozen screen supports an exact-original classical implementation improvement on two reused development media, IDs 0 and 3. It strengthens the traditional comparator; it is not a learned-method advantage or a population result. This audit used local receipt arithmetic only, without CUDA execution, new sources, training, or holdout access. Machine-readable values and receipt hashes are in [graph_cost_audit.json](graph_cost_audit.json).

## Acceptance and evidence retained

The [parity receipt](../../receipts/headroom/graph_oracle/parity_cases.json) contains 48 unique rows, all finite, passing, and explicitly `torch.equal`: 24 complex64 graph replays and 24 native complex128 fallbacks, across depths 1/16, three grids, two media, and physical/Nyquist probes. Field and physical-image relative differences are zero. The preserved script completes all 48 entry checks before the first full-solver comparison. It retains the original depth-16 algorithm and its 126 spatial FFT/IFFT calls; the prior 94-transform algebraic rewrite is not used.

The full screen invokes 96 solves: six grid/media cases × four methods × (one warmup + three timed repeats). The [solver receipt](../../receipts/headroom/graph_oracle/solver_cases.json) retains **72 timed rows**, with no accuracy rows for the 24 warmups. All 36 original/graph timed solves pass the solver, independent physical residual, evaluated solution-error bound, and field-difference gates. Their iteration counts and independent residuals match for each paired repeat. The 18 graph rows report zero norm difference from the corresponding original solution; full fields, field hashes, and explicit full-solution `torch.equal` flags are not retained, so that evidence is weaker than the explicit entry bitwise checks.

All recorded numeric row metrics are finite. Every evaluated coercivity margin is finite and positive, with minimum **0.0170082185985**. Raw final fields and direct field-finiteness flags are absent, so direct elementwise finiteness cannot be independently replayed from these rows. The largest original/graph independent relative residual is **9.7574512588e-6**, and the largest evaluated relative solution-error bound is **7.2020151017e-5**. The largest bound over *all* methods is **1.1922514926e-3**, from failed reference1 rows. These are floating evaluations of the discrete coercivity bound, not interval-certified or continuous-RTE error bounds.

`reference1_zero` fails at the 100-iteration cap on ID 0 at each grid: nine of its 18 timed rows, representing three grid cases of one independent medium. ID 3 converges. `reference2_zero` passes all 18 timed rows but is slower than the original frozen method in every case. The screen's global `all_solve_accepted=false` and `all_solve_accuracy_passed=false` correctly preserve these negative reference1 outcomes; they do not indicate graph failures.

## Warm time and recorded setup

Times below are milliseconds. Each warm time is the median of three timed repeats. “First RHS” adds recorded physical/selected-base setup and, for graph, the selected graph capture/warmup setup; it is an estimate, not a measured cold end-to-end solve. The selected graph is TSA8 at N64/N128 and TSA16 at N256.

| Grid / ID | Original warm | Graph warm | Extra graph setup | Original first RHS estimate | Graph first RHS estimate | Minimum Q for ≥0% / ≥5% total saving |
|---|---:|---:|---:|---:|---:|---:|
| 64 / 0 | 222.636 | 51.385 | 108.831 | 229.558 | 167.138 | 1 / 1 |
| 64 / 3 | 133.273 | 32.494 | 107.153 | 140.140 | 146.514 | 2 / 2 |
| 128 / 0 | 200.440 | 57.008 | 107.958 | 208.028 | 172.554 | 1 / 1 |
| 128 / 3 | 134.318 | 39.675 | 122.897 | 141.719 | 169.973 | 2 / 2 |
| 256 / 0 | 224.437 | 148.757 | 159.535 | 234.793 | 318.649 | 3 / 3 |
| 256 / 3 | 159.859 | 105.727 | 155.972 | 170.073 | 271.912 | 3 / 4 |

Taking the mean of the two case medians gives warm savings **76.433%, 71.118%, and 33.779%** at N64/N128/N256. Taking means of the first-RHS estimates instead gives **15.160%, 2.064%, and −45.866%**. Thus warm evidence passes the declared two-grid headroom screen, while single-RHS setup-inclusive evidence does not establish a ≥5% benefit at both N128 and N256.

Let Sc be common recorded physical/selected-base setup, C extra selected graph setup, and To/Tg the case warm medians. Estimated costs for Q distinct right-hand sides are `Sc + Q To` and `Sc + C + Q Tg`. Nonnegative savings require `Q(To−Tg) ≥ C`; ≥5% total savings require `Q(0.95 To−Tg) ≥ C + 0.05 Sc`. The table reports the smallest positive integer satisfying each inequality. At N256/ID3, Q=3 gives only **1.312%** estimated total saving; Q=4 gives **9.322%**.

These Q estimates assume one fixed material/grid/settings/dtype/batch shape, valid graph reuse, and future distinct-source latency approximated by repeated timings of the current single source. No distinct-RHS amortization experiment was measured. The estimates charge only the selected graph, not both graphs prepared for this screen, and exclude shared process startup/import and unmeasured global cold effects. They are workload-conditional forecasts, not observed deployment cost.

## Limits and next comparison

The [execution receipt](../../receipts/headroom/graph_execution_receipt.json) records exit 0 and process wall **38.031 s**, within the declared 300-second soft budget, although the script does not enforce that limit. Process-global peaks are **1.895 GiB allocated / 8.391 GiB reserved**, including entry captures, application diagnostics, and the screen; they are not a single method's peak. The policy code hashes match the retrieved frozen code.

Before broader reuse, strengthen captured tensor/object identity guards: a same-storage transposed square coefficient view can preserve the current pointer/version/shape signature while changing the native action. Also enforce finite positive margins explicitly, retain warmup acceptance, and store direct final-field finiteness/bitwise flags. No such mutation or nonpositive margin occurred in the frozen receipts, so these deployment-hardening gaps do not invalidate this screen.

A broader development-cohort comparison should use the successful graph backend as an expanded traditional baseline, include its setup and conditional Q costs, and retain failures. Comparing future learning only against the slower original launch implementation would overstate learned headroom. The present two-media evidence justifies this stronger control; it does not resolve angular-coarse feasibility or generalize the warm gains to unseen materials/sources.
