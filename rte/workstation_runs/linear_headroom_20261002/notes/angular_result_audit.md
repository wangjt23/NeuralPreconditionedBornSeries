# Independent angular headroom result audit — 2026-10-02

The saved full-spatial q3 angular experiment does **not** open the frozen q5 or learning gate. All tested continuations are accurate, but none provides the required ≥10% cached warm improvement at both N128 and N256. Acquisition was excluded from these optimistic cached timings. The incomplete image-range solves do not prove that the optimal q3 range is inadequate; they establish negative results for these particular acquired proposals only.

This independent audit used local receipt/code arithmetic, with no CUDA work, new sources, holdout access, training, or original-source edits. The [summary](angular_summary.json) exactly matches medians and paired baselines recomputed from [solver_cases.json](../../receipts/angular_headroom/angular_oracle/solver_cases.json). All five retrieved code hashes match [the frozen v2 policy](../../receipts/angular_headroom/angular_policy_v2.json).

## Counts and acceptance

There are **112 full-solver rows: 84 timed and 28 retained warmups**, across two reused independent media (IDs 0 and 3), nested grids 128/256, and two timing blocks. Every full row is finite, has a positive finite evaluated margin, reports zero solver failures, and passes the physical residual ≤1e-5, evaluated relative error bound ≤1e-3, and field-difference ≤1e-4 gates. The minimum evaluated margin is **0.0170082185985**; maxima are **9.7574512588e-6** physical relative residual, **7.2020151017e-5** evaluated relative solution-error bound, and **5.7388519485e-7** relative field difference from the paired graph baseline. These are floating evaluations for the discrete physical operator, without interval rounding or continuous-RTE error certification.

All four stored reduced-action/adjoint checks pass: the largest relative projection discrepancy is **7.5122e-15**. They exercise the actual stored physical FP64 operator and all spatial frequencies, with angular modes 0,+1,−1. The projected homogeneous inverse is an inner preconditioner, not the heterogeneous coarse inverse.

All eight heterogeneous Galerkin solves meet the independently recomputed coarse tolerance: relative coarse residuals **4.1590e-9–9.2298e-9**. Nonetheless, lifting the correction with coefficient one gives full physical residual ratios **2.5675–7.4539**. Accurate reduced convergence therefore does not imply full-angle physical convergence. Optimal real line scaling gives source-context residual ratios **0.9448–0.9858**; after the original four-step prefix and real image-space projection, the ratios are **0.999618–0.999978**, indicating very little useful residual reduction for those Galerkin tail directions. The intentional four-step prefix status is a truncated diagnostic, not a failure of the 112 full solves. Actual timed injection rows show no rejected proposals; acceptance alone does not imply cost benefit.

## Separate timing blocks

For each grid/method, take the median of its three timed repeats separately for each medium, then the mean of the two case medians. Each method is compared with the graph baseline measured **within the same block and case**; baselines from the earlier classical screen are not substituted.

| Cached proposal | N128 saving | N256 saving |
|---|---:|---:|
| Galerkin source initial | −1.828% | −9.776% |
| Galerkin prefix tail | −20.264% | −31.927% |
| Projected homogeneous-reference tail | −19.987% | −31.856% |
| Attained-range source initial | +1.074% | −9.807% |
| Attained-range prefix tail | −7.062% | −15.125% |

The Galerkin block baseline means are **49.611/127.859 ms** at N128/N256; the attained-range block means are **49.462/127.835 ms**. The small difference is retained, not pooled away. The +1.074% N128 result is a two-media development observation, below the gate and without population uncertainty claims. Its summary `budget_ms=+0.531328` means baseline time minus candidate time; it is not remaining inference headroom after enforcing the 10% gate. That adjusted quantity, `0.9 × baseline − candidate`, is **−4.414871 ms**. Some case regressions also exceed the policy's 10% ceiling. The mean wall-time headroom is negative at N256 for every tested angular proposal.

Cached arms include the owned device copy, physical image/projection/acceptance, initialization, and original solver continuation. They exclude proposal acquisition and coarse setup: Galerkin acquisition alone costs **99.8–293.4 ms** per context, and attained-range acquisition **312.0–765.9 ms**. Consequently, these cached comparisons give the acquired directions unusually favorable cost treatment. They do not measure a reusable fixed-material inverse or justify amortizing source-dependent acquisition over new right-hand sides. Setup/global cold effects remain separately recorded.

## Range bounds and gate decision

All eight right-preconditioned CGLS diagnostics hit the **100-step cap**. Independently evaluated normal residual ratios are **0.032085–0.387791**, far above the 1e-8 target; none claims stationarity or optimality. For the four raw source problems, the computed coercivity/Gram bracket has lower endpoint **zero** and attained upper endpoint **0.603073–0.680153**. These broad ordinary-FP64 brackets cannot exclude a substantially better optimal correction. The four real-AZ-projected prefix problems provide only attained upper bounds **0.948293–0.982250**; lower endpoints are null. The raw `sigma_min(HQ)` lower bound cannot be transferred through that projection without additional transversality evidence.

The frozen q5 condition requires both demonstrated q3-range inadequacy and positive headroom. Neither condition is established here. Learning is likewise unsupported by this screen: even free cached acquisition fails the required two-grid advantage. Retain these negative proposals and incomplete range diagnostics; do not label the angular mechanism impossible or the q3 least-squares optimum measured. A future reopening would require genuinely new decisive evidence and an explicit bounded protocol, not a post hoc relaxed gate.

The [execution receipt](../../receipts/angular_headroom/execution_receipt.json) records exit 0, no owned timeout, **54.285 s** process wall, and no training or seed9626 access. The script wall is 50.841 s. This audit recommends continuing the separately authorized classical graph cohort verification and keeping q5/learning stopped under the current angular policy.
