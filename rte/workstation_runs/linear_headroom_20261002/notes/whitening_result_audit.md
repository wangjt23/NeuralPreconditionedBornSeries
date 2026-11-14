# Independent q3 whitening audit — 2026-10-02

The numerical repair improves the angular least-squares diagnostic and produces tight **conditional raw-source** bounds, but does not open q5 or training. Cached continuation fails the frozen ≥10% warm headroom gate at both grids, even with acquisition excluded. This audit used local receipts/code only, without GPU execution or new data.

All seven frozen code hashes match [the executed policy](../../receipts/whitening/policy.json). The [solver receipt](../../receipts/whitening/results/solver.json) contains **48 unique full rows: 36 timed plus 12 retained warmups**, all finite and accurate, with zero solver failures and positive finite margins. Maximum physical relative residual is **9.7574512588e-6**, evaluated relative solution-error bound **7.2020151017e-5**, and paired field difference **3.9530262147e-7**. Minimum evaluated margin is **0.0170082185985**. These are discrete floating evaluations, not interval or continuous-RTE certificates.

## What whitening established

The fixed right action is `W(k) = [B0(k)* B0(k)]^(-1/2)`, where **B0=H0Q retains the full angular image**, rather than restricting to Q*H0Q. Every computed normal eigenvalue is positive; no clipping or physical-operator symmetrization is applied. Thus the ideal invertible right action changes conditioning while preserving the q3 range. Four checks pass, with maximum normal-whitening identity defect **2.1764e-14** and adjoint defect **3.7269e-16**. These probe checks do not certify a global floating operator norm.

All eight CGLS runs still reach the 100-step cap without the 1e-8 normal target. Their independently recomputed normal ratios improve from the prior **0.0321–0.3878** to **3.7586e-7–1.7108e-5**. The final residual is recomputed through the original physical H, not the homogeneous symbol.

For the raw problem B=HQ, t=b−Bc and g=B*t, the exact-arithmetic inequality uses **physical** coercivity m and G=Q*Q:

`sqrt(max(0, ||t||² − ||g||²/(m² λmin(G)))) / ||b|| ≤ min_c ||b−Bc||/||b|| ≤ ||t||/||b||`.

The code uses the unwhitened physical gradient from `full_adjoint`: actual negative streaming and K*=diag(weights)phase*, in the correct order. It does not substitute the homogeneous whitening eigenvalue for m²λmin(G). Lower endpoints independently reproduce using the same saved-parent source norms and the near-identity Gram. The displayed brackets, rounded here, are ordinary-FP64 conditional estimates; no directed rounding was performed.

| Raw source case | Computed relative residual bracket |
|---|---:|
| N128 ID0 | [0.658532558, 0.658532956] |
| N128 ID3 | [0.602040957, 0.602042641] |
| N256 ID0 | [0.659301761, 0.659302539] |
| N256 ID3 | [0.601709587, 0.601713421] |

Widths are **4.0e-7–3.8e-6**. With the stated floating qualification, these identify a substantial residual floor for this specific raw q3 source range, despite missing the stringent stationarity target. They do not bound runtime or the globally optimal preconditioner. The four real-AZ-projected prefix cases correctly retain **lower=null**, with attained upper ratios **0.947973–0.981283**: raw coercivity cannot pass through the projection without transversality evidence. Their optimal range quality remains unresolved.

## Cost and decision

The [summary](whitening_summary.json) exactly reproduces the mean of two within-medium medians, using this experiment's own paired graph baseline: **47.827/128.009 ms** at N128/N256.

| Cached continuation | N128 saving | N256 saving |
|---|---:|---:|
| Source initial | +1.080% | −9.709% |
| Prefix tail | −8.720% | −14.814% |

The best N128 `budget_ms=+0.516590` is ordinary baseline-minus-candidate savings; after reserving the 10% gate, remaining budget is **−4.266106 ms**. Some N256 case regressions exceed 10%. Cached arms include copying, original physical action/projection/acceptance and solver continuation, but exclude source-specific acquisition (**327.4–780.3 ms**) and whitening setup (**28.0–185.5 ms**). No distinct-RHS reuse or deployable fixed inverse was measured.

Further numerical optimization of this same raw q3 least-squares problem has little residual-quality headroom in these cases. A genuinely different candidate, if explicitly reopened, would enrich the angular/image space with material-adapted complementary components that q3 leaves, and first demonstrate cheap physical useful-image and continuation headroom against the successful graph-classical baseline. A q5 space is one possible enrichment, not a justified next job under the current gate. The projected optimum and other angular/material spaces are not universally ruled out. Current evidence supports keeping **q5/training stopped**, retaining the negative timings and conditional bounds, and using the already verified classical graph improvement.

The [execution receipt](../../receipts/whitening/execution_receipt.json) records exit 0 without timeout, **25.760 s** process wall, no training, and no test access. The retained pre-execution staging failure was a missing shared test fixture, repaired before any substantive GPU experiment; it does not change the mathematical protocol or data.
