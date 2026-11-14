# Response-space headroom prior — 2026-10-02, before GPU work

The new range `U = H0^(-1) Q` deserves a distinct geometry check: it differs from the already tested raw angular Q range by including homogeneous streaming response. However, an unchanged TSA8/TSA16 graph introduced after four classical applications has a poor **single-RHS complete-cost** opportunity. A cheaper from-start complement must be a separate control, rather than treating graph setup, proposal acquisition, or prior inference as free. All arithmetic below comes from saved receipts; no operator experiment, GPU job, training, or holdout access was performed. [HEADROOM_PRIOR.json](HEADROOM_PRIOR.json) records the input hashes and detailed scenarios.

## Complete-cost hurdle

Let Sc be shared physical/required-base setup, Cg existing graph capture/warmup, Sr additional response setup, To/Tg existing eager/graph warm medians, and Tc the **entire** candidate warm solve. Classical costs are `Sc + Q To` and `Sc + Cg + Q Tg`; candidate cost using that existing graph is `Sc + Cg + Sr + Q Tc`. To save fraction g against a successful comparator Cb(Q),

`Tc ≤ [(1−g) Cb(Q) − Sc − Cg − Sr] / Q`.

The complete warm candidate includes prefix, owned copies, response/coefficient work, physical images, projection/least squares, all remaining classical steps, and original residual/accuracy checks. Sr and any source-dependent acquisition lower the displayed optimistic budget; source-dependent acquisition cannot automatically be paid just once for Q different RHS.

On all16 development media, mean fixed eager Q1 cost is **145.034/159.859 ms** at N128/N256. Mean existing graph setup-only cost is **130.988/197.437 ms**, including shared setup. With **Sr=0**, the entire candidate warm-time budget is:

| Grid | Any Q1 saving | ≥5% Q1 saving | ≥10% Q1 saving |
|---|---:|---:|---:|
| N128 | 14.046 ms | 6.794 ms | −0.458 ms |
| N256 | −37.578 ms | −45.571 ms | −53.564 ms |

Thus even a hypothetical zero-time solver using the existing N256 capture loses against the eager Q1 comparator under these recorded setup estimates. This is a conditional accounting barrier for that composition, not a lower bound for a newly designed TSA4 graph or an impossibility theorem for response learning. Graph setup slightly understates complete construction because final signature binding is outside its recorded timer; process-cold effects are also unmeasured.

## IDs 0 and 3: late-proposal opportunity

The following stricter diagnostic hurdle uses the minimum of the two *recorded* successful classical cost estimates for each case. This per-case minimum is conservative hindsight accounting, not a tested deployable selector; the fixed eager comparator remains the Q1 portfolio control. “Warm budget” sets Sr=0 and reserves ≥5% total saving before any candidate work.

| Grid / ID | Eager Q1 cost | Graph Q1 cost | Candidate warm budget Q1 | Prefix-four proxy | Residual budget after proxy |
|---|---:|---:|---:|---:|---:|
| 128 / 0 | 216.202 ms | 188.221 ms | 50.780 ms | 16.039 ms | 34.740 ms |
| 128 / 3 | 147.625 ms | 170.740 ms | 11.569 ms | 16.030 ms | −4.460 ms |
| 256 / 0 | 242.784 ms | 348.699 ms | 31.713 ms | 66.274 ms | −34.562 ms |
| 256 / 3 | 175.068 ms | 301.230 ms | −27.886 ms | 66.258 ms | −94.145 ms |

Four classical applications are logically necessary for the original after-four proposal on these eligible contexts; physical Arnoldi images, projection, reduced solves, initialization and checks also cost work. The proxy is **four times the v1 standalone application median**, not a rigorous time lower bound or a sum of profiled categories. Important provenance correction: the retained v1 application timings are **depth16 at every grid**. N256's ~16.57 ms is the deployed TSA16 measurement; N128's ~4.01 ms is TSA16 and only a hypothetical proxy for deployed TSA8. No direct standalone TSA8 duration was retained. Different experiments, guard overhead, kernel caching, noise and stage timing prevent treating either median as an unavoidable hardware-time bound. Nevertheless, N256/ID3's setup-only budget is already negative without a timing proxy.

For an explicitly assumed fixed-material Q16 workload, both comparator and candidate can pay the same graph capture once. With Sr=0 and ≥5% total saving, **entire warm-solve** budgets are **38.395/91.349 ms** in the N128/N256 cohort means. On IDs0/3 they are **56.781/39.560 ms** at N128 and **141.657/101.071 ms** at N256. Subtracting the same prefix proxies leaves **40.742/23.530 ms** and **75.383/34.813 ms**, respectively, for everything else. Extra response setup subtracts Sr/16. These are conditional forecasts using repeated timing of one saved RHS, not measured distinct-source reuse and not permission to claim amortization.

## What previous inference can and cannot suggest

The old uninstrumented four-phase mixed-model experiment used mean **7.375 TSA8 calls versus 7.8125 classical calls** at N128, but **7.4375 TSA8 calls versus the stronger 4.5 TSA16 calls** at N256. Complete old warm means were **151.033 versus 135.830 ms** and **170.059 versus 143.695 ms**: losses of 11.193% and 18.347%. The small synchronized profile cohort recorded about 30 ms for four model forwards, with nested callback timing and perturbation caveats. Call counts, differing TSA depths, and tiny profiles cannot furnish a wall-time theorem, nor demonstrate that complex linearity alone would repay inference. The new response mechanism must beat actual complete costs, not merely reduce a horizon loss or one call count.

## Bounded control needed before learning

Use the same existing IDs0/3 and N128/N256, original physical operator/residual/error gates, and newly frozen cost protocol. Compare the genuinely different **from-start TSA4 plus response** with identical TSA4 alone and both successful frozen eager/graph recipes. Measure actual TSA4 capture/setup/application costs and response setup, acquisition, application, ownership, physical images and full continuation; the existing TSA8/TSA16 capture costs and prefix proxy are not its costs. Include zero/balanced starts consistently with the chosen controls and keep negative outcomes. First establish useful physical-image quality and optimistic cached complete-solve headroom; then charge actual acquisition/setup before any deployment or training claim. A cached proposal failing even the optimistic budget blocks that tested composition.

U is a fixed-material complex-linear action in exact arithmetic. A source-specific cached vector is not a reusable residual-linear map. Finite adaptive CGLS/GMRES has RHS-dependent coefficients/stopping and is generally nonlinear; its exact limiting least-squares map may be linear without establishing a cheap implementation. A fixed material inverse or fixed scalar-polynomial iteration is complex-linear in exact arithmetic, but stable precision, useful images and practical cost need independent evidence. No q5 expansion, training campaign, holdout use, or assumed response reuse follows from this prior.
