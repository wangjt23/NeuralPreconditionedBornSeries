# Independent response Stage A cost audit — 2026-10-02

The inspected summaries retain a negative Q1 result for every tested TSA4 and homogeneous-response control. In all four contexts, the successful original eager/graph recipes also minimize the expanded six-control comparison. The current Stage B coefficient acquisition remains stopped. These results do not prove that every heterogeneous coefficient correction in range(HU) fails: no CGLS coefficients were acquired, some setup-only necessary budgets remain positive, and first-use setup attribution is confounded.

This auditor performed local summary arithmetic only. Receipt metadata reports **96 full solves: 72 timed and 24 retained warmups**, all accurate; 20 operator checks pass; 80 standalone application rows are retained; teachers.json is empty. The frozen cost phase cannot reach online CGLS. Staging metadata records nine CPU tests passing and 191 files verified. Full-row finite/extrema/hash rescanning was interrupted by the concrete local error “exec-server transport disconnected; recovery timed out.” This is an execution-bridge limitation, not evidence of an FWI outage. No GPU or remote work was performed by this auditor.

## Recomputed Q1 costs

For each material/method, the accepted latency is the median of three solve-plus-independent-acceptance times. Recorded selected setup is constant within repeats; accepted plus setup reproduces every one of the 24 summarized method totals. The primary comparator is the lower cost of successful strong eager and graph recipes, also equal here to the minimum over all six successful classical controls. A per-case backend minimum is conservative hindsight accounting, not a measured free selector.

| Grid / ID | Original/all-six best | Best TSA4 control | Best homogeneous-response control |
|---|---:|---:|---:|
| 128 / 0 | 531.433 ms, graph | 662.377 ms, graph | 831.805 ms, graph |
| 128 / 3 | 149.647 ms, eager | 169.978 ms, eager | 184.643 ms, eager |
| 256 / 0 | 246.111 ms, eager | 341.920 ms, eager | 395.353 ms, eager |
| 256 / 3 | 181.776 ms, eager | 217.793 ms, eager | 273.378 ms, eager |

All observed candidate budgets `Cbest − Ccandidate`, including the optimistic per-case backend minima, are negative; reserving a 10% improvement makes them more negative. Both candidate backends lose individually in each context. There is no measured complete-cost headroom witness for these controls, no learning result, and no coefficient-range ceiling measurement.

TSA4 needs 23/16 calls at N128 and 24/16 at N256, versus 12/8 and 7/5 calls for the original strong recipes. Homogeneous response changes these to 23/15 and 23/16: at most one saved call, while extra response/image work increases accepted latency. Call counts do not translate into a time theorem.

## First-use and timing limits

At the first context, N128/ID0, shared physical plus coercivity setup is **353.265 ms**. The first TSA4 base costs **75.730 ms**, while the later constructed strong base costs **3.654 ms**. The first response construction is **141.811 ms**; later contexts record **6.723/17.391/17.554 ms**. TSA4 graph construction is **140.443/135.733/133.701/175.525 ms** across the four contexts. Its sizeable cost despite shallow depth is evidence to investigate construction overhead, not proof of a universal graph floor.

The driver fixes construction order (TSA4 before strong, response afterwards). CUDA/backend initialization, allocation, plan warming, and precision-specific first use can be attributed differently from independent standalone execution. The large shared first-context cost also dilutes percentage comparisons. Do not normalize these charges away post hoc or declare an intrinsic response/TSA setup ratio. Warm solve negatives remain separately recorded, so the observed weakness is not solely a setup effect.

Cost is selected construction plus warm solve and independently timed physical-residual, coercivity-error and field-finiteness acceptance. Dataset deserialization, process startup, parity/adjoint probes and comparative field-difference/bitwise QA are excluded. Hence “setup-plus-warm accepted estimate” is the precise label; the experiment did not measure matched per-arm process-cold solves. Standalone application medians, fixed-order measurements, and sums of profiled categories are empirical diagnostics, never rigorous universal time lower bounds.

## Stage B budget decision

For a source-specific coefficient candidate with selected setup S and warm solve T, the 10% opportunity is `0.9 Cbest − S − T`; acquisition, prediction/loading and other omitted work must also fit. The setup-only upper allowance `0.9 Cbest − S` for response eager/graph is:

| Grid / ID | Eager necessary warm allowance | Graph necessary warm allowance |
|---|---:|---:|
| 128 / 0 | −92.515 ms | −232.958 ms |
| 128 / 3 | 118.568 ms | −17.165 ms |
| 256 / 0 | 191.186 ms | 57.485 ms |
| 256 / 3 | 132.257 ms | −43.268 ms |

Negative allowances block those recorded setup/backend combinations even before warm work. Positive allowances are necessary budgets, not attained feasible solver witnesses or authorization to acquire teachers. Therefore the current measured-control gate remains closed while geometry and arbitrary coefficient performance remain unresolved. Any reopened Stage B needs a new explicit version, immutable Stage A receipt gate, all six successful controls, the same charged acceptance, actual acquisition accounting, and retained failures; no training follows automatically.

The selected next screen—**segmented exact-original TSA**—targets a concrete Q1 cost mechanism before coefficient data: reduce construction overhead while preserving floating-stage action, then charge the resulting setup/warm tradeoff. It must pass frozen original action parity, output ownership, guard, physical residual and solution-error gates, and compare against the same successful portfolio. Segmentation might increase replay overhead; its benefit remains unmeasured. No coefficient acquisition or broader range rejection is justified by this audit.
