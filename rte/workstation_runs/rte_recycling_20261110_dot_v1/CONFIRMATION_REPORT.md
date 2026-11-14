# Same-instance CPU confirmation

All parameters, material and four source arrays were frozen. Six candidate batches rebuilt and paid physical/acceptance setup, TSA64 cache and method factor; paired method order was reversed in the middle repetition. CPU0, one thread, no GPU.

| Pair | Order | compact4 setup | exact8 setup | compact4 total | exact8 total | Gain |
|---:|---|---:|---:|---:|---:|---:|
| 0 | compact4_inject then exact8_inner | 3.069s | 5.368s | 20.173s | 17.623s | 12.64% |
| 1 | exact8_inner then compact4_inject | 3.110s | 5.246s | 20.187s | 17.499s | 13.32% |
| 2 | compact4_inject then exact8_inner | 2.822s | 5.412s | 19.889s | 17.734s | 10.84% |

Median paired gain: 12.64% (range 10.84% to 13.32%). No confidence interval is asserted from three pairs.

Observed full batch wall, including small untimed-in-components loop/norm work, has median paired gain 12.63%; those raw wall times are also retained.

## Mechanism and acceptance

compact4_inject: outer calls [[4, 3, 4, 3], [4, 3, 4, 3], [4, 3, 4, 3]]. Inner calls [[], [], []].
exact8_inner: outer calls [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]. Inner calls [[5, 5, 5, 4], [5, 5, 5, 4], [5, 5, 5, 4]].

All 24 candidate outer solves passed; exact inner failures: 0. Worst original residual 9.08037e-06, evaluated relative-error bound 3.94535e-05, paired field difference 2.64881e-07. Source hashes and production hashes are unchanged.

Four ordinary native solves were computed only for paired field checks, using the same1e-5 outer threshold. Their research cost is separately recorded: 24.288s. Total confirmation main wall: 141.090s. They are not candidate inputs or accurate new teachers.

## Bounded decision

If every paired gain is positive with invariant calls/acceptance, clear the same-instance CPU timing gate only. Material transfer, GPU economics, single-RHS latency and production remain unproven. Do not enlarge the RHS count or relax tolerance.

Next material gate must be declared before looking: a proposed adverse direction is doubling only the original inclusion amplitude from3 to6, leaving its position, width, source arrays and all numerical parameters unchanged. This is an unscreened physical perturbation, not a statistically representative holdout. GPU work needs an exclusive coordinated slot.
