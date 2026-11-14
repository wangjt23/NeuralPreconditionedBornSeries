# Fixed-basis GEMM-layout result

## Decision

**Equivalent layout qualified; application improved versus the old nested implementation; no complete-cost win over AngularBand.** This one bounded test is closed. No alternative packing, rank/eta/leaf change, mode expansion, physical RHS or GPU job was started. The six numerical comparison workers and bridge all completed.

## What changed and what did not

The original space-major front is indexed G[i,a,j,b]. The new code computes Q*G on its left normal index with one 2D GEMM, packs the right normal index into the last dimension, uses a second 2D GEMM, and restores the original layout. Upward/downward Q operations also use a 2D GEMM. The exact identity is H[p,a,q,b] = sum_ij conj(Q[i,p]) G[i,a,j,b] Q[j,q]. All explicit packing and any implicit reshape copy are inside the measured operations. No expanded Q tensor I is stored. This is an execution change, not a lower-complexity algorithm.

All previous basis-construction and attribution-control code bytes remain identical. The fixed basis fingerprint is `0565c2b1515adb39220503903fb71a08fc31bbe35a07e8ef15e6415e6bb735c2`. Every measured nested worker independently rebuilds the same basis and pays its construction. The full scalar Q/B/D/U fingerprints, material/phase/directions/frequency fingerprints, Dhat and probes match. Basis ranks remain 8/13/16/17, eta remains 1.4091620147955772e-6, actual original derivative/directions and full 12-angle collision are unchanged.

The inherited model defect bound remains 0.000533588221037, with q^4=8.10634225236e-14. This is the previous exact-arithmetic error contract evaluated in ordinary floating point, not an interval-certified uniform floating inverse bound. The preconditioner is not the exact original inverse. Actual checks use original G, relative tolerance 1e-10 and at most 3 repairs, with no absolute floor for tiny RHS.

## Equivalence and accuracy

The paid bridge checked all 14 actual nontrivial local transformations: maximum relative difference from the old einsum was 0.0. Full raw factor actions on both the fixed single-column and five-column inputs also had maximum difference 0.0. This observed exact floating agreement is limited to these tensors/inputs, not a general bitwise theorem. The earlier small algebra unit also passed.

All 36 checked comparison applications passed. Every method used exactly one original-G repair. Nested/packed maximum original residual was 2.04979161493e-14; AngularBand maximum was 2.46093316863e-12. Maximum accepted field difference from the original nested reference was 4.55572138263e-12. Exact zero and 1e-200 nonzero columns were preserved. No rejected numerical attempt or threshold rescue occurred.

## Actual cost, both rounds preserved

All times below are milliseconds. Setup includes physical operator construction, candidate scalar nested basis and proof checks when applicable, V assembly and factor construction. It does not hide the basis or packing. AngularBand is not charged candidate-only basis work. Checked application includes all original-G residual computations and repairs; independent redundant acceptance QA is reported separately and included in the full process receipt.

| Run | Complete profiled setup | Factor | Rotation | Single calls 1/2/3 | Mixed calls 1/2/3 | Setup + all 6 checked calls |
|---|---:|---:|---:|---|---|---:|
| original1 | 175.491 | 112.157 | 54.606 | 18.719/16.096/15.976 | 21.288/20.205/20.167 | 287.942 |
| packed1 | 161.101 | 105.555 | 47.771 | 13.327/10.434/10.352 | 15.450/14.575/14.624 | 239.862 |
| angular1 | 130.101 | 114.574 | 0.000 | 20.879/7.016/6.791 | 13.547/11.198/11.160 | 200.692 |
| angular2 | 121.740 | 108.293 | 0.000 | 22.134/7.888/6.879 | 13.543/11.645/11.094 | 194.923 |
| packed2 | 171.008 | 109.529 | 47.272 | 13.127/10.504/10.319 | 15.739/14.591/14.795 | 250.083 |
| original2 | 159.588 | 99.615 | 43.061 | 18.831/16.316/15.954 | 21.339/20.309/20.303 | 272.641 |

The fixed order was original/packed/angular, then angular/packed/original. All workers were fresh processes with CPU0 affinity, one torch/BLAS/OpenMP thread and CUDA disabled. GPU was already occupied and was not used. The runtime records contain actual MKL/OpenMP team settings. No excluded warmup or extra repetitions were used. Each first call is retained. Single-column is the actual inner interface shape; the mixed batch is the previous zero + three random + tiny-column diagnostic, not a claim about batching physical RHS.

### Attribution conclusion

Packed checked application means improved versus the old nested implementation by 33.2% for the single column and 27.4% for the mixed batch. The improvements occurred in both forward/reverse rounds. Raw-apply diagnostics also improved, but those diagnostics occur after the measured calls and are not substituted as end-to-end timings.

Setup is not an established improvement: mean packed/original setup ratio is 0.9911, and the ranking reverses across rounds. Rotation totals for original were 54.606, 43.061 ms; packed were 47.771, 47.272 ms. Thus the previous 64% wall fraction cannot be treated as eliminated compute. This run measured nearly equal CPU/wall in the factor phase, but did not profile dispatch/copy/BLAS subcosts sufficiently to assign the remaining difference to one cause.

### Strong-baseline decision

Mean packed complete profiled setup was 166.054 ms versus AngularBand 125.920 ms, ratio 1.319. Mean setup plus all six checked calls was 244.972 ms versus 197.808 ms, ratio 1.238. AngularBand wins this complete diagnostic workload in both rounds.
The single-application cold-start ledger also loses: mean setup plus first checked single-column call is 179.281 ms packed versus 147.427 ms AngularBand. Mean setup plus the three observed single-column calls is 200.086 ms versus 161.714 ms. These are accounting views of the same predeclared samples, not extra runs or a new workload chosen to claim success.

The first-call effect must not be hidden: averaged over all three single-column calls, including the first, packed has ratio 0.951 to AngularBand. But AngularBand pays a larger first checked call; the separately disclosed later-call means are 10.402 ms packed versus 7.144 ms angular. The mixed-call mean ratio is 1.244. No favorable first-call weighting is used to claim a faster complete solver.

## Memory

| Run | Persistent factor bytes | Factor phase highwater bytes | Application phase highwater bytes | External whole-process peak bytes |
|---|---:|---:|---:|---:|
| original1 | 11639552 | 452579328 | 452579328 | 463970304 |
| packed1 | 11639552 | 448704512 | 448704512 | 462430208 |
| angular1 | 16272584 | 464465920 | 464465920 | 468975616 |
| angular2 | 16272584 | 464531456 | 464531456 | 469024768 |
| packed2 | 11639552 | 449236992 | 449740800 | 474796032 |
| original2 | 11639552 | 449986560 | 449986560 | 473169920 |

Persistent nested factors stay at 11,639,552 bytes versus 16,272,584 for AngularBand. The packing rewrite does not reduce this storage. Across all 14 rotations, the two explicit output-sized packs account for 31,334,400 logical allocated bytes cumulatively, not simultaneous peak and not proved extra traffic versus the original einsum (which may also copy). Actual temporaries, scalar basis and operator/reference objects are included in process RSS.

Phase highwaters are cumulative process RSS, not isolated allocation peaks. Packed has lower factor-phase highwater here, but the external whole-process peaks vary and the second packed process exceeds the angular processes. No overall process-peak memory win is claimed.

## Complete paid execution receipt

| Process | External wall s | External CPU s | Internal unpartitioned s |
|---|---:|---:|---:|
| bridge, including algebra QA and dual factors | 5.37 | 4.69 | not separately partitioned |
| original1 | 4.37 | 3.79 | 3.271659 |
| packed1 | 4.20 | 3.51 | 3.150594 |
| angular1 | 4.22 | 3.44 | 3.140025 |
| angular2 | 3.91 | 3.43 | 2.872145 |
| packed2 | 3.83 | 3.51 | 2.869881 |
| original2 | 4.29 | 3.56 | 3.035134 |

Total numerical process wall is **30.19 s**, total measured external CPU is **25.93 s**. These totals include Python/library startup, basis construction, checked and rejected work (none rejected), reference I/O, independent checks, diagnostics, allocation/GC and waiting inside the processes. They exclude development time and SSH/file-transfer overhead, which are not solver timings. The reference, all measured workers, plans and source copies were retained; raw CPU/wall times and all QA costs are in ANALYSIS.json. Unpartitioned time is retained, never subtracted from reported process totals.

## Stop and remaining scientific obstruction

This bounded exact-layout experiment has produced a usable faster nested application path, but not a complete solver promotion. The setup gain reverses and strong AngularBand still has lower complete cost and cheaper later single-column actions. Scalar basis construction and the remaining hierarchical factor/action work cannot be declared free. No claim is made that this rules out all nested representations, larger problems, or shared-basis workloads. Such a change needs an actual workload/error/cost argument before expansion.

Do not continue GEMM packing variants, rank tweaks or repeat-until-positive timings. Retain this equivalent implementation as the current nested attribution control if a substantively different mechanism is later justified. No next experiment was started or promised.

For API semantics, the official PyTorch documentation notes reshape can return either a view or a copy and 2D matmul is matrix multiplication. Those are layout semantics only; timing and numerical evidence above come from the installed FWI runtime, not the current web-doc version: [reshape](https://docs.pytorch.org/docs/stable/generated/torch.reshape.html), [matmul](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html).

## Reproduction

Run bridge first, then run_worker.py with (original,original1),(packed,packed1),(angular,angular1),(angular,angular2),(packed,packed2),(original,original2), under the frozen one-thread CPU0/CUDA-disabled environment in LAYOUT_PLAN.md. Existing result directories intentionally abort rather than overwrite. The launch manifest, raw results and process receipts identify all source bytes. Production/E source hashes remained unchanged.
