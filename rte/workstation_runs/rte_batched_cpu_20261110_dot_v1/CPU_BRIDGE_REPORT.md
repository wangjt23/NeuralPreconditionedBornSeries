# Mode batched angular inverse CPU qualification

The independent batched17 implementation passes the original-operator, sequential-inverse and zero-column bridge, followed by one fixed four-RHS complete CPU bridge. This establishes a valid execution representation for a later GPU qualification. It does not establish a GPU gain or a new paired CPU speedup. No GPU was initialized or used. Sealed E and all earlier packages remain unchanged.

## What changed and what did not

The leading tensor dimension now contains all 17 tangential modes. At each of the twelve angular chain stages, LU, triangular solves and matrix products operate on a batch of 17 independent systems. The periodic corner closure still uses one rank-256 update per mode. Both signs of k remain explicit; there is no conjugate-mode reuse. Full g=0.65 scattering and the original stored directions, weights and frequencies remain in the original-G action.

The physical contract is unchanged: N128, 12 angles, inclusion amplitude3, original sources [0,4,7,14] in order, complex128 mode inverse, P8 A P8 inner tolerance1e-4/max8, original outer tolerance1e-5/max100, evaluated relative-error bound1e-3 and relative field difference1e-4. Defect correction is tested separately for every mode and column at1e-10, with at most3 corrections. Only exact zero columns are masked; nonzero residuals use scale-normalized relative norms without an absolute floor.

`IMPLEMENTATION_GATE.md` records the two pre-run debates, controls and closing conditions. Stage alarms were180 seconds each with an aggregate numerical-stage wall budget of600 seconds. The two stages used CPU0, one thread and CUDA disabled. They completed in10.588 and40.189 seconds, total50.776 seconds, and both processes exited. GB continued on its separate resources; no isolated-host timing claim is made.

## All-mode and boundary bridge

The17 one-column qualification vectors are exactly E's fixed vectors, checked by their stored hashes. The batched original-G action equals the sequential action on those vectors at the recorded precision; the largest independent original-A0 action difference is3.256266e-16. This is a test-vector observation, not a claim of bitwise equality for every input.

Four inverse cases cover the locked vectors, all-zero blocks, mixed zero/nonzero/1e-200 columns, and mixed active/inactive modes. Across187 checked columns, including108 strict zero columns:

- Maximum inverse difference from the sealed sequential implementation:6.898000e-14.
- Maximum original-G inverse residual:3.859508e-12.
- Maximum independent original-A0 inverse residual:3.860256e-12.
- Strict zero outputs remain exactly zero; tiny nonzero columns are retained and pass the same relative criterion.

The independent checker embeds each mode into the physical field and applies the original high-precision A, subtracting the actual inclusion perturbation. It does not accept merely because the banded surrogate solves its own equation. The qualification stage constructs and pays for both sequential and batched factors. Its measured factor construction costs are1.241435 and1.439941 seconds, respectively. Batching is not a CPU construction improvement in this observation.

## One complete four-source bridge

All four candidate solutions pass original acceptance. Maximum original residual is7.418934e-6, evaluated relative-error bound3.217770e-5, and relative field difference1.626115e-7. The largest inner projected-original residual is8.882625e-5. Outer TSA64 calls remain[2,2,2,2] and inner iterations remain[5,5,5,4], matching E.

There are19 batched inverse applications, corresponding to323 logical mode inversions. Every mode requires exactly one correction. The runtime maximum original-G residual is3.930843e-12. The solve records38 batched raw band solves and38 batched original-G evaluations. The fresh setup separately pays one additional repaired17-mode qualification and17 independent original-A0 checks.

| CPU phase | Seconds |
| --- | ---: |
| Physical and acceptance setup | 0.004885 |
| TSA64 setup | 0.132186 |
| Batched factor construction | 1.366050 |
| Original A0 qualification of all17 modes | 0.550775 |
| Setup phase sum | 2.053897 |
| Four-source online solve and acceptance | 13.140786 |
| Phase sum for complete four sources | 15.194683 |
| Observed complete candidate batch wall | 15.218738 |

Use15.218738 seconds as the complete observed CPU cost. The24.055 milliseconds between the phase sum and observed wall include validation-vector preparation and bookkeeping not assigned to a component timer. They are retained, not subtracted. The first actual source has measured setup-plus-solve/check phase cost5.387908 seconds. Assigning the entire unallocated batch gap to that first source gives a conservative paid upper bound5.411963 seconds. This is not a separately captured first-source wall timestamp and is not total cost divided by four. No repeat was run to eliminate this small accounting gap.

The four ordinary native accepted reference solves cost24.555855 seconds of research time, recorded separately and never used as candidate directions or initial states. Full-stage total wall40.188739 seconds includes those references, the candidate batch, loading and receipt overhead. There are no host-to-device copies in this CPU run.

Within factor construction, measured CPU assembly takes0.152852 seconds, chain factorization/update0.315130 seconds, and corner construction/factorization0.891764 seconds. Every angular chain stage has its own LU and solve/update receipts. These internal `perf_counter` values are valid CPU measurements only. They must not be reused as GPU execution times: an eventual GPU adapter needs synchronized external phase boundaries or must label unsynchronized internal values as host-dispatch-only.

## Calls and memory

Measured solver and inverse counters are retained in `full_01/results.json`. Static inspection of the frozen successful, no-restart/no-rejection control flow additionally gives40 original-A calls for the outer solves and final acceptance,46 projected-original-A calls inside the inner solves, and17 independent setup A0 calls. These103 candidate full-operator calls are source-derived, not separately instrumented runtime counters. Reference calls are calculated from their recorded iteration counts in `analysis.json`.

The batch factor contains13 batched LU calls and23 batched LU-solve calls during construction. Its38 online raw solves entail494 batched LU-solve calls. These are source-level batched operations over17 independent systems, not measured GPU kernel launches. The underlying traced FGMRES source was statically audited after the run: `rte/npbs_unet_ablation.py`, SHA2564da8fb3a2998a02b45d508f7e8387b12c99849cfc06ff8363327b667aa17d5f2, reported modification date2026-09-12. The original four production hashes recorded before and after both stages are unchanged.

Logical stored batched factor/coefficient size is276,427,472 bytes. Process peak RSS is1,545,756,672 bytes during the qualification stage with both controls and multi-column validation, and1,268,060,160 bytes during the complete stage. These Linux process-wide high-water marks include imports, physical operators, reference data and transient workspaces. They are not isolated factor peaks and are not GPU allocation estimates.

## Decision and next gate

Outcome: CPU EXECUTION BRIDGE PASS. The new representation preserves the useful P8 solve on this fixed instance and is eligible for a coordinated GPU cost gate. The one observed CPU batch is not paired with a new compact4 or sequential batch, so differences from E are not a new timing conclusion. No extra sources, threshold changes, reserved samples or material claims are introduced.

GPU work remains blocked on explicit resource release and coordination. GB evaluation/benchmark and the prioritized Schur CPU window must finish first. Static preparation may continue without CUDA initialization. A later GPU test must use the same device-warm convention for both compact4 and candidate, synchronize actual complete-cost boundaries, pay all assembly/LU/validation/copy/repair costs, and report first-source and four-source costs separately. Stop if setup alone exceeds compact4's complete four-source cost, if any fixed accuracy/repair gate fails, or if measured complete cost has no advantage. CPU setup or dispatch observations cannot establish that GPU result.

## Reproduction and evidence

The executed sources are `batched_angular.py`, `bridge_common.py`, `qualify_batched.py` and `full_batched_cpu.py`. Both stage receipts store their exact hashes and the unchanged E source hashes. `analyze_bridge.py` statically checks all thresholds, sources, counts, cost identities, boundaries and the first-source conservative bound, producing `analysis.json`. Raw stage directories, frozen gate, report, analysis, source and checksums are in the new F evidence package. The existing repository and source bank identified by the receipts remain required inputs; the package does not duplicate the dataset.
