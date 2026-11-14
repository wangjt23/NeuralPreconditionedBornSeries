# Batched angular inverse GPU cost gate

Outcome: NO GO for this eager batched17 implementation on the RTX A6000 under the fixed four-source workload. In the first fresh, device-warm pair, captured compact4 costs0.461313 seconds end to end and the batched angular candidate costs1.119889 seconds. Both pass the original equation. The candidate is slower in both setup and online work, despite reducing the outer TSA64 calls. The predeclared rule therefore stops this implementation without a reverse pair or additional attempts.

This does not reverse the sealed CPU result E or the CPU execution bridge F. It is not a claim about every angular representation or every GPU implementation. CPU and GPU seconds are not mixed into a speedup estimate, and the earlier cold/warm-asymmetric dense GPU experiment C is not used as the current control.

## Frozen contract and admission

The problem remains the original N128 periodic Fourier RTE with12 stored directions, g=0.65, inclusion amplitude3 and source IDs[0,4,7,14] in that order. Both signed sets of tangential modes are retained explicitly, for17 modes in total. The candidate uses the unchanged complex128 F implementation and original full phase matrix, with no positive/negative k conjugate reuse or idealized directions.

The inner solve remains P8 A P8 with tolerance1e-4 and maximum8 iterations. The outer solve retains tolerance1e-5 and maximum100 iterations, plus evaluated relative-error bound <=1e-3 and relative field difference <=1e-4. Original-G block repair remains <=1e-10 with at most3 corrections per mode and column. Source hashes and original production hashes are unchanged.

GB explicitly released the GPU after its benchmark exited. Schur had also finished its numerical window. Admission found no GPU compute processes, and the gate ran as PID1275949 on CPU0 with one host thread. GB continued CPU32 auditing, which is recorded as shared-host work. GPU competitor checks occurred at measured batch entry and exit. The process exited0, and the12:06:08 server-time check found neither that PID nor a GPU compute process. Release was reported to both coordinators.

## Uniform warm convention and all paid research work

CUDA context initialization cost0.284251 seconds. A common neutral library warm-up cost51.228095 seconds. It initialized the relevant FFT, inverse, batched LU, solve and matrix-product operations using neutral tensors rather than the material or measured sources. No warm-up factor was retained. Both methods use the same device-warm convention; initialization and this common warm-up are excluded equally from their measured method costs and explicitly retained in the research accounting.

The neutral warm-up was expensive. These method timings must not be called process-cold latency, and the warm-up is not free. The complete research gate cost54.663067 seconds, including0.518835 seconds for ordinary accepted native references, all measured work and bookkeeping. It stayed below the180-second gate alarm, which is stricter than the authorized600-second total budget. No repeat was run to reduce warm-up or first-call effects.

Each measured method constructs fresh physical, TSA and inverse objects. The CPU-qualified stored operators are built on CPU and transferred, rather than recomputing material coefficients with GPU arithmetic. Core stored values are copied back and checked for exact equality; coefficient hashes match between methods. Every measured batch pays11,408,568 bytes of source/operator host-to-device copying and398,448 bytes of exact-value verification copying back. Candidate qualification additionally transfers417,792 bytes of locked mode vectors. The reference fields are checks only, never initial states or candidate directions.

## Actual synchronized costs

All table values are seconds. Factor and capture/qualification phases have explicit CUDA synchronization at their boundaries. First-source and complete four-source costs are direct synchronized observed wall timestamps, not sums that can omit asynchronous work.

| Measured quantity | Captured compact4 | Eager batched17 |
| --- | ---: | ---: |
| Host physical setup | 0.005374 | 0.007363 |
| Source and operator H2D | 0.003330 | 0.003061 |
| Exact-value D2H verification | 0.000651 | 0.000576 |
| TSA64 preparation | 0.015338 | 0.015388 |
| Synchronized factor construction | 0.063494 | 0.160579 |
| Capture or full mode/boundary qualification | 0.009224 | 0.128623 |
| Observed complete setup | 0.098079 | 0.316225 |
| Four-source online phase sum | 0.362777 | 0.803168 |
| Observed first RHS including all setup | 0.246436 | 0.595649 |
| Observed complete four RHS | 0.461313 | 1.119889 |

All source preparation, factor construction, capture where used,17 independent original-A0 checks, original-G repairs, acceptance checks and required synchronization are paid. Small phase-sum/bookkeeping differences remain in the observed complete wall. The first RHS is source0 with its entire fresh setup, not one quarter of the four-RHS total.

The candidate setup of0.316225 seconds was below compact4's complete0.461313-second batch, so the implementation did not stop at a partial setup bound. It actually completed all four sources. The complete candidate/control ratio is2.4276 for this one pair and the first-source ratio is2.4171. These are observed gate values, not a repeated-performance estimate. No reverse run was permitted after this failure, and none was run.

The F class also records internal `perf_counter` setup subphases. In the GPU receipt they are explicitly named `internal_factor_host_dispatch_only`. Those values are not GPU LU, chain or corner execution times and are not used in this table or the decision. The external synchronized factor wall is the valid cost.

## Original-equation and block evidence

All8 candidate solutions pass original acceptance. The largest original-A relative residual is9.080176e-6, evaluated relative-error bound3.945266e-5 and relative field difference2.647737e-7. The largest inner projected-original residual is8.882623e-5.

The17 independent original-A0 checks have maximum residual3.859944e-12. Every check embeds the repaired mode solution into the physical field and calls the original operator, removing the actual inclusion perturbation. The GPU all-zero and mixed zero/nonzero/1e-200 tests also pass the same per-column criterion. Only exact zero columns are masked.

Compact4 uses outer TSA64 calls[4,3,4,3]. Batched17 uses[2,2,2,2] and inner iterations[5,5,5,4], as in the CPU bridge. During the actual candidate solve,19 batched inverse applications represent323 logical mode inversions. All require exactly one correction, for38 batched raw solves and38 batched original-G evaluations. The maximum runtime original-G residual is3.934985e-12. Thus the no-go is economic, not a loss of accuracy or a changed iteration budget.

## Memory and interpretation

Logical stored inverse sizes are302,137,344 bytes for compact4 and276,427,472 bytes for batched17. Peak GPU allocated memory is554,591,232 and816,064,512 bytes, respectively. Allocated memory at batch entry is32,768,000 and49,807,360 bytes. These peaks include persistent reference fields, physical operators, solver state and construction workspaces; smaller logical factors did not imply a smaller measured peak.

Peak GPU reserved memory is713,031,680 and1,019,215,872 bytes. Allocator reservations carry across the device-warm process, so reserved bytes are not an isolated intrinsic-memory comparison. Process peak host RSS is1,393,356,800 bytes. GPU allocation and CPU RSS are separate measurements.

At the candidate's observed setup cost, its remaining online budget to match compact4 would be0.145088 seconds, versus the measured0.803168 seconds. With setup unchanged, online work would need a reduction of more than81.9%. This arithmetic identifies a substantial execution gap; it is not evidence that graph capture, kernel fusion or another untested change can close it. Both factor setup and online work currently lose.

## Closing decision

Stop this eager complex128 batched GPU implementation. Preserve the positive CPU representation evidence and use compact4 as the current GPU decision control on this frozen workload. Do not add RHS reuse, relax inner8 or residual thresholds, reopen the old dense GPU implementation, or repeat timing until a favorable sample appears.

Any reopening should be a separately debated material execution change with its own original-operator bridge and finite cost gate. Current results do not justify automatic capture, precision or kernel sweeps. No further GPU or CPU numerical work was started after release.

## Evidence

`gpu_gate_01/results.json` contains the complete pair, source/coefficient hashes, synchronized costs, original residuals, memory data and all repair traces. `GPU_GATE_PLAN.md` is the unchanged pre-run plan, whose statement that the directory was static-only describes its creation state. `gpu_gate.py` is the executed source. `analyze_gpu.py` performs static receipt and source validation without importing CUDA and prints `analysis.json`.

- Executed GPU script SHA256:90e124a5a3a2fe207df6b02f2e0560cc7745597225b5c07c7d6a8e98c0025057.
- Unchanged F batched source SHA256:b9aed56b4962ba1620bac5ff9693f7d63ce20e1909f8c5083b4065973b84bec7.

The new G archive contains the source, plan, raw receipt, report, analysis and checksum manifest. It relies on the existing project and sealed F implementation identified above; it does not duplicate the dataset or earlier evidence packages.
