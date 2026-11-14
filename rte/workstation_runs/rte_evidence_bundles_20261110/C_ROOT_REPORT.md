# C: GPU cost qualification, scoped NO-GO

Current dense complex128 exact8_inner did not clear the same-machine A6000 cost gate. Stop this implementation; no repeat or workload expansion was used to rescue it. A/B remain sealed CPU results.

Paid strong control: captured compact4 injection with DirectSegmentedTSA64, same4 source arrays and frozen parameters. Original A64 residual<=1e-5, evaluated relative-error bound<=1e-3 and paired field difference<=1e-4 are unchanged. All8 candidate outer attempts passed; no accurate new teacher/training or reserved cases.

Actual first-RHS setup+solve/check (not batch total divided by4): compact4 0.647400s; exact8_inner 0.984379s.
Complete actual four-RHS: compact4 0.873150s; exact8_inner 1.373557s.
Online four-RHS alone: compact4 0.366997s; exact8_inner 0.524638s.

Important limitation: compact4 paid process-first common initialization0.389480s; exact common initialization was warm0.025428s. The raw ratio is not an unbiased slowdown estimate. This ordering favors exact, yet it still lost. CPU and GPU times are never mixed in a denominator. No symmetric repeat is justified for promotion of this failing implementation.

All17 blocks were built: qualified exact setup0.848919s stayed narrowly below the actual compact4 complete0.873150s setup-stop threshold, so the full exact batch was run. The stop branch is separately CPU-tested. Read GPU_GATE_REPORT.md for construction/LU/copy/capture details and raw gpu_gate_01 JSON for all fields.

The GPU script ran6.656s and exited normally. A post-exit compute-process check was empty; the window was explicitly released to GB. No owned GPU task remains.

Reproduction uses analyze_gpu_gate.py and expects the immutable A confirmation settings in the adjacent rte_recycling_20261110_dot_v1 directory. SHA256SUMS verifies every archived source and receipt. No automatic rerun is scheduled.
