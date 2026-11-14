# C: bounded GPU cost qualification

## Decision

Stop the present dense complex128 GPU realization. It did not clear the frozen four-RHS speed gate. No RHS expansion, threshold change or repeated GPU campaign was used to rescue it. A/B CPU evidence remains valid and sealed; GPU economics are different.

## Actual same-machine evidence

The strong paid control is captured compact4 injection with optimized DirectSegmentedTSA64 on the same RTX A6000. Same4 sources0,4,7,14, N128/L12/g.65, prefix1, P8, inner tolerance1e-4/max8, original outer tolerance1e-5.

| Method | Paid setup seconds | Four-RHS online seconds | Complete seconds | Outer TSA64 calls |
|---|---:|---:|---:|---|
| compact4_inject | 0.506153 | 0.366997 | 0.873150 | [4, 3, 4, 3] |
| exact8_inner | 0.848919 | 0.524638 | 1.373557 | [2, 2, 2, 2] |

**Cold/warm limitation:** compact4 was the first physical construction in this process. Its common source/material/TSA setup was0.389480s; exact common setup was0.025428s. Therefore do not quote the raw total ratio as an unbiased slowdown estimate. This ordering favors exact, yet it still loses. No old GPU or CPU timing is used as denominator. A warm compact4 repeat was not needed to rescue/promote a failing candidate, so the GPU was released promptly.

The predetermined setup lower-bound threshold used the actual compact4 complete0.873150s. Exact qualified setup0.848919s narrowly stayed below it, so the complete exact batch was run. All17 blocks were constructed; this result is not a partial-factor early stop. Even with fewer outer calls, exact online work was0.524638s versus compact4 0.366997s. Thus setup is not the only bottleneck; reducing setup alone does not establish a useful GPU solver.

## First RHS is not a batch average

Actual first-RHS cost includes all method setup plus source0 solve/check: compact4 0.647400s; exact8_inner 0.984379s. Neither is the four-RHS total divided by4. The same cold/warm limitation applies.

## Paid construction, transfers and qualification

Exact factor construction 0.671414s; exact capture, validation and output-ownership checks 0.152077s. Factor storage 641,832,960bytes. All GPU timers explicitly synchronize. Common assembly 0.003613s, per-block assembly sum 0.010124s, LU sum 0.656848s.

Each method staged6,291,456 source bytes host-to-device, charged separately: compact4 0.022629s, exact0.001779s. Factors were built directly on GPU; there was no CPU-factor transfer. CUDA context initialization0.300965s is separately disclosed rather than attributed only to one candidate.

| k | Assembly seconds | LU seconds |
|---:|---:|---:|
| -8 | 0.000349 | 0.107810 |
| -7 | 0.001042 | 0.036005 |
| -6 | 0.000743 | 0.035171 |
| -5 | 0.000629 | 0.034132 |
| -4 | 0.000604 | 0.034153 |
| -3 | 0.000581 | 0.034147 |
| -2 | 0.000564 | 0.034115 |
| -1 | 0.000563 | 0.034105 |
| 0 | 0.000571 | 0.034159 |
| 1 | 0.000573 | 0.034126 |
| 2 | 0.000559 | 0.034134 |
| 3 | 0.000558 | 0.034131 |
| 4 | 0.000575 | 0.034118 |
| 5 | 0.000563 | 0.034153 |
| 6 | 0.000557 | 0.034117 |
| 7 | 0.000548 | 0.034129 |
| 8 | 0.000545 | 0.034142 |

## Correctness and scope

Acceptance thresholds are unchanged: original A64 relative residual<=1e-5, evaluated coercivity relative-error bound<=1e-3, and paired relative field difference<=1e-4, with finite output and solver success required.

All 8 candidate outer attempts passed the unchanged original-equation contract. Maximum original residual 9.07999e-06, evaluated relative-error bound 3.94519e-05, paired field difference 2.64705e-07. Exact inner calls [5, 5, 5, 4], failures 0. Exact GPU background residual 1.2659e-12; capture difference0 and fresh output ownership verified.

Before GPU admission, a small CPU test matched the sealed exact-mode implementation bit-for-bit, checked original background algebra and exercised the partial-stop branch. Ordinary native accepted solves served only as paired field checks; no accurate new teachers or training were produced. Production/source hashes and frozen numerical settings are verified.

GPU script main wall 6.656s; peak allocated 901,654,016bytes, reserved 1,568,669,696bytes. The process exited normally. A subsequent compute-process check was empty and the coordinated window was released to the existing GB work. No GPU job remains owned by this experiment.

## Next directions, not executed

Do not repeat this unchanged GPU realization. A materially new representation could reduce both dense-factor setup and application work; changing factor precision is an inherited implementation choice, but would need a separate CPU original-A qualification first and the same frozen inner/outer limits. It is not already authorized as a successful remedy by this negative result. Do not merely erase setup or add RHS reuse: even the observed online work is slower. Broader physical claims remain outside this single selected scene.

The economical CPU candidate can remain a distinct workload-specific result. Any future GPU experiment waits for a new coordinated natural window and starts with a predeclared cost and accuracy falsifier.

## Files

gpu_gate_01/settings.json includes every setup phase, all17 block records, paid batch costs and source hashes. rows.json contains per-source costs, inner/outer calls and original-equation checks. cpu_qualification_01/results.json records static/CPU qualification. analyze_gpu_gate.py regenerates analysis.json and this report, checking the immutable A settings.
