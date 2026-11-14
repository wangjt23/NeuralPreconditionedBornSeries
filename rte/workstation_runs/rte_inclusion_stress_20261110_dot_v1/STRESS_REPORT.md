# B: predetermined inclusion-strength perturbation

## What was frozen

Only the inclusion amplitude changed from3 to6. Position(.34,.60), width.08, layer profile, N128/L12/g.65, scattering scale10/3, original4 RHS arrays/order and every numerical threshold match the immutable A confirmation. The amplitude3 constructor was first verified byte-for-byte against the original. One pair, predeclared compact4 then exact8_inner, CPU0/one thread/noGPU; each method rebuilds and pays all setup.

## Result

| Method | Fresh setup seconds | Four-RHS online seconds | Complete seconds | Observed batch wall seconds | Outer TSA64 calls |
|---|---:|---:|---:|---:|---|
| compact4_inject | 3.190 | 20.133 | 23.323 | 23.338 | [4, 4, 4, 4] |
| exact8_inner | 5.199 | 15.352 | 20.551 | 20.565 | [2, 3, 2, 3] |

Observed phase-summed batch gain 11.89%; directly observed whole-batch gain 11.88%. Extra exact setup 2.009s versus online savings 4.781s. These are one predeclared pair, not repeated timing or material generalization.

Exact inner calls: [6, 6, 6, 6]; true inner residuals: [8.919349788990922e-05, 4.0437766563881286e-05, 9.829862663405952e-05, 4.006224904015579e-05]; inner failures: 0. Budget stayed tolerance1e-4/max8; there was no tuning or replacement material after the outcome.

All 8 candidate outer attempts: 0 failures. Maximum original residual 4.7548e-06, evaluated relative bound 2.73143e-05, field difference 1.72737e-07. Source hashes, frozen settings and production hashes verified unchanged.

Four ordinary native accepted-reference solves were used only for paired field checks, separately costing 24.585s. Total stress-script main wall 72.278s. No accurate new teachers or training.

## Bounded interpretation

This result tests only the predetermined doubled-inclusion perturbation. It is not evidence for arbitrary materials, sources, higher grids or GPU acceleration. Preserve any failures and do not adjust the eight-step budget to rescue this test. GPU economics remain a separate coordinated gate.

## Evidence

stress_cpu_01/settings.json: material/source hashes, frozen parameters and source code hashes. rows.json: all residuals, checks, inner/outer calls and per-source times. batches.json: paid phase costs and observed whole-batch wall. reference_checks.json: ordinary reference solves. analyze_stress.py reproduces this report and verifies equality with the immutable A settings.
