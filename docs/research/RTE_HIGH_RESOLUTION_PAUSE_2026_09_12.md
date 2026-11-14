# Higher-resolution research paused at user request

Pause time: **2026-09-12 21:08 Asia/Shanghai**. Do not resume or launch experiments
without a new user instruction. The two running CPU benchmarks received
SIGSTOP and were verified in process state `T`. No unrelated job was signalled.
All training jobs from this continuation had already completed.

## Suspended processes

Host boot ID: `fab98596-547b-468c-94f5-f046a7cb9b42`.

| PID | /proc start ticks | Experiment | Durable progress |
|---|---:|---|---|
| 3418342 | 712682326 | `resolution_n128_classical_tuning_cpu_20260912` | 84 tuning rows: two complete media, 42 candidates each |
| 3419160 | 712703308 | `resolution_n64_final_cpu_20260912` | Tuning complete; 22 comparison rows: two complete media, 11 selected methods each |

The processes remain in memory, use no CPU while stopped, and retain their
allocated memory. They are not durable process checkpoints and will not survive
a host reboot. Verify the boot ID, start ticks, command and state before any
future PID-targeted action; do not trust a stale PID alone.

**Timing caveat:** resuming a process during a timed region can include the
suspension interval in its wall time. Do not blindly use resumed timing rows.
The safest final measurement is a fresh run in a new directory, preserving these
partial outputs. These benchmark runners do not implement disk-based resume.
No resume or restart was performed after the pause request.

The N64 final run was stopped during a telemetry rewrite; its on-disk
`telemetry.json` is currently empty. Earlier telemetry is still in the suspended
process's memory. Do not interpret the empty file as evidence of an idle device
or fabricate the missing observations. The completed per-medium result/sample
CSVs and saved datasets/checkpoints remain present.

## Completed training (all seed 402)

Directories below are under `rte/workstation_runs/`; each has `last.pt`,
`best.pt`, history and configuration. Solver-aware runs also save actual datasets.

| Directory | Updates completed | Selected update |
|---|---:|---:|
| `nto4_n64_seed402_20260912` | 800 | 800 |
| `ct4_n64_seed402_20260912` | 800 | 200 |
| `nto4_krylov_n64_seed402_20260912` | 400 | 400 |
| `ct4_krylov_n64_seed402_20260912` | 400 | 400 |
| `sct4_krylov_n64_seed402_20260912` | 400 | 400 |
| `sct4_krylov_n128_seed402_20260912` | 200 | 200 |
| `ct4_krylov_n128_seed402_20260912` | 200 | 150 |

The first two use 512 N64 training materials and the old surrogate objective.
The next three use the same materials and a differentiable three-step FGMRES
objective. N128 adaptation uses 256 training materials. All are development
runs; additional independent training seeds have not been run.

## Findings and pending work

- FP32 arithmetic caused a substantial residual floor at N128. The precision
  audit is saved in `resolution_precision_audit_20260912`; mixed FP64 outer /
  FP32 preconditioning keeps tolerance 1e-5 unchanged for every method.
- Full-angle two-level N32 neural correction did not outperform its classical
  counterpart in the initial screens. Preserve these negative results in
  `resolution_n64_screen_cpu_20260912`, `resolution_n128_screen_cpu_20260912`,
  and `resolution_n128_mixed_screen_cpu_20260912`.
- SCT adds learned complex frequency/angle modulation to CT's existing cached
  streaming multipliers without additional residual-path FFTs or dense angular
  matrix products. It is implemented and trained, **not yet a demonstrated
  significant wall-time improvement**.
- N64 CPU classical tuning completed in
  `resolution_n64_classical_tuning_cpu_20260912`; deeper HSS8 was selected.
- The final 24-medium N64/N128 CPU/GPU comparison is **not complete**. N64 has
  only the two durable comparison media above. No final GPU or N128 comparison
  was launched. Do not extrapolate a speedup from training loss/iterations.

See [the protocol](RTE_HIGH_RESOLUTION_PROTOCOL_2026_09_12.md) for data seeds,
candidate grids, precision treatment and acceptance gates. Implementation:
`multilevel_neural_transport.py`, `mixed_precision_transport.py`,
`spectral_collision_transport.py`, `repeated_transport.py`,
`train_krylov_transport.py`, `audit_resolution_precision.py`,
`run_neural_resolution_study.py`, and `make_resolution_specs.py` in `rte/`.

The full suite passed 88 tests before the final two test additions. No new test
or experiment was launched after the user's pause request. The earlier N32
results and original handoff remain historical evidence, not high-resolution
confirmation. No final high-resolution report/hash manifest has yet been made.
