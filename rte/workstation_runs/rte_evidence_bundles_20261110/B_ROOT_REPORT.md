# B: one predeclared inclusion-strength perturbation

These are CPU-only measurements. CPU and GPU times must never be mixed in a speed ratio. The paid strong control is compact4 injection with TSA64 for this frozen physical scene and the exact same four source arrays. Original A64 residual<=1e-5, evaluated relative-error bound<=1e-3 and paired field difference<=1e-4 are preserved. First-RHS cost means all setup plus that actual solve/check; it is never the four-RHS total divided by4.

Only inclusion amplitude3 became6, with position/width/grid/RHS/settings fixed and verified. Fixed compact-then-exact order, one fresh fully paid pair. Complete four-source costs: compact4 23.323s, exact8_inner 20.551s. Gain 11.89%. This is not material generalization or repeated timing.

Actual first-source setup+solve/check: compact4 8.222974s; exact8_inner 8.482422s. It is not divided by4.

All8 candidate outer checks passed. Exact inner steps[6,6,6,6], maximum inner residual9.83e-5 against unchanged1e-4/max8. No rescue tuning. Read STRESS_REPORT.md and stress_cpu_01/ JSON. analyze_stress.py additionally expects the immutable A settings next to this directory, available in package A.
