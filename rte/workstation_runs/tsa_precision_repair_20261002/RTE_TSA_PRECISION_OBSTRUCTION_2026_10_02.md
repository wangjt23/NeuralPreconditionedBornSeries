# RTE TSA precision obstruction — 2026-10-02

The isolated FP64-sum repair **failed the unchanged parity gate: 43/48 checks passed**. The same five physical-source, depth16, complex64-input cases failed. Stop this candidate; keep the existing classical portfolio. No full-solve accuracy/timing comparison, promotion, training, or seed-9626 access followed the failed entry gate. This is an application-parity failure, not evidence of FGMRES convergence failure or a proof that every wider-precision reformulation is uneconomic.

## Workspace and scope

Existing SSH alias FWI succeeded with configured forwarding disabled and host-key checking retained: user `wjt`, host `FWI-Standard-PC-i440FX-PIIX-1996`. The repository is `/data/wjt/NeuralPreconditionedBornSeries`, origin `git@github.com:wangjt23/NeuralPreconditionedBornSeries.git`, branch `main`, HEAD `028caea3f35d100225f656e2b1e910c1366aaafe`. RTE means **radiative transfer equation**. New work is isolated in `rte/workstation_runs/tsa_precision_repair_20261002`; original source, dirty work, models and earlier reports are preserved. No OL_preconditioner work was resumed.

The problem is the existing periodic 2D, 12-angle discretization, anisotropy .65 and epsilon .1. All checks use the same two development media IDs 0/3 from seed4626, nested N64/N128/N256 fields, depth1/16, original source and deterministic grazing-Nyquist probes, complex64/128 inputs. No fresh physical draws or held-out evidence were used. The 48-row gate is identical: finite outputs; field and original physical-image relative differences at most `1e-4` for complex64 or `1e-9` for complex128; intended transform counts. Validation-parent hashes and all seven regenerated provenance fields were checked exactly before the gate.

## Diagnosis and error budget

Three retained diagnostic executions refined the same five failures. The final diagnostic verifies independently constructed original/fused reference caches are **bit-identical**, evaluates 80 stages, and closes 15 sampled stage budgets. Diagnostics use the actual FP32-assembled streaming symbol promoted to FP64; acceptance still uses the original physical FP64 operator H, whose products are formed from promoted stored factors. No ideal directions, phase symmetrization or replacement physical operator is introduced.

For `L=S+gamma I`, `D=L−A`, and the exact linear action R of the stored inverse bytes, both exact-arithmetic formulas share `F=G+RDG`, where `G=L⁻¹`. At common rounded input d and field g:

```
h_fused − h_original = (L g − d) + e_local + e_A − e_subtract
u_fused − u_original = R(h_fused − h_original) + delta_rho_R + delta_beta_add
delta_z_next = (I−FA) delta_z + F(delta_tau_outer) + delta_epsilon_inner + delta_alpha_sum
```

The associated bound is `||H delta_z_J|| <= sum_k ||H (I−FA)^(J−1−k)|| ||forcing_k||`; no contraction, independent-noise assumption or spectral-radius-only argument is used. The numerical ledgers are FP64 estimates, not interval certificates. Terms are vectors and can cancel, so their norms are not additive causal percentages.

All five old failures reproduced exactly. Local defect cancellation ratios were only 1.056–1.672. In the final difference images, 89.53–94.80% of power lies above each grid's N/4 frequency mask; Nyquist lines account for 22.43–27.95%. These overlapping, grid-relative bands show high-frequency amplification within each grid. They do not compare fixed physical bands across grids or single out a vendor FFT kernel.

At N256/ID0/stage0, the reference-action rounding difference has physical-image norm .015182, versus .000202 for the transported local-defect difference and .000621 for stage-addition rounding. The forward/division/inverse-FFT ledger identifies contributions to `L g−d`, but that discarded identity residual alone does not explain the actual stage discrepancy.

The closed late-stage budget locates a larger **outer physical-defect rounding difference**. At N256/ID0/stage15 its transported H-image norm is .077245, compared with .020183 for accumulation, .0000616 for native inner error, and .079898 for the actual discrepancy. It is the largest term in all five final-stage budgets. The maximum recurrence closure H-image norm is `1.9684e-11`. This accounts for original-versus-frozen-fused trajectories; it does not uniquely split the outer FFT/action/subtraction kernels, measure the repaired trajectory's ledger, or rank either branch's absolute error against an exact inverse.

## One frozen selective-precision attempt

`SumStableAlgebraicTSA` keeps G, scattering, local defect, cached R, and the outer physical action at the input precision. Only stage sums, outer accumulation and output use complex128. The accumulator is cast back to the input dtype before the original outer action. This targets a sizeable accumulation term at lower cost than wider physical/reference FFTs. Its complex128 output is explicit and matches the deployment wrapper's existing output precision. Relaxation remains 1; the original cache refresh and stored-physics guard are inherited.

The policy and source hashes were frozen before execution. The complete 48 checks finished in 8.535 seconds internally, 12.163 seconds including process/import startup; exit2 signifies the completed gate failure. All outputs were finite and all transform counts correct. Depth16 still uses **94 spatial transforms versus the original 126**, with no FP64 transforms for complex64 input. All24 complex128-input checks pass, retaining the stored FP32 inverse rather than rebuilding it.

| Grid / ID | Previous fused image difference | FP64-sum image difference | Limit |
|---|---:|---:|---:|
| 64 / 0 | 1.397991e-4 | 1.380606e-4 | 1e-4 |
| 128 / 0 | 2.566321e-4 | 2.575576e-4 | 1e-4 |
| 128 / 3 | 1.049790e-4 | 1.012756e-4 | 1e-4 |
| 256 / 0 | 5.386726e-4 | 5.180030e-4 | 1e-4 |
| 256 / 3 | 2.121982e-4 | 2.308752e-4 | 1e-4 |

The worst image difference remains **5.18 times the gate**; two cases worsen. The global field-difference maximum `5.039868e-6` belongs to a passing grazing probe, not one of these failed rows. Complex128 maxima are field `9.3911e-15`, image `1.4295e-12`.

## Costs, decision and limits

At N256, the repair adds 32 field promotions and 15 state casts per application: a conceptual **846 MiB** of read-plus-write conversion traffic before cache effects. The deployment wrapper already promotes the original output once, so that shared-wrapper comparison would remove one conversion from the extra ledger. The wider accumulator occupies12 MiB, 6 MiB more than a complex64 accumulator; no wider persistent inverse cache is added. Each arm retains the original72 MiB inverse. This ledger is not measured DRAM traffic or a latency prediction.

Separate depth16 construction receipts, two media per grid, have median original/candidate setup including the candidate guard of 3.025/3.314ms at N64, 3.725/3.772ms at N128 and 5.665/6.011ms at N256. They are incidental preparation observations, not a dedicated warm performance study. The combined 48-check process peaked at442.66 MiB allocated and610 MiB reserved. The diagnostic peaked at874.66 MiB allocated and1078 MiB reserved; these are process peaks with both arms/instrumentation, not candidate-only costs.

The concrete obstruction is a failed physical-image gate despite wider sums and extra conversion traffic. Stop this lowest-cost repair and make no whole-solver speed claim. Wider reference/outer-action execution would pay for different kernels and casts and requires a new explicit precision policy; its feasibility and cost are unmeasured. Do not infer universal impossibility from this bounded negative result, loosen the gate, consume seed9626, or start a neural/coarse training campaign to bypass it.

The original FGMRES code, FP64 physical A/RHS, residual tolerance and acceptance/rejection semantics were not edited. No full solve ran because the entry gate failed. A strict conditional six-pair protocol is retained for future authorized work; this experiment supplies no final-solution accuracy, cost saving, OOD or confirmatory claim.

## Evidence and reproduction

The bundle contains the failed old48 cases, frozen policy, exact executed helpers, three diagnosis receipts, closed stage budgets, all48 repair rows, process logs and preservation hashes. Run only in the identified existing FWI repository using `/data1/software/miniconda3/bin/python` (Python3.12.4, Torch2.5.1/CUDA12.4, NumPy1.26.4, SciPy1.13.1), with the recorded repository/code directories in PYTHONPATH. `check_sum_stable_tsa.py --output <new-empty-directory> --device cuda` reproduces the gate using the hash-pinned existing validation parents. It intentionally exits2 on failure after writing all rows. The isolated modules require the existing dirty project state and inputs; this compact report bundle does not contain datasets or model weights.

Final preservation and Library round-trip receipts accompany the report. Private Library storage does not authorize promotion, public sharing or further compute.
