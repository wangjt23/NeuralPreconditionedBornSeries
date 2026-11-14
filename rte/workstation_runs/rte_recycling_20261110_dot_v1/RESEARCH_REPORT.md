# RTE: paid recycling and a coupled spectral-subspace candidate

## Decision

Continue the coupled P8 A P8 inner-solve candidate through a frozen timing gate. Do not promote the default solver or claim GPU acceleration.

On the same one-CPU, actual four-RHS development workload, compact4 injection cost 20.851s and exact8_inner cost 17.924s, including setup, inner/outer work and original-equation acceptance: a one-pass observed saving of 14.04%. This is not a repeated or independent performance confirmation.

The key change is actually solving inclusion-induced coupling among retained tangent Fourier modes. A more accurate single background direction, including one Born correction, did not save outer calls.

## Protected contract and scope

- Original stored periodic Fourier RTE A u=b, N128, L12, g=.65. Stored directions, frequencies, phase and Nyquist remain unchanged.
- Previously selected two-layer high-scattering medium, scale10/3, absorption about.05, weak local inclusion; actual development source IDs0,4,7,14.
- Original A64 residual<=1e-5; evaluated coercivity relative-error bound<=1e-3; field difference<=1e-4 against the paired accepted native solution. No new accurate teachers.
- Single CPU thread; later stages pinned to CPU0, separate from Schur cores16-31. Other work could occupy different cores; timing remains exploratory.
- No GPU, training, reserved seed9626 or N256 work. Earlier A6000 milliseconds are not used in any speed ratio here.

## Stage1: paid online Krylov recycling

Maintain A U=C and real-stacked C^T C=I. Compare solution initialization, paid full-Krylov initialization, and GCRO-style projected directions z-U C^T A z. Only earlier accepted sources supply the basis. Charge first source, captured images, copies, MGS, projections and updates.

| Method | Four-source TSA64 calls | Complete CPU seconds | Final rank |
|---|---|---:|---:|
| native | [5, 5, 5, 5] | 24.956 | 0 |
| krylov_init | [5, 5, 5, 5] | 25.625 | 15 |
| solution_init | [5, 5, 5, 5] | 24.721 | 3 |
| gcro | [5, 5, 5, 5] | 26.163 | 15 |
| capture_control | [5, 5, 5, 5] | 25.252 | 0 |

All20 outer attempts passed. AU-C defects were below4.9e-13 and orthogonality defects below8.1e-14. Source4 retained0.999998 of the initial residual norm after prior-source projection; source14 retained about0.692 but still used5 calls. This closes only this paid span and four-source order, not reuse, local ROM, or other source families. These are one mechanism pass, recorded repeat=-1; small timing differences are not interpreted as wins.

## Stage2: exact background is more accurate but does not save calls

Write A=A0+V. A0 uses the observed layered material and exact stored normal Fourier derivative. Each tangent mode gives a1536-dimensional normal-angle block. Factor the17 blocks |k|<=8 and define G8=P8 A0^{-1} P8. It is not the inverse of heterogeneous A. Single-block/full-background residual checks are about1e-14 to8e-14.

| Method | TSA64 calls | Four-RHS complete CPU seconds |
|---|---|---:|
| native | [5, 5, 5, 5] | 25.118 |
| compact4_inject | [4, 3, 4, 3] | 20.658 |
| exact8_inject | [4, 3, 4, 3] | 22.681 |
| exact8_born | [4, 3, 4, 3] | 22.785 |

Exact8 prefix quotient-residual squared ratios0.0033-0.0242 beat compact4 ratios0.299-0.565; one Born correction improves them to0.00046-0.00402. All three still use[4,3,4,3] outer calls. Better direction metrics do not establish solve savings. The extra exact-factor setup is not repaid by this single-direction implementation.

## Stage3: solve the actual coupled P8 A P8 problem

Frozen inner policy: one injection after outer prefix1; zero inner initial state; relative tolerance1e-4; maximum8 inner iterations; no sweep. Exact and compact controls use identical projected original A, identical projected RHS and budgets. Only their preconditioner differs. Every outer solve still stops relative to original b.

| Method | Outer TSA64 | Inner steps | Setup seconds | Four-source online seconds | Total seconds |
|---|---|---|---:|---:|---:|
| native | [5, 5, 5, 5] | [] | 0.148 | 24.317 | 24.465 |
| compact4_inject | [4, 3, 4, 3] | [] | 3.161 | 17.690 | 20.851 |
| compact8_inner | [4, 3, 4, 3] | [8, 8, 8, 8] | 3.161 | 22.705 | 25.866 |
| exact8_inner | [2, 2, 2, 2] | [5, 5, 5, 4] | 5.282 | 12.642 | 17.924 |

The actual four RHS pay 2.121s extra setup and save 5.048s online, net 2.927s. The arithmetic break-even from these observed average online costs is 1.68 RHS; this is not a promise for a new source family. Single-RHS Q1 is slower, so this is a workload-throughput candidate, not improved single-source latency.

Exact inner steps[5,5,5,4] all meet the inner threshold, with true residuals1.17e-5 to8.88e-5. The matched compact inner control exhausts8 steps on every source, retains0.269-0.434 inner residual and does not save outer calls. All four inner failures are preserved; their outer solves still pass.

This is a classical spectral reduced-space/FGMRES result, not neural performance and not a novelty claim. The stored phase is a full12x12 matrix; no unverified angular low-rank/separability property is assumed.

## Factor assembly, LU, copies and memory

Each1536x1536 complex128 LU plus pivots uses37,754,880bytes;17 blocks use641,832,960bytes. A separately paid5.479s calibration decomposes assembly and LU below. It does not replace the main runs' actual paid setup. Common assembly: 0.171s; total assembly: 1.365s; LU: 4.103s. Cross-device copies:0s (CPU throughout).

| k | Assembly seconds | LU seconds |
|---:|---:|---:|
| -8 | 0.0646 | 0.2477 |
| -7 | 0.0685 | 0.2346 |
| -6 | 0.0723 | 0.2377 |
| -5 | 0.0710 | 0.2372 |
| -4 | 0.0708 | 0.2359 |
| -3 | 0.0712 | 0.2505 |
| -2 | 0.0711 | 0.2440 |
| -1 | 0.0711 | 0.2419 |
| 0 | 0.0711 | 0.2439 |
| 1 | 0.0713 | 0.2423 |
| 2 | 0.0710 | 0.2390 |
| 3 | 0.0711 | 0.2393 |
| 4 | 0.0706 | 0.2392 |
| 5 | 0.0696 | 0.2408 |
| 6 | 0.0699 | 0.2429 |
| 7 | 0.0699 | 0.2440 |
| 8 | 0.0697 | 0.2425 |

## Accounting and limits

Complete outer attempts: 52; failed outer attempts: 0. Recorded main-script wall time totals 329.510s. Stage3 maxima: original residual 9.0804e-06, evaluated error bound 3.9454e-05, paired field difference 2.6488e-07.

- Deployment batch cost includes material/acceptance preparation, TSA cache, method factors, inner/outer iterations, projections/FFTs/conversions and final checks. Only the actual same4RHS share one setup.
- Process startup, reading the source bank, receipt writes and research-only diagnostics are excluded from the deployment batch metric; main wall records disclose additional experimental work. Reference/diagnostic data never construct candidate directions.
- No default production solver edits. End hashes of models.py, source_initialization.py, augmented_fgmres.py and train_source_predictor.py match start hashes.
- No independent material generalization, isolated repeated timing or A6000 confirmation has been performed at this checkpoint. No production promotion.

## Next smallest gate

First freeze all parameters and RHS, alternate/randomize compact4 and exact8_inner for at least3 CPU0 repeats, and charge fresh setup each batch. Preserve inner residual/call consistency, not only time. Then define an independent unscreened material perturbation before any generalization test. GPU use must wait for a coordinated exclusive slot; compare with the same hardware optimized TSA64/compact4 and charge exact-factor construction. Do not enlarge RHS count or alter tolerance to manufacture a win.

## Reproduction

- CONCEPT_PORTFOLIO.md records three rounds of debate before each advance.
- analyze_results.py uses the standard library to verify executed source hashes and regenerate analysis.json.
- Raw settings/rows/audits/contexts/results are retained under each *_01 directory.
- FWI root: /data/wjt/NeuralPreconditionedBornSeries/rte/workstation_runs/rte_recycling_20261110_dot_v1.
- Existing Python: /data1/software/miniconda3/bin/python. Set OMP_NUM_THREADS, MKL_NUM_THREADS and OPENBLAS_NUM_THREADS to1; CUDA_VISIBLE_DEVICES empty; taskset -c 0. Scripts refuse existing output directories.
