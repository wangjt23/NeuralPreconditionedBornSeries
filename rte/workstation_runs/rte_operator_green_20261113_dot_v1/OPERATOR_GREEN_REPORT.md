# Operator-first normal Green: weighted proof and cost obstruction

**Decision: PROOF / CONCEPT MILESTONE, not solver promotion.** A concrete inverse-action representation was constructed without the full dense Green inverse. Both fixed original-A0 tests passed. Its present factor layout loses to the existing angular-band baseline in storage, setup and probe application. No rank/tolerance sweep or outer-RTE trial followed.

## What changed

The failed prior normal H representation compressed an already computed inverse, controlling inverse energy rather than original-operator defect. Here the known scalar Fourier differentiation matrix is compressed first, while the entire local 12-angle collision matrix stays intact. A recursive Woodbury elimination factors that approximate operator directly. Its construction does not use response snapshots, teacher solutions or any full (12N)-dimensional inverse. Dense original LU is used only as a control.

The invariant object is the actual layered A0(k), N128, modes k=0 and 8, full g=.65 phase, actual min/gather profiles, stored rounded directions, stored normal frequencies and the complex negative Nyquist. All original-A0 residual tests retain 1e-10 and max 3 repairs. This is not a full heterogeneous 2D solve, a learned solver, a new physical regime or a many-RHS deployment claim.

## Finite-dimensional proof

In space-major order, G=D tensor Cn + blockdiag_j(t_j I-s_j K+i*k_t Ct). D is the Fourier derivative with actual real stored frequencies; Cn and Ct contain the actual real angular velocities. Define alpha=min_j[t_j-s_j*lambda_max((K+K*)/2)]. Its evaluated value here is 0.0499615845142.

For every x, Re<x,Gx> >=alpha||x||², since normal and tangent streaming are skew-Hermitian. Replace D by a skew-Hermitian Dtilde, retain local collision exactly, and call the result Gtilde. The same alpha applies, so ||Gtilde^{-1}||<=1/alpha. Consequently, in exact arithmetic,

    ||I-G Gtilde^{-1}|| <= max|omega_n| ||D-Dtilde|| / alpha = q.

For x0=Gtilde^{-1}b and original-equation residual repairs x_{j+1}=x_j+Gtilde^{-1}(b-Gx_j), the relative residual is at most q^(j+1). This is the implication that the earlier unweighted inverse compression lacked. It applies to the actual stored-value discrete model, not an assumed continuum Green function.

Construction: first Dbar=(D-D*)/2. On each binary-tree sibling pair, truncate one scalar off-diagonal SVD and set the other block to its negative adjoint. Keep dense diagonal leaves. The paired error [[0,E],[-E*,0]] has 2-norm ||E||; same-level pairs are disjoint. Therefore ||D-Dtilde|| is bounded by ||D-Dbar|| plus the sum across depths of the largest per-block Frobenius truncation error. Ranks are outputs of a predeclared absolute budget alpha*1e-3/(2*max|omega_n|*4), not searched parameters.

The measured derivative skew-projection discrepancy is 3.52472e-13; assembled Dtilde has exactly zero measured skew defect. The hierarchy bound is 1.90792611e-05. An independent sqrt(1-norm*infinity-norm) bound is tighter, 1.68961546e-05, yielding q=0.000338182921 and q^4=1.30799681e-14.

**Proof scope:** the inequalities are analytical; alpha, SVD factors and norm bounds were evaluated in ordinary floating point, not interval arithmetic. For a computed action Bhat, E=I-Gtilde Bhat gives I-G Bhat=(I-G Gtilde^{-1})+(G Gtilde^{-1})E, so the bound becomes q+(1+q)||E||. Actual probes measure the corresponding per-input defect, not a uniform floating-machine theorem. Original residuals are still checked after every repair.

## Concrete no-full-inverse factor

At a node, Gtilde=[[A,U_L V_R*],[U_R V_L*,B]]. Child solves construct W_L=A^{-1}U_L,W_R=B^{-1}U_R, then M_L=V_L*W_L,M_R=V_R*W_R. Factor S=I-M_L M_R. With yL=A^{-1}bL,yR=B^{-1}bR and cL=V_L*yL,cR=V_R*yR, solve a=S^{-1}(cL-M_L cR), b=cR-M_R a; output yL-W_L b and yR-W_R a. This eliminates the paired 24r update through a 12r Schur system. All W, M, Schur LU and pivots are counted. No extra low-rank compression occurs inside the factor.

The exact child operators remain accretive and invertible; that does not guarantee benign small-Schur floating conditioning. A separate deterministic recursive algebra unit test passed with residual 2.714e-16 and dense-solution difference 3.443e-16.

## Fixed numerical evidence

Three predetermined normalized complex random columns, one exact-zero column and one 1e-200 scaled column were used per object. No physical RHS or reserved data was accessed. The same probes and original-G contract were used for all methods. Both controls and candidate preserve zero/tiny columns.

| k | Raw original residual, max | Raw Gtilde solve residual, max | Original residual after repair, max | Repairs |
|---|---:|---:|---:|---:|
| 0 | 1.82002e-07 | 7.68784e-13 | 2.02365e-14 | 1 |
| 8 | 2.9194e-08 | 5.51388e-15 | 2.39665e-15 | 1 |

The independent embedding in the original full A0 implementation agreed to 2.87e-16. Dense LU and angular-band both passed. The candidate is an approximate inverse before repair, not a strict one-shot inverse; the final outputs met the original target. This 2-object success is not a uniform discretization or full-solver certificate.

## Cost obstruction, not a speedup

| k | Candidate factor s | Angular-band factor s | Dense LU factor s | Candidate checked batch s | Angular checked batch s | Dense checked batch s |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.241816 | 0.090120 | 0.242577 | 0.036094 | 0.014491 | 0.014397 |
| 8 | 0.234596 | 0.080144 | 0.243042 | 0.036674 | 0.014232 | 0.014003 |

These are single-order diagnostic timings, not repeated benchmark estimates. Each checked batch includes original residual repairs on the five columns. Shared known-derivative construction cost 0.016690s and its proof checks 0.006801s are additional, not free. No heterogeneous-RTE Q1/four-source time is inferred.

Each candidate mode has factor tensor storage of 34,499,952 bytes including the shared scalar tree, versus 16,272,584 for angular-band and 37,754,880 for original dense LU. The candidate is 2.1201x the strong baseline. Sharing the 141,312-byte derivative tree across modes changes little.

| Candidate factor component | Bytes |
|---|---:|
| W_bytes | 17,989,632 |
| ML_MR_Schur_LU_bytes | 13,996,800 |
| leaf_LU_bytes | 2,359,296 |
| pivots_bytes | 12,912 |
| shared_scalar_tree_bytes | 141,312 |

The top scalar sibling rank is 33, inflated to 396 by the full angular tensor product. That top node alone owns 17,260,848 bytes, already exceeding the whole angular-band factor. Under this fixed budget/layout, optimizing only lower-tree leaves cannot eliminate the storage gap. This is a scoped layout obstruction, not a theorem against every structured representation.

The current scalar constructor stores an N-by-N derivative and pays scalar SVDs, O(N²) storage/O(N³) worst-case work. It avoids the much larger (12N)-square inverse and all inverse snapshots, but is not a demonstrated near-linear constructor. Recursive Green factors may store O(sum_nodes[(12n)(12r)+3(12r)²]) numbers. A Cauchy/ACA constructor could reduce scalar setup work but cannot by itself remove the measured W/Schur-factor storage.

## One bounded scalar cause check, not a rescue experiment

The completed scalar spectra show a rounding floor in the actual stored-frequency derivative. A separately declared, paid scalar-only comparison retained the same hierarchy and tolerance but compared actual frequencies with ideal integer frequencies. It did not build an ideal-frequency Green factor, run an RTE solver, or test more tolerances.

Actual top rank 33 becomes 16 for ideal frequencies. The stored-minus-ideal Fourier derivative 2-norm is exactly the maximum frequency difference in the mathematical unitary-FFT model, evaluated here as 2.38898735e-05. A triangle bound that explicitly includes this difference gives model q=0.000851211743, q^4=5.24989267e-13. Thus keeping that difference for original-equation repairs is analytically possible as a preconditioner design; silently dropping it as an exact identity is not.

Using the same factor layout, scalar ranks predict 20,618,080 factor bytes, still larger than the 16,272,584-byte strong baseline. This is a validated tensor-count formula, not measured ideal-factor performance or stability. No ideal Green factor was launched. The check explains one cost source without justifying another solver campaign.

An unused Cauchy-formula diagnostic in the completed run rounded the scalar 2*pi coefficient in float32 before promotion; it did not enter D, factors, solves, bounds or acceptance. The separate replay corrected that diagnostic only: ideal Cauchy-vs-FFT relative difference 4.23485e-15. Original raw results and source remain preserved. The actual derivative was always built from stored frequencies.

## All numerical process costs

| Process | Wall s | User s | Sys s | Peak RSS KiB | Outcome |
|---|---:|---:|---:|---:|---|
| initial_unit_failure | 3.98 | 2.84 | 0.34 | 431,684 | exit1 |
| completed_two_mode_check | 28.62 | 4.63 | 1.03 | 619,716 | exit0 |
| scalar_only_replay | 3.50 | 2.87 | 0.32 | 434,604 | exit0 |

Total numerical process wall: 36.10s. The initial attempt failed during the small algebra unit, before physical cases, due to a PyTorch Kronecker-product noncontiguous-stride issue. Its source/log/results are retained. Replacing that implementation with an explicit equivalent einsum fixed it without changing a mathematical parameter. The corrected unit and two fixed modes then ran once. No failed cost is omitted.

All workers used one CPU core 0, one library thread and CUDA disabled. No other job was stopped. Main wall time contains a large unprofiled component, especially 16.99 s in k0; main CPU user+sys was only 5.66 s. The evidence does not identify the whole unprofiled cost as numerical work or any specific competing process. It is included rather than discarded. Timings cannot establish a robust speed gain on this shared host. Filesystem transfer and analysis latency are not included in numerical-process timers.

| Exclusive wall ledger | Seconds |
|---|---:|
| unit_test_seconds | 0.018668 |
| physical_setup_seconds | 4.874407 |
| known_derivative_build_and_compression_seconds | 0.016690 |
| derivative_proof_checks_seconds | 0.006801 |
| both_modes_original_matrix_for_control_seconds | 0.315953 |
| both_modes_operator_bridges_seconds | 0.379264 |
| both_modes_recursive_operator_factor_seconds | 0.476412 |
| both_modes_dense_original_LU_seconds | 0.485619 |
| both_modes_existing_angular_band_factor_seconds | 0.170265 |
| both_modes_all_probe_repair_and_independent_check_seconds | 0.149962 |
| both_modes_case_other_seconds | 17.230212 |
| main_process_unprofiled_seconds | 3.441685 |
| main_external_envelope_seconds | 1.054062 |
| initial_failed_process_seconds | 3.980000 |
| scalar_only_replay_process_seconds | 3.500000 |

## What is now proved, observed and still missing

- **Proved mathematically:** skew-preserving operator compression leaves accretivity intact; the displayed original-A0 right-defect and refinement bounds; the recursive no-full-inverse factor identity.
- **Observed:** two exact-current-mode implementations passed original tolerances with one repair; finite coefficient/storage formulas and all costs above; frequency rounding causes much of the scalar top-rank inflation at this fixed budget.
- **Not proved:** a uniform floating certificate, economical 17-mode scaling, a faster full 2D solver, or superiority to the strong GPU/CPU full-solver baselines.
- **Scoped obstruction:** this direct tensorized Woodbury layout has too-large factor memory and slower observed setup/action. No unconstrained rank, leaf or precision search is justified.

The next meaningful advance would have to alter the factor representation itself, for example avoid storing all angular-expanded W blocks while preserving the displayed original-weighted error chain. Merely switching scalar construction to Cauchy interpolation, discarding stored-frequency differences, or expanding to more modes is not enough. No such improved factor has been established or launched. Preserve this as a useful proof/construction milestone and a paid negative deployment result.

## Classical context and integrity

The architecture is classical hierarchical operator factorization, not a claim of a new HODLR algorithm. [Aminfar, Ambikasaran and Darve](https://arxiv.org/abs/1403.5337) discuss HODLR direct solvers and their use as preconditioners; [Ho and Greengard](https://arxiv.org/abs/1110.3105) construct structured inverse actions from compressed original operators. Their benchmark ranks/costs do not transfer to this RTE system. The task-specific content here is the actual skew/absorption-based weighted certificate, no-full-inverse construction, and its measured economic obstruction.

Raw main SHA256: `b7b287718541e3c8fe562fb69740eb659ebe311bd7a1edbee21d17e23dc0cb4b`.
Scalar diagnostic SHA256: `9baedeecb767627655112321469bae65583e14c0c4f4c61b745e5bbd928f4c61`.
Production and sealed E source hashes remained unchanged. Owned completed workers were checked exited. Reports, source, fixed plans, receipts, raw outputs and this generated analysis are stored locally and on the existing FWI project server. No Library upload or attachment is attempted.
