# Removing expanded response storage: real memory saving, expensive recomputation

**Completed decision:** the exact no-W/no-M representation works and preserves the original-equation acceptance contract, but is a memory–compute tradeoff, not a faster solver. Three fresh sequential workers completed the single fixed N128, k=0 comparison. No other mode, tolerance, leaf size or compression rank was tried.

## State and unchanged model

The previous operator-first construction keeps actual stored frequencies, all 12 angular channels, full anisotropic phase, the original min/gather background, and the nonzero grazing velocities. It compresses only the known scalar normal derivative while preserving skew-Hermitian structure. This follow-up changes only the inverse-factor storage/evaluation of the same Gtilde. Cached and recomputing derivative matrices have the same byte hash: `b929406e6df55e68851a75528aa30b8815ef55e7f1ca0449b6fb33aae52140e3`.

The unchanged exact-arithmetic model bound is ||I-G Gtilde^-1|| <= q=0.000338182921. The same original residual target 1e-10 and max 3 repairs apply. Floating implementations have additional per-input E=I-Gtilde Bhat; the earlier q+(1+q)||E|| accounting remains necessary. No uniform floating or interval certificate is claimed. No physical source, held-out dataset, training, full dense Green inverse or outer heterogeneous-RTE solve was used.

## Exact construction without stored W or M

For Gtilde=[[A,U_L V_R*],[U_R V_L*,B]], define S=I-(V_L*A^-1 U_L)(V_R*B^-1 U_R). Store only pivoted leaf factors and S LU. U and V actions retain the scalar spatial factors and actual angular diagonal, without keeping their angular-expanded tensors. For each RHS:

    yR = B^-1 qR
    zL = A^-1 (qL - U_L V_R* yR)
    a  = S^-1 V_L* zL
    tR = B^-1 U_R a
    xR = yR - tR
    xL = zL + A^-1 U_L V_R* tR

Substitution into the two block equations proves the identity. This removes W_L,W_R,M_L,M_R from persistent storage. It does NOT remove the Schur LU. To avoid claiming a low persistent footprint after a high-memory constructor, each S was constructed in fixed 12-column panels, one scalar interface coordinate times all angles, using child solves. The expanded W/M matrices were never constructed then freed. Every streamed child solve is paid in setup.

This evaluation circuit makes two left and two right child solves. Its recurrence is T(n)=4T(n/2)+local work, versus two child solves plus cached W/M contractions. With the fixed four-level tree, one application makes 256 leaf visits rather than 16. This is a property of this circuit, not a universal lower bound on every structured solver, nor a prediction of 16x total arithmetic or wall time. Removing cached contractions offsets some additional work.

The new small recursive algebra unit passed against both a dense solve and the cached factor. On the real object, independently instrumented calls measured exactly 31/341 node visits and 16/256 leaf visits for cached/recomputing respectively. Instrumentation was removed for all reported application timings.

## Accuracy and matched cost

Each process paid its own physical setup and method-specific setup. Angular-band did not pay a derivative tree it does not use. Three checked applications of the same five input columns (three normalized complex random, exact zero, 1e-200 nonzero) were retained, including the first call. No warmup result was removed. A small accepted cached reference was written/read only after construction for equality checks; this cost is recorded.

| Method | Factor MB | Factor s | Physical + used tree + factor s | Three checked applications, s | Mean s | Max original residual |
|---|---:|---:|---:|---|---:|---:|
| cached | 34.499952 | 0.249031 | 0.278742 | 0.036457, 0.033766, 0.033175 | 0.034466 | 2.02365e-14 |
| recompute | 7.179312 | 2.793176 | 2.823660 | 0.245138, 0.242697, 0.241989 | 0.243275 | 1.9793e-14 |
| angular | 16.272584 | 0.100392 | 0.112518 | 0.025775, 0.012472, 0.011540 | 0.016596 | 2.46093e-12 |

All nine checked applications passed, all with one original-G repair, including zero/tiny handling. Recomputing differs from the accepted cached solution by at most 1.06305e-14. Its raw Gtilde solve residual reached 5.35093e-12, versus 7.68784e-13 cached; this extra floating error was not assumed away.

Persistent tensor storage is 0.2081x cached and 0.4412x angular-band. The exact reduction removes 17,989,632 W bytes and 9,331,200 M bytes; the recomputing action additionally retains the 192-byte actual direction backing storage. The remaining representation still includes 4,665,600 bytes of expanded Schur LU, 2,359,296 leaf LU bytes, 12,912 pivot bytes and 141,312 scalar-tree bytes.

Observed streamed factor setup was 11.216x cached and 27.823x angular-band. Across all three applications, mean time was 7.058x cached and 14.659x angular-band. These are this single-object diagnostic's observations, not a general timing theorem or a full-RTE throughput result. A tighter memory limit could make the tradeoff useful; that is not the present speed objective.

## Resident memory, including setup workspace

| Method | Factor-phase process high-water MiB | Application-phase high-water MiB | Complete external process peak MiB |
|---|---:|---:|---:|
| cached | 457.594 | 457.594 | 474.191 |
| recompute | 415.297 | 415.910 | 439.797 |
| angular | 445.207 | 445.207 | 449.098 |

Logical factor bytes and whole-process RSS are different measurements. All three workers started separately, so no method carries another factor in its reported footprint. RSS includes PyTorch/runtime, original operator, buffers and allocator history. The cached process later performs additional structure diagnostics, so its final process peak is not the like-for-like factor peak. The external receipt also covers final serialization/shutdown after the last internal sample. The memory saving is real in both stored tensors and measured process footprint, but is much smaller in full-process percentage than the factor ratio.

## Why two easy ways to remove Schur storage are not established

**1. Primitive angular banding is not closed under inversion.** In the actual stored data, the 12 entries of Omega=diag(omega_n) are distinct (minimum separation 5.56363e-08), and every phase entry is nonzero (minimum 0.0174768). Polynomial interpolation in Omega produces each coordinate projector E_ii. Then E_ii K E_jj / K_ij=E_ij. Hence the algebra generated by Omega and K is all 12-by-12 matrices. A small fixed exact angular algebra cannot be inferred from the original few Kronecker terms or near-banded angular Fourier representation.

This theorem does not prove that the particular current W/S tensors need full approximate separation rank. The exact distinctness partly comes from tiny stored rounding differences. Any approximation exploiting near symmetry is a different preconditioner and needs an added original-A0-weighted defect bound. Existing angular-band already exploits primitive angular structure before elimination; it does not license silently truncating inverse-response factors.

| Root factor object in angular Fourier coordinates | Frobenius fraction outside cyclic nearest-angular band |
|---|---:|
| root_WL_angular_fourier | 0.027451161 |
| root_WR_angular_fourier | 0.027860090 |
| root_S_minus_I_angular_fourier | 0.021625757 |

These nonzero off-band measurements are diagnostic, not an original-weighted certificate for any truncated inverse. No band-truncated W/S solver was run.

**2. A matrix-free Schur inverse needs a new convergence argument.** Removing S LU leaves S-vector actions, each requiring two child inverse actions. On the existing 396-dimensional root, ||S||F=320.606914; therefore by the triangle inequality and ||X||F<=sqrt(R)||X||2,

    ||I-S||2 >= (||S||F-sqrt(R))/sqrt(R) = 15.1111036 > 1.

Thus taking I as an inverse and invoking a Euclidean operator-norm-contractive Neumann argument is unavailable. This is not proof that the spectral radius is >1, and does not rule out another interface preconditioner. Original-G accretivity alone is insufficient: rescaling equivalent interface coordinates can change Euclidean Schur conditioning without changing G. No matrix-free interface iteration was launched.

## What the nested-basis escape would actually require

A nested HSS/ULV elimination is a materially different representation that could avoid this recomputation circuit. It cannot simply reuse the old sibling ranks. An exact nested basis must span restrictions of ancestor interface bases as well as its sibling interaction. The four fixed 32-cell restrictions were checked, without a rank/tolerance search:

| Quarter | Inherited numerical rank | Smallest inherited singular value | Combined inherited+sibling numerical rank | Smallest combined singular value |
|---|---:|---:|---:|---:|
| 0 | 32 | 3.11955e-09 | 32 | 1.12172e-06 |
| 1 | 32 | 3.11955e-09 | 32 | 1.12172e-06 |
| 2 | 32 | 3.11955e-09 | 32 | 1.12172e-06 |
| 3 | 32 | 3.11955e-09 | 32 | 1.12172e-06 |

All are numerically full row rank under the recorded standard floating-point rank threshold. The conditional algebraic statement is simple: a rank-32 inherited map on 32 cells needs 32 interface directions for exact representation, so there are no redundant normal directions to eliminate at that node. The numerical observation is not an exact-rational rank certificate and not a theorem that approximate nesting is impossible. Smaller nested interfaces would require recompressing the inherited+local spaces and charging that new error to the original-A0 bound; the old rank14/8 sibling counts and q cannot just be reused.

Classical nested-factor methods are described in [Chandrasekaran, Gu and Pals, A Fast ULV Decomposition Solver for Hierarchically Semiseparable Representations](https://epubs.siam.org/doi/10.1137/S0895479803436652). That is distinct from Hermitian/skew-Hermitian splitting, also called HSS in this repository. Their small-rank premise and cost claims have not been established for this actual RTE derivative/factor.

## Complete numerical process accounting

| Worker | Wall s | User s | Sys s | External peak RSS KiB | Exit |
|---|---:|---:|---:|---:|---:|
| cached | 4.36 | 3.53 | 0.48 | 485,572 | 0 |
| recompute | 7.52 | 6.72 | 0.41 | 450,352 | 0 |
| angular | 3.73 | 3.15 | 0.41 | 459,876 | 0 |

Total numerical process wall = 15.61s. This includes the small unit, all three methods, all repeated checks, reference I/O, call instrumentation and structural diagnostics. No numerical worker failed. All used one existing CPU core 0, one library thread and CUDA disabled. Source/runtime checks found no production or sealed E changes. No other job was stopped. Transfers and local scalar report generation are outside numerical timers.

Exclusive per-worker wall ledgers:

### cached

| Item | Seconds |
|---|---:|
| physical_setup_seconds | 0.011722 |
| derivative_tree_and_bound_seconds | 0.017989 |
| factor_seconds | 0.249031 |
| reference_save_seconds | 0.030851 |
| raw_defect_diagnostic_seconds | 0.018625 |
| three_checked_applications_seconds | 0.103398 |
| three_independent_checks_seconds | 0.009516 |
| unit_seconds | 0.021452 |
| count_instrumented_application_seconds | 0.015483 |
| structure_audit_seconds | 0.102948 |
| internal_other_seconds | 3.226085 |
| external_envelope_seconds | 0.552899 |

### recompute

| Item | Seconds |
|---|---:|
| physical_setup_seconds | 0.012248 |
| derivative_tree_and_bound_seconds | 0.018236 |
| factor_seconds | 2.793176 |
| reference_read_seconds | 0.006505 |
| raw_defect_diagnostic_seconds | 0.122997 |
| three_checked_applications_seconds | 0.729824 |
| three_independent_checks_seconds | 0.009607 |
| count_instrumented_application_seconds | 0.121079 |
| internal_other_seconds | 2.963895 |
| external_envelope_seconds | 0.742433 |

### angular

| Item | Seconds |
|---|---:|
| physical_setup_seconds | 0.012126 |
| factor_seconds | 0.100392 |
| reference_read_seconds | 0.004848 |
| three_checked_applications_seconds | 0.049787 |
| three_independent_checks_seconds | 0.009482 |
| internal_other_seconds | 3.013795 |
| external_envelope_seconds | 0.539569 |

## Bounded conclusion

- **Established:** an exact representation identity eliminates expanded W/M storage; the same model error bound is retained and the original floating acceptance tests pass. The memory reduction and extra work are both real.
- **Not established:** eliminating all Schur storage cheaply, a small exact angular closure, or small inherited nested interfaces. The specific simple contraction argument fails; broader methods are not ruled out.
- **Current decision:** do not deploy this recomputing variant as a speed improvement, and do not add modes or scan parameters. Preserve it as a verified memory–time tradeoff and structural diagnostic.
- **Necessary new premise to reopen:** construct nested/recompressed interfaces with an explicit additive original-operator error ledger and a factor-layout count that can beat the strong angular-band path before implementing a larger solver. No such candidate has been established or launched.

## Integrity

- `cached/results.json`: `a76dea463f796b4e5b53f061a361ab3b9f1258d0167c3db6f133abff17b26133`
- `recompute/results.json`: `8ca43c526880326d867ea5793926e5fae08894b21cbb7875012d2ab54cd6cce9`
- `angular/results.json`: `0f9e498f545bc6412452eb029b85d42bb187005e5fe250409c19e24f3dcc6a0c`

All fixed plans, exact source copies, raw outputs, timing receipts and analysis are saved in the isolated local/FWI research folder. No Library upload is attempted.
