# Error-controlled nested interfaces constructed; strong baseline still faster

**Result:** a concrete nested normal-interface representation, its inherited-error ledger, and a local elimination factor were constructed and tested. The prior exact inherited-rank obstruction was overcome by explicitly budgeted recompression, not by assuming old ranks. Persistent factor storage and arithmetic proxies improve, but the present implementation does not beat the strong baseline in complete profiled setup or checked application time. No sweep or mode expansion followed.

## Fixed scope and why an actual factor check was justified

One N128, k=0 original layered mode, full 12 angles, actual g=.65 phase, stored directions and original residual tolerance 1e-10 with max 3 repairs. No physical RHS, teacher, held-out set, full dense Green inverse or outer heterogeneous-RTE run. The known angular-band implementation is the decision baseline.

The scalar-only preflight passed its three predeclared conditions: original-weighted model bound, full persistent-factor storage below angular-band, and lower application arithmetic proxy. Those checks justified precisely one nested-factor validation and a fresh angular baseline, not a broader campaign.

## Explicit error ledger

Let D be the original stored-frequency derivative and D0 the higher-precision integer-frequency reference. The reference frequency vector is explicitly stored/evaluated in double precision. In the mathematical unitary-Fourier model, for any two real frequency vectors, ||D-D0||2 equals the maximum frequency difference. Here delta_frequency=2.38898735461e-05. D0 is used only inside the approximate inverse; every residual and repair uses the original stored-frequency G.

At a non-root node I, let QI=diag(U_l,U_r), with identity bases at the leaves. Compute an SVD of QI*D0[I,Ic], keep Z_I with local discarded Frobenius norm <=eta, and set U_I=QI Z_I. A full orthonormal complement is retained for local elimination; a full-rank transfer uses the identity rather than an unnecessary rotation. Define e_I=||(I-U_I U_I*)D0[I,Ic]||F. Orthogonal components give

    e_I² <= e_l² + e_r² + eta²,   with leaf e_I <= eta.

The child exterior contains the parent exterior, which is required for this inequality. Sibling reconstruction uses the ORIGINAL D0, not an already truncated sibling:

    Dhat_IJ = U_I (U_I* D0_IJ U_J) U_J*,
    Dhat_JI = -Dhat_IJ*.

Its block error is at most e_I+e_J. Same-level paired errors are disjoint, so their operator norm is the maximum block norm, not the sum over all nodes. For the fixed N128 / leaf8 hierarchy:

    delta_nested <= 2 eta (sqrt(15)+sqrt(7)+sqrt(3)+1).

The allocation constant is 18.5015709297. With alpha=0.0499615845142, max|omega_n|=1 and unchanged q_design=1e-3, the one-time choice was eta=1.4091620148e-06. No eta, rank or leaf scan was performed.

Dhat is skew-Hermitian, and all local full-angle collision/tangent terms are retained. Thus Ghat has the same absorption/accretivity lower bound alpha. Consequently,

    ||I-G Ghat^-1|| <= max|omega_n|*(delta_frequency+delta_nested)/alpha = q.

The assembled scalar hierarchy bound is 2.76903945506e-06, including its tiny skew-projection discrepancy. It yields q=0.000533588221037 and q^4=8.10634225236e-14. This bounds initial application plus three original-G residual repairs in exact arithmetic. The a priori allocated nested budget was 2.60717109682e-05; the measured bound is smaller, not a post-hoc tolerance change.

**Numerical scope:** alpha, norms and SVDs were evaluated in ordinary floating point, not interval arithmetic. The displayed theorem concerns the mathematical represented operator. A computed inverse action has additional E=I-Ghat Bhat, giving q+(1+q)||E||; the tests below measure per-input floating defects and original residuals, not a uniform machine certificate.

## Actual nested ranks and local factor

| Normal cells | New nested interface rank |
|---|---:|
| 8 | 8 |
| 16 | 13 |
| 32 | 16 |
| 64 | 17 |

These are new nested ranks 8/13/16/17. They are neither the old sibling ranks 33/14/8/6 nor a claim that the old inherited bases had low exact rank. The root has no exterior basis; its two rank17 interfaces produce a 408-dimensional root system.

At each non-root node, combine child interfaces, apply [Z_perp,Z] tensor I12, and split e=12(m-r) redundant coordinates from s=12r retained coordinates. Redundant coordinates have no exterior coupling in Ghat. Store the LU of A_ee, A_se and W_local=A_ee^-1 A_es; pass S=A_ss-A_se W_local upward. W_local is only a local redundant/interface coupling, not the previous response from an entire physical subtree to a wide interface. Both coupling sides are stored and charged because the system is not assumed Hermitian.

A solve uses one upward elimination and one downward recovery. Actual instrumentation observed 31 nodes and 16 leaves in each direction, instead of the previous no-W circuit with 256 leaf solves per application. Instrumentation was not active in application timings. No full physical-subtree response W, expanded M_L/M_R, or dense original inverse is formed.

There is also a favorable exact-arithmetic stability property: orthogonal transformations preserve accretivity, and for y set x=(-A_ee^-1 A_es y,y). Then Re(y*S y)=Re(x*A x)>=alpha||y||². Therefore principal blocks and all successive local Schur complements retain at least alpha. This does not by itself certify floating LU stability.

The deterministic small recursive unit exercised nonzero redundant subspaces and agreed with its independently assembled dense Ghat solve to 1.0115e-15; its residual was 1.54026e-15. It formed only a small control solve with three RHS, never a full inverse.

## Predicted versus actual cost

| Quantity | Nested | Strong angular-band |
|---|---:|---:|
| Persistent factor bytes | 11,639,552 | 16,272,584 |
| Setup complex multiply-add proxy | 143,095,680 | 202,724,693 |
| Application proxy per RHS | 883,488 | 1,015,808 |

Actual nested persistent tensor bytes exactly equal the preflight estimate. The count includes every redundant/root LU, both local coupling tensors, scalar orthogonal transformations and pivots. Child Schur matrices are temporary during construction, not silently retained as free caches. Setup workspace is covered by process measurements.

The proxies explicitly include local rotations, LU, coupling solves, Schur products and the root factor. The angular proxy includes its chain, periodic-corner response construction and corner solve, not just nominal LU calls. Scalar SVD/assembly and QA remain separately paid. Proxies are arithmetic counts, not elapsed-time predictions.

| Method | Factor s | Physical + basis + bound QA + factor s | Three checked applications s | Mean s | Worst original residual |
|---|---:|---:|---|---:|---:|
| nested | 0.134921 | 0.203768 | 0.023831, 0.020821, 0.020451 | 0.021701 | 2.04979e-14 |
| angular | 0.112003 | 0.125563 | 0.027702, 0.013371, 0.012003 | 0.017692 | 2.46093e-12 |

Both methods ran in fresh sequential one-thread CPU 0 processes. The original same fixed three normalized random complex columns, an exact-zero column and a 1e-200 nonzero column were used. All six checked batches passed, each with one original-G repair. No first application or warmup cost was dropped. The angular result differed from the accepted nested reference by at most 4.55572e-12. The nested raw Ghat solve defect was at most 6.37263e-13, before original-G refinement.

The current nested prototype uses 0.7153x persistent factor storage, but complete profiled setup is 1.6228x and mean checked application time is 1.2266x the baseline. These one-object measurements do not establish a general runtime ranking; they do establish no observed speed win here despite lower arithmetic proxies.

## Measured implementation obstacle

| Nested factor construction component | Seconds |
|---|---:|
| assembly_seconds | 0.022055 |
| rotation_seconds | 0.086256 |
| LU_seconds | 0.010446 |
| W_solve_and_coupling_copy_seconds | 0.007999 |
| Schur_update_seconds | 0.007213 |
| other_factor_seconds | 0.000953 |

Normal-basis rotations account for 63.93% of this factor-construction time. This is a measured cost concentration, not a claim that all online overhead has the same cause. The smaller root/local LU work did not compensate for the rotation implementation, assembly and new basis construction. The online phase was not independently decomposed into rotation/solve costs, so no unsupported online attribution is made.

A mathematically equivalent execution-layout change might reduce this overhead, but none was tested here. Such a claim would require the same bases, coefficients and original residuals, plus a new complete-cost comparison. It is not a reason to scan ranks, change tolerances, or expand modes to seek a positive result.

## Memory beyond stored factors

| Method | Current RSS after factor MiB | Factor-phase high-water MiB | Complete external process peak MiB |
|---|---:|---:|---:|
| nested | 429.594 | 429.594 | 454.445 |
| angular | 427.238 | 443.379 | 447.145 |

Stored-factor memory falls by about 28.5%. The factor-phase high-water measurement also falls, but the complete external process peak is slightly higher for nested. Whole-process peaks include imports, basis work, unit/check/reference activity, allocator history and final serialization/shutdown; they are not interchangeable with factor tensor bytes. No overall process-memory win is claimed.

## Full research cost, including the slow preflight

| Process | Wall s | User CPU s | System CPU s | External peak RSS KiB | Exit |
|---|---:|---:|---:|---:|---:|
| preflight | 153.26 | 3.17 | 0.53 | 436,696 | 0 |
| nested | 5.61 | 3.34 | 1.23 | 465,352 | 0 |
| angular | 3.61 | 3.03 | 0.41 | 457,876 | 0 |

Total numerical-process wall: 162.48s. The preflight's 153.26 s wall, including unprofiled waits, is retained in full; user+system CPU was only 3.70 s and its measured physical/scalar/bound stages total about 0.123 s. The evidence does not attribute the missing time to a specific host cause. It is neither discarded nor called solver computation.

The factor-stage workers used aggregate /proc load/memory/CPU snapshots instead of a whole-host process scan, identically for both methods. This is documented in FACTOR_EXECUTION_NOTE.md and does not change any numerical qualification. Before factor launch, a one-second sample found CPU 0 idle; the GPU was busy and was left untouched with CUDA disabled. No other job was stopped or new resource acquired.

The ledgers below include physical/basis/factor setup, all three applications, independent original checks, the algebra unit, reference I/O and separately instrumented application. Transfers and local report generation are outside numerical timers.

### preflight

| Exclusive item | Seconds |
|---|---:|
| physical_setup_seconds | 0.049629 |
| scalar_nested_construction_seconds | 0.070209 |
| bound_and_cost_check_seconds | 0.003326 |
| internal_other_seconds | 98.818718 |
| external_envelope_seconds | 54.318118 |

### nested

| Exclusive item | Seconds |
|---|---:|
| physical_setup_seconds | 0.022874 |
| scalar_nested_construction_seconds | 0.041333 |
| bound_and_cost_check_seconds | 0.004640 |
| factor_seconds | 0.134921 |
| reference_save_seconds | 0.252893 |
| raw_defect_diagnostic_seconds | 0.012052 |
| three_checked_applications_seconds | 0.065103 |
| independent_checks_seconds | 0.009301 |
| algebra_unit_seconds | 0.041811 |
| instrumented_application_seconds | 0.009168 |
| internal_other_seconds | 4.353719 |
| external_envelope_seconds | 0.662185 |

### angular

| Exclusive item | Seconds |
|---|---:|
| physical_setup_seconds | 0.013559 |
| factor_seconds | 0.112003 |
| reference_read_seconds | 0.005844 |
| three_checked_applications_seconds | 0.053075 |
| independent_checks_seconds | 0.009454 |
| internal_other_seconds | 2.892712 |
| external_envelope_seconds | 0.523352 |

## Bounded result and next condition

- **Constructive progress:** inherited/interface recompression has a complete original-A0 error ledger; new low nested ranks were actually constructed; local orthogonal elimination avoids the expanded-response and four-child-recomputation problems.
- **Verified implementation:** fixed original-mode accuracy, zero/tiny behavior, predicted persistent memory and upward/downward visit counts all passed. No full-RTE, multi-mode or uniform floating certificate is asserted.
- **Current obstacle:** complete measured setup/application still lose to angular-band, and complete process peak memory does not improve. The decisive observed setup cost is basis rotation, not an impossibility of nested compression.
- **Stop:** no scan, alternate eta, extra k or N256 trial. No production promotion. A future equivalent batched/layout implementation would need to beat the strong baseline on unchanged full costs; it has not been implemented or promised here.

The general nested-factor architecture is classical. [Chandrasekaran, Gu and Pals](https://epubs.siam.org/doi/10.1137/S0895479803436652) describe HSS/ULV solvers; this work does not claim to invent them. The task-specific evidence is the actual discrete frequency/error accounting, inherited-rank construction, local factor and measured cost boundary.

## Integrity

- `preflight/results.json`: `8e8983bbd543f796fc55d5e6ba13c39ecd58392aa3dbccaad819c0c4da9afa21`
- `nested/results.json`: `7b5cf9f9821a962cfd911d05a16905efc1f2c00fd471cdd8a8b545a7283e07f7`
- `angular/results.json`: `29f7a97e12609846034bf98c34832e1bbec714d081a9cea08d7c16fdb406f096`

Production and sealed E source hashes are unchanged. All source, predeclared plans, raw receipts, references and generated analysis are retained in the isolated local/FWI research folder. No Library upload or attachment is attempted.
