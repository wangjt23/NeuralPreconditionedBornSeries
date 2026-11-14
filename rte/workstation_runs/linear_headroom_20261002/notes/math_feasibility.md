# RTE headroom: a classical frontier and a physical range oracle

Independent source/retained-result analysis, 2026-10-02. No GPU job, teacher acquisition, new numerical experiment, training or holdout access was performed. All mathematical statements below concern fixed discrete matrices in exact arithmetic; saved FP64 diagnostics are empirical unless their rounding is enclosed.

**Do not infer a representation barrier from the stopped nonlinear tail model or failed94-transform sum repair. First refresh the shallow classical frontier. Then measure the best physical residual attainable in a genuinely different full-spatial/angular-coarse range, before choosing a learned representation.** There is currently no measured positive cost headroom for that range.

## Current budget and the missing classical comparison

The current `benchmark_tail_solver.CLASSICAL` family contains TSA4/8/16/32, reference16/32 and HSS16/32 balanced starts, TSA8/16 zero starts, and reference/moment-tail controls. It omits reference1/2/4 and TSA1/2 zero/balanced starts (`remote_reference/rte/benchmark_tail_solver.py:31–33`). `train_source_predictor.build_base` already supports those depths without changing the physical operator. The earlier small-grid classical audit did include depths1/2/4 (`audit_transport_baselines.py`), but its different CPU fixtures/cohorts do not answer the current N128/N256 question.

The sixteen-medium saved development means are:

| Grid | Frozen portfolio warm ms | TSA16-plain warm ms | Maximum warm ms for5% saving versus saved portfolio |
|---|---:|---:|---:|
| N128 | 135.829950 | 137.723079 | 129.038453 |
| N256 | 143.694969 | 144.751221 | 136.510220 |

Thus the saved portfolio's absolute5% margins are6.791498 ms and7.184748 ms per RHS. A new correction with unchanged classical work has only this small margin available. Once a faster shallow/compiled classical recipe is established, recompute the denominator and margin: these old thresholds cannot support a claim against a weaker portfolio. Source: `receipts/rte/workstation_runs/direction_diagnostic_20261002/results/runtime_summary.csv`.

For a fixed material and declared B distinct RHS, let S be extra method-specific setup, C its warm online correction cost, and D the actual saved classical/solver time. A5% total-cost opportunity requires

    S/B + C <= D − .05*T_classical,

with the right side positive. For one RHS take B=1. Warm repetitions of one RHS do not increase B. This is an exact accounting condition, not an iteration-count surrogate. Charge wide-precision kernels/casts, physical verification, basis acquisition and memory traffic. The existing sum-only intervention retains126→94 transform savings but fails43/48 image gates; it supplies no solve-time D.

The saved TSA16 calls across the sixteen media are72 total at N128 and N256 (mean4.5). Only six media still need work after four calls, and exactly16 calls remain in total. A perfect zero-cost one-shot correction introduced that late could remove **at most16/72=22.22% of those classical calls** on this cohort. This is a call-count ceiling, not a22.22% wall-time bound; the true residual/check/projection work remains. It explains why a late correction cannot be credited with replacing the entire TSA16 solve. A fixed action applied from the start has a different opportunity.

## A quantitative polynomial oracle without fine teachers

For a fixed complex-linear one-stage action F and physical A, define T=AF. Stationary depth J is

    B_J = F sum_(j=0)^(J−1)(I−T)^j,
    A B_J = I−(I−T)^J.

An unrestarted m-step right-preconditioned GMRES correction using B_J lies in `F K_(Jm)(T,r0)`: each polynomial in AB_J is a polynomial of degree at most Jm in T, with the correct zero-degree residual constraint. Therefore exact GMRES on the primitive F after Jm calls can attain a residual **no larger** than stationary-depth-J-preconditioned GMRES after m calls. This is a subspace inclusion, not a floating-point/parity, wall-time or restart guarantee. It also holds in the repository's real-stacked polynomial spaces. Compare the same r0; balanced versus zero starts are separate controls.

The cheapest nontrivial oracle is a single ≤16-direction cycle on TSA1 versus one TSA16 call. No restart truncates that16-direction space. If minimum residual reaches a decisive tolerance after substantially fewer primitive calls, stationary repetition wastes information and the refreshed shallow classical solver may exploit it immediately. Reference1 is also cheap, but it is a different primitive; there is no automatic inclusion between its Krylov space and TSA16's space. Evaluate it rather than assume it is weaker.

Ideal primitive+one-image work counts are:

| Primitive | Primitive transforms | J stationary stages plus one image | J Krylov primitive calls plus J images |
|---|---:|---:|---:|
| Reference1 R |2 |4J |4J |
| TSA1 F |6 |8J |8J |

The equality omits least-squares/orthogonalization, extra true-residual actions and precision differences. The actual solver recomputes physical residuals, so count/time those calls rather than adopting this ideal table as a latency prediction. Inner GMRES/Anderson coefficients depend on the RHS and are generally **not** a fixed complex-linear residual map. A scalar polynomial frozen after material-specific calibration would be linear; fitting its coefficients on its evaluation RHS is only an optimistic oracle and its calibration/setup cost is real.

This shallow-frontier test is the first recommendation. Preserve original FP32 cached matrices, original FP64 physical A, original-RHS tolerance1e-5 and every failure. It is a new classical mechanism, rather than a relaxed acceptance of the failed algebraic TSA replacement. It needs no new teacher or learned checkpoint.

## What the exact historical controls already ruled out

The following are different small development cohorts and are retained as mechanism evidence, not current timing denominators:

* Teacher-error POD ranks2/4/8 and first-RHS Krylov rank8 all remained at five outer iterations on twelve translated N64 RHS. Rank8 captured75.8% of error energy but only30.2% of residual energy. The exact best correction in that fixed image space leaves69.8% mean squared normalized residual, or **83.55% RMS residual**. This is a ceiling for that tested space, not every rank-eight space. Teacher acquisition was approximately4.71 s/material plus0.10 s verification.
* Exact strict-band radial Schur correction changed stiff N128 iterations15.5→15.0 and slowed the solve; compressed/direct projected actions agreed to8.4e-16. Directional294-dimensional Schur Galerkin dictionaries reduced stiff N64 iterations15→12.5, yet remained slower than full-angle controls. Consistency alone did not create headroom.
* Exact m16 harmonic Schur response at N128 required45.5 iterations/11.077 s versus TSA8's9.5/1.907 s. The original full odd-reset composition failed, and harmonic lifting repaired that semantic defect while remaining slow.
* The actual residual-weighted hierarchy lost its advantage at N32: stiff FP32 hierarchy7.17 iterations/232.97 ms versus TSA8's9.33/204.60 ms, with127.2 MiB cache/material and tens of seconds acquisition/compression. Moderate harder-media failures survived FP64 caches. The matched rank32 global repair took249.74 ms versus TSA8's212.79 ms. A small field/inverse approximation error did not bound its physical defect.
* The N128 wider-band source-information oracle did expose information headroom: directional4096/angle5.5 steps/96.26 ms versus zero9.75/165.32 ms. It excluded fine-teacher acquisition and used different materials/baselines; it is not current portfolio headroom. An actual classical coarse64 solve cost644.81 ms.

Sources: `remote_reference/docs/research/RTE_LEARNED_GLOBAL_RESPONSE_DESIGN_2026_09_13.md:217–255`, `RTE_PARITY_STRUCTURE_REVIEW_2026_09_19.md:179–239`, `RTE_RECURSIVE_RESPONSE_RESULTS_2026_09_19.md:59–191`, and `RTE_SOURCE_INITIALIZATION_REVIEW_2026_09_26.md:65–97`. Same-spectral sparse CPU ILU/LU is implemented/tested in `spectral_sparse_reference.py`; its current hardware performance is unmeasured and cannot be silently counted as either a defeated or deployable GPU control.

## A genuinely different range: all spatial modes, few angular modes

Let Q be an L×q fixed angular lifting, initially modes0,±1 (q=3), acting independently at **every fine-grid spatial point**. Retain its actual Gram Gq=Q*Q. For the unchanged physical operator

    A = diag(Omega_x)*partial_x + diag(Omega_y)*partial_y + tI−sP,

the exact angular Galerkin operator is

    A_c = Mx*partial_x + My*partial_y + tGq−sKq,
    Mx=Q*diag(Omega_x)Q, My=Q*diag(Omega_y)Q, Kq=Q*P Q.

It acts on three channels with all spatial frequencies and variable material coefficients. Form the small angular matrices from the actual stored directions/weighted phase in FP64; do not regenerate ideal physical geometry or assume transformed P is diagonal. This differs from the failed radial coarse spaces, fixed POD, symmetrized parity Schur references and homogeneous projected-moment inverses. At N256 it has196,608 unknowns,25% of the full786,432; this dimension reduction is not itself a timing result.

Physical coercivity is inherited: if `Re<u,Au> >= m||u||²`, then `Re<c,A_c c> >= m<c,Gq c>`. A_c still has skew streaming and is generally nonsymmetric. MINRES/CG on A_c is not justified merely by this coercivity; use a nonsymmetric coarse solver. Its exact inverse would be a fixed material-dependent complex-linear action. A finite adaptive inner GMRES solve is a variable preconditioner and must be charged/handled accordingly.

An angular Galerkin correction enforces `Q* residual=0`; it need **not** minimize the full physical residual and can perform poorly despite a tight coarse residual. The physically attainable range oracle below distinguishes poor inversion from poor range.

## Attainable physical range bound, requiring no fine solution

After a declared classical prefix, let r be its original physical residual, B=AQ, and seek

    e_min = min_c ||r−Bc||.

This is the best any correction in range(Q) can do at that point, regardless of its architecture or coarse inverse. B and B* have analytical actions; no basis-by-basis fine solves or dense full matrix is needed. In particular

    B* u = Q*[-streaming(u)+t*u−s*P* u]

for a full field u, with the original complex Nyquist convention. A normal-equation/LSQR implementation must verify this adjoint before using it. The nonsymmetric Galerkin inverse and a physical least-squares range oracle are separate controls.

For an approximate least-squares coefficient chat, compute t_res=r−B chat and gradient g=B* t_res. Physical coercivity gives `sigma_min(B)>=m sqrt(lambda_min(Gq))`. Orthogonality of the optimum residual gives the quantitative interval

    sqrt(max(0, ||t_res||²−||g||²/[m² lambda_min(Gq)]))
      <= e_min <= ||t_res||.

Proof: with cstar the unique minimizer, `B*(r−Bcstar)=0` and `||B(chat−cstar)||<=||g||/sigma_min(B)`; the optimum residual and that error image are orthogonal. The interval distinguishes an unconverged range solve from actual lack of range headroom. FP64 estimates of m/Gram/gradient produce an empirical interval, not an interval-certified theorem application.

The existing minimum evaluated margin across48 physical cases is0.01270603136. For approximately orthonormal Q, an image uncertainty radius≤1% of||r|| requires `||g||<=1.2706031e-4||r||`. If a bounded range solve cannot reach a useful interval, label the oracle incomplete; its failure is not evidence that the range is ineffective. Raw low-angular residual energy supplies no substitute, because streaming maps Q into additional angular modes.

**This lower-singular-value bound applies only to raw B=AQ.** In a Krylov-expanded-space oracle that reoptimizes existing coefficients, the new image is `Bbar=(I−Pi_AZ)AQ`. Projection can destroy rank, so coercivity of A gives no positive lower bound for sigma_min(Bbar). A valid projected interval needs an independently established transversality bound `||(I−Pi_AZ)y||>=beta||y||` on the relevant retained image range, yielding `sigma_min(Bbar)>=beta*m*sqrt(lambda_min(Gq))`, or another validated lower singular-value bound after null-space filtering. Without that, only the achieved projected residual is an optimization upper bound; a gradient-based gap using raw m is invalid. The repository's real-orthogonal AZ projection also must not be silently substituted for a complex projector in these formulas.

Use the same two existing development media at N128/N256, their saved physical RHS and original residual acceptance. Compare actual angular Galerkin correction, the raw best-in-range interval, and the refreshed shallow classical continuation. Include any fine complement rather than resetting unrelated angular response. A free oracle that cannot eliminate meaningful remaining classical work blocks training of that tested composition/cost budget on those cases, rather than every trajectory or expanded-space use of Q. A positive free oracle still needs its actual3-channel solve/setup/application cost measured before it establishes advantage.

## Rank headroom for a fixed cached residual action

For any fixed complex-linear M0 and invertible A, let E0=I−AM0. Any additive rank-k action M=M0+UV* obeys

    ||I−AM||_2 >= sigma_(k+1)(E0).

This optimum is attainable algebraically: choose leading right singular vectors V and `U=A⁻¹V−M0V`, giving defect `E0(I−VV*)`. Acquisition of A⁻¹V is expensive oracle information, not free learned setup. For an orthonormal forward probe block Omega,

    sigma_(k+1)(E0 Omega) <= sigma_(k+1)(E0).

A large observed singular value can reject a proposed **uniform** rank-k contraction claim with only forward actions. A small value cannot certify global compressibility. Actual rounded FP32 M0 is not exactly linear; use a clearly defined fixed stored-matrix action or report the nonlinear/replay discrepancy rather than invoking this theorem directly on a black-box rounded map.

Uniform norm contraction eta supplies the sufficient exact-GMRES bound `||r_j||<=eta^j||r0||`. At zero start, three directions reaching1e-5 have sufficient eta≤0.0215443; four have eta≤0.0562341. Failure of those sufficient norm conditions is **not** a proof of GMRES failure or absence of source-specific headroom. The existing global POD/hierarchy results do not measure this spectrum after current TSA16.

## Decision boundary

The selected next cheap diagnostic is the shallow R1/R2/R4/TSA1 classical accuracy-cost frontier, with the16-primitive Krylov residual oracle as a mathematical control. It resolves a concrete omitted baseline and can reveal stationary-polynomial inefficiency without training or teacher acquisition. If a useful learned/material cache target is still needed, the all-spatial/angular-Q physical range interval is the next defensible feasibility test; its Galerkin accuracy and full residual ceiling must be reported separately.

Require reproducible original-RHS1e-5 convergence, zero discarded failures and a measured net setup-plus-solve/warm advantage against the **refreshed** successful portfolio before claiming headroom. Freeze the material, method, cost scope and decision thresholds before those root-owned bounded runs. No architecture, coefficient training, larger coarse space, new source family or holdout access is implied by this note. A failed bounded oracle establishes a barrier for its mechanism/cohort/cost budget, not a universal impossibility result.

Execution status at handoff: the root reports that its execution bridge disconnected before new runs. Earlier FWI access succeeded; a transport-disconnected tool error alone establishes neither an FWI outage nor loss of SSH/network connectivity. The math/source assessment above is saved locally; all proposed shallow/angular diagnostics remain unexecuted here.
