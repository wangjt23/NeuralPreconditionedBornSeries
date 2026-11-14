# Homogeneous-response range: mechanism, limits and cost opportunity

Source/math assessment, 2026-10-02. No GPU, new numerical experiment, project-source sampling, fine teacher, training or seed 9626 access was performed here. Preserve the q3 Galerkin/whitening negatives and classical graph receipts. This note evaluates a different range, without claiming novelty or positive economics.

## Difference from fixed angular q3

Let H be the unchanged physical FP64 periodic operator, with actual promoted stored directions, frequencies, phase P=phase_matrix diag(weights), and heterogeneous coefficients t,s. Let H0 use the same streaming/phase and spatial means t0,s0, with no artificial shift. For the fixed angular lifting Q, define at every original spatial frequency

    U(k) = H0(k)^-1 Q
    B = H U = Q + (H-H0)U.

U has twelve output angles and three coefficient channels. Its angular output range depends on frequency, carrying the homogeneous streaming/scattering response of low-angular source components. Fixed-angular Q excludes that higher-angular response. U still has only three independent columns per frequency; it does not supply the entire heterogeneous inverse or automatically resolve every grazing component.

H-H0 is the actual local collision difference `(t-t0)I-(s-s0)P`. Multiplication by spatially varying coefficients couples Fourier harmonics in B. The intended new information is the coefficient field y needed so Uy corrects the heterogeneous residual, beyond the analytic choice based only on Q*r. It is not the low angular phase eigenvalues or homogeneous transport inverse, both cheap classical information.

**The range already exists classically.** `models.py:reference_transport_solve(rhs,harmonics=1)` projects the RHS into low angular modes before applying the full homogeneous inverse: R0 QQ* rhs for an orthonormal Fourier basis. Its output belongs to range(U). This differs from `reference_moment_transport_solve`, whose output remains in range(Q). Exact Gram retention gives the projector QG^-1Q*, G=Q*Q. Use an FP64 homogeneous reference built from these actual stored factors for numerical parity; the existing deployed FP32 reference and its rounded means are separately preserved as strong classical controls.

The candidate is therefore a **new tested heterogeneous coefficient solve/control in an established response range**. In homogeneous material, B=Q, and exact raw least squares gives y=G^-1Q*r, reducing to the existing truncated homogeneous response. If it cannot improve a strong full-angle homogeneous-reference continuation in heterogeneous media after paying costs, there is no learned advantage to claim.

## Discrete guarantees and conditioning caveats

If Re<x,Hx> >=m||x||² with positive evaluated stored-phase margin, then

    sigma_min(HU) >= m sigma_min(U)
    sigma_min(U) >= sqrt(lambda_min(G)) / max_k ||H0(k)||2.

The second inequality follows from `||Qy||=||H0(k)U(k)y||<=||H0(k)|| ||U(k)y||` at each frequency; consistent Fourier Parseval yields the global bound. A safe computable alternative to expensive exact per-frequency norms is `max|actual_stream|+t0+s0||P||2`, an upper bound formed from actual stored tensors. The denominator generally grows with resolution, so this valid bound can be loose. Direct empirical singular-value checks of cached U may improve the estimate but are ordinary FP64 calculations, not certified enclosures.

For t_res=r-B y and g=B*t_res, define a as the stated lower bound. Exact least squares satisfies

    sqrt(max(0,||t_res||²-||g||²/a²)) <= min_z ||r-Bz|| <= ||t_res||.

A capped solve with a loose bracket is underresolved; a large attained residual is not an optimal-range rejection. Old real-AZ projection can destroy rank, so neither this a nor raw physical coercivity transfers to `(I-P_AZ)HU`. Projected cases retain attained upper bounds unless separately verified transversality exists.

An optional diagnostic reparametrization is V=U(U*U)^-1/2 at every frequency, giving V*V=I and raw sigma_min(HV)>=m. It preserves the same range. Its gradient can be evaluated after a solve as a differently conditioned gap diagnostic, or its action can be used as a separately charged conditioner; it is not automatically part of the frozen follow-up. It need not improve iteration conditioning: H0U=Q is already well scaled in the homogeneous limit, whereas normalizing U itself can reintroduce frequency scaling. No universal 100-step or heterogeneous condition guarantee is supplied.

The optimized identity B=Q+(H-H0)U and its adjoint require actual-operator parity against HU and U*H*. In floating point, H0U is approximate and cancellation removes a nonzero identity defect. The earlier algebraic-TSA FP32 failure is a reason to retain the original H action as acceptance authority, rather than granting a cancellation identity exact deployment status. No geometry/phase symmetrization or acceptance relaxation is permitted.

## Why one-shot negatives do not settle the from-start composition

The completed fixed-Q whitening diagnostic found raw-source residual floors around 0.60–0.66 and no cached continuation gate pass; projected prefix optima remained unresolved. Those findings concern range(Q) and its tested source/prefix composition. They do not transfer to range(U), nor to a response correction applied from the first direction with a cheap full-angle complement.

For a frozen cheap complement M4 (TSA4), a coherent exact composition is

    M = M4 + U B† (I-H M4).

Here B† is the physical least-squares pseudoinverse. The residual after the coarse correction is `(I-P_range(B))(r-H M4 r)`, so it cannot exceed the cheap complement's residual in exact arithmetic. It preserves the full complement, rather than resetting omitted angles. This proves neither fewer FGMRES directions nor lower time. Finite CGLS coefficients depend on the RHS and make a nonlinear oracle; a genuinely residual-linear target requires material-frozen linear factors or a fixed linear approximation to B†. A from-start study is a distinct scope requiring its own bounded gate; it is not silently ruled out or authorized by a failed one-shot test.

## Primary predecessors and actual competitors

[Fan–An–Ying](https://arxiv.org/abs/1802.03061) already combines angular reduction, homogeneous FFT inversion and heterogeneous recursive skeletonization. [Dölz–Palii–Schlottbom](https://arxiv.org/html/2102.09038v2) supplies anisotropic angular-subspace correction under a different variational discretization. These prevent treating kinetic response coarsening as unexplored merely because this particular Fourier/S1 implementation is new.

[Peng FGMRES–ROMSAD](https://arxiv.org/html/2410.08735v3) and [Tang–Peng trajectory-aware ROM](https://arxiv.org/html/2509.05001v1) are close correction-space/trajectory competitors, with different transport discretizations and parameter/setup assumptions. [McMg](https://arxiv.org/html/2606.30495v2) establishes a material-cache/residual-linear multilevel design in Helmholtz; it does not prove this RTE range's speed or complex linearity. [Dimola et al.](https://arxiv.org/html/2507.15452v1) makes Krylov-geometry/continuation training an established direction, not a new theorem here. These sources motivate mechanism-specific controls and total-cost measurement, not copying an architecture name.

## Economics and conditional next ranking

The primary workload is **one RHS for a new medium**. Compare all candidate setup plus its complete solve with the lower complete cost of successful original eager and exact-original replay recipes. Replay capture is not free: the verified 16-medium graph means improve warm time greatly but lose setup-plus-one-RHS means at N128/N256. Q=16 is a conditional secondary forecast only; one timed RHS, global-phase aliases, or sixteen different media do not constitute sixteen independent-source solves per medium. Charge acquisition, encoding/factors, model loading, every U/U* application, physical defect, helper/LS/check, memory and failures. Positive cached-proposal savings merely specify a budget for those omitted costs.

If the bounded response oracle has no usable budget, rank the next mechanisms by a new cost opportunity, not another training run:

1. **A genuinely cheaper fixed-material full-angular action replacing deep TSA from the start.** It addresses all solver calls and can retain modes that the q3 ranges omit. Require fewer costly full-resolution passes or demonstrably cheaper angular/storage factors than existing cached CT/NTO; those implementations and the failed dense/hierarchical response controls remain mandatory. A free exact full-angular inverse oracle is tautological and supplies no representation feasibility. First exhibit a constrained representation/action-budget oracle against the updated complete-cost portfolio.
2. **TSA16-trajectory-specific correction learning, only after a charged information ceiling passes.** Existing mixed training was largely behind TSA8, and current stronger TSA16 tails showed little saved work. Alignment could improve useful directions, but the late eligible subset and network cost sharply limit economics. Before training, demonstrate a feasible correction on those actual ordered TSA16 trajectories that repays a specified inference/setup cost. Established trajectory-aware ROM and differentiable-Krylov controls must be credited.

This ranking is provisional. Neither restoring generic CT/NTO caching nor blindly retraining a residual FNO changes the mechanism. A future candidate must show both a concrete operation/storage advantage and matched-accuracy cost headroom before learning is justified.
