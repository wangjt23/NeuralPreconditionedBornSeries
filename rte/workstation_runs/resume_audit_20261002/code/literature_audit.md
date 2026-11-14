# RTE learned-preconditioner literature audit

Prepared 2026-10-02. Primary papers were checked against the confirmed steady radiative-transfer project: periodic two-dimensional pseudo-spectral space, twelve equispaced S1 ordinates, normalized circular phase at fixed g = 0.65, and A = Omega dot grad + sigma_t I - sigma_s P. Repository evidence comes from `remote_reference/docs/research/RTE_KRYLOV_TAIL_RESULTS_2026_09_26.md`; no independent remote experiment was run by the literature reviewer.

The repository already has safeguarded tail injection, projection against the retained Krylov image space, and two-step solver continuation training. Its unseen nested-grid study reports zero failures but neural runtime losses of 4.78%, 5.84%, and 10.05% at N64, N128, and N256. This supports a reproducible negative result; resolution-robust speedup remains unproved.

## Relevant predecessors and substantive differences

| Primary source | Established mechanism | Difference and implication here |
|---|---|---|
| [Melchers, Dolean, Abdelmalik, 2605.19867v3](https://arxiv.org/html/2605.19867v3) | Compatible integration against the output basis, source-term linearity, and coarse-space training coverage improve neural two-level corrections. Fixed coarse dimension limits difficult high-frequency problems. | Diffusion/advection-diffusion/Helmholtz, largely iteration evidence with preliminary runtime. Matched transfer plus linear residual action is established architectural guidance, not a new RTE claim or kinetic robustness theorem. |
| [Dimola, Coclite, Zunino, 2507.15452](https://arxiv.org/html/2507.15452v1) | Static pretraining followed by a dynamic principal-angle objective through differentiable FGMRES, Arnoldi, and Givens rotations. | Their 3D–1D mixed-dimensional systems differ physically. Training against actual Krylov progress is already established; the present continuation loss is a transport-specific application. |
| [Peng, FGMRES–ROMSAD, 2410.08735v3](https://arxiv.org/html/2410.08735v3), [Tang–Peng, trajectory-aware ROM, 2509.05001](https://arxiv.org/html/2509.05001v1) | Reduced models approximate ideal kinetic corrections and combine them with classical synthetic acceleration; deployed-preconditioner trajectories matter. | Upwind DG, isotropic scattering, sphere ordinates differ from periodic S1. They are the closest learned-correction predecessors and motivate a simple snapshot-ROM ablation and training behind the actual tuned classical action. |
| [Song, Liu, Zhang, filtered MgNet, 2604.23265](https://arxiv.org/html/2604.23265v1) | Coefficient-conditioned learned multilevel operations with filtered transport residual training. | Reported 32x32/four-angle/ten-sample Block-Jacobi iteration comparisons and explicit transport/interface failures provide limited evidence. This establishes neural RTE multilevel prior art, not superiority to tuned TSA/HSS. |
| [Fan, An, Ying, 1802.03061](https://arxiv.org/abs/1802.03061) | Angular Fourier reduction, homogeneous FFT solution, and heterogeneous recursive skeletonization. | Particularly close to this project's exact homogeneous reference and proposed compressed global multiple-scattering action. Reusable classical setup must be credited in repeated-source comparisons. |
| [Lathouwers–Perkó angular multigrid, 2010.04559](https://arxiv.org/abs/2010.04559), [Dargaville et al. AIRG, 2301.05521v4](https://arxiv.org/abs/2301.05521) | Angular coarsening addresses forward-peaked scatter; nonsymmetric streaming/removal multigrid combined with DSA approaches scalable refinement. | Their DG/unstructured operators are not direct periodic-FFT baselines. Direction-aware multilevel transport splitting is established; changing g or angular resolution would require these stronger comparisons. |

These sources establish several ingredients separately. A useful distinction would require a transport-specific scalable correction mechanism, a clearly delimited guarantee, or measured cost advantage. This audit makes no priority or absence-of-prior-art claim.

## Strong baseline and mathematical distinctions

For the present discretization, the immediate comparator is the frozen validation-selected classical portfolio already measured: homogeneous-reference tail at N64/N128 and TSA16 with zero initialization at N256, alongside tuned HSS/TSA depths. Calling an untuned scalar diffusion solve the strongest classical baseline would be misleading. Small-grid sparse LU provides an independent accuracy reference; factorization setup and memory prevent assuming it remains competitive at larger N.

Classical DSA can be discretization-consistent and heterogeneous/void-compatible: [Haut et al.](https://arxiv.org/abs/1810.11082), [Southworth et al.](https://arxiv.org/abs/2001.09196). Anisotropic angular subspace correction also has convergence theory in its own setting: [Dölz et al.](https://arxiv.org/abs/2102.09038). These results explain why heterogeneity or anisotropy alone is not an unexplored classical deficiency. Their discretization-specific proofs do not automatically apply to this Fourier/S1 operator.

The [Bai–Golub–Ng HSS theorem](https://epubs.siam.org/doi/10.1137/S0895479801395458) applies if streaming is skew-Hermitian and collision is Hermitian positive definite. Here this follows when P is symmetric contractive and a_min = min(sigma_t - sigma_s) > 0. For a fixed positive shift alpha, the classical contraction is bounded by max_lambda |(alpha-lambda)/(alpha+lambda)| in the norm ||(alpha I+S)x||. Varying learned shifts/gains does not inherit that estimate merely from positivity; the source already acknowledges this limitation.

Direct operator checks also give ||u-u_hat|| <= ||b-Au_hat||/a_min. Therefore report trusted-reference full-flux and angular-mean errors alongside true residual: a relative tolerance alone can yield a loose resolution-dependent relative-error bound. In S1 the isotropic second angular moment is I/2; a homogeneous diffusion coefficient uses the actual discrete first phase eigenvalue lambda_1 in D = 1/[2(sigma_t-lambda_1 sigma_s)], rather than importing a three-dimensional 1/3 factor or assuming lambda_1 = g.

## Selected next study and limited grazing insight

The chosen first step is the existing cheap architecture with equal-budget continuation training: N64-only versus paired N64/N128, matched update/sample budgets, both validation resolutions used for selection, then the predeclared N128/N256 runtime gate. Keep unused test seed 9626 untouched. This isolates resolution data from architecture and retains a negative outcome if the gate fails. It does not itself establish architectural novelty.

If that study fails, diagnose saved heterogeneous tails by physical Fourier bands and ordinate streaming ratio 2 pi |Omega_l dot k|/mean(sigma_t), then consider a material-conditioned complex-linear directional multiscale remainder. Compare it against a simple non-neural directional enrichment or snapshot ROM.

A direct homogeneous calculation motivates that diagnostic: for Omega_l = (1,0), k = (0,m), and u = e_l exp(2 pi i m y), streaming vanishes although |k| is arbitrarily large. Taking b = C u, any inverse action that annihilates inputs outside a fixed radial Fourier band has error ratio 1/||C e_l|| on these modes. This is an elementary full-inverse obstruction, not a novel theorem or evidence that the post-classical remainder fails. The homogeneous reference already solves this example exactly. Only heterogeneous saved-tail evidence can connect grazing channels to the observed refinement loss.

## Cost and reproducibility acceptance

Charge data/teacher generation, training, tuning, model loading, material-dependent setup, transfers, inference, every operator/Krylov operation, and failed solves. Use the same hardware, precision, true residual tolerance, error checks, and material/source splits. Report paired runtime uncertainty, memory, and failure counts rather than only iteration savings.

For material instances m with Q_m repeated sources, compare

    T_learned = T_offline + sum_m [S_learned,m + Q_m t_learned,m]
    T_classical = sum_m [S_classical,m + Q_m t_classical,m].

Offline includes the actual teacher/data/training costs; setup includes reusable material-specific work on both sides. If online learned time is slower and its setup saving cannot compensate, there is no favorable amortization count. When online savings are positive and setup is comparable, the break-even number of solves is approximately offline cost divided by the per-solve saving. Present frozen-model online results separately from this total-cost question.

Remaining gaps include angular refinement, stronger forward scatter, reduced absorption, new coefficient geometries, independent solution-error checks, and repeated-source amortization. The fixed L/g/epsilon cohort establishes none of these broader robustness claims; periodic geometry supplies no inflow boundary-layer evidence.
