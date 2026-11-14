# Han et al. literature audit and actual-discretization closure test

## Decision

Relevant research reference, but not a directly equivalent new strong baseline for the current RTE. The cheapest decisive next step was one exact-discretization closure witness, not a solver port or tolerance sweep. That witness has now been executed and falsifies the required cheap scalar closure. Preserve AngularBand as the normal-Green decision baseline and the already qualified full-RTE classical controls. No performance claim from this paper is transplanted into our timings.

## Verified source boundary

The verified record is Han, Einkemmer, Guo and Peng, arXiv:2610.07653v1 (6 October 2026), a 28-page preprint. Its setting is isotropic scattering/source, inflow boundaries and upwind DG with spherical quadrature. Compression targets transport operations, not the scalar Krylov basis. Richardson accuracy tightens; inexact-FGMRES accuracy relaxes. Theorems 4.1 and 5.1 are conditional, and the implemented tolerance rules are explicitly heuristic. Numerical tests use scalar-flux iterate differences, rather than original angular-equation residuals, as stopping criteria. They compare on identical DSA preconditioners, use ten random seeds, and count FGMRES RHS sweeps. Reported implementations use MATLAB on an M4 MacBook, with hundreds of angles. I found no separate setup/peak-memory ledger in the text. These are not matched timings for our periodic 12-angle solver. [Official full PDF](https://arxiv.org/pdf/2610.07653v1).

I read the main equations, Algorithms 3.1/5.1, numerical tables, and Appendices A–E from the official PDF and cross-checked the HTML. The downloaded PDF SHA256 was `e1246d6e441b8897b2a90c6845a6ba3c3246b93bb51533b49313b7fd08d37d09`. The author lists the work as a preprint; no directly linked RTE implementation was found on the checked publication/code pages. This is not a claim that no code exists elsewhere. [Author publication page](https://zhichaopengmath.github.io/publications/), [author code page](https://zhichaopengmath.github.io/code/).

## Independent algebraic audit of transfer to our operator

Write the actual current background operator as A=L-SK, where L=diag(L_j) contains the original stored-frequency periodic Fourier transport and total collision; S is the scattering profile; K is the stored weighted 12-angle phase. Let R be the actual angular average and P=1 R/(R1).

If the scalar transport update depends only on R psi for every angular field, its map H=R L^-1 S K must annihilate ker(R), equivalently H(I-P)=0. A single nonzero result on one zero-mean angular field disproves this identity. This is a necessary-condition test, not a speed test and not a claim about all scalar Schur representations.

The stronger scattering-lift identity K=U R is impossible when rank(K)>1. General exact K=U V* yields the moment system (I-V*L^-1 S U)phi=V*L^-1 q. With rank12 this carries twelve moment fields, so it does not reduce the angle dimension. One could instead eliminate the other eleven fields into a scalar Schur complement, but the complementary coupled inverse must then be represented and paid. It is not the independent-angle sweep scalarization being considered here. Full phase rank does NOT prove that every solution matrix has rank12.

A periodic Fourier directional operator is globally coupled, not the acyclic upwind/inflow matrix to which a single ordered sweep applies. Replacing it by upwind DG would be a changed discretization, not an accepted baseline for the original equation. Current angle count12 also gives a different compression/overhead budget; no large-angle speed ratio is inherited.

## Actual one-witness result

Frozen inputs: N128, tangent k=0, actual g=.65, the same material background and stored weights/directions/frequencies as the completed layout experiment. We chose h=(e_0-e_1)/sqrt(2) and f_j=exp(2 pi i j/128) before running; psi=h tensor f. No vector, frequency, material, rank or tolerance was selected after seeing a result. We assembled the twelve original directional 128x128 matrices, applied one batched LU factorization/solve, and checked them against the original FFT actions.

- Input scalar-average norm: **0.0**. The rank-one null-control source was also exactly zero.
- Actual phase rank: **12**; smallest singular value **0.149984632846**.
- ||K(I-P)||2: **0.655024955183**.
- Output-average norm ||R L^-1 S K psi||2: **0.0208250912877**.
- Normalized output ||RY||/(||R|| ||Y||): **0.0270545623737**.
- Largest directional original-FFT relative residual: **6.72564858934e-16**.
- Dense/FFT action bridge: **3.22693050367e-16**.

The zero field has the same scalar input as this witness and maps to zero, whereas the witness maps to a nonzero scalar output. The difference is far above the checked numerical error. Hence the cheap scalar update is not determined by the scalar input for this actual discrete background. This disproves direct use of that closure; it does not disprove an approximate preconditioner subsequently checked against original A.

## Two mathematical safeguards from the audit

1. SPD diffusion does not automatically give an Euclidean-SPD DSA map. Independently, for D=D*>0 and S=S*>0, B=I+D^-1 S has B*=I+S D^-1, so symmetry requires an extra condition. For example D=[[2,1],[1,2]], S=diag(1,2) gives B=[[5/3,-2/3],[-1/3,7/3]], which is not symmetric. If S>0, SB=S+S D^-1 S gives a possible S-weighted formulation, but the norms/operator and the contraction estimate must be transformed consistently. S-weighting degenerates if scattering vanishes. Even after choosing a valid norm, contraction needs an independent proof. A bound rho+aK eta<=1 yields only nonexpansion at equality; strict contraction needs a strict margin. These are algebraic cautions, not a claim that the entire paper is invalid.

2. For a computed Arnoldi relation A Z+E=Q H and an imperfect initial residual, direct substitution gives r_true-r_comp=(b-b_tilde)+e_0+E y. The triangle bound is therefore ||b-b_tilde||+||e_0||+sum_k ||e_k|| |y_k|. A tiny recursive residual cannot erase RHS bias, and a sampled-angle indicator is not itself a certified whole-action norm. Orthonormality, floating errors, restarting and all additional truncations have to be accounted for.

The useful design distinction is that an inexact preconditioner is not an inexact original matvec. If z_k is stored exactly as produced and original A z_k is then evaluated, varying the internal preconditioner does not add the above E term in exact arithmetic. The current FGMRES architecture already supports this distinction. It does not justify changing the protected original residual thresholds or assuming that a cheaper internal operation lowers complete cost. The already rejected fixed coarse-fine/second-injection schedules are not reopened by the existence of a related paper.

## Original-equation transfer ledger for any future moment approximation

This is our own acceptance identity. Let K_r=U V*, z be a reconstructed angular field, phi its retained moment variable, delta=Lz-SUphi-q the transport residual, and s=phi-V*z the moment mismatch. Then

    A z-q = delta + S U s + S (K_r-K) z.

All three terms are charged. For an independently valid accretivity constant alpha of the selected original operator, ||z-psi_exact|| <= ||A z-q||/alpha. A scalar iterate-difference criterion alone controls none of these three terms uniformly. This ledger states what a faithful port would have to certify before comparing speed. No low-rank phase replacement or adaptive-tolerance solver was built in this audit.

## Complete cost and reproducibility

The one numerical process took **3.62 s external wall**, **3.29 s external CPU**, with external peak RSS **448,098,304 bytes**. It includes imports, material construction, identities, small algebra, directional LU/solve and original FFT checks. CPU0 was idle on the admission snapshot; the GPU was busy and was left untouched. There were no rejected numerical attempts and no full RTE solve.

| Phase | Wall seconds |
|---|---:|
| physical_and_identity_wall_seconds | 0.012117923 |
| witness_and_small_algebra_seconds | 0.003117742 |
| directional_matrix_assembly_seconds | 0.010503894 |
| twelve_directional_LU_and_solve_wall_seconds | 0.005434849 |
| closure_and_original_checks_seconds | 0.003022856 |

Logical diagnostic arrays: {"D": 262144, "directional_matrices": 3145728, "LU": 3145728, "pivots": 6144, "witness": 24576, "source": 24576, "response": 24576}. These are not a new solver factor-memory measurement. Full process RSS contains runtime/operator state. Public-paper download and reading time are not solver execution time.

Source, plan, original production/E source and actual coefficient fingerprints were checked. Original physics and tolerances were unchanged. The existing-source reference bridge and new raw result SHA256 are retained in ANALYSIS.json. Run run_closure.py only in a fresh result directory with the environment in CLOSURE_PLAN.md; it intentionally refuses to overwrite evidence.

## What to do next

Do not port this scalar/DG solver merely to install a literature-labelled baseline. Do not relax the existing inner-G contract or restart rank/packing searches. The useful outcome now is the exact closure obstruction and the full-angular residual ledger. A future candidate must provide a concretely cheaper representation of the retained angular response or a genuinely available reuse workload, and then beat both the faster packed nested control and AngularBand under the unchanged original-equation checks. No further experiment is running or promised.
