# Operator-first normal Green representation: proof and fixed falsifier

This is a new operator-factor construction, not a rank rescue of the failed inverse H matrix. No production edits. No original RHS, teachers, training, heldout data, outer PDE experiments, or parameter sweep. N128, tangent modes 0 and 8 only. The existing min/gather layer, stored directions, full rank12 phase, stored Fourier frequencies and nonzero complex Nyquist remain unchanged.

## Concept portfolio and selection

1. **Fourier low/high Schur split plus a simple transport tail.** Deferred. With normal collisions about 6.55..666.55 and absorption about .05, the obvious streaming-dominance condition does not supply an affordable low-frequency core on N128. Stored grazing channels are not exactly zero. Failure of a sufficient bound is not a theorem against all tail preconditioners.
2. **Remove only tiny grazing normal streaming, then eliminate its2 angular channels locally.** There is a useful bound: the removed operator is skew-Hermitian, so coercivity is unchanged; its norm is at most max_grazing|omega_n| max|xi|. This can be a certified preconditioner simplification, not an exact identity for original A0. It does not yet construct the remaining10-angle normal inverse. Not implemented here.
3. **Selected: compress known normal derivative, then factor the operator recursively.** Unlike compressing an already computed inverse, all compressed inputs are available before any inverse. Full local angular collision is exact. The tolerance is dictated once by original A0-weighted error and accretivity, not by inverse energy or a fitted rank.

## Actual discrete identity

In normal-space-major order, for each fixed tangent k,

G = D_N tensor C_n + blockdiag_j[V_j(k)],
V_j(k)=t_j I - s_j K + i*k_t*C_t.

C_n and C_t contain the actual rounded real directions. D_N=F*diag(i*xi_stored)F. K is the actual full phase times quadrature weights. The Hermitian part of G is local. Define

alpha = min_j [t_j - s_j lambda_max((K+K*)/2)] >0.

Then Re<x,Gx> >=alpha||x||². The exact same lower bound applies if D_N is replaced by any skew-Hermitian Dtilde. No continuum or ideal-direction identity is used.

For ideal integer frequencies, D off-diagonal is a periodic Cauchy kernel: D_jl=2*pi*i*(-1)^(j-l)*z_l/(z_j-z_l), z_j=exp(2*pi*i*j/N), and D_jj=-i*pi. The original stored-frequency discrepancy must not be dropped. Our prototype constructs the actual scalar D by FFT, so includes this discrepancy and the complex Nyquist directly. The Cauchy identity motivates structured separation; it is not used to change the operator.

## Weighted theorem and contract

Let Dbar=(D_N-D_N*)/2, then form Dtilde by a binary normal hierarchy. Each upper sibling block is compressed once by SVD; the lower is its negative adjoint. All local diagonal blocks and all V_j are retained. For depth h=log2(N/8), give each compressed block Frobenius error budget

eta = alpha*1e-3 / (2*max|omega_n|*h).

A full-rank block remains an exact dense leaf, a structural fallback fixed before results. At any depth, paired errors are block diagonal, with operator norm bounded by the largest sibling error Frobenius norm. Consequently

||D_N-Dtilde|| <= ||D_N-Dbar|| + sum_depth max_block ||E_block||_F.

For B=Gtilde^{-1} in exact arithmetic,

||I-G B|| <= max|omega_n| ||D_N-Dtilde|| / alpha = q.

Thus x0=Bb and x_{j+1}=x_j+B(b-Gx_j) have ||b-Gx_j||/||b|| <=q^(j+1). q<=1e-3 is sufficient for max3 repairs with a1e-10 target, with theoretical headroom. B alone is an approximate inverse/preconditioner for G, not an exact original inverse. Original residual checks are mandatory; no threshold change.

This is an analytical finite-dimensional theorem, conditional on exact B and the stated norm bound. Floating norm/SVD evaluations are not interval certificates. The implementation's nested solves can introduce defects; actual original-G tests may falsify stability. We separately measure Gtilde action residual and original-G residual; zero and1e-200 nonzero columns are included. Probe success is not a uniform machine bound.

## Concrete recursive factor, no dense Green inverse

At each node Gtilde=[[A,U_L V_R*],[U_R V_L*,B]]. The scalar derivative low-rank factors are tensorized with the exact12-angle diagonal C_n; no angular compression. Child factors apply A^-1 and B^-1. Define W_L=A^-1 U_L, W_R=B^-1 U_R, M_L=V_L*W_L, M_R=V_R*W_R. Factor S=I-M_L M_R, dimension12r. For each RHS first compute yL,yR and cL=V_L*yL,cR=V_R*yR, then a=S^-1(cL-M_L*cR), b=cR-M_R*a; output xL=yL-W_L*b,xR=yR-W_R*a. Leaves use pivoted LU. This is a constructive algebraic identity provided its child and small factors are nonsingular. No (12N)x(12N) inverse is formed or sampled for construction.

The scalar SVD constructor is deliberately simple and paid: O(N²) scalar storage and O(N³) worst-case work, not a claimed near-linear builder. If rank r_l stays small, recursive representation stores roughly sum_nodes[(12n_node)(12r_node)+3(12r_node)²] numbers plus leaf factors; application has corresponding quadratic contractions. The r*12 inflation and small dense Schur factorizations may destroy a saving. We will report exact owned bytes, ranks, construction and application times; no asymptotic claim before evidence. Future Cauchy interpolation/ACA would be a different constructor and must be charged and verified.

## Fixed actual check and controls

- N128,k0 andk8, two source-independent objects; original A0 full embedding bridge.
- One shared scalar derivative compression; per-mode factors include its amortized cost clearly, not free reuse.
- Controls: current AngularBandMode and original dense LU. Dense LU is an accuracy/control solve, not construction data for candidate; no full inverse.
- Three predetermined normalized complex random columns, exact zero,1e-200 scaled column. Same original-G1e-10,max3repairs contract.
- Algebra unit check on one small deterministic skew/local-block example; no selection of a best seed or order.
- Original and approximate action bridges, actual floating factor defect, source hashes, factor bytes, RSS and all setup/qualification costs retained.
- Stop after two objects. A passed representation is a PROOF/CONCEPT MILESTONE only. If more costly than strong control, it is not a solver promotion. No N256,17-mode expansion or outer PDE job automatically follows.

## Prior art / scope

This is an application of classical hierarchical operator factorization and recursive Woodbury, not a claim to invent HODLR methods. Aminfar, Ambikasaran and Darve describe HODLR direct factorization and use as a preconditioner (arXiv:1403.5337). Ho and Greengard construct inverse actions by recursively compressing an original operator, with substantial preprocessing (arXiv:1110.3105). Their PDE/integral benchmarks do not establish rank, stability or cost for this RTE matrix. The task-specific proposed contribution is the discrete skew-preserving error-to-original-A0 certificate and its cost test on actual stored coefficients.

### Floating implementation qualification

For a computed inverse action Bhat, let E=I-Gtilde*Bhat. Then I-G*Bhat=(I-G*Gtilde^{-1})+(G*Gtilde^{-1})E, hence ||I-G*Bhat||<=q+(1+q)||E||. For a nonlinear floating action the same identity is interpreted on each input vector, not as a measured uniform operator certificate. The fixed probes and each repaired residual are evaluated separately. The hierarchical factor uses a12r-dimensional elimination of the paired24r system; the stored W_L,W_R,M_L,M_R and12r LU are all counted. Exact accretivity of child systems does not guarantee well-conditioned small Schur arithmetic.
