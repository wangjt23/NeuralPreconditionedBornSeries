# Nested interface recompression: fixed error ledger and feasibility gate

## Protected contract and change

One actual N128, k=0 background, full12 angles and actual g=.65 phase. Original operator, stored directions/frequencies, source availability and original-G residual1e-10/max3 repairs unchanged. No rank, tolerance, leaf, mode or source sweep. No full dense Green inverse. Strong decision baseline remains current AngularBandMode. The previous cached/no-W HODLR paths remain evidence, not the decision baseline.

We change the representation to genuinely nested normal interfaces and local redundant-variable elimination. Previously inherited rank32 at32-cell nodes was an exact-reuse obstruction, not proof against approximate recompression. Here every new inherited/interface error is explicitly charged. The existing four-level layout and leaf8 are fixed, but all interface ranks are recomputed from the new error allocation, not copied from old rank33/14/8/6.

## The actual stored-frequency discrepancy is paid

Let D be the original stored-frequency Fourier derivative, D0 the mathematical integer-frequency derivative on the same grid, v=max|omega_n|. For unitary mathematical Fourier F,

||D-D0||2 = max_n |xi_stored[n]-2*pi*n| = delta_freq.

Use D0 only to construct the approximate inverse. The original D remains in every physical action/residual/repair. The old scalar cause check established delta_freq~2.389e-5; this run recomputes it. Preserve q_design=1e-3, not the looser largest permitted value. The remaining derivative budget is alpha*q_design/v-delta_freq; abort if nonpositive. Ordinary floating construction and factor errors are not interval-certified and are checked separately.

## Nested row construction and proof

At every non-root node I with children l,r, let QI=diag(U_l,U_r); leaves use the identity. Compute the SVD of QI* D0[I,Ic], retain Z_I with discarded Frobenius norm<=eta, and set U_I=QI Z_I. Retain the complete orthonormal complement for local elimination. If all columns are required, use the identity transfer rather than an unnecessary rotation. The root has no exterior and needs no basis.

Define e_I=||(I-U_I U_I*)D0[I,Ic]||F. Orthogonal decomposition gives

e_I² <= e_l²+e_r²+eta²,

with leaf e_I<=eta. Parent exterior is a subset of each child's exterior. Reconstruct every sibling block from the ORIGINAL D0,

Dhat_IJ=U_I(U_I*D0_IJ U_J)U_J*, Dhat_JI=-Dhat_IJ*.

Keep skew-Hermitian diagonal leaves. Then sibling error<=e_I+e_J. Same-level paired errors are disjoint and their operator norm is the maximum paired block norm. For N128/leaf8,

||D0-Dhat||2 <= C*eta, C=2*(sqrt15+sqrt7+sqrt3+1).

Thus set eta=(alpha*q_design/v-delta_freq)/C once. We also evaluate the assembled scalar approximation's actual hierarchy and row/column norm bounds, adding the tiny skew-projection discrepancy of the constructed D0. This is a conditional mathematical error theorem with ordinary-FP evaluated constants, not an interval machine proof. Failure stops the candidate; no tolerance rescue.

Ghat=Dhat tensor Omega + original local collision/tangent blocks retains the same accretivity alpha. Therefore ||I-G Ghat^-1||<=v*(delta_freq+delta_nested)/alpha. The original residual refinement theorem remains applicable. No free exactness or angular-rank reduction is assumed.

## Local nested elimination

For a node whose child interfaces total m scalar coordinates and output interface rank r, transform by [Z_perp,Z] tensor I12 and partition local coordinates into e=12(m-r) redundant and s=12r retained. The redundant variables have no exterior coupling in the constructed Ghat. Factor A_ee, store its LU, A_se and W=A_ee^-1 A_es, and pass S=A_ss-A_se W to the parent. This W is a LOCAL redundant/interface factor, not the previous full physical-subtree expanded response W. Both coupling sides are paid; the local system is not assumed Hermitian.

Apply one upward elimination and one downward recovery. At the root factor the entire interface Schur of dimension12*(r_left+r_right). There is no four-child-solve recomputation. For any accretive block matrix, x=(-A_ee^-1 A_es y,y) gives Re(y*S y)=Re(x*A x)>=alpha||y||². Hence unitary transforms, principal blocks and successive Schur complements retain alpha in exact arithmetic. This helps the structure but does not replace floating original-G checks.

## First, one scalar preflight only

Recompute actual ranks, all inherited residuals, orthogonality, sibling errors, Dhat skew and the complete original-weighted model bound. Account every local factor and root dense interface:

complex tensor entries = sum_nonroot[e²+2es+m²(if nontrivial Q)] + root_dimension²,

plus pivots. Scalar Q has m² entries, not (12m)². Identity transfers are not stored. Include local Schur/rotation/SVD workspace in later measured process peaks; persistent counts are not peak memory.

Also compute transparent arithmetic proxies, not wall-time predictions: nested setup includes both orthogonal transforms, local LU, W solves, Schur updates and root LU; application includes local LU solves, both coupling products, Q transforms and root solve. Derivative SVD/assembly and QA costs remain separate, fully paid. The angular baseline proxy is derived from the actual12-block chain/corner implementation, including construction of its corner response, not just its LU calls.

Proceed to at most one k0 nested-factor check with a fresh angular baseline only if (i) the original-weighted bound passes, (ii) predicted persistent factor bytes are below16,272,584, and (iii) application arithmetic proxy is below the existing angular action. Setup proxy must be disclosed even if it is worse. This is a bounded reason to test, not proof of speed. Otherwise stop with the precise failed condition. No N256, other k, changed eta, or alternate basis is tried.

If proceeded: first a deterministic small algebra unit; two sequential fresh CPU0/one-thread/CUDA-disabled workers, every method paying its own setup. Same3 predetermined random complex probes plus zero and1e-200 nonzero column, three checked applications including first call, original residual1e-10/max3 repairs. Count actual upward/downward node visits separately from timings. Report all setup, checks, references, memory, failures and total numerical process wall. Keep production/E hashes unchanged. Do not promote from this single-mode diagnostic to full RTE.
