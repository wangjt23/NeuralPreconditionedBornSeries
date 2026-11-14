# Frozen-layout execution test

## Question and protected contract
Can replacing generic normal-index einsum by two explicit two-dimensional GEMMs reduce measured setup/application cost of the qualified nested Green representation, without changing its basis or error contract? This is a layout/kernel test, not a new rank or mathematical approximation. Do not infer an algorithmic complexity reduction.

Fixed N128, k=0, 12 actual stored directions, g=.65, actual original G, nested basis construction source byte-identical to the previous study; eta=1.4091620147955772e-6, ranks 8/13/16/17, leaf8, q_design1e-3. Original-G tolerance1e-10/max3 repairs. The derivative approximation and every basis/coupling tensor get fingerprints. Each measured worker rebuilds and pays its own basis. AngularBand does not pay the candidate basis. No rank/eta/mode/source sweep, dense inverse, GPU, or production changes.

## Exactly one selected execution layout
For space-major G_(i,a),(j,b) and scalar Q, target H_(p,a),(q,b)=sum_ij conj(Q_i,p) G_i,a,j,b Q_j,q.
1. left = (Q.conj().T @ G.reshape(m,l*m*l)).reshape(m,l,m,l).
2. pack left.permute(0,1,3,2).contiguous().reshape(m*l*l,m); right=pack@Q.
3. restore right.reshape(m,l,l,m).permute(0,1,3,2).contiguous().reshape(ml,ml).
All packing is charged. No expanded Q tensor I_l is stored. Up/down use Q* or Q @ b.reshape(m,l*nrhs), with any implicit reshape copy also inside apply. All local elimination/LU/couplings/tree topology stay unchanged.

This layout exposes large 2D matrix products and avoids generic einsum dispatch, with equal arithmetic count and additional explicit packing rather than free conversion. Shape values m=16,26,32 and l=12 come from the previous fixed tree. No alternative layout will be tried to select the fastest result.

## Why a fresh attribution control is necessary
The old 64% rotation fraction is a wall-time observation, not a demonstrated compute fraction: equal-size nodes differed by more than an order of magnitude. New runs record CPU and wall time as well as resource snapshots and all individual results. We do not subtract waiting or cherry-pick timings.

## Preflight, then bounded measured workers
One paid bridge process: existing small algebra unit, fixed actual tree, compare all14 local transforms of the packed factor against the old einsum, compare complete factor actions and checked original-G solutions; check zero/tiny inputs. No timing conclusion from the dual-factor bridge. If any bridge exceeds1e-12 transform relative error,1e-10 raw action difference, or original checks, stop.

Then6 sequential fresh workers, fixed order original,packed,angular,angular,packed,original. One physical/basis/factor setup each. Every worker uses CPU0, single BLAS/OpenMP/torch thread, CUDA disabled. No warm-up is excluded; first checked call is kept. Each runs three predetermined repetitions of single-column and the old five-column batch (zero,3 random,1e-200 scaled column), alternating single then batch. Single-column matches the actual inner interface workload better; neither is a full17-mode or physical four-source solve. Compare accepted reference from the bridge, charging its load and QA as research costs. Record raw application diagnosis separately after checked calls, no free diagnostic warmup.

Acceptance: full original residual<=1e-10, same max3 repairs, reference field relative<=1e-9, exact zero and nonzero tiny column, finite; packed raw action relative to original<=1e-10. Basis, phase/material/directions, probe hashes exact; floating factor outputs need not be bit-identical.

Record physical setup, scalar basis/setup, proof checks, factor and complete profiled setup, all original checks/repairs/application, reference I/O, diagnostics, full process wall+CPU including import/QA/wait, persistent factor bytes, before/after RSS and phase/full process highwater. Same process includes no competing factor. Logical factor bytes are not RSS. Source files and production/E hashes checked.

Two reversed rounds are a screen, not a robust benchmark. Report each result and stop rather than repeating until positive. A ranking reversal or gain comparable to variation means insufficient speed evidence. Even a positive result is only a fixed single-mode layout result, not full RTE promotion. No more experiments are implicitly scheduled.
