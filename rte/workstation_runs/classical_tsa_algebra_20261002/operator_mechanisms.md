# Step back: TSA16 operator mechanisms and one defensible diagnostic

Independent source/proof assessment, 2026-10-02. No GPU experiment, training, new physical data, original-source modification, or new CPU numerical experiment was performed for this note.

**Prioritize an algebraically equivalent classical TSA implementation before constructing another learned correction or coarse inverse.** The current strongest N256 method repeats a linear action with redundant streaming FFTs. A conservative identity removes sixteen streaming applications from each TSA16 call, retains its fifteen outer physical defect updates, and introduces no model/setup amortization assumption. Operation savings are proved below; field equivalence in floating point and actual latency still require the bounded diagnostic.

## What is verified about the bottleneck

`train_source_predictor.py:121–127` constructs `ClassicalTransportStack(low, 'tsa', 16, 0.)`. `transport_baselines.py:25–49` builds a same-grid/same-dtype reference, uses zero reference shift/floor for this setting, and eagerly prepares the cached homogeneous angular inverse. `models.py:245–256` stores that inverse after setup; `models.py:315–333` applies it by spatial FFT, per-frequency angular matrix-vector multiplication, and inverse FFT. **The warm solver is not rebuilding or refactoring the inverse on every application.** Setup is separately recorded.

The full uninstrumented N256 benchmark reports frozen TSA16-zero at 143.695 ms, with a mean 4.5 classical calls over the sixteen development media. The supplementary two-eligible-case TSA16-plain profiles report mean preconditioning 168.847 ms out of 200.846 ms profiled wall, approximately 84%; six mean classical calls are used in that different, small cohort. This establishes classical preconditioning as the dominant recorded stage there. Synchronization changes execution, and the cases differ from the full cohort. It does **not** establish which fraction of preconditioning belongs to FFTs, the angular matrix-vector product, Python/kernel launch overhead, or data-layout operations: those internal sub-stages were not separately timed.

One TSA16 call currently contains 16 `G` applications, 16 homogeneous `R` applications, and **31 physical `A_op` applications**: sixteen inside the TSA single stage and fifteen between accumulated updates. Each contains one spatial FFT/inverse-FFT pair, acting on all twelve ordinates. Thus the current call contains 63 such pairs, or 126 transform calls, excluding setup and outer FGMRES actions/checks. This count follows directly from `transport_baselines.py:52–70` and `models.py:197–221,315–333`; it is independent of a timing model. `FrozenTSA` (`global_source_predictor.py:93–100`) repeats the same unfused inner and outer physical actions.

## Conservative exact identity

For a fixed material and common stored spatial/angular geometry, write

\[
A=S+tI-sK,\qquad L=S+t_0I,\qquad G=L^{-1},
\]

where `K_ij = phase_matrix_ij * weights_j`, `t0 = base.reference.sigma0`, and `t,s` are the physical coefficients. Let `R` be the cached homogeneous transport/scattering inverse used by the existing stage. Its exactness is **not required for the following identity**; only `LG=I` and linear application of `R` are needed. The original single stage is

\[
g=Gd,\qquad Fd=g+R(d-Ag).
\]

Since `d=L g`,

\[
d-Ag=(L-A)g=(t_0-t)g+sKg.
\]

Therefore the same stage can compute the defect entirely locally:

```python
g = base.reference.G(d)
local_defect = (base.reference.sigma0 - physical.sigma_t) * g
local_defect = local_defect + physical.scattering_op(g)
u = g + base.reference.reference_transport_solve(local_defect)
```

Retain the original outer recurrence `d = rhs - physical.A_op(z)` before stages 2–16, the same depth/relaxation, coefficient/cache lifecycle, precision, and final independent physical check. The new count is 16×2 + 16×2 + 15×2 = **94 spatial transforms**, down from 126. This saves 32 transforms, or **25.4% of this transform count**, not 25.4% of solve time. Scattering remains local and still costs work; all sixteen angular reference applications remain. The wall-time ceiling is the time spent in the sixteen removed streaming actions minus any added local/implementation overhead. Existing coarse profile categories cannot give that ceiling numerically.

The source already provides `V_op` (`models.py:210–213`), but examined TSA, model TSA/DSA, and `FrozenTSA` paths do not use this identity. **Do not blindly use `physical.V_op(g)`: its own `sigma0` may differ from `base.reference.sigma0`, notably because the classical reference sets `eta_floor=0` whereas the physical operator may retain its own floor.** Use the reference's actual removal coefficient with the physical collision/scattering tensors. Verify equality of stored streaming symbols, not merely shape/angle count. The current construction regenerates matching geometry for the normal FP32 base; a promoted or externally altered operator requires an explicit check.

Floating point is the remaining issue. In FP32, `LGd-d` is small rather than zero, so the changed defect differs from the original by that inverse/FFT roundoff. Strong scattering and sixteen stages may amplify it. This is not bitwise equivalence, and theoretical phase symmetry does not remove the numerical issue. Preserve the original FP32 preconditioner/FP64 physical operator convention; `mixed_precision_transport.py:9–16` deliberately preserves the original rounded geometry rather than reconstructing ideal FP64 angles.

## Why not start with a learned linear surrogate or another coarse space

For fixed material, grid, depth and cache state, TSA16 is already complex-linear in exact arithmetic. If `F` is its single-stage map, its depth action is

\[
B_{16}=\sum_{j=0}^{15}(I-FA)^j F
       =F\sum_{j=0}^{15}(I-AF)^j.
\]

A material-conditioned encoder could in principle cache a cheaper linear approximation to this map, or to a residual-relevant component of the heterogeneous inverse. That is a different mechanism from the tested nonlinear one-shot tail network. It is useful only if setup/storage and action cost are sufficiently small, and if it preserves useful `A`-images across sources/trajectories. A dense full map is already prohibitive: at N256/L12 it has 786,432 squared complex entries, about **4.95 TB at complex64**. Low-rank or hierarchical structure needs evidence in the physical residual norm; raw field/Frobenius compression alone is insufficient.

The September19 saved report already supplies adverse evidence: naive 1% Schur inverse far-block truncation stalls on all six moderate-regime sources and three of six stiff sources; successful tighter residual-weighted compression still estimates 96.3%/76.3% of dense storage. It is representation evidence, not a fast deployable surrogate. The existing exact N8 radial Schur coarse correction changes N128 stiff mean iterations only 15.5→15.0 and slows the solve. Directional dictionaries improve some iteration counts but still lose to full-angle controls. These are earlier, small development surveys with different cohorts, not new estimates for the current sixteen media.

Coarse-grid inconsistency can matter, but it cannot be assumed to explain those negative results. `multilevel_neural_transport.py:16–38` uses strict Fourier transfers, while lines66–69 cell-average the coefficients; the rediscretized coarse action generally differs from `RAP`. In contrast, `galerkin_transport.py:25–53` preserves stored geometry and computes dealiased retained coefficient products with a direct `RAP` comparator. The previous exact Schur/coarse probes agree with direct actions to approximately `8.4e-16`, yet do not produce speed gains. Radial coarse bands also omit high-transverse-frequency grazing modes: small `|Omega·k|` need not imply small `|k|`. The full homogeneous reference already treats those modes, so that observation does not establish a missing post-TSA correction.

An even/odd Schur system offers a Hermitian positive reference and six angular components per cell, but `parity_transport.py:57–80` deliberately symmetrizes rounded collision blocks/antipodal geometry. It is a **reference**, not the exact stored physical matrix. The saved September19 direct Schur solves leave original physical residuals `1.6e-5–3.5e-5`, despite tight inner convergence, so independent outer physical correction remains necessary. Finite inner PCG is also source-dependent; it must not be treated as a fixed complex-linear surrogate merely because the Schur operator is linear.

## Angular/Fourier reformulation and a more aggressive identity

For ideal equally spaced circular ordinates, `K` is circulant and diagonal in the angular Fourier basis. Multiplication by `kx cos(theta)+ky sin(theta)` couples neighboring angular harmonics, making the homogeneous symbol cyclic tridiagonal in that basis. This is a possible classical alternative to a dense twelve-angle application. The stored rounded `K` and directions are only approximately ideal: dropping off-diagonal transformed entries or regenerating ideal geometry changes the discrete reference. At L12, extra angular transforms and cyclic solves may cost more than the existing cached small dense matrix-vector product. No profile presently establishes the angular product as the bottleneck. Angular Fourier reduction/FFT homogeneous solvers are established mechanisms in [Fan, An and Ying](https://arxiv.org/abs/1802.03061); angular multigrid addresses different forward-scattering DG regimes in [Lathouwers and Perkó](https://arxiv.org/abs/2010.04559). Neither paper supplies a speed guarantee for this stored Fourier/S1 implementation.

A stronger algebraic cancellation exists but should remain a later option. Define `A0=S+t0 I-s0 K`, `R=A0^-1`, and `Delta=A0-A`, then

\[
F=R(I+\Delta G),\qquad I-AF=\Delta(F-G).
\]

The recurrence `g=Gd; u=R(d+Delta*g); z+=u; d=Delta*(u-g)` would remove all31 internal streaming actions, reducing63 FFT pairs to32. It requires exact homogeneous-inverse identities and recursive residual propagation. The cached FP32 inverse is approximate, so drift may accumulate; for homogeneous `Delta=0`, it suppresses refinement of inverse roundoff that the original fresh `rhs-Az` recurrence performs. Also, `base.reference.A_op` remains heterogeneous because setup stores the original coefficients: **it is not `A0`**. Use `_reference_transport_matrix` to define the homogeneous symbol. This more aggressive recurrence should not be mixed into the first controlled test.

## One recommended diagnostic and stopping rule

**Test only the conservative inner-defect identity, with all outer residual updates retained.** First use N8/N16 CPU unit fixtures (homogeneous and contrasting materials, L12, signed complex probes) to compare the actual original and isolated fused actions at FP64 and the deployed FP32→FP64 convention. Check stored streaming-symbol identity, cache invalidation, the reference-vs-physical `sigma0` distinction, zero/phase/additivity behavior, depth1/16 output difference, and `||LGd-d||/||d||`. Use explicit small-matrix actions where practical; do not call a projected/coarse inverse an exact full inverse. These would be implementation verification fixtures, not a new physical dataset or training campaign. No such new computation was performed by this reviewer.

If those checks pass, the consuming executor can evaluate the **same frozen development fields and classical selections**, with the original true-RHS tolerance and measured full-solve cost, without new teachers or test seed9626. Record changed fields/residuals/failures and equality tolerance before judging latency; preserve negative outcomes. Only a warmed paired full-solver comparison can establish whether the proven elimination of redundant FFT work helps. If equivalence/accuracy fails, diagnose the inverse/geometry discrepancy instead of loosening the physical tolerance. If latency fails, retain the code as a negative implementation result and do not infer that a learned surrogate will be cheaper.

Primary local evidence: `remote_reference/rte/transport_baselines.py`, `models.py`, `global_source_predictor.py`, `mixed_precision_transport.py`, `multilevel_neural_transport.py`, `galerkin_transport.py`, `parity_transport.py`; `remote_reference/docs/research/RTE_PARITY_STRUCTURE_REVIEW_2026_09_19.md`; and the verified `direction_diagnostic_20261002/results` and `eligible_profiles` receipts. All current compute/cost statements refer to recorded executions, and all reformulation/latency opportunities above remain source-derived until the specified diagnostic runs.
