# Locked proposal: compressed P8 Fourier inner coordinates

Status: proposed, not executed. Parent review is required before implementation/numerical work. This document does not reopen G or alter sealed B/E/G/H. Follow `CONCEPT_PORTFOLIO.md` for the two-round direction debate and split restrictions.

## 1. Exact finite-dimensional identity

Use normal index n and tangent index j to avoid ambiguity about x/y names. N_n=N_t=128, L=12, and J=(-8,...,8), m=17. Store coefficients z as [batch1,angle12,normal128,mode17], with signed modes in that order and original FFT slots k mod128.

Let F be the orthonormal128-point tangent FFT. Let I_J insert the17 entries into those signed slots and Q_J select them. Define E=F* I_J, R=Q_J F and P=ER. Thus R E=I, E*E=I and P=P8 in exact arithmetic. The inner problem is A_c z=R r, with A_c=R A E. The injected physical correction is E z. This retains exactly the same P8 subspace and original physical A.

For either actual promoted stored coefficient sigma, form

  c_sigma[n,d] = FFT_backward(sigma[n,:])[d mod128] /128,
  C_sigma[n,k,l] = c_sigma[n,k-l],  k,l in J.

Then R diag(sigma) E z = C_sigma z. Only d=-16,...,16 is needed. This is a finite-projection identity, NOT a claim that sigma is33-band-limited. Negative differences wrap into the original128 slots. The sampled periodic problem already has its original128-point cyclic aliasing. Do not replace this by a17-point cyclic convolution, zero-pad/dealias the physics, re-evaluate the analytic Gaussian, or assume exact separability/proportionality of the rounded total/scattering arrays. Build both C_t and C_s independently from `high.sigma_t` and `high.sigma_s`.

Let K_lq=stored_phase_lq*stored_weight_q. In exact arithmetic the reduced action is

  A_c z = streaming_c(z) + C_t z - C_s(K z).

K is the actual full12-by12 angular kernel, not rank1. Its action commutes with the spatial coefficient matrices because those are scalar in angle. To follow production rounding as closely as possible, implement angular weighting then the original full phase contraction in the same order as `models.py:182-187`, instead of relying on a pre-multiplied K to be bitwise equal.

For streaming, use the actual promoted `high._stream_symbol` restricted to the17 tangent slots. Compute a backward-normalized normal FFT, multiply that stored-derived symbol, then a matching inverse normal FFT. This preserves the stored normal Nyquist and the original order of direction-times-frequency addition. Do not replace the Nyquist with zero, regenerate ideal directions/frequencies or split the symbol using a new rounded formula. Exact arithmetic gives R streaming(Ez)=streaming_c(z); two-dimensional versus separated FFT execution still needs numerical qualification.

## 2. Background inverse and Krylov semantics

The reduced preconditioner calls each unchanged E `AngularBandMode.repaired` on z[...,k], keeping its original-G residual target1e-10, at most3 corrections, full original phase, stored directions and strict-zero handling. This avoids only the full tangent transform/projection wrappers. No factor is reused across measured batches for free.

The existing full-grid comparison path is exactly the sealed E inner action `project8(high.A_op(v))` and preconditioner `project8(factor.apply(project8(v)))`. Use it unchanged as the strongest attribution control. Its repeated projections are algebraically redundant but have actual finite-precision and cost effects that cannot be silently erased from the historical result.

The existing `solve_from_initial` and `fgmres_real_stacked_with_trace` remain imported unchanged. Their real-channel representation uses real Euclidean products, equivalent to Re(u* v) on complex fields. The orthonormal Fourier embedding induces a real isometry, preserving norms, dot products and exact-arithmetic Krylov recurrences. This does not promise bitwise reductions or identical floating-point stopping. Keep the existing floors, breakdown1e-14, restart25, zero initial state and all stopping semantics; do not rewrite them for the compressed state.

Each candidate injection must additionally lift its completed correction and independently check `||P8(r-A Ez)||/||P8 r|| <=1e-4` using the original full A, before returning it to the unchanged outer solver. That extra check is paid. The original outer residual/field/bound checks remain full physical calculations. Zero projected RHS returns exact zero; nonzero checks use scaled relative norms without an added absolute floor. Tiny-vector action/inverse tests do not claim that the production Krylov solver's existing1e-30 floor has been changed.

## 3. Dimensions and operation ledger

- One complex128 physical vector has196,608 elements/3,145,728 bytes. One reduced vector has26,112 elements/417,792 bytes. The state ratio is128/17, about7.53, not a predicted solver speedup.
- Two sets of33 coefficient frequencies need135,168 bytes. Expanded two128-by17-by17 complex128 matrices need1,183,744 bytes. A retained12-by128-by17 streaming symbol needs417,792 bytes. Charge actual persistent and temporary storage, not merely these analytic sizes.
- One reduced A action uses204 length128 normal FFTs and204 inverse FFTs,26,112 symbol products,313,344 full-angular complex multiply-accumulate terms and887,808 coefficient-matrix multiply-accumulate terms. Angular weights add26,112 scalar products. These are algebraic operation counts, not BLAS/kernel dispatch counts or FLOP-based timing predictions.
- The current full-grid inner A uses a full forward/inverse2D FFT and a tangent projection FFT pair:9,216 length128 transforms in the straightforward separable count, versus408 in the reduced action. Native FFT implementation choices can fuse operations, so this count is explanatory rather than a timing model.
- The existing preconditioner wrapper performs three full tangent FFT/inverse pairs: input projection, E application and output projection. The reduced path removes those wrappers but retains every17-mode raw block solve, original-G action and repair. E factor storage is not reduced by this concept.
- With m_i inner iterations, the existing imported solver invokes the inner A exactly2m_i+2 times and its inverse m_i times, absent early/breakdown branches. The frozen amplitude3 pattern totals19 inverse applications and46 reduced-A applications over four sources. New independent full projected-A injection checks add four full-A applications. Record actual counters, not only this source-derived expectation.

The saved conditional free-inner ceiling is reported in `SOURCE_AUDIT.json`. It limits a representation-only claim with the outer call pattern fixed. Do not add this ceiling to a GPU setup saving, conflate it with neural gain or extrapolate to new source counts.

## 4. Frozen qualification before complete timing

New isolated directory only. Read and verify production, B/E dependency, amplitude3 material/background and all four source hashes from the sealed receipts. Require normal axis0. No N256/9626 or new material generation. CPU0, one thread, CUDA disabled; verify current admission and coordinate with other memory-heavy work. No package installation.

Use one predeclared finite probe set, not a sweep:

1. The17 complex vectors generated in E's existing validation loop with seed2026111004, identical generation order and per-mode hashes, combined into one all-mode state and also retained for the usual independent17 A0 inverse checks.
2. Two deterministic endpoint states supported at tangent mode-8 and+8, with a normal Nyquist sign vector. Use fixed angle0 and angle3 respectively to expose angular mixing and grazing directions. No conjugate reduction.
3. Exact zero, all-mode state scaled by1e-200, and a mixed-column zero/ordinary/tiny case. Synthetic action probes are not extra physical RHS to amortize the solve.

For every nonzero action probe compare A_c z with R A(Ez), using production high.A_op as the independent oracle. Require scaled relative difference <=1e-10. Verify E A_c z against P8 A(Ez) too, so gather/scatter conventions cannot agree by sharing the same mistake. Zero outputs must be exact zero and finite. Record both input and output hashes, norms, per-probe differences and actual stored coefficient fingerprints.

For the same probes compare the directly compressed E inverse with R[project8(E_factor.apply(project8(Ez)))]. Require field and projected-original-A-image relative differences <=1e-10, and retain every per-mode original-G repair record. Run the existing independent full-A0 check for all17 modes, threshold1e-10, plus the zero/tiny/mixed checks. Do not use the angular band residual as the acceptance oracle.

A failed identity, mode, action, inverse, zero/tiny or budget gate ends this implementation attempt with its receipt. Do not loosen the threshold or tune a mode subset. Qualification may share one freshly built E factor for attribution, but those factors are not carried into measured batches.

## 5. Conditional complete-cost gate

Only after the bridge passes, obtain four ordinary accepted native reference outputs for field checks; charge their research cost separately and never use them as directions/initial states. No high-accuracy teacher.

Run one fresh triplet in the locked order: compact4, unchanged full-grid E, reduced-coordinate E. Each batch rebuilds its physical/TSA/factor objects. E batches both pay all17 independent A0 checks; the reduced batch additionally pays its coefficient construction and all entry/action qualification needed by the chosen deployment contract. Do not prewarm with measured material or retain useful factors/RHS state. Use the same CPU warm convention for all methods and report any common initialization separately.

Require unchanged amplitude3 calls: compact4 outer[4,3,4,3]; both E paths outer[2,2,2,2], inner[5,5,5,4]. Record original inner residuals, inverse application counts, every original-G repair and actual outer-A calls. A changed pattern stops the attribution gate even if final acceptance passes. This protects the intended representation-only claim; it is not a statement that another convergent algorithm is invalid.

Require all original final gates: outer relative residual1e-5, evaluated relative-error bound1e-3, field difference1e-4, inner1e-4/max8, outermax100 and repaired original-G1e-10/max3. Failure, exception or a setup lower bound already reaching either completed control stops immediately: the candidate must beat both, so the smaller completed control is the relevant lower-bound cutoff. If complete reduced observed wall is not strictly lower than BOTH fresh full-grid E and compact4, stop without a reverse run.

Only if the first triplet passes every bridge, acceptance and full-cost condition may one reverse triplet run: reduced E, full-grid E, compact4, again fresh. Opposite timing conclusions mean uncertain and stop, not more repeats until favorable. This is a bounded representation qualification, not a population timing claim or material generalization.

For each batch measure direct first-RHS wall from the beginning of setup through accepted source0 and direct complete-four-RHS wall. Never divide by four. Include coefficient FFT/assembly, E chain/corner LU, original-A0/action checks, all repairs, conversions, Krylov work, original full-A validation and bookkeeping. Report online/setup breakdowns, logical factor and reduced-coefficient bytes, process peak RSS, exact source/background hashes and current shared-host context. No timing sums may replace observed wall.

## 6. Budget and decision

Proposed numerical ceiling: CPU0/one thread/no CUDA, each stage180 seconds and aggregate600 seconds, including probes, ordinary references and all timed work. Stages are qualification, references plus forward triplet, then conditional reverse triplet. Apply hard elapsed-time limits, save partial receipts on failure and stop only the owned job. Estimated duration is a few minutes based on existing CPU receipts, not a promised runtime. No GPU admission is requested.

If action equivalence succeeds but costs fail, preserve the finite-projection identity as a representation result and stop this implementation. If both triplets pass, the allowed conclusion is a cheaper original-equation-accepted CPU implementation of the same fixed P8 enrichment on one development material. Do not claim held-out, N256, neural, GPU or general-material performance. Any later H stress transfer, new coefficient regime, scale test or GPU port requires a new debate and explicit coordinated plan.
