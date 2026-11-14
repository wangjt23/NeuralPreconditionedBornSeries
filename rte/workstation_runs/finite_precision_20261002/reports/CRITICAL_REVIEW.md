# Critical finite-precision review — 2026-10-02

This is new analysis, not a change to the frozen complex-Krylov run. That run correctly stopped after eight auxiliary `low_H64` records violated its fixed 2e−6 engineering cap; all 112 capture checks, original physical H128, actual deployed preconditioner and its physical composite passed. Zero full solves ran. No original source/policy was edited, new field experiment executed, SSH/GPU used, data exported, training performed or seed9626 touched for this review.

## 1. What is necessary, and what the old cap does not establish

Freeze the target **nominal discrete operator A** using the original stored coefficients and frequency convention. Specify whether its symbol is the exact product of stored directions/frequencies or the actual assembled high-precision rounded symbol. These differ slightly; their difference belongs in the budget if the former is the target. Angular scattering is the original `phase_matrix @ (weights * x)` followed by sigma_s, not a newly constructed coarse operator. Coefficient-product rounding likewise needs a declared target/budget. Source q and every returned field are the stored finite arrays, interpreted exactly for the algebraic statements below.

Nominal A must be complex-linear for the usual complex image-space interpretation. Actual flexible Arnoldi can use **any finite, owned returned direction z_j**, including one obtained with a varying/nonlinear/FP32 preconditioner, provided its actual physical image is formed and the resulting field is checked. Preconditioner complex linearity is needed for the simple fixed-Krylov inclusion theorem, not for every flexible solve. FP32 preconditioner error changes the directions and convergence/economics; it is not automatically an outer physical-action residual-gap term. Never charge the discrepancy from an ideal M as though actual z_j had not been used.

The old 2e−6 low-H commutation cap is **neither necessary nor sufficient** for physical acceptance. It is a prespecified diagnostic rule, which remains binding for its historical run. A common complex-linear bias in a computed action passes every homogeneity/additivity test while giving the wrong physical residual: `(1+c)A` is a simple algebraic counterexample. Conversely, a nonlinear preconditioner can be legitimate in flexible GMRES because each actual z_j is separately imaged by A. Thus finite commutation defects do not give an absolute bound on `||Ahat(x)−Ax||`, or a theorem that a future complex solve succeeds/fails. No cap should be raised retrospectively to fit the observed 5.0361e−6 witness.

The outer physical action in the prepared companion is the **original complex128 `high.A_op`**, not low_H64. Its `applied` wrapper casts returned images to the physical RHS dtype. Low_H64 participates inside the existing TSA preconditioner. The failed auxiliary test therefore does not show that the outer operator or actual deployed preconditioner violated their declared checks.

## 2. Adversarial Arnoldi residual-gap identity

For one cycle, treat stored vectors, coefficients and matrices as exact arrays. Let z_j be the actual owned directions, `what_j=Ahat(z_j)=A z_j+e_j`, and

    What = V Hbar + F,
    rhat_0 = r_0 + e_0 = beta v_1 + d_0,
    xhat = x_0 + Z y + c,

where F is the arithmetic Arnoldi relation defect, d_0 the initial normalization defect, e_0 the computed versus nominal cycle-start residual error, and c the field reconstruction rounding. With `h=beta e_1−Hbar y`, the **exact identity** is

    q−A xhat = V h + d_0−e_0−F y+E y−A c,
    E=[e_1,...,e_k].

Therefore a sufficient worst-case bound is

    ||q−A xhat|| <= ||V||_2 ||h||
      + ||d_0|| + ||e_0||
      + sum_j |y_j| (||f_j||+epsilon_A(z_j))
      + L_A ||c||,

if `epsilon_A(z_j)>=||e_j||` and `L_A>=||A||_2`. This allows all local errors to align adversarially with the residual. Do not replace the weighted sum by a random-cancellation or root-sum-of-squares estimate without additional assumptions. Tiny least-squares error changes actual y/h; once those stored quantities are used, it need not be counted again as a separate duplicate term. If h itself is evaluated in floating point, its evaluation uncertainty is additional.

One-pass MGS does not ensure `||V||=1`. A certified Gram-defect bound g gives `||V||<=sqrt(1+g)`. An ordinary computed Gram residual is diagnostic until its own product/reduction uncertainty is included. Small singular values and large coefficients can amplify otherwise tiny action/orthogonalization errors; recording only a per-direction relative defect omits this conditioning. SVD/rank safeguards in the companion are explicit algorithmic differences and costs, not exact projection guarantees.

Across restarts, repeat this statement with the actual cycle-start field/residual. There is no safe unqualified sum of independent cycle gains. The original helper checks `b−Ahat(correction)` during refinement and `q−Ahat(initial+correction)` at the end. Their split/direct gap includes initial action, field-addition and action errors. Fresh direct final evaluation is essential. Optional injection rejection preserves the original frontier; a numerical monotonicity assertion needs its own residual uncertainty rather than a nominal `1+1e−10` comparison alone.

## 3. Fresh physical acceptance and exact slack formula

For the **stored returned field x**, a fresh check eliminates dependence on historical Arnoldi errors for final accuracy. Suppose

    rhat = fl(q−Ahat(x)),
    ||Ahat(x)−Ax|| <= E_A(x),
    ||fl(q−Ahat(x))−(q−Ahat(x))|| <= E_sub.

Let `R_U` bound the exact norm of stored rhat from its computed norm, `Q_L>0` bound ||q|| from below, `X_L>0` bound ||x|| from below, and `m_L>0` be a genuine lower bound on nominal discrete coercivity. Then

    R_true <= B := R_U + E_A(x) + E_sub.

For residual tolerance tau=1e−5 and relative solution-error target eps=1e−3, a **joint sufficient acceptance condition** is

    B <= T := min(tau Q_L, [eps/(1+eps)] m_L X_L).

Indeed `||u−x||<=B/m_L` and `||u||>=X_L−B/m_L`; the second threshold gives `B/(m_L X_L−B)<=eps` and a positive denominator. No continuum, intensity-positivity or interval claim follows automatically.

The available **action/check slack** is

    S := T−R_U; require S>=0 and E_A(x)+E_sub<=S.

This is derived from the requested acceptance criteria, not a fitted phase cap. At an observed residual exactly at tau Q_L, the residual slack is zero. A blanket promise that FP64 roundoff is harmless is insufficient there. Conversely, requiring every arbitrary preconditioned probe to have a solution residual below1e−5 would be a category error: those probes are directions, not approximate solutions. Final-field slack and direction-image fidelity have different roles.

For an accepted proposal to be **provably physically monotone**, use an upper bound on the new residual and a lower bound on the old one, e.g. `Rnew_U+Enew <= Rold_L−Eold`, rather than comparing two unqualified computed ratios. The final source residual/error criteria remain decisive even when no monotonicity guarantee is asserted. Paired-field difference≤1e−4 is an additional gate; two error bounds≤1e−3 do not imply it. Its direct field-difference norm needs its own rounding bound, or a sufficient triangle bound `(B_candidate+B_baseline)/(m_L Xbaseline_L)<=1e−4`.

**Ordinary versus certified quantities.** The existing `m_hat=min(t−s lambda_max(Herm(P)))` is an ordinary FP64 evaluation. A rigorous m_L requires eigenvalue/coefficient/min-product uncertainty, for example a verified upper bound on the Hermitian eigenvalue plus conservative product/reduction rounding. Positive evaluated margin alone is not an interval certificate. Likewise, Q_L/X_L/R_U require an explicit norm-evaluation model. Standard finite-normal arithmetic assumptions permit conservative gamma bounds for products/sums/sqrt, but the actual GPU reduction/overflow/subnormal behavior must meet those assumptions. Do not relabel empirical norm or shadow differences as proven upper errors.

## 4. A noncircular absolute-action threshold

There is a useful conservative source-scale condition without assuming a small unknown solution norm. Suppose the complete fresh-check error obeys

    E_check(x) <= a Q + b ||x||,

with dimensionless a and an **absolute operator-scale** b, while the observed-residual norm is bounded by `rho_hat Q`. Coercivity gives

    (m_L−b)||x|| <= (1+a+rho_hat) Q,   provided b<m_L.

Combining this with the residual bound yields the sufficient condition

    a + rho_hat + b(1+a+rho_hat)/(m_L−b) <= tau,

equivalently

    b <= m_L (tau−rho_hat−a)/(1+tau),

with positive RHS. This is a derived conditioning-aware threshold for the **outer physical check**, not a relative commutation cap on low_H64. It remains conditional on proven a/b/m_L and norm bounds. It does not by itself guarantee the additional relative solution-error criterion; retain the actual X_L threshold above.

For example, if `||Ahat(x)−Ax||<=eta_A||x||` and real/complex subtraction has `E_sub<=u_c(Q+||Ahat(x)||)`, then one may take `a=u_c`, `b=eta_A+u_c(L_A+eta_A)`. All constants require their stated arithmetic model. With ordinary minimum margin0.0170082, even rho_hat=0 allows b only about1.7008e−7 in operator units. With rho_hat near tau the allowance shrinks sharply. This explains the need for high-precision physical residuals; it does not require an exactly linear FP32 preconditioner.

An a priori action model may bound `L_A<=Omega+Tmax+Smax||P||_2` for the actual stored Fourier symbol and angular matrix. In unitary Fourier notation, forward FFT, diagonal multiply and inverse FFT error factors epsilon_f,epsilon_d,epsilon_i imply the conditional streaming coefficient

    eta_stream <= [(1+epsilon_i)(1+epsilon_d)(1+epsilon_f)−1] Omega,

plus any target-symbol assembly discrepancy. Collision multiplication/einsum and final addition/subtraction require corresponding componentwise/product/reduction terms, using absolute phase/weight matrices when cancellation occurs. Actual Torch/CUDA FFT error constants are **not supplied by old commutation receipts**. A chosen multiple of unit roundoff or an observed shadow discrepancy is not a vendor-independent rigorous worst-case bound. No interval/verified FFT contract has been established in this review.

## 5. What existing scalar receipts say, and the prospective boundary

The following uses prior accepted real-direct-segmented repeat0 fields, not a new solution or confirmation. `mX/Q` is algebraically reconstructed from reported residual rho and evaluated bound B as `rho(1+1/B)`; margins agree across the old/current setup receipts. All numbers are ordinary computed diagnostics. For these old fields the error-bound allowance `eps/(1+eps)*mX/Q` is about1.20e−4–1.35e−4, so the1e−5 source residual condition is stricter.

| N / ID | Computed margin | Prior rho | Nominal remaining residual slack /Q | Absolute-action b ceiling at that rho, a=0 |
|---|---:|---:|---:|---:|
|128 /0|0.0170082|7.00242e−6|2.99758e−6|5.09829e−8|
|128 /3|0.0407708|4.79180e−6|5.20820e−6|2.12341e−7|
|256 /0|0.0170082|9.75745e−6|2.42549e−7|4.12528e−9|
|256 /3|0.0407708|1.10543e−6|8.89457e−6|3.62635e−7|

This table is scale context, **not a certificate or an admission threshold for an unobserved complex trajectory**. The tight N256/ID0 field makes using a generic1e−5 budget without subtracting observed residual especially unsafe. The low_H64 source-phase witness measures a different input, action, normalization and arithmetic; its2e−6 cap cannot be substituted for any of these outer-check allowances.

A defensible new diagnostic/protocol must declare before measurements: exact nominal target; error-model/interval assumptions; how L_A,m_L and norm bounds are obtained; admissible field/coefficient-conditioning domain; per-direction versus final-field budget; the source-scaled joint slack formula; what quantities are only ordinary estimates; no-waiver stop rules; full acquisition/verification costs; and the unchanged original residual/error/paired-field gates. Future coefficients/norms are unknown: a finite set of passing probes cannot guarantee an unbounded future Arnoldi trajectory. Either enforce derived live conditioning/slack guards or state a bounded diagnostic conclusion only.

The mathematical distinction justifies reconsidering what is necessary **in a newly authorized prospective protocol**. It does not erase the earlier failure, authorize an ad hoc threshold increase, demonstrate an affordable complex method, or supply a positive GO today. Without actual measured budgets tied to the acceptance slack—and justified bounds or an explicitly limited empirical claim—no stronger fidelity/performance conclusion is available. No full solves are authorized by this analysis task.

Source/API basis: existing `remote_reference/rte/models.py` (`_stream_symbol`, `streaming_op`, `A_op`, `G`, cached reference solve), `remote_reference/rte/source_initialization.py`, `remote_reference/rte/augmented_fgmres.py`, and preserved `deliverables/complex_krylov/complex_augmented_fgmres.py`. Scalar context: `receipts/complex_execution/results/{admissibility,setup}.json` and `receipts/gain_two_execution/results/solver.json`.
