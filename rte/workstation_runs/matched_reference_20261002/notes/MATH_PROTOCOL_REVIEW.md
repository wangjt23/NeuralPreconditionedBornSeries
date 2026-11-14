# Matched full-angle reference-defect diagnostic

Prepared 2026-10-02. Mathematical/source review only: no SSH, GPU execution, original-source export, new fine solution, training or seed9626 access. The new runner was not present when this design was prepared; this note is a protocol, not implementation or execution verification. Preserve all previous angular, whitening, response and timing negatives.

## Controlled comparison and theoretical scope

Let H be the unchanged actual stored-buffer FP64 physical operator. Let R0 be **the same already-cached FP32 full-angle homogeneous reference action** in both arms. Let Pi be the angular-index projector onto modes0,+1,-1. The physical directions/weighted phase are not replaced by ideal geometry. For a shared shallow action M4 and input r, define

    g=M4 r,  d=r-Hg,
    zq=R0(Pi d),  zf=R0 d,
    pq=g+zq,  pf=g+zf.

Use identical means, inverse cache, source/RHS cast order, precision and outer physical H. In the deployed FP32 path, first cast d once, then use the existing harmonic-input projection before the common reference solve; do not compare `Pi128 then cast` with `cast then Pi32`. Output casts/ownership also match. A separate optimized q3 symbol is a different representation and needs parity/cost evidence; its altered rounding cannot be silently substituted here.

**The q3 arm truncates angular input. Its output is full-angle** and has frequency-dependent angular response. `Pi R0 d` is an output-angle restriction and a different method. A large higher-angular output fraction in zq is expected when streaming couples angular modes; it does not invalidate the q3 input range. A matched q3/full difference therefore cannot be attributed to output masking.

For the mathematical exact inverse of a matched homogeneous operator H0,

    H0 R0=I,  H0 zf=d,  H0 zq=Pi d.

Consequently the unprojected homogeneous unit-correction residuals are0 and `(I-Pi)d`. This gives an exact homogeneous input-truncation statement, without a continuum or heterogeneous convergence claim. The inverse range `U(k)=R0(k)Q` is established classical response information. [Fan–An–Ying](https://arxiv.org/abs/1802.03061) is a close angular-reduction/FFT predecessor; no novelty claim is made.

The **stored** FP32 inverse and rounded FFT applications are approximate. With the mathematical cached map, E=H0R0−I and D=H−H0 give

    H R0 d = d + E d + D R0d,
    H R0 Pi d = Pi d + E Pi d + D R0 Pi d.

Record the homogeneous closure defect and heterogeneous image contribution separately. H0 must use actual stored streaming/weighted phase and the declared mean coefficients; a native regenerated ideal geometry is not a parity control. Differences between the low reference's rounded means/stream and the promoted physical means/stream belong in E. If measured via `H0z=Hz-Dz`, retain its ordinary-FP64 cancellation qualification.

In exact arithmetic the matched image difference is `H R0(I-Pi)d`. FP32 application/casts have an additivity gap: explicitly compare `zf-zq` with `R0(d-Pi d)` when that extra diagnostic action is affordable. Retain the gap; do not deploy a cancellation identity from it. Unit full correction can be worse than unit truncated correction. Writing `tq=d-Hzq` and `wo=H(zf-zq)`,

    ||d-Hzf||² - ||d-Hzq||² = ||wo||² - 2 Re<tq,wo>.

Omitted input energy alone says nothing about this sign under heterogeneity.

## Probe/frontier scope

Freeze **IDs0/3 at N128/N256**, the existing development arrays and hashes, with two contexts per grid/media pair: raw original source and the captured ordered frontier after four frozen strong-classical calls. These are eight contexts but only two independent media; no source-phase aliases create extra samples. Keep the actual N128 balanced initialization/reference-tail semantics and N256 zero start recorded. Already-converged/zero-defect prefixes supply no tail opportunity; retain them in full-solve means and mark the geometry ineligible.

Distinguish three possible inputs: original source b, current physical residual `b-Hx`, and the next normalized Arnoldi input. They are not interchangeable. A residual-conditioned correction diagnostic uses the residual; a from-start preconditioner is called on the Arnoldi input. Preserve the frozen state, stored solver residual, recomputed physical residual and their difference. For a full-solver comparison, the original true source-normalized residual remains acceptance authority.

Raw source/frontier geometry is primary. An existing actual grazing probe can remain an engineering parity check, clearly separate from independent medium/source evidence. Do not let a synthetic grazing success override the real-frontier result. All q/full arms share the same g and defect d; the strong-classical baseline remains its independently frozen original recipe.

## Real-AZ progress and permitted oracle diagnostics

Let V contain real-orthonormal **old search-image** vectors spanning HZ, and let J=I−VV^T in the real-stacked space. It is not the projector onto all Arnoldi vectors. Recompute/record its real Gram and residual orthogonality defects; if rank or orthogonality is unreliable, report attained values without optimality language.

For a complete candidate p, use `rbar=Jr` and `w=JHp`. The unrestricted real optimal one-direction coefficient is

    alpha = Re<rbar,w> / ||w||²,
    min_alpha ||rbar-alpha w||²
      = ||rbar||² - (Re<rbar,w>)²/||w||².

Evaluate the remaining vector explicitly rather than subtracting nearly equal scalars. Negative alpha is allowed by the repository's real FGMRES; clipping to nonnegative is a different constraint. A zero/nearly dependent useful image is ineligible, with its absolute norm recorded rather than a clamp-created progress claim. Complex scalar LS, or adding i p as a second direction, would expand the admissible real search span and is not this matched diagnostic.

**Score the full composite pq/pf against r for the deployed one-direction question.** Score zq/zf against the defect d only as a separately labeled increment diagnostic. Replacing one by the other gives misleading progress. A fixed unit coarse update and a real line-search oracle answer different questions; the latter's source-dependent scalar generally does not define a fixed complex-linear preconditioner.

A two-column real LS on `[JHpq,JHpf]` measures attained joint progress in those two actual proposals, retaining rank/conditioning and an explicit reconstructed residual. It is a small-span diagnostic, not a full-spatial coefficient solve or an optimal FGMRES preconditioner. If near dependence prevents a resolved rank-revealing solve, keep the attained upper only. The span generally contains a full-angle response outside range(R0Q); its floor is **not** a bound for the q3 response range. Additional spans containing g separately are yet another composition and must be labeled/charged separately.

No raw coercivity lower bound survives arbitrary real-AZ projection. J may remove rank, is generally real-linear rather than complex-linear, and its useful image requires the real inner product. Do not attach the old raw HQ/HU residual bracket to JHU. A small attained projected residual is positive evidence for that proposal; a large one does not prove the entire projected range useless.

## Measurements that separate the mechanisms

For each context/arm retain absolute input/defect/correction norms plus:

- Input omitted fraction `||(I-Pi)d||/||d||`, projection leakage and cast/additivity gaps.
- Homogeneous cache closure defects; `D z` norms and their residual alignment; q3 output's higher-angular fraction `||(I-Pi)zq||/||zq||`.
- Raw physical unit-defect ratio `||d-Hz||/||d||`; composite physical image and residual ratio.
- Useful real-AZ norm/retention, real signed cosine, unrestricted alpha, explicit unit/optimal one-direction ratios for both increment and composite when reported.
- Two-direction resolved rank/condition and attained LS ratio, with its exact span stated.
- Actual correction calls, projection/physical-image/helper/copy costs, setup/memory, whole-solver continuation, failures and acceptance.

Different projected-image norms are not themselves progress; alignment matters. Full proposal membership outside U can be established in the ideal homogeneous algebra when `(I-Pi)d!=0` and R0 is invertible. In heterogeneous material that membership does **not** show the true needed correction lies outside U. Fixed coefficients `Q*d` might simply be poor. Resolving all coefficient fields would need a separate correctly converged range oracle; this bounded matched comparison does not acquire one and does not reopen CGLS or q5 automatically.

## Deployment, economics and negative conclusions

The deployable arms are the fixed q3/full coefficient rules, with identical shallow action, initialization, outer solver and any tail policy. Retain M4 alone as the common-action control. An optional oracle line search/span is labeled separately and charged if actually deployed; cached oracle coefficients are neither a reusable inverse nor a learning result.

The existing reference implementation applies full-angle spatial FFTs and the same L-by-L inverse in both arms, with additional angular projection in the q3 arm. Do not claim a three-versus-twelve-channel compute advantage unless the actual selected implementation provides one. Full angle may be cheaper in this matched code despite a larger input space. Reuse/setup is credited equally to classical actions.

Compare complete Q1 costs with every successful frozen classical backend, including the verified direct-segmented improvement. The16-medium confirmation reports classical gains of20.784%/19.645% at N128/N256; these strengthen the comparator, not the correction-space claim. [Confirmation audit](../direct_capture/confirmation_audit.md) Timing repeats reduce to a within-medium median; charge actual used setup, initialization, copies, inference/acquisition, physical images, LS/guards and independent acceptance. Q16 remains conditional until distinct-source reuse is measured.

Keep the original FP64 physical residual1e-5, evaluated field-error bound1e-3 and paired field difference1e-4. Exact finite-operator coercivity identities are mathematical statements; numerically evaluated margins/gradients/error bounds here are **ordinary FP64 discrete estimates, not interval certificates, continuum-RTE errors or uniform convergence proofs**.

Decisive interpretations are:

1. **Full improves unit/allowed image progress and complete cost:** omitted-input response information matters for this fixed composition; freeze the classical improvement before any expansion. No learned superiority follows.
2. **Full improves image quality but loses cost:** the tested mechanism contains useful information whose selected implementation/composition is uneconomical. Only a concrete cheaper-action budget can justify further representation research.
3. **Full helps oracle scaling/span but not fixed deployment:** coefficient/combination choice is a possible explanation; the oracle alone supplies neither a fixed residual-linear map nor economically feasible acquisition.
4. **Neither helps after real-AZ projection:** reject those actual frontier proposals/compositions. Do not reject every coefficient field in U, all full-angular correctors, or projected optima that were not resolved.
5. **Geometry/accuracy/parity is unresolved or fails:** retain the numerical failure and stop that arm's economic interpretation; never change physical A or acceptance to rescue it.

This separates angular information, approximate reference/heterogeneity coupling and complete cost while keeping the experiment finite. No new teacher bank or training follows from a positive diagnostic alone. [Peng ROMSAD](https://arxiv.org/html/2410.08735v3) and [Tang–Peng trajectory-aware ROM](https://arxiv.org/html/2509.05001v1) already establish correction-space/trajectory mechanisms under different transport discretizations and setup assumptions.
