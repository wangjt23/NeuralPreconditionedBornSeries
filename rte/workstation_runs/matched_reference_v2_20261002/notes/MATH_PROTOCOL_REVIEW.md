# Matched full-angle/reference-q3 mathematical review, v2

Prepared 2026-10-02 from local code/protocol evidence. No SSH, GPU experiment, source export, new teacher, training, Library mutation or seed9626 access. The original [v1 review](../full_angle/MATH_PROTOCOL_REVIEW.md) is preserved. This self-contained note corrects its operator attribution and qualifies the measured-image span fit. The v1 CUDA mask-construction failure occurred before scientific results; the separately frozen cached-mask repair needs its own parity/capture evidence.

## Actual deployed pair

Both arms use the same prepared FP32 TSA4 object and the same FP32 full-angle reference inverse R0. With incoming r32, the implemented fixed map is

    g32=TSA4(r32), d32=r32-low.A_op(g32),
    zq32=R0(Pi3 d32), zf32=R0 d32,
    pq=promote128(g32+zq32), pf=promote128(g32+zf32).

Pi3 retains angular-index Fourier modes0,+1,-1. It truncates reference **input**, while both responses retain all output angles and spatial frequencies. The physical directions/phase are unchanged. The original stored positive uniform weights justify the DFT projector's weighted-orthogonal interpretation; they do not change the original real-stacked solver's coefficient convention.

Precision, casts, reference means/cache and addition/defect order match. A segmented prefix's owned128 result is cast back before the literal FP32 defect/addition. A native128 fallback promotes the existing inverse, rather than rebuilding a new double-precision one. An optimized channel-reduced q3 symbol would be a different representation requiring separate parity and cost evidence.

The scientific image authority is always the original promoted-buffer FP64 **H**. The diagnostic records `dtrue=r-H(promote(g32))` separately from d32 and scores the complete pq/pf through H against r. A nonzero low-versus-high defect gap qualifies physical coefficient interpretations; it does not invalidate the matched full/q3 contrast.

## Correct reference decomposition

Define H0 **exactly as the implementation does**: the stored FP32-assembled reference symbol from `_reference_transport_matrix()`, promoted to complex128 and applied with diagnostic spatial FFTs. It uses its original reference stream, rounded means, weighted phase and cache conventions. Define the nominal cached inverse map R0 from those stored inverse entries. In exact finite-dimensional algebra,

    E=H0 R0-I, D=H-H0,
    H R0 d=d+E d+D R0d,
    H R0 Pi3 d=Pi3 d+E Pi3 d+D R0 Pi3 d.

**E concerns inverse-cache/reference application closure. D contains both heterogeneous coefficient variation and stored-operator differences** between physical H and reference H0, including streaming assembly, mean/phase/quadrature arithmetic. It is incorrect to assign those physical-versus-reference differences to E under this definition. It is also incorrect to identify the observed remaining discrepancy solely as heterogeneity.

Production FP32 FFTs/casts/multiplications add rounding; their measured closure is input-dependent and is not a proof of a globally exact linear map. The implemented `h0(zf)-d32` and `h0(zq)-Pi3 d32` are reference-consistent field closure probes in ordinary FP64. They neither certify a global operator norm nor isolate pure heterogeneous physics. Measured additivity gaps in proposals and actual-H images remain explicit.

For an exact inverse of H0, `H0 R0=I`: reference images are d and Pi3 d. Only in the **matched homogeneous limit H=H0** are physical unit residuals0 and `(I-Pi3)d`. Homogeneous material alone does not erase the actual stored-operator mismatch. This is a finite discrete identity, not a novel transport theorem or continuous-RTE certificate.

The full-q3 difference isolates omitted **input** information through the same response map. The q3 output need not lie in the fixed angular space Q: streaming produces higher output harmonics. `Pi3 R0 d` would be a different output restriction. A large q3 higher-output fraction is therefore compatible with the intended method.

Unit full correction can be worse under physical H. With `tq=dtrue-Hzq` and `wo=H(zf-zq)`, exact image algebra gives

    ||dtrue-Hzf||²-||dtrue-Hzq||²
      =||wo||²-2 Re<tq,wo>.

Large omitted-input energy does not establish useful alignment or predict the sign.

## Frozen contexts and real-AZ geometry

Use existing IDs0/3 at N128/N256: raw source and the captured ordered four-call strong-classical frontier, eight targets from only two independent media. Retain N128 balanced initialization/reference-tail and N256 zero start. Store the original solver residual, independently recomputed `b-Hx`, their gap, and real Gram/orthogonality defects of captured image basis V. A four-call truncated diagnostic's nonconverged status is not a failed full solver. A converged/zero or unavailable prefix provides no eligible tail opportunity.

Do not interchange source, residual and normalized Arnoldi input. Fixed preconditioner deployment acts on the Arnoldi input; a residual-conditioned diagnostic probes a different state. All actual complete solves retain the frozen original solver semantics.

Let J remove the retained **real** search-image span HZ, rather than all Arnoldi vectors. For complete proposal p, use `rbar=Jr`, `w=JHp`. Unrestricted real one-direction LS has

    alpha=Re<rbar,w>/||w||²,
    remaining=rbar-alpha w.

Evaluate the remaining vector explicitly. Negative alpha is allowed; nonnegative clamping is a different constraint. Complex alpha or an extra i p direction expands the admissible real span. Record absolute norms and eligibility for near-zero useful images, rather than allowing denominator clamps to imply useful progress.

The complete proposal pq/pf is scored against r. Increment zq/zf is scored against dtrue and reported separately. The latter cannot substitute for deployed composite geometry. Unit and line-search results answer different questions. Outer FGMRES already optimizes real scaling: line-search beating the unit update is not evidence that learning a mere scalar rescaling will improve Krylov directions.

Raw coercivity does not transfer through arbitrary real-AZ projection. J may remove rank and is generally real-linear, not complex-linear. A large attained projected residual cannot reject the full q3 response range. A useful measured image supports that actual direction, not a uniform convergence theorem.

## Two-image fit: exact permitted conclusion

The driver fits the two **measured projected images** of pq/pf, with a predeclared rank cutoff and reported Gram spectrum. This is an attained result in that two-column real image span. It is not a full-spatial coefficient solve, a q3/full response-range optimum, or a globally optimal FGMRES action. The full-angle column generally extends outside range(R0Q), so the span floor is not even a bound for that q3 range's best correction.

**The fitted field combination has not been independently evaluated through H.** The reported residual is reconstructed from the separately measured images. Floating-point additivity and projection/rank effects therefore qualify an actual-field interpretation. Label it `attained measured-image span`, retain coefficients/conditioning, and do not report it as a verified deployed correction or acceptance-qualified solve. An independent H evaluation of the combined field would be a separately budgeted future verification; this review performs none.

Direct actual-H scoring of each primary deployed composite remains valid. Missing combination verification does not discredit those individual images or the fixed-map full-solver comparison.

## Costs, gates and conclusions

Both matched reference arms still execute the full-angle spatial FFT and L-by-L inverse; q3 adds angular projection FFTs. A q3 input-space dimension is not a q3-channel performance claim. Actual constructors, masks/guards/copies, initialization, physical images, continuation and independent acceptance are charged. Geometry acquisition, H0 construction, four-call prefix and small fits are separate diagnostic costs; nested component times must not be counted twice. They are excluded from fixed-map deployment only because those maps do not use the fitted oracle.

The future comparator includes every successful classical backend, including direct segmented's verified16-development-medium advantage. Preserve recorded setup-plus-warm labels and first-use ordering; these are not matched process-cold estimates. Timing repeats are medians within media, and Q16 is conditional until distinct-source reuse is measured.

The frozen matched screen retains the10% Q1 **and** warm-accepted target at both grids, no individual Q1 regression>10%, and same-backend comparisons separately from backend selection. Keep the original physical residual<=1e-5, evaluated relative error bound<=1e-3 and paired field difference<=1e-4. These margins/bounds are **ordinary FP64 discrete evaluations**, not interval certificates, continuum error estimates or universal convergence proofs.

Possible conclusions concern this fixed coefficient rule and composition: omitted inputs improve useful signal; truncation removes unhelpful signal; the homogeneous rule is insufficient; or costs exceed attained savings. Neither a poor full-input result nor a two-proposal fit establishes an optimal-range obstruction or a requirement for residual-history data. For fixed linear H, the exact inverse is memoryless; history can matter to economical finite-capacity/trajectory-aware corrections, which needs separate evidence and budget.

[Fan–An–Ying](https://arxiv.org/abs/1802.03061) already supplies angular-reduction/homogeneous-FFT precedent. [Peng ROMSAD](https://arxiv.org/html/2410.08735v3) and [Tang–Peng trajectory-aware ROM](https://arxiv.org/html/2509.05001v1) supply correction/trajectory competitors with different transport discretizations and setup assumptions. No priority claim, teacher bank or training authorization follows from this bounded diagnostic.
