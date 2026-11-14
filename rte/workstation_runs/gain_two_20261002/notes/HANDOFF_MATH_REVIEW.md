# Independent handoff mathematics review

Reviewed 2026-10-02: `receipts/build_gain_two_handoff.py`, the generated `RTE_FWI_Gain_Two_Screen_2026_10_02.md`, `MECHANISM_AND_STOP_RULE.md`, `FALLBACK_REVIEW.md`, the original stationary/complex-FGMRES code and the new setup receipts. **No material mathematical blocker found.** This is a source/algebra review; no new numerical experiment or frozen-file edit was performed.

The ordered identities are correct for fixed nominal exact operators and relaxation one, which the executed setup receipts confirm. With `F=G+R0(I-H_l G)` and `S=I-H_l F`,

    S=(I-H_l R0)(I-H_l G),
    M4=F sum_(j=0)^3 S^j,   I-H_l M4=S^4,
    M2=M4+R0 C2 S^4,       C2=2I-Pi3,
    I-H_l M2=(I-H_l R0 C2)S^4,
    I-H_h M2=(I-H_l R0 C2)S^4-(H_h-H_l)M2.

No commutation is needed. TSA4 is not `F^4`; the complete terminal map generally is not a polynomial in one common operator. The solution-error identity in the mechanism note likewise has the correct order. In the ideal matched homogeneous inverse, a general tentative field leaves `(1-beta)(I-Pi3)d`; the ideal TSA primitive itself already has `F=R0`. Scattering does not imply `R0=G`. Positivity of `C2` establishes no contraction or solver-cost theorem.

The report correctly limits these identities to nominal maps. FP32 FFTs, local operations, subtraction, prefix backcasts and the final addition are rounded and are not exactly complex-linear. Native128 promotes the original complex64 inverse rather than rebuilding it. `H_l`, the actual stored reference matrix `H0`, and the promoted-buffer physical operator `H_h` remain distinct; cache closure and their assembly differences cannot be attributed solely to material heterogeneity. Actual `H_h(p2)` supplies geometry and acceptance: the rounded reassociation `2*p_full-p_q3` cannot replace it. The finite-discrete coercivity/error estimate is correctly described as ordinary FP64 evaluation, not an interval or continuum certificate.

The complex-Krylov fallback is conditional and unexecuted. For one shared frozen basis `W=H_h Z`, the complex coefficient fit contains the real fit, hence `min_C ||r-Wc|| <= min_R ||r-Wa||`. The nominal unrestarted fixed-Krylov version additionally requires a fixed complex-linear preconditioner `M` and effective operator `H_h M`. **Optional wording clarification:** spell out this effective operator when the report says “fixed complex-linear action.” Its existing restart/flexible/injection/rounding caveats already rule out a general call-count or time guarantee.

The original complex FGMRES preserves conjugate products, complex Hessenberg entries and complex least-squares coefficients. Any future numerical fallback must also preserve complex coefficients, or use the full real block system; discarding imaginary entries would solve a different fit. No added fallback was executed or certified here. Retaining actual `Z` and unprojected `W`, handling complex rank and checking the combined physical field are necessary. A real-AZ projector need not commute with multiplication by `i`; real-stacked fields retain their imaginary channels and can already be globally phase-equivariant.

The numerical success and failed cost gate support stopping this tested fixed-gain/composition family on the reused development cohort. They prove neither learned superiority nor an impossibility for other gains, representations, coefficient fields or materials. Conditional whole-solve ceilings are accounting constraints for the current setup; eliminating acquisition cost alone does not remove its measured warm-cost failure. The handoff preserves these boundaries and does not authorize the proposed fallback or new compute.
