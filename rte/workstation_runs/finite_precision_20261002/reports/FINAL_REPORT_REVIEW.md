# Independent final report mathematics and numerical-claim review

Reviewed `RTE_FWI_Finite_Precision_2026_10_02.md` and the scalar JSON receipts in `receipts/finite_execution/results`. This review performs only scalar reductions of recorded results: no original RTE action, solver, SSH, GPU, training, new field or threshold change.

**Conclusion:** the coefficient-bound definitions, conditional error/residual-gap statements, recorded numerical claims and retained certified-admission **NO-GO** are consistent. One illustrative-example distinction was sent to the parent for clarification; it does not alter the actual-column proof or numerical results.

## Mathematical scope

The target H uses the actual assembled physical D64, and L uses actual assembled D32. Promoting an application input does not change low.float_dtype and therefore does not replace D32 by D64. The coefficient norm bound uses the exact stored phase/column weights; its row/column absolute-sum bound does not require exact symmetry. The real nonnegative materials and purely imaginary fixed streaming symbol give the stated conservative coercivity inequality. The implemented rational square-root ceiling and outward scalar conversions are conservative versions of the symbolic bounds.

The flexible residual-gap identity correctly uses actual returned Z and actual computed physical images. Preconditioner nonlinearity/rounding can change those directions and their quality but supplies no additional independent low-H-homogeneity term in that exact identity. The coefficient-weighted image/Arnoldi errors and final field-update error remain necessary in an implementation bound. A fresh final residual enclosure can certify an individual returned field without a prior Arnoldi-trajectory bound; prospective convergence or coordinate-residual admission needs the additional appropriate bounds. No such solve or certificate is claimed here.

The unchanged-tolerance reserve uses rho_upper, pays subtraction/action/norm contracts inside the fixed one percent, and retains a strictly positive solution-error denominator. The gamma dot/reduction hypotheses and FFT factors remain conditional. The reported equal-FFT limits are sufficient requirements of this chosen model, not established backend constants or necessary error tolerances for the true kernels. Small observed comparisons cannot establish any of those uniform contracts.

## Counterexample distinction

The first inspected draft used M(r)=r³ and a zero M to illustrate why homogeneity alone is neither necessary nor sufficient for flexible-GMRES quality. Those are examples about the **complete preconditioner M**, whereas the preserved failed predicate was about the **auxiliary low-H action**. They should be labelled generic illustrations rather than direct witnesses for the project low-H predicate. A nonzero rank-deficient linear M=diag(1,0), H=I, b=(1,1) is a clearer insufficiency illustration: the first direction is nonzero, residual becomes (0,1), and the next returned direction vanishes, so the solver cannot complete accurately.

A direct necessity illustration for an auxiliary action is the scalar toy H=1, a(r)=r³, G=R0=I, F(r)=r+(r-a(r))=2r-r³. At normalized first Arnoldi vector1, F(1)=1 gives an exact one-step flexible solution, even though a(i)=-i differs from i*a(1)=i. This is an abstract TSA-like composition, not the actual transport coefficients, an RTE physical regime, or evidence that the present factors converge. The actual-Z identity is the substantive project argument; no counterexample waives the historical frozen engineering rule.

## Recorded numerical claims checked

- Counts agree:72 full-grid H actions plus24 toy H actions,8 M actions,48 NumPy references,12 paired fixture cases/24 exact applications, zero full solves. Internal M/capture H work is explicitly outside the direct-H count and inside its measured action/setup scopes.
- All72 full-grid action rows have finite matched and overall comparisons. Every input-cast discrepancy is exactly zero. The C64-origin full-grid fields and exactly representable toy dyadics do not test a nonrepresentable C128-to-C64 cast.
- Physical128 maximum discrepancy per input norm is1.2104016191e-13 and1.2068129900e-13 on N128 IDs0/3, and2.6143848966e-13 and2.6120321639e-13 on N256. These match the report table and remain differences between two rounded implementations.
- The prospective allowances1.7007057437e-9,4.0769188043e-9,1.7006446696e-9,4.0768576071e-9 and sufficient equal-FFT limits7.7307635322e-13 through3.7091333150e-12 match recorded coefficient-model outputs. Observed/allowance ratios are engineering ratios, not certified residual gaps.
- The low32 maxima7.6216931054e-5 and1.4398523829e-4 per input norm, low-native128 maxima1.2081987264e-13 and2.6136381894e-13 against matched D32, and low-native128-to-physical discrepancies up to7.5340364867e-6 and1.5010017742e-5 agree with receipts. They neither recompute nor supersede the old homogeneity failure.
- N256/ID0 M_source has image-relative low32 error3.8023875061e-4 but error/(Lambda*input norm)3.5935578189e-8, with indicator10581.12238. Its interpretation as cancellation/scale sensitivity is appropriate; it is not a certified condition number or component-level localization of FFT versus contraction error.
- Exact toy upper errors7.4366553596e-6 for low32 and1.4157154069e-14 for physical128 match receipts. Direct stored-bit Fraction subtraction and outward norm ratios certify those particular fixed-symbol4x4 differences only. No uniform/full-grid/continuum inference is licensed.
- Settings timestamp50.66211526s and peak allocated361017856B/reserved731906048B match receipts. Component/JSON timings are partial research-process scopes; the launcher's52.07058s is the authoritative owned process total and no solve-economics conclusion is inferred.

The original failed entry, ordinary original FP64 residual/field acceptance, held-out seal, strongest classical denominator and zero-full-solve outcome remain unchanged. This review adds no new GO, backend constant, solver claim or learned-method novelty claim.
