# Design only: residual-to-solution certificate for the stored discrete RTE

This note was initially a proposed cheap mathematical diagnostic. The dated
root-execution follow-up below records its subsequent validation execution.
It concerns the repository's complex, finite-dimensional periodic
Fourier operator. It does not establish positivity of a physical intensity,
continuous-RTE error, or convergence of a learned stationary iteration.

## Assumptions and theorem

Use the actual stored discrete scattering matrix

\[
P=\texttt{phase\_matrix}\,\operatorname{diag}(\texttt{weights}),\qquad
H_P=(P+P^*)/2,\qquad \mu=\lambda_{\max}(H_P).
\]

The present quadrature has constant angular weights, so its discrete weighted
norm is a constant multiple of the Euclidean norm. All relative norm ratios
are unchanged. This formula must be rederived for nonuniform quadrature rather
than assuming Euclidean self-adjointness there.

Write the stored operator as

\[
A=S+\sigma_t(x)I-\sigma_s(x)P,
\qquad m=\min_x\{\sigma_t(x)-\sigma_s(x)\mu\}.
\]

Assume the stored coefficients are real, `sigma_s >= 0`, the Fourier streaming
symbol is purely imaginary with real wavenumbers/directions, and `m > 0`.
The even-grid Nyquist convention does not prevent skew-Hermitian streaming
on this complex discrete space. No exact symmetry, circulant structure,
antipodal identity, or exact normalization of the rounded `P` is needed.

For every field `v`,

\[
\operatorname{Re}\langle v,Av\rangle
=\langle v,(\sigma_t I-\sigma_s H_P)v\rangle
\ge m\|v\|^2.
\]

Cauchy–Schwarz gives `||Av|| >= m ||v||`, so the discrete square matrix is
invertible and `||A^{-1}|| <= 1/m`. If `u_star` solves `A u_star = q` and an
approximation `u_hat` has original-equation residual `r=q-A u_hat`, then

\[
\|u_\star-u_{\rm hat}\|\le\delta:=\|r\|/m,
\qquad
\frac{\|u_\star-u_{\rm hat}\|}{\|u_\star\|}
\le \rho\frac{\|q\|}{m\|u_\star\|},
\quad \rho=\frac{\|r\|}{\|q\|}.
\]

The exact solution norm is generally unavailable. A computable sufficient
relative bound follows when `||u_hat|| > delta`:

\[
\frac{\|u_\star-u_{\rm hat}\|}{\|u_\star\|}
\le \frac{\delta}{\|u_{\rm hat}\|-\delta}.
\]

If that denominator is nonpositive, report the absolute bound and an
unavailable relative certificate; do not divide or declare failure of the
solver. A residual tolerance of `1e-5` does not itself imply `1e-5` field error.

## Proposed bounded diagnostic

1. Promote the stored `P`, coefficients, directions and wavenumbers exactly
   as `rte/mixed_precision_transport.py:8` does. Do not regenerate geometry.
2. Compute `mu` with a twelve-by-twelve Hermitian eigensolve, then per-medium
   `m`. Record phase asymmetry and the margin `m`; if `m <= 0`, this sufficient
   bound is unavailable, without implying singularity.
3. On an already-computed training or validation solve, independently form
   the true FP64 original-RHS residual, then record `rho`, `m`, `delta`, field
   norm and the computable relative bound. No teacher or additional large
   solve is required. Keep this work outside the historical timing window.
4. Verify the formula on small synthetic operators and check the operator
   adjoint/skew identity in complex arithmetic. If checking against a teacher,
   account for the teacher's own residual bound instead of treating it as an
   exact solution.

This is an exact-arithmetic theorem applied through floating-point estimates.
A strict machine-verified numerical certificate additionally needs reliable
upper rounding of `mu`, lower rounding of `m`, and residual-rounding control.
An ordinary FP64 eigensolve should be described as a numerical diagnostic,
with its positive margin recorded, rather than as an interval proof.

Sources: `rte/models.py:89` (actual phase construction), `:182` (scattering),
`:189` (streaming symbol), `:202` (physical action), and
`rte/mixed_precision_transport.py:8` (exact promotion of stored geometry).

## Prepared script, not executed by the audit author

`audit_coercivity.py` imports the current repository's `rte` package. From the
FWI repository root, the consuming executor can run the small CPU check:

```bash
PYTHONPATH=. /data1/software/miniconda3/bin/python PATH_TO/audit_coercivity.py --manufactured-test
```

After the resource check and current training have finished, the optional full
validation diagnostic is:

```bash
PYTHONPATH=. /data1/software/miniconda3/bin/python PATH_TO/audit_coercivity.py --device cuda:0 --output NEW_AUDIT_DIRECTORY
```

It uses only frozen classical selections and the exact seed-4626 sixteen-case
validation cohort at N64/N128/N256. `--count` limits the audited prefix without
changing the generated cohort. It creates no teachers, neural model, training,
tuning or held-out test data. Diagnostic setup/solve durations are explicitly
separate from historical benchmark times. Failed/nonfinite cases are retained.

## Root-execution follow-up — 2026-10-02

The root executor subsequently ran the diagnostic on FWI after the controlled
training study, using the frozen classical selections on all sixteen existing
validation media at N64/N128/N256. No teachers or held-out test data were
generated. All 48 solves meet the original-equation residual tolerance. The
minimum recorded coercivity margin is 0.01270603; the maximum computed relative
solution-error bound `delta / (||u_hat|| - delta)` is 0.0002368861.

Thus the current validation solves have an evaluated sufficient discrete
solution-error bound as well as a true-residual check. The theorem remains an
exact-arithmetic statement, and its recorded FP64 estimates remain numerical
diagnostics without interval rounding. Neither these values nor the residual
checks establish positivity of the physical solution or error for a continuous
RTE. Diagnostic durations are separate from the 480 measured validation solves
in the neural runtime comparison, whose gate failed and left seed 9626 unused.

The consuming root retains the execution receipts under
`receipts/rte/workstation_runs/resume_audit_20261002/`. The original design and
reproduction commands remain above as a record of what was specified before
execution.

Independent inspection of the retrieved `solution_bounds/cases.json` confirms
16 cases per grid, all 48 positive margins and available relative bounds, zero
failures, and exact agreement of the recorded relative bound with the displayed
formula. The maximum independently checked relative residual is
9.9850115718e-6. The actual rounded phase has Hermitian-part maximum eigenvalue
1.0000000393225126, relative asymmetry 5.4284406229e-8, and maximum row-normalization
error 1.0554989582e-7. These small defects explain why the bound uses the stored
Hermitian part rather than replacing it by an ideal phase with eigenvalue one.
No serious inconsistency was found in the compact numerical receipts.
