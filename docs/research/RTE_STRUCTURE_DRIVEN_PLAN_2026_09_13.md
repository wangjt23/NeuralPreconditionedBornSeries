# Next plan: target physical stiffness and heterogeneous coupling

This is a proposed next campaign, **not work already completed**. The
[September 13 results](RTE_STRUCTURE_AND_RESOLUTION_RESULTS_2026_09_13.md)
reject the idea that more pixels or more spectral-head capacity alone produce
a large AI benefit.

## 1. Separate grid size from physical difficulty

Use the same continuous material/source realizations sampled at N64/128/256,
not independently translated/randomized grids. Keep separate training,
selection and untouched confirmation seeds. Vary a diffusive scaling

\[
\sigma_t^\epsilon=\sigma_t/\epsilon,\quad
\sigma_s^\epsilon=\sigma_t/\epsilon-\epsilon\sigma_a,
\quad\epsilon\in\{1,.25,.1\}.
\]

The existing analytic audit predicts a much worse HSS bound as epsilon
decreases; this is a hypothesis about where learning could help, not a measured
AI result. Also vary coefficient contrast separately from epsilon. Keep FP64
outer residual checks, fixed tolerance and all failed cases. A change in g
needs a separate angular-resolution study; L12 is not automatically adequate
for a strongly forward-peaked kernel.

## 2. Strengthen the classical comparison first

The current DSA controls use spatially averaged diffusion coefficients. In
the diffusive family, implement a variable-coefficient scalar baseline using
the transport rate from the actual discrete circular kernel:

\[
\sigma_{tr}=\sigma_t-p_1\sigma_s,\qquad
\mathcal L_D\phi=\sigma_a\phi-
\nabla_h\!\cdot\left[(2\sigma_{tr})^{-1}\nabla_h\phi\right].
\]

This is a proposed diffusion approximation, not the exact finite-resolution
transport Schur complement. Construct the discrete diffusion part as a
positive gradient-adjoint product, audit its Nyquist convention, and compare
FFT-preconditioned CG/multilevel inversion against the existing HSS8 and TSA.
Do not claim AI benefit by comparing learned heterogeneous diffusion only with
an averaged-coefficient classical approximation.

## 3. Learn the expensive coupling, retaining full angular coverage

Candidate architecture: a **coefficient-conditioned macro correction with a
full-angle kinetic path**.

1. Apply an inexpensive full-angle HSS/CT correction.
2. Recompute the true physical defect. Project its scalar component for a
   heterogeneous macro correction; keep every angular component available to
   the fine path.
3. A material encoder predicts a compact coarse interpolation/operator across
   multiple physical scales. Cache it once per medium. Use a consistent
   Galerkin coarse action, not an independently averaged rediscretization.
4. Lift the macro correction and apply a full-angle post-correction, all inside
   the existing true-residual-checked flexible Krylov solver.

For a symmetric diffusion surrogate, R=P* and P* L_D P offer useful structure;
they do not prove convergence of the whole transport preconditioner. Train the
learned macro inverse on actual slow defects and short outer-Krylov solves,
with enough full-band probes to expose a lost range. Include a non-neural
Galerkin counterpart with the same fine stages and setup accounting.

This avoids the historical mistake of making the *entire* solution correction
live in a truncated angular subspace. It also avoids assuming that the new
full-angle Galerkin oracle is deployable: its 63 inner iterations were far too
expensive. Learning must compress that costly solve into a genuinely cheap
application; otherwise use the classical baseline.

For moderately scattering or ballistic cases, do not force a scalar macro
model: the structural audit found only 15–38% scalar-angular error. A separate
direction-aware kinetic correction may be needed. Any regime selection should
use material features, be fixed on validation, and include routing cost.

## Acceptance and ordering

Start with a bounded classical stiffness survey and a same-medium grid check,
then one matched learned-macro pilot. Measure cold preparation plus first RHS,
same-RHS warmed timing, and separately a genuine multiple-RHS workload.
Promote only if paired time-to-tolerance improves by at least 30% against the
strongest tuned classical control with no failure or tail regression, and if
extra AI capacity beats the matched CT control. Require additional independent
training seeds and an uncontended GPU window before a general speedup claim.

Do not launch a larger data/epoch sweep merely because a surrogate loss falls.
The previous campaigns show that loss, iteration count, and deployment time
can rank the models differently.
