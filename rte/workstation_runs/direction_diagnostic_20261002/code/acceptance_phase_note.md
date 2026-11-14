# Acceptance, continuation, and phase symmetry

This note concerns the offline continuation helper and the existing real-stacked
one-shot augmented FGMRES. It establishes exact-arithmetic identities and the
conditions needed to compare their trajectories. It reports no new training,
data generation, runtime result, or held-out result.

## Source boundary

The original `remote_reference/rte/tail_lookahead.py:26–55` reconstructs an
Arnoldi span from the retained search-image basis and prefix residual. A zero
or dependent proposal appends a zero frontier, so later calls apply the
preconditioner to zero instead of following the deployed fallback.

The deployed `remote_reference/rte/augmented_fgmres.py:70–76` rejects zero and
nonfinite proposals, then resumes the original ordered Arnoldi input without
restarting. Lines 87–95 test dependence on the **search-image space**; lines
99–107 jointly fit old and new directions and reject a residual increase.
Lines 137–141 independently check the final physical residual against the
original right-hand side.

The measured cohort had zero rejected injections, according to the parent
audit. Correcting rejection continuation closes a correctness gap; it does
not explain the recorded runtime slowdown on that cohort.

## State and acceptance contract

For each sample, regard a complex field as a real vector, with

\[
\langle x,y\rangle_{\mathbb R}=\operatorname{Re}(x^*y),
\qquad \|x\|^2=x^*x.
\]

Let \(Z_j\) be the old search directions, \(S_j=\operatorname{span}_{\mathbb R}
(AZ_j)\), and let the columns of \(Q\) be real-orthonormal and span \(S_j\).
The prefix residual \(r\) is orthogonal to \(S_j\). The helper receives the
physical action \(A\), classical preconditioner \(B\), proposed correction
\(z\), residual \(r\), image basis \(Q\), and the original **ordered Arnoldi
frontier**. Passing the full ordered basis \(V_{j+1}\) is sufficient; at an
eligible prefix, retaining its original next input \(v_j\), together with the
reconstructible span, is sufficient for rejection fallback.

Acceptance is independent for each batch sample:

1. Reject a zero or nonfinite proposal. Reject a nonfinite physical image too;
   invalid values must be sanitized or excluded before they can contaminate
   another sample's action, projection, norm, or derivative.
2. Form \(u=Az\) and \(w=(I-P_Q)u\), using the real projector
   \(P_Qx=\sum_k q_k\langle q_k,x\rangle_{\mathbb R}\). Reject images dependent
   on \(S_j\), including zero images. A practical relative threshold is an
   implementation policy, rather than an exact mathematical identity.
3. For an accepted image, enlarge the image space and minimize the residual
   over it. Advance the Arnoldi frontier using
   \(f=(I-P_{V_{j+1}})u\), provided \(f\ne0\).
4. On rejection, leave the old residual, image space, and ordered frontier
   unchanged. The next classical call is \(B(v_j)\). A rejected attempt
   consumes no accepted-direction slot and does not restart the solver.

An offline output convention may retain an injection-attempt slot: its squared
residual ratio is one when rejected, followed by the ratios after actual
classical calls. This convention must be stated so it is not confused with
the deployed trace, which records accepted directions only.

The deployment scales the proposal by \(1/\|r\|\) at
`augmented_fgmres.py:73–74`. In exact arithmetic this is positive real scaling,
so it changes neither the image space nor its normalized new frontier. A
helper claiming numerical trace parity must also align precision, scaling,
orthogonalization, rank thresholds, and breakdown rules.

## Why the old span is insufficient on rejection

At an eligible, non-breakdown prefix with \(\dim S_j=j\),
\(\dim\operatorname{span}_{\mathbb R}(V_{j+1})=j+1\), and nonzero residual,

\[
\operatorname{span}_{\mathbb R}(V_{j+1})
=S_j\oplus\operatorname{span}_{\mathbb R}(r).
\]

Indeed, \(r\) lies in the old Arnoldi span, is orthogonal to \(S_j\), and supplies
the one missing dimension. Thus \(Q,r/\|r\|\) reconstruct an orthonormal basis
for the old span. They do not identify its ordered last vector \(v_j\).
Different orthonormal orderings of the same span generally give different
\(B(v_j)\), even for linear \(B\). Rejection fallback therefore requires the
original frontier, rather than an arbitrary basis for its span. The span
identity itself must not be assumed for ineligible or breakdown states.

Dependence on the full Arnoldi span is **not** grounds for rejecting a proposal.
For a perfect tail correction, \(Az=r\), so its image is inside the old Arnoldi
span but outside \(S_j\). Accepting it removes the entire residual, while its
new Arnoldi frontier is zero: this is a happy breakdown. Under the eligible
span identity, any accepted image inside the old span has
\(w=\beta r\) with nonzero real \(\beta\), and also eliminates the residual in
exact arithmetic. Do not call \(B(0)\) to manufacture further continuation.

## Monotonicity and symmetries

Assume \(A\) is real-linear, \(Q\) is real-orthonormal, and
\(r\perp_{\mathbb R}S_j\). For \(w\ne0\), the expanded-space minimizer is

\[
\alpha=\frac{\langle w,r\rangle_{\mathbb R}}{\|w\|^2},
\qquad r_+=r-\alpha w,
\qquad
\|r_+\|^2=\|r\|^2-
\frac{\langle w,r\rangle_{\mathbb R}^2}{\|w\|^2}
\le\|r\|^2.
\]

This is minimization over the real expanded image space
\(S_j+\operatorname{span}_{\mathbb R}(Az)\). Real coefficients and a real Gram
matrix are consistent with the deployed solver; allowing complex coefficients
would generally enlarge the search space and change the method.

Replacing \(z\) by \(cz\), for any nonzero **real** \(c\), gives the same
expanded image space and immediate residual: \(w\) becomes \(cw\) and
\(\alpha\) becomes \(\alpha/c\). For negative \(c\), the normalized new
Arnoldi frontier changes sign. Later residual trajectories remain equivalent
when \(B\) is odd, in particular when \(B\) is real-linear, and the subsequent
algebra and stopping rules respect that sign change. An arbitrary nonlinear
\(B\) need not preserve this later equivalence. Absolute floors and floating
point thresholds can also break scaling invariance near zero.

For a unit phase \(\gamma=e^{i\phi}\), assume \(A\) and \(B\) are
complex-linear at fixed material. Rotate the complete state together:
\((r,z,Q,V)\mapsto(\gamma r,\gamma z,\gamma Q,\gamma V)\).
The real inner product and norms are unchanged, the physical images and
projectors rotate, and all real least-squares coefficients are unchanged.
Acceptance decisions and residual ratios are therefore identical in exact
arithmetic; fields and frontiers rotate by \(\gamma\). A learned proposal
must itself satisfy this joint phase identity for the complete deployed
pipeline to inherit it. Real Gram fitting cannot impose it on a proposal
that lacks it. Complex linearity of \(B\) is a sufficient condition here;
a nonlinear phase-equivariant \(B\) can also preserve phase symmetry.

## Limits of the claims

The identities above do not promise bitwise agreement in finite precision.
One-pass versus two-pass orthogonalization, near rank loss, nonfinite actions,
and threshold branches need explicit tests. Acceptance masks are piecewise
decisions; derivatives through a fixed accepted branch do not establish smooth
gradients at an acceptance boundary.

The helper's projected residuals are offline algebraic quantities. Without
the old solution directions and original physical right-hand side, it cannot
independently reconstruct and verify the deployed physical solution or its
original-RHS stopping tolerance. The final deployed residual check remains
necessary. Residual acceptance alone is also not a solution-error bound.

An accepted direction adds a valid immediate search image. It can still alter
the later classical frontier unfavorably, consume time, or change behavior
under restart and finite budgets. Neither immediate monotonicity nor symmetry
is a guarantee of future convergence, fewer classical calls, or runtime
improvement. No new held-out seed or expensive training is justified by this
correctness repair alone.
