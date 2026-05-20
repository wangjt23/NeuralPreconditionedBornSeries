# Moment-Space Learned Synthetic Acceleration For RTE

This note records the design decision behind the new closed angular moment TSA
path and the first learned structural hooks.

## Literature-Grounded Design

Classical diffusion synthetic acceleration is effective because it targets the
diffusion-limit scalar-flux slow mode of source iteration. In forward-peaked or
strongly anisotropic scattering, higher angular modes also become slow, so a
scalar DSA correction is not enough. Transport synthetic acceleration, angular
multigrid, variable Eddington factor/quasidiffusion, and reduced-order
synthetic-acceleration methods all point to the same design rule: keep a
low-order transport/moment solve in the preconditioner and learn only the
uncertain low-order structure.

The learned component should therefore augment
\[
  B_Q r
  =
  G_\eta r+
  Q(Q^*A_{\rm ref}Q)^{-1}Q^*(I-A G_\eta)r,
\]
where \(Q\) is an angular Fourier or moment basis. This is preferable to a
generic raw-ordinate U-Net because it exposes the slow angular subspace directly.

## Implemented Components

- `transport_tsa_moment` in `rte/models.py` implements the closed Galerkin
  moment solve \(Q(Q^*A_{\rm ref}Q)^{-1}Q^*\). It preserves the existing
  `transport_tsa` behavior, whose `tsa_harmonics=K` option only projects the RHS
  before the full angular inverse.
- `MomentNPBSUNetPreconditioner` in `rte/npbs_unet_ablation.py` keeps the
  defect2 structure, but restricts the learned correction to angular Fourier
  modes \(0,\pm1,\ldots,\pm K\) before prolonging back to ordinates.
- `ModeWeightedMomentTSAPreconditioner` is a tiny learned structural model. It
  starts exactly as closed TSA-MK and learns bounded real weights for each
  absolute angular mode in the synthetic correction.
- `CoeffWeightedMomentTSAPreconditioner` keeps the closed moment solve fixed,
  but learns nonlinear coefficient-dependent physical-space weights
  \(\Gamma_\theta(x,|m|;\sigma_t,\sigma_s,g,\nabla\sigma_t)\) on the moment
  correction. This is the first implemented component that explicitly learns
  nonlinear medium dependence while staying linear in residuals for fixed
  coefficients. By default it freezes the scalar \(m=0\) weight and exposes
  transport/synthetic relaxation parameters, because the smoke tests show that
  undamped moment TSA can be unstable as a stationary iteration.
- `rte/verify_moment_tsa.py` checks projection equivalence, Galerkin residual,
  complex linearity, and angular-mode residual energy.
- `rte/run_hard_case_experiments.py` now includes `TSA-M2` and `TSA-M4`.

## First Smoke-Test Results

The deterministic verifier passed at machine precision on a small anisotropic
case:

- projection error: `6.27e-16`
- Galerkin residual in retained moment space: `2.12e-15`
- complex-linearity error: `4.50e-16`

A size-16 hard-case smoke run showed that closed moment TSA is a meaningful but
not automatically dominant baseline. In anisotropic and mixed cases, TSA-M4 is
better than TSA-M2, but full TSA still gives the best FGMRES residual. This
supports learning angular subspaces or weights rather than assuming fixed low
Fourier modes are sufficient.

A tiny learned anisotropic run also showed that closed TSA-M2 can require
damping for stationary use. Unit-relaxation depth-2 moment TSA can grow the
stationary residual, while `w=0.4` stabilizes the same two-stage correction.
With only five training steps, `MomentUNet-K2-w0.4` improved one-step residual
relative to damped TSA-M2 in the smoke run, but its FGMRES residual was still
worse than the damped classical baseline. This is a sanity check, not evidence
of superiority.

The fixed-\(Q\) components are intentionally conservative and do not learn a
full nonlinear subspace \(Q_\theta(\sigma)\). The coefficient-weighted moment
model is the first intermediate step: it lets the medium choose local bounded
mode amplitudes without changing the Galerkin inverse. A full learned
restriction/prolongation pair should come later and should preserve the scalar
moment, conjugate mode pairs, and \(R_\theta P_\theta\approx I\) on retained
moments.

## Next Experiments

1. Run `transport_tsa_moment` sweeps for `K=1,2,3,4` against DSA, current
   RHS-projected TSA-H, and full TSA.
2. Use real-stacked FGMRES for learned real/imag networks, or constrain learned
   layers to be complex-linear.
3. Train `MomentUNet-K` with damping, rollout stability loss, and residuals from
   solver trajectories.
4. Train `WeightedTSA-MK` longer and compare learned mode weights against fixed
   damping and residual-polynomial baselines.
5. Only after fixed moment TSA and mode-weighted TSA are understood, test
   learned restriction/prolongation \(R_\theta,P_\theta\). Keep \(R_\theta\) and
   \(P_\theta\) initialized to \(Q^*,Q\), preserve the scalar moment, and
   constrain conjugate angular pairs.
6. For heterogeneous/interface cases, test variable or learned DSA closure with
   positive \(D_\theta(x)\) and \(\sigma_{a,\theta}(x)\), but report the cost of
   replacing the current FFT scalar diffusion solve.

## References

- Adams and Larsen, “Fast iterative methods for discrete-ordinates particle
  transport calculations”: https://www.sciencedirect.com/science/article/pii/S0149197001000233
- Morel and Manteuffel-style angular multigrid for forward-peaked scattering:
  https://www.osti.gov/biblio/5308325
- Transport synthetic acceleration and S2SA methods for anisotropic transport:
  https://www.tandfonline.com/doi/abs/10.13182/NSE00-A2108
- Forward-peaked scattering acceleration with higher angular moments:
  https://arxiv.org/abs/1907.13013
- Variable Eddington factor/quasidiffusion and second-moment acceleration for
  low-order transport closures: https://arxiv.org/abs/2104.07826 and
  https://arxiv.org/abs/2409.06162
- Reduced-order model synthetic acceleration and trajectory-aware ROMSA for
  learned or data-informed low-order spaces:
  https://www.sciencedirect.com/science/article/pii/S0021999124005515 and
  https://arxiv.org/abs/2509.05001
