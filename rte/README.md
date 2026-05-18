# RTE NPBS Prototype

This module is a minimal implementation of the NPBS residual-metric idea for a
steady-state radiative transfer equation (RTE) with periodic spatial boundary
conditions and discrete ordinates.

The discrete system is

```text
(Omega_l . grad + sigma_t) psi_l - sigma_s * phi = q_l,
phi = sum_m w_m psi_m.
```

`PeriodicRTE2D` chooses a constant-coefficient transport reference

```text
L_eta,l = Omega_l . grad + sigma_ref + eta
```

and applies `G_eta = L_eta^{-1}` by FFT. This gives the same residual-coordinate
identity used by NPBS:

```text
I - G_eta V_eta = G_eta A,   V_eta = L_eta - A.
```

It also provides a second preconditioning map, `transport_dsa`, for experiments
where the NPBS metric is induced by a synthetic-acceleration-like
preconditioner:

```text
B_transport r = G_eta r
B_transport_dsa r = G_eta r + lift(D_ref^{-1} moment(r - A G_eta r))
```

Here `D_ref^{-1}` is a constant-coefficient periodic diffusion solve for the
low-order scalar-flux error, and `lift` maps scalar-flux corrections back to
isotropic angular-flux corrections. This is a periodic diffusion-augmentation
prototype, not a boundary-compatible production DSA discretization.

Run the verifier from the repository root:

```bash
python -m rte.verify_rte_npbs
```

The verifier checks the Born residual identity and confirms residual decrease
for both raw transport and transport+DSA preconditioned Richardson iterations on
a moderate-scattering case.

This is a prototype, not a production RTE solver. The current implementation is
a periodic pseudo-spectral algebraic benchmark; it does not model upwind sweeps,
vacuum/inflow/reflecting boundaries, positivity preservation, or production
transport-consistent DSA. The next useful extensions are anisotropic phase
functions, non-periodic inflow boundaries, transport-consistent DSA/TSA, and
rollout/replay training of learned residual correctors.
