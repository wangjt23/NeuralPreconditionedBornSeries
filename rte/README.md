# RTE NPBS Prototype

## Current Entry Points

The [workstation handoff](../docs/research/RTE_WORKSTATION_HANDOFF_2026_09_12.md)
contains the latest shared CT ensemble results, actual data snapshots, checkpoint
locations, verification commands and prioritized workstation experiments.
Local training is stopped; GPU validation and further training are pending.

Start with the [2026-09-12 audit](../docs/research/PROJECT_AUDIT_2026_09_12.md)
and [self-contained mathematical note](../docs/research/RTE_RESIDUAL_SUBSPACE_NOTE.tex).
The [artifact index](archive/ARTIFACT_INDEX.md) groups historical code, notes,
results and traces. The [large-experiment protocol](LARGE_ANGULAR_SPECTRAL_EXPERIMENT_PLAN.md)
documents corrected epoch semantics, data splits, controls and restart behavior.

The new [collision/transport architecture](../docs/research/RTE_COLLISION_TRANSPORT_ARCHITECTURE.md)
uses exact angular collision and spatial streaming factors with cached coefficient-conditioned gains.
Its [controlled results](../docs/research/RTE_COLLISION_TRANSPORT_RESULTS.md) include train/heldout losses,
classical shift tuning, spatial/global gain ablations, a cached U-Net control, and negative experiments.
This is an additional per-medium prototype, not a replacement for DSA/TSA in all regimes.

From the repository root:

```bash
python -m unittest discover -s rte/tests -v
python -m rte.verify_rte_npbs
python scripts/summarize_rte_evidence.py --check
python -m rte.run_residual_subspace_experiment --steps 300 --seeds 1 2 3
python scripts/summarize_residual_subspace.py
python -m rte.probe_residual_amplification
```

The residual-subspace laptop runner compares classical two/four-stage TSA, raw-loss learned
corrections, and physical residual-minimized variants. Saved results show
one-step improvements but **not a learned solver speedup over TSA4**.
`--models legacy_raw legacy_replay` isolates an exploratory difficult-residual
sampling repair; use a separate output directory for that experiment.

## Operator

This module is a minimal implementation of the NPBS residual-metric idea for a
steady-state radiative transfer equation (RTE) with periodic spatial boundary
conditions and discrete ordinates.

The isotropic discrete system is

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
transport-consistent DSA. Anisotropic circular phase functions and several
rollout/replay variants are already implemented. Physical angular S2 quadrature,
non-periodic inflow boundaries and transport-consistent DSA/TSA remain separate
extensions.
