import argparse

import torch

from rte.models import PeriodicRTE2D


def relative_error(x, y):
    return (torch.linalg.norm(x - y) / torch.clamp(torch.linalg.norm(y), min=1e-30)).item()


def make_coefficients(size, dtype):
    grid = torch.linspace(0.0, 1.0, size, dtype=dtype)
    x, y = torch.meshgrid(grid, grid, indexing="ij")
    bump = torch.exp(-80.0 * ((x - 0.55) ** 2 + (y - 0.45) ** 2))
    sigma_t = 1.0 + 0.25 * torch.sin(2.0 * torch.pi * x) * torch.cos(2.0 * torch.pi * y)
    sigma_t = sigma_t + 0.25 * bump
    albedo = 0.45 + 0.1 * torch.sin(2.0 * torch.pi * y) ** 2
    sigma_s = torch.clamp(albedo * sigma_t, max=0.85 * sigma_t)
    return sigma_t.unsqueeze(0).unsqueeze(0), sigma_s.unsqueeze(0).unsqueeze(0)


def make_source(size, n_angles, dtype):
    source = torch.zeros(1, n_angles, size, size, dtype=torch.complex128 if dtype == torch.float64 else torch.complex64)
    source[..., size // 3, size // 2] = 1.0 + 0.0j
    return source


def assert_close(label, lhs, rhs, tol):
    err = relative_error(lhs, rhs)
    print(f"{label}: {err:.3e}")
    if err > tol:
        raise SystemExit(f"{label} failed: {err:.3e} > {tol:.3e}")


def check_operator_identities(rte, psi, rhs, tol):
    assert_close("G(L_eta psi) identity", rte.G(rte.L_eta_op(psi)), psi, tol)
    assert_close("L_eta(G rhs) identity", rte.L_eta_op(rte.G(rhs)), rhs, tol)
    assert_close("A + V = L_eta identity", rte.A_op(psi) + rte.V_op(psi), rte.L_eta_op(psi), tol)

    lhs = rte.G(rte.V_op(psi) + rte.source) - psi
    born_rhs = rte.G(rte.source - rte.A_op(psi))
    assert_close("RTE Born residual identity", lhs, born_rhs, tol)

    scalar = rte.scalar_flux(psi)
    lifted = rte.lift_scalar_flux(scalar)
    assert_close("scalar_flux(isotropic_lift(phi)) identity", rte.scalar_flux(lifted), scalar, tol)

    for kind, tsa_harmonics in (("transport", None), ("transport_dsa", None), ("transport_tsa", 2)):
        du = torch.randn_like(psi)
        residual = rte.source - rte.A_op(psi)
        lhs = rte.apply_preconditioner(
            rte.source - rte.A_op(psi + du),
            kind=kind,
            tsa_harmonics=tsa_harmonics,
        )
        rhs_coord = rte.apply_preconditioner(
            residual,
            kind=kind,
            tsa_harmonics=tsa_harmonics,
        ) - rte.apply_preconditioner(
            rte.A_op(du),
            kind=kind,
            tsa_harmonics=tsa_harmonics,
        )
        assert_close(f"{kind} residual-coordinate propagation", lhs, rhs_coord, tol)

        a = torch.randn((), dtype=psi.real.dtype).item()
        b = torch.randn((), dtype=psi.real.dtype).item()
        r1 = torch.randn_like(psi)
        r2 = torch.randn_like(psi)
        lhs = rte.apply_preconditioner(
            a * r1 + b * r2,
            kind=kind,
            tsa_harmonics=tsa_harmonics,
        )
        rhs_lin = a * rte.apply_preconditioner(
            r1,
            kind=kind,
            tsa_harmonics=tsa_harmonics,
        ) + b * rte.apply_preconditioner(
            r2,
            kind=kind,
            tsa_harmonics=tsa_harmonics,
        )
        assert_close(f"{kind} linearity", lhs, rhs_lin, tol)


def main():
    parser = argparse.ArgumentParser(description="Verify the RTE NPBS/Born identity.")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--n-angles", type=int, default=16)
    parser.add_argument("--eta-scale", type=float, default=0.3)
    parser.add_argument("--anisotropy-g", type=float, default=0.0)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--float32", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    dtype = torch.float32 if args.float32 else torch.float64
    cdtype = torch.complex64 if args.float32 else torch.complex128
    torch.manual_seed(args.seed)

    sigma_t, sigma_s = make_coefficients(args.size, dtype)
    source = make_source(args.size, args.n_angles, dtype)
    rte = PeriodicRTE2D(
        shape=(args.size, args.size),
        dx=1.0 / args.size,
        n_angles=args.n_angles,
        anisotropy_g=args.anisotropy_g,
        eta_scale=args.eta_scale,
        fdtype=dtype,
    ).setup(sigma_t=sigma_t, sigma_s=sigma_s, source=source)

    psi = torch.randn(1, args.n_angles, args.size, args.size, dtype=cdtype)
    rhs = torch.randn_like(psi)
    check_operator_identities(rte, psi, rhs, args.tol)

    _, history = rte.preconditioned_richardson(maxiter=args.maxiter, tol=1e-6, preconditioner="transport")
    print(f"transport Richardson initial/final relres: {history[0]:.3e} -> {history[-1]:.3e}")
    if history[-1] >= history[0]:
        raise SystemExit("transport Richardson residual did not decrease")

    _, dsa_history = rte.preconditioned_richardson(
        maxiter=args.maxiter,
        tol=1e-6,
        relaxation=0.8,
        preconditioner="transport_dsa",
    )
    print(f"transport+DSA Richardson initial/final relres: {dsa_history[0]:.3e} -> {dsa_history[-1]:.3e}")
    if dsa_history[-1] >= dsa_history[0]:
        raise SystemExit("transport+DSA residual did not decrease")

    _, tsa_history = rte.preconditioned_richardson(
        maxiter=args.maxiter,
        tol=1e-6,
        relaxation=0.8,
        preconditioner="transport_tsa",
        tsa_harmonics=2,
    )
    print(f"transport+TSA Richardson initial/final relres: {tsa_history[0]:.3e} -> {tsa_history[-1]:.3e}")
    if tsa_history[-1] >= tsa_history[0]:
        raise SystemExit("transport+TSA residual did not decrease")

    residual = rte.source - rte.A_op(torch.zeros_like(rte.source))
    transport_norm = rte.metric_norm(residual, kind="transport").item()
    dsa_norm = rte.metric_norm(residual, kind="transport_dsa").item()
    tsa_norm = rte.metric_norm(residual, kind="transport_tsa", tsa_harmonics=2).item()
    print(
        "metric norm of initial residual: "
        f"transport={transport_norm:.3e}, transport_dsa={dsa_norm:.3e}, transport_tsa={tsa_norm:.3e}"
    )
    print("RTE NPBS verifier passed.")


if __name__ == "__main__":
    main()
