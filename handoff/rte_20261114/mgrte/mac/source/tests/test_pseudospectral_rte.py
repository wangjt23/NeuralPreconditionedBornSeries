import scipy.io
import torch

from RTE_pseudospectral import PseudoSpectralRTE2d
from gmres import fgmres
from rte_config import REPO_ROOT


def load_dom_n1():
    quadrature = scipy.io.loadmat(REPO_ROOT / "discretized_parameters" / "quadrature2DN1.mat")
    kappa = scipy.io.loadmat(REPO_ROOT / "discretized_parameters" / "KappaN1g0.mat")["Kappa"]
    ct = torch.tensor(quadrature["ct"]).squeeze(-1).to(torch.float32)
    st = torch.tensor(quadrature["st"]).squeeze(-1).to(torch.float32)
    omega = torch.tensor(quadrature["omega"]).squeeze(-1).to(torch.float32)
    Kappa = torch.tensor(kappa).to(torch.float32)
    return ct, st, omega, Kappa


def constant_coeff(batch_size=1, mesh_size=4):
    coeff = torch.zeros(batch_size, 3, mesh_size, mesh_size)
    coeff[:, 0] = 2.0
    coeff[:, 1] = 0.4
    coeff[:, 2] = 0.0
    return coeff


def test_pseudospectral_reference_inverse_constant_coefficients():
    ct, st, omega, Kappa = load_dom_n1()
    op = PseudoSpectralRTE2d(mesh_size=4).setup(constant_coeff(), ct, st, omega, Kappa)
    rhs = torch.randn(1, ct.numel(), 4, 4)
    reconstructed = op.G_inv(op.G(rhs))
    rel = torch.linalg.norm(reconstructed - rhs) / torch.linalg.norm(rhs)
    assert rel.item() < 1e-5


def test_pseudospectral_A_matches_reference_for_constant_coefficients():
    ct, st, omega, Kappa = load_dom_n1()
    op = PseudoSpectralRTE2d(mesh_size=4).setup(constant_coeff(), ct, st, omega, Kappa)
    u = torch.randn(1, ct.numel(), 4, 4)
    rel = torch.linalg.norm(op.A_op(u) - op.G_inv(u)) / torch.linalg.norm(op.G_inv(u))
    assert rel.item() < 1e-5


def test_pseudospectral_green_preconditioner_is_exact_for_constant_reference():
    ct, st, omega, Kappa = load_dom_n1()
    op = PseudoSpectralRTE2d(mesh_size=4).setup(constant_coeff(), ct, st, omega, Kappa)
    rhs = torch.randn(1, ct.numel(), 4, 4).to(torch.complex64)
    _, history = fgmres(op.A_op, rhs, op.G, tol=1e-6, maxiter=3, verbose=False)
    assert history[-1] < 1e-5
