"""Source-conditioned moment shapes, with exact small projected coefficients.

Four angular groups: scalar, cos(theta), sin(theta), and remaining moments.
The network predicts shapes; the classical residual chooses their amplitudes.
An analytical-reference-only version is mandatory as a non-AI control.
"""

import math

import torch

from rte.augmented_fgmres import augmented_fgmres


def moment_components(value):
    l = value.shape[1]
    theta = torch.arange(l, device=value.device, dtype=value.real.dtype)*(2*math.pi/l)
    patterns = torch.stack((torch.ones_like(theta)/math.sqrt(l),
        theta.cos()*math.sqrt(2/l), theta.sin()*math.sqrt(2/l)))
    coefficients = torch.einsum('kl,blxy->bkxy', patterns.to(value.dtype), value)
    components = coefficients[:, :, None]*patterns[None, :, :, None, None]
    return torch.cat((components, (value-components.sum(1))[:, None]), 1)


def combine_moment_space(action, proposal, residual, basis, ridge=1e-12):
    """Batched differentiable four-column least squares after removing A Z.

    action accepts [B,L,N,N]. Four calls keep the material batch unchanged.
    Each column is normalized before the tiny solve, avoiding angular-scale
    imbalance; the returned ratio is evaluated explicitly in residual space.
    """
    components = moment_components(proposal.to(torch.complex128))
    images = torch.stack([action(components[:, j]) for j in range(4)], 1)
    if basis is not None:
        inner = torch.einsum('bk... ,bj... ->bkj', basis.conj(), images).real
        images = images-torch.einsum('bk...,bkj->bj...', basis, inner.to(basis.dtype))
    norms = images.abs().square().flatten(2).sum(2).sqrt().clamp_min(1e-30)
    normalized = images/norms[:, :, None, None, None]
    gram = torch.einsum('bi...,bj...->bij', normalized.conj(), normalized).real
    right = torch.einsum('bi...,b...->bi', normalized.conj(), residual).real
    identity = torch.eye(4, dtype=gram.dtype, device=gram.device)
    weights = torch.linalg.solve(gram+ridge*identity, right.unsqueeze(-1)).squeeze(-1)/norms
    correction = (components*weights[:, :, None, None, None]).sum(1)
    remaining = residual-(images*weights[:, :, None, None, None]).sum(1)
    ratio = remaining.abs().square().flatten(1).sum(1)/residual.abs().square().flatten(1).sum(1).clamp_min(1e-30)
    return correction, ratio


@torch.no_grad()
def moment_injected_solve(high, low, base, initial, model=None, tolerance=1e-5, maxiter=100):
    context = {}
    def capture(value):
        context.update(value)
    def inject(x, residual):
        reference = base.reference.reference_transport_solve(residual.to(torch.complex64))
        proposal = reference if model is None else model(low.sigma_t, low.sigma_s, residual.to(torch.complex64), reference)
        basis = context['image_basis'].transpose(0, 1)
        correction, _ = combine_moment_space(high.A_op, proposal, residual, basis)
        return correction
    return augmented_fgmres(high.A_op, high.source,
        lambda r: base(r.to(torch.complex64)).to(torch.complex128), initial,
        inject=inject, tolerance=tolerance, maxiter=maxiter, capture_prefix=capture)
