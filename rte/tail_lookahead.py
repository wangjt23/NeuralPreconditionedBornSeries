"""Differentiable continuation from the retained classical search-image space.

At the prefix, span(V) = span(Q, r), with Q spanning A Z and r orthogonal
to Q. Thus Q and r suffice to reconstruct the following Arnoldi subspaces
without saving the original large Z/V arrays. All coefficients are real,
matching the deployed real-stacked flexible solver.
"""

import torch


def inner(x, y):
    return (x.conj()*y).real.flatten(1).sum(1)[:, None, None, None]


def norm(x):
    return x.abs().square().flatten(1).sum(1).sqrt()[:, None, None, None]


def orthogonalize(value, basis):
    for vector in basis:
        value = value-vector*inner(vector, value)
    return value


def continuation_loss(action, precondition, proposal, residual, image_basis, steps=2):
    """Squared residual ratios after injection and each subsequent B call.

This is an offline loss, not extra deployment work. Q and residual must come
from an eligible non-breakdown prefix. Near-dependent images are safely ignored
in the loss; the deployed solver retains its own independent rejection checks.
"""
    if steps < 0:
        raise ValueError('nonnegative lookahead length required')
    residual, proposal = residual.to(torch.complex128), proposal.to(torch.complex128)
    denominator = norm(residual).clamp_min(1e-30)
    images = list(image_basis.unbind(1))
    arnoldi = images+[residual/denominator]
    remaining, ratios = residual, []
    for step in range(steps+1):
        z = proposal if step == 0 else precondition(arnoldi[-1]).to(torch.complex128)
        az = action(z)
        threshold = (1e-12*norm(az)).clamp_min(1e-30)
        image = orthogonalize(az, images)
        # A second pass stabilizes the image-space projector used by the loss.
        image = orthogonalize(image, images)
        length = norm(image)
        q = torch.where(length > threshold, image/length.clamp_min(threshold), torch.zeros_like(image))
        remaining = remaining-q*inner(q, remaining)
        ratios.append((norm(remaining)/denominator).flatten().square())
        images.append(q)
        next_vector = orthogonalize(az, arnoldi)
        length = norm(next_vector)
        arnoldi.append(torch.where(length > threshold,
            next_vector/length.clamp_min(threshold), torch.zeros_like(next_vector)))
    return torch.stack(ratios, 1)
