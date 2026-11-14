"""Acceptance-aware offline continuation with the retained ordered frontier.

This diagnostic does not replace the deployed solver or a differentiable loss.
Its real image-space least-squares arithmetic is equivalent in exact arithmetic
on an eligible, non-breakdown prefix. The original last Arnoldi vector is needed
to continue a rejected proposal; span(Q, residual) alone does not identify it.
Batch members are independent. A/B must preserve this batch independence.
"""

import torch


REJECTION_CODES = {0: "accepted", 1: "zero_proposal", 2: "nonfinite",
                   3: "dependent_image", 4: "residual_increase", 5: "already_zero_residual"}


def real_inner(first, second):
    return (first.conj() * second).real.flatten(1).sum(1)


def field_norm(value):
    return torch.linalg.vector_norm(value.flatten(1), dim=1)


def expand(value, field):
    return value.reshape((-1,) + (1,) * (field.ndim - 1))


def orthogonalize(value, basis):
    # Two MGS passes improve the diagnostic projection. Deployment currently
    # uses one pass, so finite-precision trajectories need not be bitwise equal.
    for _ in range(2):
        for vector in basis:
            value = value - expand(real_inner(vector, value), value) * vector
    return value


def finite_rows(value):
    return torch.isfinite(value).flatten(1).all(1)


def unit_or_zero(value, threshold):
    length = field_norm(value)
    valid = torch.isfinite(length) & (length > threshold)
    safe_length = torch.where(valid, length, torch.ones_like(length))
    normalized = value / expand(safe_length, value)
    return torch.where(expand(valid, value), normalized, torch.zeros_like(value)), valid


@torch.no_grad()
def corrected_continuation(action, precondition, proposal, residual, image_basis,
                           original_frontier, steps=2, dependence_rtol=1e-12,
                           increase_rtol=1e-10):
    """Return injection and subsequent classical squared residual ratios.

    Inputs:
      action, precondition: independent batched complex field maps A and B.
      proposal/residual/original_frontier: [batch, *field_shape] tensors.
      image_basis: [batch, k, *field_shape], real-orthonormal old span(AZ).
      original_frontier: the actual unit last Arnoldi vector, retained at prefix.
      steps: number of classical B calls AFTER the attempted injection.

    The prefix residual must be orthogonal to image_basis, and its union with
    that basis must span the old Arnoldi space. The original_frontier lies in
    this space. Those mathematical prefix assumptions are supplied by caller;
    the function checks tensor shape and finite data, not an expensive Gram audit.

    Rejection is per sample: zero/nonfinite/AZ-dependent/increasing proposals
    leave the old residual, image space and ordered frontier intact. A proposal
    in old Arnoldi span but outside AZ is accepted (happy breakdown if it closes
    the residual). Invalid/padding vectors do not become mathematical frontiers.

    Return dictionary contains ratios [batch, steps+1], residuals and classical
    directions, acceptance/rejection code, happy-breakdown flags, classical-call
    counts and per-step breakdowns. Zero padding in recorded batched directions
    denotes no logical call; it is not added to the search space. Inactive rows
    may still receive vectorized B(0), so counts are mathematical, not timing.
    """
    if not isinstance(steps, int) or steps < 0 or dependence_rtol <= 0 or increase_rtol < 0:
        raise ValueError("nonnegative integer steps, positive dependence tolerance and nonnegative increase tolerance required")
    if residual.ndim < 2 or residual.shape[0] < 1:
        raise ValueError("residual must be a nonempty batched field")
    if proposal.shape != residual.shape or original_frontier.shape != residual.shape:
        raise ValueError("proposal, residual and original frontier shapes must agree")
    if image_basis.ndim != residual.ndim + 1:
        raise ValueError("image basis must have shape [batch, k, *field_shape]")
    expected_basis = (residual.shape[0], image_basis.shape[1], *residual.shape[1:])
    if tuple(image_basis.shape) != expected_basis:
        raise ValueError("image basis must have shape [batch, k, *field_shape]")
    residual = residual.to(torch.complex128)
    proposal = proposal.to(device=residual.device, dtype=residual.dtype)
    image_basis = image_basis.to(device=residual.device, dtype=residual.dtype)
    original_frontier = original_frontier.to(device=residual.device, dtype=residual.dtype)
    if not bool(finite_rows(residual).all()) or not bool(torch.isfinite(image_basis).all()):
        raise ValueError("prefix residual and image basis must be finite")
    residual_norm = field_norm(residual)
    if not bool(torch.isfinite(residual_norm).all()):
        raise ValueError("prefix residual norm must be finite")
    nonzero_residual = residual_norm > 0
    original_valid = finite_rows(original_frontier) & (field_norm(original_frontier) > 0)
    if bool((nonzero_residual & ~original_valid).any()):
        raise ValueError("an eligible nonzero prefix needs a finite nonzero original frontier")
    denominator = torch.where(nonzero_residual, residual_norm, torch.ones_like(residual_norm))
    images = list(image_basis.unbind(1))
    arnoldi = images + [residual / expand(denominator, residual)]
    remaining = residual.clone()
    frontier = torch.where(expand(original_valid, original_frontier), original_frontier,
                           torch.zeros_like(original_frontier))
    batch = len(residual)
    calls = torch.zeros(batch, dtype=torch.long, device=residual.device)
    reason = torch.full((batch,), 3, dtype=torch.long, device=residual.device)

    proposal_finite = finite_rows(proposal)
    proposal_length = field_norm(torch.where(expand(proposal_finite, proposal), proposal, torch.zeros_like(proposal)))
    proposal_nonzero = proposal_length > 0
    proposal_safe = torch.where(expand(proposal_finite & proposal_nonzero & nonzero_residual, proposal),
                                proposal, torch.zeros_like(proposal))
    az = action(proposal_safe).to(torch.complex128)
    if az.shape != residual.shape:
        raise ValueError("action changed field shape")
    az_finite = finite_rows(az)
    az = torch.where(expand(az_finite, az), az, torch.zeros_like(az))
    image = orthogonalize(az, images)
    az_norm = field_norm(az)
    q, independent = unit_or_zero(image, dependence_rtol * az_norm)
    admissible = (nonzero_residual & proposal_finite & proposal_nonzero & az_finite &
                  torch.isfinite(az_norm) & (az_norm > 0) & independent)
    candidate = remaining - expand(real_inner(q, remaining), remaining) * q
    candidate_finite = finite_rows(candidate)
    nonincreasing = field_norm(candidate) <= residual_norm * (1 + increase_rtol)
    accepted = admissible & candidate_finite & nonincreasing
    reason = torch.where(~proposal_nonzero, torch.ones_like(reason), reason)
    reason = torch.where(~proposal_finite | ~az_finite | ~torch.isfinite(az_norm),
                         torch.full_like(reason, 2), reason)
    reason = torch.where(admissible & (~candidate_finite | ~nonincreasing),
                         torch.full_like(reason, 4), reason)
    reason = torch.where(accepted, torch.zeros_like(reason), reason)
    reason = torch.where(~nonzero_residual, torch.full_like(reason, 5), reason)
    remaining = torch.where(expand(accepted, remaining), candidate, remaining)
    accepted_q = torch.where(expand(accepted, q), q, torch.zeros_like(q))
    # Zero rows are padding for independent batch members and span nothing.
    if bool(accepted.any()):
        images.append(accepted_q)
    next_vector = orthogonalize(az, arnoldi)
    next_frontier, next_valid = unit_or_zero(next_vector, dependence_rtol * az_norm)
    next_valid &= accepted
    injection_breakdown = accepted & ~next_valid
    if bool(next_valid.any()):
        arnoldi.append(torch.where(expand(next_valid, next_frontier), next_frontier,
                                  torch.zeros_like(next_frontier)))
    frontier = torch.where(expand(accepted, frontier), next_frontier, frontier)
    happy = accepted & ~next_valid & (field_norm(remaining) <= dependence_rtol * denominator)
    # Accepted non-happy breakdown also stops this local Arnoldi continuation;
    # it is exposed as a breakdown rather than silently replacing its frontier.
    active = nonzero_residual & (~accepted | next_valid)
    ratios = [(field_norm(remaining) / denominator).square()]
    residuals = [remaining.clone()]
    directions, active_steps, breakdowns = [], [], []
    for _ in range(steps):
        active_before = active.clone()
        calls += active_before.long()
        input_frontier = torch.where(expand(active_before, frontier), frontier, torch.zeros_like(frontier))
        z = precondition(input_frontier).to(torch.complex128)
        if z.shape != residual.shape:
            raise ValueError("preconditioner changed field shape")
        z_finite = finite_rows(z)
        z = torch.where(expand(z_finite & active_before, z), z, torch.zeros_like(z))
        az_step = action(z).to(torch.complex128)
        if az_step.shape != residual.shape:
            raise ValueError("action changed field shape")
        az_finite = finite_rows(az_step)
        az_step = torch.where(expand(az_finite & active_before, az_step), az_step, torch.zeros_like(az_step))
        az_length = field_norm(az_step)
        useful = orthogonalize(az_step, images)
        q_step, independent = unit_or_zero(useful, dependence_rtol * az_length)
        expand_space = active_before & z_finite & az_finite & torch.isfinite(az_length) & (az_length > 0) & independent
        candidate = remaining - expand(real_inner(q_step, remaining), remaining) * q_step
        expand_space &= finite_rows(candidate) & (field_norm(candidate) <= field_norm(remaining) * (1 + increase_rtol))
        remaining = torch.where(expand(expand_space, remaining), candidate, remaining)
        if bool(expand_space.any()):
            images.append(torch.where(expand(expand_space, q_step), q_step, torch.zeros_like(q_step)))
        next_vector = orthogonalize(az_step, arnoldi)
        next_frontier, next_valid = unit_or_zero(next_vector, dependence_rtol * az_length)
        next_valid &= expand_space
        if bool(next_valid.any()):
            arnoldi.append(torch.where(expand(next_valid, next_frontier), next_frontier,
                                      torch.zeros_like(next_frontier)))
        frontier = torch.where(expand(next_valid, frontier), next_frontier, frontier)
        breakdowns.append(active_before & ~next_valid)
        active_steps.append(active_before)
        active = next_valid
        directions.append(z)
        ratios.append((field_norm(remaining) / denominator).square())
        residuals.append(remaining.clone())
    empty_directions = residual.new_empty((batch, 0, *residual.shape[1:]))
    empty_masks = torch.empty(batch, 0, dtype=torch.bool, device=residual.device)
    return dict(ratios=torch.stack(ratios, 1), residuals=torch.stack(residuals, 1),
                proposal_accepted=accepted, proposal_rejected=nonzero_residual & ~accepted,
                proposal_rejection_code=reason, projected_proposal_image=image,
                happy_breakdown=happy, accepted_frontier_breakdown=injection_breakdown,
                classical_calls=calls,
                classical_directions=torch.stack(directions, 1) if directions else empty_directions,
                classical_step_active=torch.stack(active_steps, 1) if active_steps else empty_masks,
                classical_breakdowns=torch.stack(breakdowns, 1) if breakdowns else empty_masks,
                final_frontier=torch.where(expand(active, frontier), frontier, torch.zeros_like(frontier)),
                final_residual=remaining)
