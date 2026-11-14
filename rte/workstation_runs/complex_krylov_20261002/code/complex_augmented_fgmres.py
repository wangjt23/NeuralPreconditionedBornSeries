"""Isolated complex-coefficient companion to rte.augmented_fgmres.

This changes the coefficient field, not the transport operator or preconditioner.
The physical action must be complex-linear; flexible/nonlinear preconditioners
are allowed because every actual returned direction is imaged by that action.
No claim of exact complex-linearity is made for a rounded FP32 preconditioner.

The original initial field, source normalization, one injection after the global
classical-call prefix, iteration budgets and final physical check are retained.
MGS uses Hermitian products; small LS uses a complex SVD pseudoinverse with a
fixed dimension-times-machine-epsilon rank threshold.
An injection inside span(V) is eligible if outside span(AZ). A rejected proposal
does not advance or replace the ordered Arnoldi frontier. Nonconverged Arnoldi
breakdown stops with an explicit failure reason rather than false convergence.

capture_prefix keeps the original payload and additionally retains actual
directions Z, unprojected images HZ, the ordered basis/frontier, small H and the
current cycle's starting field/residual. Tensors remain on-device; callers own
serialization. Capture work is diagnostic and must be billed separately.
"""

import math

import torch


def _small_lstsq(matrix, right):
    """Complex rank-aware small LS; never discard imaginary coefficients."""
    eps = torch.finfo(matrix.real.dtype).eps
    rcond = max(matrix.shape) * eps
    # CUDA's default lstsq is a full-rank-only driver. SVD of only the tiny
    # Hessenberg handles dependent complex columns on both CPU and CUDA.
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    cutoff = rcond * singular.max() if singular.numel() else singular.new_tensor(0.)
    keep = singular > cutoff
    reciprocal = keep.to(singular.dtype) / singular.clamp_min(torch.finfo(singular.dtype).tiny)
    y = vh.mH @ ((u.mH @ right) * reciprocal)
    return y, int(keep.sum())


@torch.no_grad()
def complex_augmented_fgmres(action, rhs, precondition, initial=None, inject=None,
                             inject_after=4, tolerance=1e-5, maxiter=100, restart=25,
                             capture_prefix=None, capture_injection=None):
    """Original core call contract, complex coefficients and optional diagnostics.

    Status retains iterations/classical_calls/injection_calls/accepted/rejected,
    true_residual/failed/trace. Added fields expose rank loss, cycles, breakdown
    and injection event receipts. ``failed`` is decided only by a fresh physical
    original-RHS residual. A complex-linear H is an external entry requirement;
    no expensive admissibility probe is silently added to the solve.
    """
    if maxiter < 1 or restart < 1 or inject_after < 1:
        raise ValueError("positive iteration, restart and prefix budgets required")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("positive finite tolerance required")
    if not rhs.is_complex():
        raise TypeError("complex physical RHS required")
    if not bool(torch.isfinite(rhs).all()):
        raise ValueError("finite physical RHS required")
    source_norm = torch.linalg.vector_norm(rhs)
    stages = []
    if not bool(source_norm > 0):
        return torch.zeros_like(rhs), dict(iterations=0, classical_calls=0,
            injection_calls=0, injection_accepted=0, injection_rejected=0,
            true_residual=0., failed=0, trace=[0.], rank_deficient_steps=0,
            breakdowns=0, breakdown_reason=None, cycles=0, injection_stages=stages)
    initial = torch.zeros_like(rhs) if initial is None else initial.to(dtype=rhs.dtype, device=rhs.device)
    if initial.shape != rhs.shape or not bool(torch.isfinite(initial).all()):
        raise ValueError("initial field must match the finite physical RHS")

    def applied(value):
        result = action(value)
        if result.shape != rhs.shape or not result.is_complex():
            raise ValueError("physical action must return a matching complex field")
        return result.to(dtype=rhs.dtype, device=rhs.device)

    b = rhs - applied(initial)
    correction = torch.zeros_like(rhs)
    residual = b.clone()
    ratio = float(residual.norm() / source_norm)
    trace = [ratio]
    classical_calls = attempted = accepted = rejected = total = 0
    rank_losses = breakdowns = cycles = 0
    breakdown_reason = None
    captured = False

    def record(reason, accepted_attempt, before, proposal, z=None, image=None,
               orthogonal_image=None, column=None, dependence=None,
               candidate_ratio=None, frontier=None, arnoldi=None):
        event = dict(reason=reason, accepted=bool(accepted_attempt), iteration_before=total,
            classical_calls=classical_calls, injection_call=attempted,
            relative_residual_before=before,
            relative_residual_after=candidate_ratio if accepted_attempt else ratio,
            candidate_relative_residual=candidate_ratio)
        stages.append(event)
        if capture_injection is not None:
            capture_injection(dict(**event, proposal=proposal, z=z, image=image,
                orthogonal_image=orthogonal_image,
                image_column=None if column is None else column.clone(),
                dependence_column=None if dependence is None else dependence.clone(),
                frontier_before=frontier,
                arnoldi_basis_before=None if arnoldi is None else torch.stack(arnoldi)))

    while total < maxiter and math.isfinite(ratio) and ratio > tolerance:
        cycles += 1
        cycle_start = correction.clone()
        residual_start = b - applied(cycle_start)
        beta = residual_start.norm()
        if not bool(torch.isfinite(beta)) or not bool(beta > 0):
            breakdown_reason = "nonfinite_cycle_residual" if not bool(torch.isfinite(beta)) else "zero_cycle_residual"
            break
        capacity = min(restart, maxiter - total)
        vectors, directions, images = [residual_start / beta], [], []
        h = rhs.new_zeros(capacity + 1, capacity)
        j = 0
        stop_cycle = False
        while j < capacity and total < maxiter and ratio > tolerance:
            neural = inject is not None and not attempted and classical_calls >= inject_after
            before, frontier = ratio, vectors[j]
            if neural:
                attempted += 1
                proposal = inject(initial + correction, residual)
                if proposal.shape != rhs.shape:
                    raise ValueError("injected direction shape differs from the physical field")
                if not bool(torch.isfinite(proposal).all()) or not bool(proposal.norm() > 0):
                    rejected += 1
                    record("invalid_proposal", False, before, proposal, frontier=frontier, arnoldi=vectors)
                    continue
                z = proposal.to(dtype=rhs.dtype, device=rhs.device) / residual.norm()
            else:
                proposal = None
                z = precondition(frontier)
                classical_calls += 1
                if z.shape != rhs.shape:
                    raise ValueError("preconditioned direction shape differs from the physical field")
                z = z.to(dtype=rhs.dtype, device=rhs.device)
            raw_image = applied(z)
            if not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(raw_image).all()):
                if neural:
                    rejected += 1
                    record("worse_or_nonfinite_residual", False, before, proposal, z, raw_image,
                           frontier=frontier, arnoldi=vectors)
                    continue
                breakdown_reason = "nonfinite_classical_direction"
                stop_cycle = True
                break
            image_norm = raw_image.norm()
            w = raw_image
            column = rhs.new_zeros(j + 2)
            for i, vector in enumerate(vectors):
                column[i] = torch.sum(vector.conj() * w)
                w = w - column[i] * vector
            next_norm = w.norm()
            column[j + 1] = next_norm
            small = torch.cat((h[:j + 2, :j], column[:, None]), dim=1)
            dependence = None
            if neural:
                # Dependence is on AZ, not on V. An exact residual teacher can
                # have zero new Arnoldi frontier and still close the search.
                dependence = column.clone()
                if j:
                    old_q, _ = torch.linalg.qr(h[:j + 1, :j], mode="reduced")
                    dependence[:j + 1] -= old_q @ (old_q.mH @ column[:j + 1])
                if not bool(dependence.norm() > 1e-12 * image_norm.clamp_min(1e-30)):
                    rejected += 1
                    record("dependent_image", False, before, proposal, z, raw_image, w,
                           column, dependence, frontier=frontier, arnoldi=vectors)
                    continue
            right = rhs.new_zeros(j + 2)
            right[0] = beta
            y, rank = _small_lstsq(small, right)
            if rank < j + 1:
                rank_losses += 1
            candidates = torch.stack(directions + [z])
            candidate = cycle_start + (candidates * y.reshape((-1,) + (1,) * rhs.ndim)).sum(0)
            next_residual = b - applied(candidate)
            next_ratio = float(next_residual.norm() / source_norm)
            finite_candidate = bool(torch.isfinite(candidate).all()) and bool(torch.isfinite(next_residual).all())
            if neural and (rank < j + 1 or not finite_candidate or next_ratio > ratio * (1 + 1e-10)):
                rejected += 1
                reason = "dependent_image" if rank < j + 1 else "worse_or_nonfinite_residual"
                record(reason, False, before, proposal, z, raw_image, w, column,
                       dependence, next_ratio, frontier, vectors)
                continue
            if not finite_candidate:
                breakdown_reason = "nonfinite_classical_candidate"
                stop_cycle = True
                break
            if neural:
                record("accepted", True, before, proposal, z, raw_image, w, column,
                       dependence, next_ratio, frontier, vectors)
            h[:j + 2, j] = column
            directions.append(z)
            if capture_prefix is not None and not captured:
                images.append(raw_image)
            correction, residual, ratio = candidate, next_residual, next_ratio
            total += 1
            accepted += int(neural)
            trace.append(ratio)
            # Keep the original absolute classical/accepted-direction cutoff.
            # Complex-rank safeguards change LS handling, not this threshold.
            # Never carry a numerically deficient H into the next injection's
            # AZ-QR or prefix image-basis construction. Its SVD candidate is
            # still physically checked before explicit unsuccessful breakdown.
            has_frontier = rank == j + 1 and bool(next_norm > 1e-14)
            if has_frontier:
                vectors.append(w / next_norm)
            j += 1
            if capture_prefix is not None and not captured and classical_calls >= inject_after:
                captured = True
                basis = None
                if has_frontier:
                    image_q, _ = torch.linalg.qr(h[:j + 1, :j], mode="reduced")
                    basis = torch.einsum("vk,v...->k...", image_q, torch.stack(vectors[:j + 1]))
                capture_prefix(dict(solution=initial + correction, residual=residual,
                    image_basis=basis, iterations=total, classical_calls=classical_calls,
                    relative_residual=ratio, frontier=vectors[j] if has_frontier else None,
                    arnoldi_basis=torch.stack(vectors), hessenberg=h[:j + 1, :j].clone(),
                    directions=torch.stack(directions), images=torch.stack(images),
                    cycle_start_solution=initial + cycle_start, cycle_start_residual=residual_start))
            if ratio <= tolerance:
                break
            if not has_frontier:
                breakdowns += 1
                breakdown_reason = "rank_deficient_arnoldi" if rank < j else "nonconverged_arnoldi_breakdown"
                stop_cycle = True
                break
        if j == 0 or stop_cycle:
            break
    result = initial + correction
    physical_residual = rhs - applied(result)
    true_residual = float(physical_residual.norm() / source_norm)
    failed = int(not bool(torch.isfinite(result).all()) or not math.isfinite(true_residual) or true_residual > tolerance)
    return result, dict(iterations=total, classical_calls=classical_calls,
        injection_calls=attempted, injection_accepted=accepted, injection_rejected=rejected,
        true_residual=true_residual, failed=failed, trace=trace,
        rank_deficient_steps=rank_losses, breakdowns=breakdowns,
        breakdown_reason=breakdown_reason, cycles=cycles, injection_stages=stages)
