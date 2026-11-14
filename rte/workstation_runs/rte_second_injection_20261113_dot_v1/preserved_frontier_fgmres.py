"""Real-stacked FGMRES with one optional, residual-conditioned search direction.

The existing Krylov basis is retained. A learned proposal is NOT an accepted
solution, and final convergence is always checked against the original RHS.
The unaugmented arithmetic follows the repository's traced FGMRES (one MGS
pass); additional safeguards apply only to the optional direction.
"""

import torch

from rte.npbs_unet_ablation import complex_to_real_channels, real_channels_to_complex


def projected_direction_loss(action, proposal, residual, image_basis):
    """Independent batched, differentiable optimal one-direction residual.

    image_basis [B,K,L,N,N] is real-orthonormal and spans A Z, NOT all Arnoldi
    vectors. Explicit residual evaluation avoids cancellation in 1-cos^2.
    """
    proposal, residual = proposal.to(torch.complex128), residual.to(torch.complex128)
    w = action(proposal)
    if image_basis is not None and image_basis.shape[1]:
        coefficients = (image_basis.conj()*w[:, None]).real.flatten(2).sum(2)
        w = w-(image_basis*coefficients[:, :, None, None, None]).sum(1)
    denominator = w.abs().square().flatten(1).sum(1).clamp_min(1e-30)
    alpha = (residual.conj()*w).real.flatten(1).sum(1)/denominator
    remaining = residual-alpha[:, None, None, None]*w
    return remaining.abs().square().flatten(1).sum(1)/residual.abs().square().flatten(1).sum(1).clamp_min(1e-30)


@torch.no_grad()
def augmented_fgmres(action, rhs, precondition, initial=None, inject=None, inject_after=4,
                     tolerance=1e-5, maxiter=100, restart=25, capture_prefix=None,
                     inject_schedule=None, observer=None):
    if maxiter < 1 or restart < 1 or inject_after < 1:
        raise ValueError('positive iteration, restart and prefix budgets required')
    schedule = (inject_after,) if inject_schedule is None else tuple(inject_schedule)
    if any(type(v) is not int or v < 1 for v in schedule) or any(a >= b for a, b in zip(schedule, schedule[1:])):
        raise ValueError("injection schedule must contain strictly increasing positive integers")
    channels, dtype = rhs.shape[1], rhs.dtype
    to_complex = lambda value: real_channels_to_complex(value, channels, dtype)
    to_real = complex_to_real_channels
    ar = lambda value: to_real(action(to_complex(value)))
    source_norm = torch.linalg.vector_norm(rhs)
    if not bool(source_norm > 0):
        return torch.zeros_like(rhs), dict(iterations=0, classical_calls=0, injection_calls=0,
            injection_accepted=0, injection_rejected=0, true_residual=0., failed=0, trace=[0.])
    initial = torch.zeros_like(rhs) if initial is None else initial
    defect = rhs-action(initial)
    b = to_real(defect)
    correction = torch.zeros_like(b)
    residual = b.clone()
    ratio = float(residual.norm()/source_norm)
    trace = [ratio]
    classical_calls = attempted = accepted = rejected = total = 0
    captured = False
    pending_classical_vector = None
    while total < maxiter and ratio > tolerance:
        cycle_start = correction.clone()
        residual_start = b-ar(cycle_start)
        beta = residual_start.norm()
        if not bool(beta > 0):
            break
        capacity = min(restart, maxiter-total)
        vectors, directions = [residual_start/beta], []
        h = b.new_zeros(capacity+1, capacity)
        j = 0
        while j < capacity and total < maxiter and ratio > tolerance:
            neural = inject is not None and attempted < len(schedule) and classical_calls >= schedule[attempted]
            if neural:
                # The second injection must not discard the classical input
                # that the original one-shot path was about to precondition.
                if attempted == 1:
                    pending_classical_vector = vectors[j].clone()
                attempted += 1
                proposal = inject(initial+to_complex(correction), to_complex(residual))
                if proposal.shape != rhs.shape:
                    raise ValueError('injected direction shape differs from the physical field')
                if not bool(torch.isfinite(proposal).all()) or not bool(proposal.norm() > 0):
                    rejected += 1
                    if observer is not None:
                        observer(dict(event='injection_rejected', reason='invalid_or_zero', classical_calls=classical_calls, attempts=attempted, residual=to_complex(residual).clone(), ratio=ratio))
                    continue
                # Match the scale of a preconditioned unit Arnoldi vector.
                z = to_real(proposal.to(dtype))/residual.norm()
            else:
                next_input = vectors[j] if pending_classical_vector is None else pending_classical_vector
                pending_classical_vector = None
                z = to_real(precondition(to_complex(next_input)))
                classical_calls += 1
            w = ar(z)
            image_norm = w.norm()
            column = b.new_zeros(j+2)
            for i, vector in enumerate(vectors):
                column[i] = (w*vector).sum()
                w = w-column[i]*vector
            column[j+1] = w.norm()
            next_norm = float(column[j+1])
            if neural:
                # A perfect teacher may be inside span(V), but outside span(AZ).
                # Test dependence on AZ before invoking a full-rank-only LS.
                orthogonal = column.clone()
                if j:
                    image_q, _ = torch.linalg.qr(h[:j+1, :j], mode='reduced')
                    orthogonal[:j+1] -= image_q@(image_q.T@column[:j+1])
                if not bool(orthogonal.norm() > 1e-12*image_norm.clamp_min(1e-30)):
                    rejected += 1
                    if observer is not None:
                        observer(dict(event='injection_rejected', reason='dependent', classical_calls=classical_calls, attempts=attempted, residual=to_complex(residual).clone(), ratio=ratio))
                    continue
            h[:j+2, j] = column
            right = b.new_zeros(j+2)
            right[0] = beta
            y = torch.linalg.lstsq(h[:j+2, :j+1], right).solution
            candidates = torch.stack(directions+[z])
            candidate = cycle_start+(candidates*y.reshape((-1,)+(1,)*b.ndim)).sum(0)
            next_residual = b-ar(candidate)
            next_ratio = float(next_residual.norm()/source_norm)
            if neural and (not torch.isfinite(next_residual).all() or next_ratio > ratio*(1+1e-10)):
                h[:, j].zero_()
                rejected += 1
                if observer is not None:
                    observer(dict(event='injection_rejected', reason='nonfinite_or_residual_increase', classical_calls=classical_calls, attempts=attempted, residual=to_complex(residual).clone(), ratio=ratio))
                continue
            directions.append(z)
            correction, residual, ratio = candidate, next_residual, next_ratio
            total += 1
            accepted += int(neural)
            trace.append(ratio)
            if observer is not None and (neural or classical_calls <= 2):
                observer(dict(event='injection_accepted' if neural else 'classical_accepted', classical_calls=classical_calls, attempts=attempted, residual=to_complex(residual).clone(), ratio=ratio))
            if next_norm > 1e-14:
                vectors.append(w/column[j+1])
            j += 1
            if capture_prefix is not None and not captured and classical_calls >= inject_after:
                captured = True
                # QR of the SMALL Hessenberg gives a basis of search images.
                image_q, _ = torch.linalg.qr(h[:j+1, :j], mode='reduced')
                v = torch.stack(vectors[:j+1])
                if len(v) != j+1:
                    # Already-converged happy breakdown: no eligible tail.
                    basis = None
                else:
                    real_basis = torch.einsum('vk,v...->k...', image_q, v)
                    basis = torch.stack([to_complex(row) for row in real_basis])
                capture_prefix(dict(solution=initial+to_complex(correction), residual=to_complex(residual),
                    image_basis=basis, iterations=total, classical_calls=classical_calls,
                    relative_residual=ratio))
            if ratio <= tolerance:
                break
            if next_norm <= 1e-14:
                # Unsuccessful classical breakdown: mirror the existing solver.
                break
        if j == 0 or (ratio > tolerance and next_norm <= 1e-14):
            break
    result = initial+to_complex(correction)
    true_residual = float((rhs-action(result)).norm()/source_norm)
    return result, dict(iterations=total, classical_calls=classical_calls,
        injection_calls=attempted, injection_accepted=accepted, injection_rejected=rejected,
        true_residual=true_residual, failed=int(not true_residual <= tolerance), trace=trace)
