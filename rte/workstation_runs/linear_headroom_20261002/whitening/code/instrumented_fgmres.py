"""Isolated diagnostic clone of rte.augmented_fgmres.augmented_fgmres.

The numerical operations and acceptance decisions match the saved reference.
Optional synchronized profiling perturbs execution: its wall/stage timings are
diagnostic receipts, never substitutes for separately measured unprofiled solve
times. Stage counts count timed blocks (projection/LS may have multiple blocks
per direction). Callback work and Python bookkeeping are outside stage totals.

capture_prefix retains the original payload and adds the ordered Arnoldi basis
[K+1,B,L,N,M], its last frontier [B,L,N,M], and the small Hessenberg matrix.
On happy breakdown no next frontier exists, so frontier/image_basis are None.

capture_injection receives exact physical z and A(z), the actual post-MGS image,
the small dependence column, the ordered frontier before the attempt, and its
accepted/rejected status. An invalid/zero proposal is rejected before A(z), so
z/image are None. Tensors stay on the input device; consumers own copying or
serialization. Status.injection_stages contains scalar receipts for every
attempt, including the retained post-attempt residual and candidate residual
when that candidate was evaluated. The original module is never modified.
"""

from contextlib import contextmanager
import time

import torch

from rte.npbs_unet_ablation import complex_to_real_channels, real_channels_to_complex


STAGES = ("precondition", "action", "neural_callback", "projection_LS", "true_checks")


class StageProfiler:
    def __init__(self, rhs, enabled):
        self.device, self.enabled = rhs.device, enabled
        self.seconds = dict.fromkeys(STAGES, 0.)
        self.counts = dict.fromkeys(STAGES, 0)
        if enabled:
            self.synchronize()
        self.started = time.perf_counter()

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return
        self.synchronize()
        began = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self.seconds[name] += time.perf_counter() - began
            self.counts[name] += 1

    def finish(self, status):
        if self.enabled:
            self.synchronize()
            status["profile"] = dict(stage_seconds=dict(self.seconds), stage_counts=dict(self.counts),
                                     profiled_wall_seconds=time.perf_counter() - self.started,
                                     note="synchronized diagnostic execution; stage counts are timed blocks; capture callbacks/bookkeeping excluded from stage totals; not unprofiled latency")
        return status


@torch.no_grad()
def instrumented_fgmres(action, rhs, precondition, initial=None, inject=None, inject_after=4,
                        tolerance=1e-5, maxiter=100, restart=25, capture_prefix=None,
                        profile=False, capture_injection=None):
    """Reference core signature plus profile and capture_injection diagnostics."""
    if maxiter < 1 or restart < 1 or inject_after < 1:
        raise ValueError('positive iteration, restart and prefix budgets required')
    profiler = StageProfiler(rhs, profile)
    injection_stages = []
    channels, dtype = rhs.shape[1], rhs.dtype
    to_complex = lambda value: real_channels_to_complex(value, channels, dtype)
    to_real = complex_to_real_channels
    ar = lambda value: to_real(action(to_complex(value)))
    source_norm = torch.linalg.vector_norm(rhs)
    if not bool(source_norm > 0):
        return torch.zeros_like(rhs), profiler.finish(dict(iterations=0, classical_calls=0, injection_calls=0,
            injection_accepted=0, injection_rejected=0, true_residual=0., failed=0, trace=[0.],
            injection_stages=injection_stages))
    initial = torch.zeros_like(rhs) if initial is None else initial
    with profiler.stage("true_checks"):
        defect = rhs-action(initial)
    b = to_real(defect)
    correction = torch.zeros_like(b)
    residual = b.clone()
    ratio = float(residual.norm()/source_norm)
    trace = [ratio]
    classical_calls = attempted = accepted = rejected = total = 0
    captured = False

    def record_attempt(reason, accepted_attempt, before, proposal, z=None, image=None,
                       orthogonal_image=None, column=None, dependence=None, candidate_ratio=None,
                       frontier=None, arnoldi=None):
        scalar = dict(reason=reason, accepted=bool(accepted_attempt), iteration_before=total,
                      classical_calls=classical_calls, injection_call=attempted,
                      relative_residual_before=before,
                      relative_residual_after=candidate_ratio if accepted_attempt else ratio,
                      candidate_relative_residual=candidate_ratio)
        injection_stages.append(scalar)
        if capture_injection is not None:
            capture_injection(dict(**scalar, proposal=proposal,
                z=None if z is None else to_complex(z),
                image=None if image is None else to_complex(image),
                orthogonal_image=None if orthogonal_image is None else to_complex(orthogonal_image),
                image_column=None if column is None else column.clone(),
                dependence_column=None if dependence is None else dependence.clone(),
                frontier_before=None if frontier is None else to_complex(frontier),
                arnoldi_basis_before=None if arnoldi is None else torch.stack([to_complex(v) for v in arnoldi])))

    while total < maxiter and ratio > tolerance:
        cycle_start = correction.clone()
        with profiler.stage("true_checks"):
            residual_start = b-ar(cycle_start)
        beta = residual_start.norm()
        if not bool(beta > 0):
            break
        capacity = min(restart, maxiter-total)
        vectors, directions = [residual_start/beta], []
        h = b.new_zeros(capacity+1, capacity)
        j = 0
        while j < capacity and total < maxiter and ratio > tolerance:
            neural = inject is not None and not attempted and classical_calls >= inject_after
            if neural:
                attempted += 1
                before = ratio
                frontier_before = vectors[j]
                with profiler.stage("neural_callback"):
                    proposal = inject(initial+to_complex(correction), to_complex(residual))
                if proposal.shape != rhs.shape:
                    raise ValueError('injected direction shape differs from the physical field')
                if not bool(torch.isfinite(proposal).all()) or not bool(proposal.norm() > 0):
                    rejected += 1
                    record_attempt("invalid_proposal", False, before, proposal,
                                   frontier=frontier_before, arnoldi=vectors)
                    continue
                # Match the scale of a preconditioned unit Arnoldi vector.
                z = to_real(proposal.to(dtype))/residual.norm()
            else:
                with profiler.stage("precondition"):
                    z = to_real(precondition(to_complex(vectors[j])))
                classical_calls += 1
            with profiler.stage("action"):
                w = ar(z)
            # MGS rebinds w, leaving this exact applied image intact for capture.
            raw_image = w if neural else None
            image_norm = w.norm()
            with profiler.stage("projection_LS"):
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
                    dependent = not bool(orthogonal.norm() > 1e-12*image_norm.clamp_min(1e-30))
            if neural and dependent:
                rejected += 1
                record_attempt("dependent_image", False, before, proposal, z, raw_image, w,
                               column, orthogonal, frontier=frontier_before, arnoldi=vectors)
                continue
            with profiler.stage("projection_LS"):
                h[:j+2, j] = column
                right = b.new_zeros(j+2)
                right[0] = beta
                y = torch.linalg.lstsq(h[:j+2, :j+1], right).solution
                candidates = torch.stack(directions+[z])
                candidate = cycle_start+(candidates*y.reshape((-1,)+(1,)*b.ndim)).sum(0)
            with profiler.stage("true_checks"):
                next_residual = b-ar(candidate)
                next_ratio = float(next_residual.norm()/source_norm)
            if neural and (not torch.isfinite(next_residual).all() or next_ratio > ratio*(1+1e-10)):
                h[:, j].zero_()
                rejected += 1
                record_attempt("worse_or_nonfinite_residual", False, before, proposal, z, raw_image, w,
                               column, orthogonal, next_ratio, frontier_before, vectors)
                continue
            if neural:
                # Record against the accepted residual before advancing total or
                # mutating the ordered basis, so iteration/frontier remain exact.
                record_attempt("accepted", True, before, proposal, z, raw_image, w,
                               column, orthogonal, next_ratio, frontier_before, vectors)
            directions.append(z)
            correction, residual, ratio = candidate, next_residual, next_ratio
            total += 1
            accepted += int(neural)
            trace.append(ratio)
            if next_norm > 1e-14:
                vectors.append(w/column[j+1])
            j += 1
            if capture_prefix is not None and not captured and classical_calls >= inject_after:
                captured = True
                # QR of the SMALL Hessenberg gives a basis of search images.
                with profiler.stage("projection_LS"):
                    image_q, _ = torch.linalg.qr(h[:j+1, :j], mode='reduced')
                    v = torch.stack(vectors[:j+1])
                    if len(v) != j+1:
                        # Already-converged happy breakdown: no eligible tail.
                        basis = None
                    else:
                        real_basis = torch.einsum('vk,v...->k...', image_q, v)
                        basis = torch.stack([to_complex(row) for row in real_basis])
                ordered = torch.stack([to_complex(row) for row in vectors[:j+1]])
                frontier = to_complex(vectors[j]) if len(vectors) == j+1 else None
                capture_prefix(dict(solution=initial+to_complex(correction), residual=to_complex(residual),
                    image_basis=basis, iterations=total, classical_calls=classical_calls,
                    relative_residual=ratio, frontier=frontier, arnoldi_basis=ordered,
                    hessenberg=h[:j+1, :j].clone()))
            if ratio <= tolerance:
                break
            if next_norm <= 1e-14:
                # Unsuccessful classical breakdown: mirror the existing solver.
                break
        if j == 0 or (ratio > tolerance and next_norm <= 1e-14):
            break
    result = initial+to_complex(correction)
    with profiler.stage("true_checks"):
        true_residual = float((rhs-action(result)).norm()/source_norm)
    return result, profiler.finish(dict(iterations=total, classical_calls=classical_calls,
        injection_calls=attempted, injection_accepted=accepted, injection_rejected=rejected,
        true_residual=true_residual, failed=int(not true_residual <= tolerance), trace=trace,
        injection_stages=injection_stages))
