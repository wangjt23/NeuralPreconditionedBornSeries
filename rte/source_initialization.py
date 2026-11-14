"""One-shot source-conditioned initialization with unchanged physical refinement.

The stopping threshold is relative to the ORIGINAL source, not the smaller
correction RHS. All angular/spatial components remain available to FGMRES.
"""

import torch

from rte.npbs_unet_ablation import fgmres_real_stacked_with_trace


def solve_from_initial(action, rhs, precondition, initial=None, tolerance=1e-5, maxiter=100):
    norm = torch.linalg.vector_norm(rhs)
    if not bool(norm > 0):
        return torch.zeros_like(rhs), dict(iterations=0, true_residual=0., initial_residual=0., failed=0)
    if initial is None:
        initial = torch.zeros_like(rhs)
        defect = rhs
    else:
        defect = rhs-action(initial)
    defect_norm = torch.linalg.vector_norm(defect)
    ratio = float(defect_norm/norm)
    if ratio <= tolerance:
        result, used = initial, 0
    else:
        correction, trace, _, _ = fgmres_real_stacked_with_trace(action, defect, precondition,
            tol=tolerance/ratio, maxiter=maxiter, restart=25, max_restarts=(maxiter-1)//25)
        result, used = initial+correction, len(trace)-1
    residual = float(torch.linalg.vector_norm(rhs-action(result))/norm)
    return result, dict(iterations=used, true_residual=residual, initial_residual=ratio,
                        failed=int(not residual <= tolerance))


def constant_initial(operator, rhs):
    """Cheap analytic constant field matching the integrated physical source."""
    constant = torch.ones_like(rhs)
    image = operator.A_op(constant)
    coefficient = rhs.flatten(1).sum(1)/image.flatten(1).sum(1)
    return coefficient[:, None, None, None]*constant


def characteristic_project(operator, value, modes=1024, transverse_weight=.01):
    """Directional compact teacher control; no learned coefficients involved."""
    n = operator.shape[0]
    wave = operator._stream_symbol(len(value), value.dtype, value.device).imag
    radius = operator.kx.square()+operator.ky.square()
    score = wave.square()+transverse_weight*radius
    indices = score.flatten(-2).argsort(dim=-1, stable=True)[..., :min(modes, n*n)]
    spectrum = torch.fft.fftn(value, dim=(-2, -1)).flatten(-2)
    selected = torch.gather(spectrum, -1, indices.expand_as(spectrum[..., :indices.shape[-1]]))
    kept = torch.zeros_like(spectrum).scatter(-1, indices.expand(len(value), -1, -1), selected)
    return torch.fft.ifftn(kept.reshape_as(value), dim=(-2, -1))
