"""Fine transport complement around a fixed-size full-angle Schur response.

This isolates a plausible neural target: the material-dependent coarse inverse.
The exact Galerkin/Cholesky version is an essential classical control. No
neural weights or speedup claim are contained in this module.
"""

import torch

from rte.multilevel_neural_transport import prolong_field, restrict_field


class CachedCoarseResponse:
    def __init__(self, coarse, response=None, dtype=torch.complex64):
        self.size, self.parity = coarse.size, coarse.parity
        self.response, self.dtype = response, dtype
        self.inverse = torch.cholesky_inverse(coarse.factor).to(dtype) if response is None else None

    def __call__(self, rhs):
        self.parity.check_current()
        coarse = restrict_field(rhs, self.size)
        flat = coarse.flatten(1).to(self.dtype)
        value = flat@self.inverse.T if self.response is None else self.response(flat)
        value = restrict_field(value.to(rhs.dtype).reshape_as(coarse), self.size)
        return prolong_field(value, self.parity.rte.shape[0])


class CoarseResponseTransport:
    def __init__(self, fine, parity, coarse_response, base, post=False, recover_odd_rhs=True,
                 safeguard=False):
        self.fine, self.parity = fine, parity
        self.coarse_response, self.base, self.post = coarse_response, base, post
        self.recover_odd_rhs = recover_odd_rhs
        self.safeguard = safeguard

    def __call__(self, rhs):
        self.parity.check_current()
        value = self.base(rhs.to(torch.complex64)).to(rhs.dtype)
        defect = rhs-self.fine.A_op(value)
        even = self.coarse_response(self.parity.reduced_rhs(defect))
        # Full elimination (the initial, negative control) resets the fine odd
        # response. Harmonic-only lifting instead preserves the fine complement:
        # Z e = [e, -Co^-1 D e], A0 Z e = [H e, 0] in parity coordinates.
        odd_rhs = defect if self.recover_odd_rhs else torch.zeros_like(defect)
        correction = self.parity.recover(even, odd_rhs)
        if self.safeguard:
            # Optimal real damping in the unchanged physical residual norm.
            # This adds one A application and makes M nonlinear; use FGMRES.
            image = self.fine.A_op(correction)
            numerator = (image.conj()*defect).real.flatten(1).sum(1)
            denominator = image.abs().square().flatten(1).sum(1)
            alpha = torch.where(denominator > 0,
                numerator/denominator.clamp_min(torch.finfo(denominator.dtype).tiny), 0.)
            correction = alpha[:, None, None, None]*correction
        value = value+correction
        if self.post:
            defect = rhs-self.fine.A_op(value)
            value = value+self.base(defect.to(torch.complex64)).to(rhs.dtype)
        return value
