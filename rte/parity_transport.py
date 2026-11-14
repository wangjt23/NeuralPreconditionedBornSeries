"""Even/odd Schur reference for the existing periodic full-angle operator.

Antipodal pairing turns an ideal symmetric collision/streaming operator into
[[Ce, D], [D, Co]], with D*=-D. The Schur action Ce-D Co^-1 D is Hermitian
positive definite. Stored quadrature has small rounding asymmetries: this
module explicitly defines a symmetric reference, never changes the physical A.
"""

import math

import torch


def pcg(action, rhs, precondition, steps=20, tolerance=0.):
    """Independent complex PCG systems along batch, with safe zero-RHS masks."""
    inner = lambda a, b: (a.conj()*b).real.flatten(1).sum(1)
    shape = (-1,)+(1,)*(rhs.ndim-1)
    tiny = torch.finfo(rhs.real.dtype).tiny
    x, residual = torch.zeros_like(rhs), rhs.clone()
    norm2 = inner(rhs, rhs)
    z = precondition(residual)
    direction, rz = z.clone(), inner(residual, z)
    used = 0
    for step in range(steps):
        active = inner(residual, residual) > norm2*tolerance**2
        if tolerance > 0 and not bool(active.any()):
            break
        applied = action(direction)
        denominator = inner(direction, applied)
        alpha = torch.where(active & (denominator > 0), rz/denominator.clamp_min(tiny), 0.)
        x = x+alpha.reshape(shape)*direction
        residual = residual-alpha.reshape(shape)*applied
        z = precondition(residual)
        next_rz = inner(residual, z)
        beta = torch.where(active & (rz > 0), next_rz/rz.clamp_min(tiny), 0.)
        direction = z+beta.reshape(shape)*direction
        rz, used = next_rz, step+1
    true_defect = rhs-action(x)
    relative = (inner(true_defect, true_defect)/norm2.clamp_min(tiny)).sqrt()
    return x, relative, used


class ParitySchur:
    def __init__(self, operator):
        if operator.n_angles % 2:
            raise ValueError('antipodal pairing requires an even angular count')
        if not torch.allclose(operator.weights, operator.weights[0].expand_as(operator.weights)):
            raise ValueError('this reference requires uniform angular weights')
        self.rte, self.half = operator, operator.n_angles//2
        self.generation = operator.setup_generation
        self.inputs = (operator.sigma_t, operator.sigma_s, operator.directions,
                       operator.phase_matrix, operator.weights, operator.kx, operator.ky)
        self.versions = tuple(x._version for x in self.inputs)
        h = self.half
        phase = operator.phase_matrix*operator.weights[None]
        # Ue* P Ue and Uo* P Uo, then Hermitian projection. Cross-parity
        # and same-parity streaming defects are not silently folded into A.
        pe = (phase[:h, :h]+phase[:h, h:]+phase[h:, :h]+phase[h:, h:])/2
        po = (phase[:h, :h]-phase[:h, h:]-phase[h:, :h]+phase[h:, h:])/2
        pe, po = (pe+pe.T)/2, (po+po.T)/2
        t = operator.sigma_t[:, 0, :, :, None, None]
        s = operator.sigma_s[:, 0, :, :, None, None]
        identity = torch.eye(h, dtype=t.dtype, device=t.device)
        ce, co = t*identity-s*pe, t*identity-s*po
        # Cholesky validates positivity of the symmetric surrogate.
        torch.linalg.cholesky(ce)
        self.ce = ce.to(operator.complex_dtype)
        self.co = co.to(operator.complex_dtype)
        self.co_inverse = torch.cholesky_inverse(torch.linalg.cholesky(co)).to(operator.complex_dtype)
        directions = (operator.directions[:h]-operator.directions[h:])/2
        wave = directions[:, 0][None, :, None, None]*operator.kx+directions[:, 1][None, :, None, None]*operator.ky
        self.stream = 1j*wave.to(operator.complex_dtype)
        mean_ce, mean_co = ce.mean((1, 2), keepdim=True), co.mean((1, 2), keepdim=True)
        mean_co_inverse = torch.linalg.inv(mean_co)
        k = wave.permute(0, 2, 3, 1)
        symbol = mean_ce+k[..., :, None]*mean_co_inverse*k[..., None, :]
        self.reference_inverse = torch.cholesky_inverse(torch.linalg.cholesky(symbol)).to(operator.complex_dtype)

    def check_current(self):
        current = (self.rte.sigma_t, self.rte.sigma_s, self.rte.directions,
                   self.rte.phase_matrix, self.rte.weights, self.rte.kx, self.rte.ky)
        if self.rte.setup_generation != self.generation or any(a is not b for a, b in zip(current, self.inputs)) \
                or self.versions != tuple(x._version for x in current):
            raise RuntimeError('material or geometry changed; reconstruct parity reference')

    @staticmethod
    def local(matrix, value):
        return torch.einsum('bxyij,bjxy->bixy', matrix, value)

    def split(self, value):
        return (value[:, :self.half]+value[:, self.half:])/math.sqrt(2), \
               (value[:, :self.half]-value[:, self.half:])/math.sqrt(2)

    @staticmethod
    def join(even, odd):
        return torch.cat((even+odd, even-odd), dim=1)/math.sqrt(2)

    def derivative(self, value):
        return torch.fft.ifftn(self.stream*torch.fft.fftn(value, dim=(-2, -1)), dim=(-2, -1))

    def schur(self, even):
        return self.local(self.ce, even)-self.derivative(self.local(self.co_inverse, self.derivative(even)))

    def reference(self, rhs):
        return torch.fft.ifftn(self.local(self.reference_inverse,
            torch.fft.fftn(rhs, dim=(-2, -1))), dim=(-2, -1))

    def reduced_rhs(self, rhs):
        even, odd = self.split(rhs)
        return even-self.derivative(self.local(self.co_inverse, odd))

    def recover(self, even, rhs):
        _, odd_rhs = self.split(rhs)
        odd = self.local(self.co_inverse, odd_rhs-self.derivative(even))
        return self.join(even, odd)

    def full_reference_action(self, value):
        even, odd = self.split(value)
        return self.join(self.local(self.ce, even)+self.derivative(odd),
                         self.derivative(even)+self.local(self.co, odd))

    def solve(self, rhs, steps=20, tolerance=1e-7, precondition=None):
        self.check_current()
        even, residual, used = pcg(self.schur, self.reduced_rhs(rhs),
            self.reference if precondition is None else precondition, steps, tolerance)
        return self.recover(even, rhs), residual, used


class ParityPreconditioner:
    def __init__(self, parity, steps=8):
        self.parity, self.steps = parity, steps
        self.inner_iterations = 0

    def __call__(self, rhs):
        solution, _, used = self.parity.solve(rhs, self.steps, 1e-7)
        self.inner_iterations += used
        return solution
