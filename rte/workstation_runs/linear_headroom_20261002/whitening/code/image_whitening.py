"""Homogeneous full-image whitening for the unchanged angular range oracle.

For B0(k)=H0(k) Q (L-by-q), the cached right action is
W(k)=[B0(k)^H B0(k)]^(-1/2). It preconditions least squares on H Q,
not the angular Galerkin equation Q^H H Q. All physical factors are taken
from the supplied original promoted FP64 operator. No eigenvalue clipping,
physical symmetrization, spatial truncation, or change of residual norm occurs.
"""

import torch


class ImageWhitening:
    """Material-cached complex128 q-channel FFT action and its actual adjoint.

    Supply an AngularGalerkin instance. Cached physical/basis mutations refresh
    through its prepare/signature contract; changed grids or precision are
    rejected there. A single medium can act on multiple RHS. Construction
    checks every frequency's computed normal eigenvalues before inversion.
    """

    def __init__(self, coarse):
        self.coarse = coarse
        self.physical = coarse.physical
        self.n_modes = coarse.n_modes
        self.cached_signature = None
        self.prepare()

    @torch.no_grad()
    def prepare(self):
        self.coarse.prepare()
        signature = self.coarse.cached_signature
        if signature == self.cached_signature:
            return self
        op, q = self.physical, self.coarse.basis
        if q.dtype != torch.complex128:
            raise ValueError("whitening requires the actual complex128 angular basis")
        # This is H0 Q, before angular restriction. In particular it retains
        # the higher angular streaming components discarded by Q^H H0 Q.
        stream = op._stream_symbol(1, torch.complex128, op.sigma_t.device)
        mean_t = op.sigma_t.mean((-2, -1), keepdim=True).to(torch.complex128)
        mean_s = op.sigma_s.mean((-2, -1), keepdim=True).to(torch.complex128)
        phase = op.phase_matrix.to(torch.complex128)
        pq = phase @ (op.weights[:, None] * q)
        diagonal = (stream + mean_t).permute(0, 2, 3, 1)
        b0 = (diagonal[..., None] * q[None, None, None]
              - mean_s[..., None] * pq[None, None, None])
        if not bool(torch.isfinite(b0).all()):
            raise ValueError("homogeneous full-image symbol is nonfinite")
        normal = b0.mH @ b0
        # Hermitian roundoff cleanup applies only to the mathematically
        # Hermitian normal Gram, never to the physical H or its phase.
        hermitian = (normal + normal.mH) / 2
        values, vectors = torch.linalg.eigh(hermitian)
        if not bool(torch.isfinite(values).all() and (values > 0).all()):
            raise ValueError("full-image normal matrix is not numerically positive definite")
        matrix = (vectors * values.rsqrt()[..., None, :]) @ vectors.mH
        if not bool(torch.isfinite(matrix).all()):
            raise ValueError("full-image whitening matrix is nonfinite")
        self.full_reference_symbol = b0
        self.homogeneous_image_symbol = b0
        self.whitening_symbol = matrix
        self.symbol = matrix
        self.matrix = matrix
        self.eigenvalues = values
        self.minimum_reference_eigenvalue = float(values.min())
        self.maximum_reference_eigenvalue = float(values.max())
        self.normal_hermitian_relative_defect = float(
            (normal - normal.mH).norm() / normal.norm().clamp_min(1e-300))
        self.cached_signature = signature
        return self

    def _validate(self, value):
        self.prepare()
        self.coarse._validate_field(value, self.n_modes)

    @staticmethod
    def _spectral(matrix, value):
        spectrum = torch.fft.fftn(value, dim=(-2, -1), norm="backward")
        result = torch.einsum("bhwij,bjhw->bihw", matrix, spectrum)
        return torch.fft.ifftn(result, dim=(-2, -1), norm="backward")

    def solve(self, value):
        """W value; 'solve' denotes the fixed right-preconditioning action."""
        self._validate(value)
        return self._spectral(self.matrix, value)

    action = solve
    __call__ = solve

    def adjoint_solve(self, value):
        """W^H value, using the actual cached adjoint including roundoff."""
        self._validate(value)
        return self._spectral(self.matrix.mH, value)

    adjoint = adjoint_solve


FullImageWhitening = ImageWhitening
