"""Full-spatial angular Galerkin control of the actual FP64 physical RTE.

Q contains angular-index Fourier columns 0,+1,-1 by default. Every spatial
mode, including the original even-grid Nyquist modes, is retained. Directions,
phase, quadrature and coefficients come from the supplied physical operator;
no ideal geometry, scalar scattering closure or spatial coarse grid is used.
The supplied operator should be high_precision_copy(low), or a native FP64
PeriodicRTE2D. A homogeneous projected inverse is only an inner preconditioner
for heterogeneous Aq. Reduced convergence is not full physical convergence.
"""

import math

import torch


class AngularGalerkin:
    """Fixed-material Aq=Q^H H Q, with original physical H and complex128 inputs.

    Fields have shape (B, number_of_modes, H, W); full fields use n_angles
    channels. A one-material operator may apply to multiple RHS; otherwise the
    RHS batch must equal the material batch. prepare() refreshes cached matrices
    and the homogeneous inverse after stored-buffer mutations/replacements.
    Grid, angle count or configured precision changes require reconstruction.
    """

    def __init__(self, physical, modes=(0, 1, -1)):
        self.physical = physical
        self.modes = tuple(modes)
        if not self.modes or any(type(mode) is not int for mode in self.modes):
            raise ValueError("angular modes must be a nonempty tuple of integers")
        if len({mode % physical.n_angles for mode in self.modes}) != len(self.modes):
            raise ValueError("angular modes must be distinct modulo n_angles")
        self.n_modes = len(self.modes)
        self._grid = (tuple(physical.shape), physical.dx, physical.n_angles,
                      physical.float_dtype, physical.complex_dtype)
        self.cached_signature = None
        self._cached_inputs = None
        self._validate_physical()
        device = physical.sigma_t.device
        angles = 2 * math.pi * torch.arange(physical.n_angles, dtype=torch.float64,
                                           device=device) / physical.n_angles
        indices = torch.tensor(self.modes, dtype=torch.float64, device=device)
        self.basis = torch.exp(1j * angles[:, None] * indices[None]) / math.sqrt(physical.n_angles)
        self.prepare()

    def _validate_physical(self):
        op = self.physical
        if (tuple(op.shape), op.dx, op.n_angles, op.float_dtype, op.complex_dtype) != self._grid:
            raise RuntimeError("physical grid/angles/precision changed; reconstruct AngularGalerkin")
        if op.float_dtype != torch.float64 or op.complex_dtype != torch.complex128:
            raise ValueError("supply the original promoted physical FP64 operator")
        names = ("sigma_t", "sigma_s", "directions", "phase_matrix", "weights", "kx", "ky")
        if any(not hasattr(op, name) for name in names):
            raise ValueError("physical material setup is required")
        if any(getattr(op, name).dtype != torch.float64 for name in names):
            raise ValueError("physical coefficient and geometry buffers must be float64")
        if op.sigma_t.shape != op.sigma_s.shape or tuple(op.sigma_t.shape[-2:]) != tuple(op.shape):
            raise ValueError("physical material shapes differ from the grid")
        if op.sigma_t.device != op.sigma_s.device:
            raise ValueError("physical coefficient devices differ")

    @staticmethod
    def _tensor_signature(value):
        return (value.data_ptr(), value._version, value.dtype, str(value.device),
                tuple(value.shape), tuple(value.stride()))

    @torch.no_grad()
    def prepare(self):
        """Refresh exact stored projections and a per-material mean inverse."""
        self._validate_physical()
        op = self.physical
        inputs = (op.sigma_t, op.sigma_s, op.directions, op.phase_matrix,
                  op.weights, op.kx, op.ky, self.basis)
        signature = (self._grid, getattr(op, "setup_generation", 0), self.modes,
                     tuple(self._tensor_signature(value) for value in inputs))
        if signature == self.cached_signature:
            return self
        device = op.sigma_t.device
        if self.basis.dtype != torch.complex128 or self.basis.device != device:
            raise ValueError("angular basis must retain physical-device complex128 precision")
        q = self.basis
        qh = q.conj().T
        self.gram = qh @ q
        if not bool(torch.linalg.eigvalsh((self.gram + self.gram.conj().T) / 2).min() > 0):
            raise ValueError("angular basis is rank deficient")
        directions = op.directions.to(device=device)
        phase = op.phase_matrix.to(device=device, dtype=torch.complex128)
        weights = op.weights.to(device=device)
        self.mx = qh @ (directions[:, 0, None] * q)
        self.my = qh @ (directions[:, 1, None] * q)
        # Weight before phase, as in the actual H scattering operation.
        self.kq = qh @ (phase @ (weights[:, None] * q))
        # Project H's assembled FP64 stream directly. Mx*kx+My*ky is equivalent
        # in exact arithmetic, but reassociates the actual assembly roundoff.
        stream = op._stream_symbol(1, torch.complex128, device)
        self.streaming_symbol = torch.einsum("li,blhw,lj->bhwij", q.conj(), stream, q)
        self._t = op.sigma_t.to(torch.complex128)
        self._s = op.sigma_s.to(torch.complex128)
        self.mean_t = op.sigma_t.mean((-2, -1), keepdim=True)
        self.mean_s = op.sigma_s.mean((-2, -1), keepdim=True)
        matrix = (self.streaming_symbol
                  + self.mean_t[..., None] * self.gram
                  - self.mean_s[..., None] * self.kq)
        self.reference_inverse = torch.linalg.inv(matrix)
        self._phase, self._weights = phase, weights
        self.cached_signature, self._cached_inputs = signature, inputs
        return self

    def _validate_field(self, value, channels):
        self.prepare()
        op = self.physical
        if value.dtype != torch.complex128:
            raise ValueError("angular Galerkin fields must be complex128")
        if value.ndim != 4 or value.shape[1] != channels or tuple(value.shape[-2:]) != tuple(op.shape):
            raise ValueError("field must have shape (B, channels, physical H, physical W)")
        if value.device != op.sigma_t.device:
            raise ValueError("field and physical material devices differ")
        if value.shape[0] < 1 or (op.sigma_t.shape[0] != 1 and value.shape[0] != op.sigma_t.shape[0]):
            raise ValueError("field batch must equal material batch unless material batch is one")

    @staticmethod
    def _angular(matrix, value):
        return torch.einsum("ij,bjhw->bihw", matrix, value)

    @staticmethod
    def _spectral(matrix, value):
        spectrum = torch.fft.fftn(value, dim=(-2, -1), norm="backward")
        result = torch.einsum("bhwij,bjhw->bihw", matrix, spectrum)
        return torch.fft.ifftn(result, dim=(-2, -1), norm="backward")

    def lift(self, value):
        """Q value, with no spatial restriction or filter."""
        self._validate_field(value, self.n_modes)
        return self._angular(self.basis, value)

    def restrict(self, value):
        """Q^H value under the Euclidean full-field inner product."""
        self._validate_field(value, self.physical.n_angles)
        return self._angular(self.basis.conj().T, value)

    def action(self, value):
        """Aq value = S_q value + t Gram value - s K_q value."""
        self._validate_field(value, self.n_modes)
        return (self._spectral(self.streaming_symbol, value)
                + self._t * self._angular(self.gram, value)
                - self._s * self._angular(self.kq, value))

    __call__ = action

    def adjoint(self, value):
        """Aq^H, retaining the adjoint of each actual stored reduced matrix."""
        self._validate_field(value, self.n_modes)
        return (self._spectral(self.streaming_symbol.conj().transpose(-2, -1), value)
                + self._t * self._angular(self.gram.conj().T, value)
                - self._s * self._angular(self.kq.conj().T, value))

    def full_action(self, value):
        """H Q value through the original physical H (full-angle acceptance)."""
        return self.physical.A_op(self.lift(value))

    def full_adjoint(self, value):
        """(H Q)^H value = Q^H H^H value, with actual phase/quadrature order."""
        self._validate_field(value, self.physical.n_angles)
        # K=phase*diag(weights), hence K^H=diag(weights)*phase^H.
        angular = self._angular(self._phase.conj().T, value)
        scatter_adjoint = self._weights[None, :, None, None] * angular
        physical_adjoint = (-self.physical.streaming_op(value)
                            + self._t * value - self._s * scatter_adjoint)
        return self.restrict(physical_adjoint)

    def reference_solve(self, value):
        """Cached inverse of mean-material Aq; not a heterogeneous inverse."""
        self._validate_field(value, self.n_modes)
        return self._spectral(self.reference_inverse, value)

    def reference_adjoint_solve(self, value):
        """Adjoint of the cached mean-material inverse application."""
        self._validate_field(value, self.n_modes)
        return self._spectral(self.reference_inverse.conj().transpose(-2, -1), value)

    def coercivity_bound(self):
        """Per-material conservative accretivity estimate from stored matrices.

        min_x[t*lambda_min(Herm Gram)-s*lambda_max(Herm Kq)]. A positive value
        bounds Re<v,Aq v>/||v||^2 in exact arithmetic. This ordinary FP64
        computation is an estimate, not a directed-rounding certificate.
        """
        self.prepare()
        gram_min = torch.linalg.eigvalsh((self.gram + self.gram.conj().T) / 2).min()
        scatter_max = torch.linalg.eigvalsh((self.kq + self.kq.conj().T) / 2).max()
        local = self.physical.sigma_t * gram_min - self.physical.sigma_s * scatter_max
        return local.flatten(1).amin(1)


FullSpatialAngularGalerkin = AngularGalerkin
