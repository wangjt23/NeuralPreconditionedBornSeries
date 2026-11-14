"""Conservative TSA stage optimization for the stored periodic RTE operator.

For G=(S+gamma I)^-1 and A=S+t I-s K, r-A G r equals
(gamma-t) G r+s K G r in exact arithmetic. This eliminates the inner streaming
FFT pair while retaining the reference inverse, physical angular action,
outer physical-defect updates, relaxation, preparation and cache refresh.
Finite precision changes the eliminated inverse/FFT roundoff, so this is not
a bitwise equivalence claim. The original implementation is the parity control.
"""

import torch

from rte.transport_baselines import ClassicalTransportStack


class AlgebraicTSA(ClassicalTransportStack):
    """TSA-only subclass; same arguments/defaults and inherited outer iteration."""

    def __init__(self, rte, kind="tsa", depth=1, eta_scale=0., relaxation=1.):
        if kind != "tsa":
            raise ValueError("AlgebraicTSA supports only kind='tsa'")
        self._stream_guard_signature = None
        self._stream_guard_inputs = None
        super().__init__(rte, kind, depth, eta_scale, relaxation)

    def _check_stored_physics(self):
        op, ref = self.rte, self.reference
        inputs = (op.directions, op.kx, op.ky, op.phase_matrix, op.weights,
                  ref.directions, ref.kx, ref.ky, ref.phase_matrix, ref.weights)
        grids = tuple((item.shape, item.dx, item.n_angles, item.float_dtype,
                       item.complex_dtype, item.anisotropy_g,
                       getattr(item, "setup_generation", 0)) for item in (op, ref))
        signature = (grids, tuple((value.data_ptr(), value._version, value.dtype,
                                  str(value.device), tuple(value.shape)) for value in inputs))
        if signature == self._stream_guard_signature:
            return
        if op.shape != ref.shape or op.n_angles != ref.n_angles:
            raise RuntimeError("physical/reference grid or angle count differs")
        if op.float_dtype != ref.float_dtype or op.complex_dtype != ref.complex_dtype:
            raise RuntimeError("physical/reference configured dtypes differ")
        if op.sigma_t.device != ref.sigma_t.device:
            raise RuntimeError("physical/reference coefficient devices differ")
        # Compare the symbols actually used by A and G, including stored rounded
        # directions and the even-grid Nyquist convention. Equality is exact;
        # no tolerance, symmetrization, or regenerated geometry is substituted.
        stream = op._stream_symbol(1, op.complex_dtype, op.sigma_t.device)
        reference_stream = ref._stream_symbol(1, ref.complex_dtype, ref.sigma_t.device)
        if not torch.equal(stream, reference_stream):
            raise RuntimeError("stored physical/reference streaming symbols differ")
        if not torch.equal(op.phase_matrix, ref.phase_matrix) or not torch.equal(op.weights, ref.weights):
            raise RuntimeError("stored physical/reference phase or quadrature differs")
        self._stream_guard_signature = signature
        # Prevent allocator address reuse from impersonating an earlier tensor.
        self._stream_guard_inputs = inputs

    def single(self, residual):
        self._check_stored_physics()
        z = self.reference.G(residual)
        gamma = self.reference.sigma0.to(dtype=z.dtype, device=z.device)
        sigma_t = self.rte.sigma_t.to(dtype=z.dtype, device=z.device)
        defect = (gamma - sigma_t) * z + self.rte.scattering_op(z)
        return z + self.reference.reference_transport_solve(defect)
