"""Matched original-dtype TSA4 plus full/q3 cached-reference correction.

These are classical controls. ``harmonics=1`` filters the INPUT angular DFT
modes 0,+1,-1, then uses the original full-angle cached inverse; the response
is not confined to those three angles/modes. Stored uniform quadrature makes
this DFT projector the weighted orthogonal projector for the index-ordered
Fourier basis. No physical directions, phase, inverse or arithmetic is rebuilt.
"""

import time
from unittest.mock import patch

import torch

from direct_full_graph import _capture_full_action
from direct_segmented_tsa import DirectSegmentedTSA
from graph_classical_v2 import GraphClassicalV2
from rte.transport_baselines import ClassicalTransportStack
from segmented_tsa import SegmentedTSA


class ReferenceDefectAction:
    """Literal M4 r + R0 (r - H M4 r), with an optional input-q3 filter.

    The prefix is the same original TSA4 object or its DirectSegmentedTSA
    wrapper. Its result is cast BACK to the incoming dtype before forming the
    original physical defect and adding the correction. Native complex128
    uses the same original operations and the promoted ORIGINAL complex64
    inverse cache, not a separately rebuilt double-precision inverse.

    __call__ returns the input dtype. owned128 supplies a fresh solver-owned
    complex128 output. All applications are serial when the prefix is graphed.
    """

    _OP_TENSORS = GraphClassicalV2._OP_TENSORS
    _OP_SETTINGS = GraphClassicalV2._OP_SETTINGS

    def __init__(self, base4, prefix_backend, harmonics=None):
        started = time.perf_counter()
        if (type(base4) is not ClassicalTransportStack or base4.kind != "tsa"
                or base4.depth != 4 or base4.rte.float_dtype != torch.float32):
            raise ValueError("supply the original FP32 ClassicalTransportStack TSA4")
        if harmonics is not None and (type(harmonics) is not int or harmonics != 1):
            raise ValueError("matched modes are only full (None) and q3 (1)")
        prepared_signature = (
            getattr(base4.rte, "setup_generation", 0),
            tuple((value.data_ptr(), value._version, value.dtype,
                   str(value.device), value.shape)
                  for value in (base4.rte.sigma_t, base4.rte.sigma_s)),
        )
        grid = (base4.rte.shape, base4.rte.dx, base4.rte.n_angles,
                base4.rte.anisotropy_g, base4.rte.float_dtype)
        if base4._grid != grid or base4._signature != prepared_signature:
            raise ValueError("supply the unchanged already-prepared original TSA4")
        if prefix_backend is not base4:
            if (type(prefix_backend) is not DirectSegmentedTSA
                    or prefix_backend.base is not base4):
                raise ValueError("prefix must be the same TSA4 or its DirectSegmentedTSA")
            prefix_backend.prepare()
        self.base4, self.prefix_backend = base4, prefix_backend
        self.rte, self.reference = base4.rte, base4.reference
        self.depth, self.kind = base4.depth, "tsa4_reference_defect"
        self.eta_scale, self.relaxation = base4.eta_scale, base4.relaxation
        self.harmonics = harmonics
        self.mode = "full" if harmonics is None else "q3"
        cache = getattr(self.reference, "_reference_transport_inverse", None)
        if not isinstance(cache, torch.Tensor) or cache.dtype != torch.complex64:
            raise ValueError("prepare and retain the original complex64 reference inverse")
        if self.rte.n_angles < 4:
            raise ValueError("q3 requires distinct original indices 0,+1,-1")
        # These CPU-returning checks occur only before capture, never in the
        # application/prepare path. Account for them in synchronized setup.
        for operator in (self.rte, self.reference):
            weights = operator.weights
            if (weights.ndim != 1 or weights.numel() != operator.n_angles
                    or not bool(torch.isfinite(weights).all())
                    or not bool((weights > 0).all())
                    or not torch.equal(weights, weights[0].expand_as(weights))):
                raise ValueError("legacy q3 equivalence requires actual uniform positive weights")
        self._captured_operators = (base4, self.rte, self.reference, prefix_backend)
        self._captured_tensors = self._operator_tensors()
        self._captured_backend_objects = tuple(
            getattr(prefix_backend, name, None)
            for name in ("first_graph", "step_graph", "capture_stream")
        )
        self.cached_signature = self._current_signature()
        self.application_fft_counts = dict(fftn=17, ifftn=17,
            fft=int(harmonics == 1), ifft=int(harmonics == 1))
        self.info = dict(
            mode=self.mode, harmonics=harmonics,
            prefix_backend="original" if prefix_backend is base4 else "direct_segmented",
            prefix_depth=4, original_dtype_defect_and_addition=True,
            prefix_output_cast_back_to_input_dtype=True,
            reference_inverse_dtype=str(cache.dtype),
            reference_inverse_cache_bytes=cache.numel()*cache.element_size(),
            original_reference_cache_reused=True,
            q3_filters_input_only=True, full_angle_output=True,
            projection="index-DFT; actual stored uniform quadrature",
            application_fft_counts=self.application_fft_counts,
            serial_application_only=prefix_backend is not base4,
            setup_timing="host construction; externally synchronize GPU setup",
        )
        self.setup_seconds = time.perf_counter() - started

    @staticmethod
    def _tensor_signature(value):
        return SegmentedTSA._tensor_signature(value)

    def _operator_tensors(self):
        result = []
        for operator in (self.base4.rte, self.base4.reference):
            result.extend(getattr(operator, name, None) for name in self._OP_TENSORS)
        result.extend((getattr(self.base4.reference, "sigma0", None),
                       getattr(self.base4.reference, "sigma_s_ref", None),
                       getattr(self.base4.reference, "_reference_transport_inverse", None)))
        return tuple(result)

    def _current_signature(self):
        base = self.base4
        settings = (id(base), id(base.rte), id(base.reference), id(self.rte),
                    id(self.reference), id(self.prefix_backend), self.depth, self.kind,
                    self.eta_scale, self.relaxation, self.harmonics, self.mode,
                    base.kind, base.depth, base.eta_scale, base.relaxation,
                    base._grid, base._signature)
        operator_settings = tuple((name, getattr(operator, name, None))
                                  for operator in (base.rte, base.reference)
                                  for name in self._OP_SETTINGS)
        tensors = tuple(self._tensor_signature(value) for value in self._operator_tensors())
        if self.prefix_backend is base:
            backend = None
        else:
            backend = (self.prefix_backend._signature(), self.prefix_backend.signature,
                       tuple(id(getattr(self.prefix_backend, name, None))
                             for name in ("first_graph", "step_graph", "capture_stream")))
        return settings, operator_settings, tensors, backend

    def prepare(self):
        """Reject changed state before the original prefix can refresh caches."""
        try:
            current = self._current_signature()
        except (AttributeError, TypeError) as error:
            raise RuntimeError("matched reference state changed; reconstruct") from error
        if current != self.cached_signature:
            raise RuntimeError("matched mode/operator/cache/backend changed; reconstruct")
        return self

    def _validate_field(self, rhs):
        if rhs.dtype not in (torch.complex64, torch.complex128):
            raise ValueError("matched action accepts complex64 or native complex128")
        if (rhs.ndim != 4 or rhs.shape[1] != self.rte.n_angles
                or tuple(rhs.shape[-2:]) != tuple(self.rte.shape)
                or rhs.device != self.rte.sigma_t.device or rhs.shape[0] < 1
                or (self.rte.sigma_t.shape[0] != 1
                    and self.rte.sigma_t.shape[0] != rhs.shape[0])):
            raise ValueError("field shape/device/batch must match the original physical operator")

    @torch.no_grad()
    def __call__(self, rhs):
        self.prepare()
        self._validate_field(rhs)
        g = self.prefix_backend(rhs).to(dtype=rhs.dtype)
        defect = rhs - self.rte.A_op(g)
        correction = self.reference.reference_transport_solve(defect, harmonics=self.harmonics)
        return g + correction

    @torch.no_grad()
    def owned128(self, rhs):
        # For wide input, g+correction already owns new independent storage.
        return self(rhs).to(torch.complex128)


class RawReferenceDefectGraph(GraphClassicalV2):
    """Whole literal compound capture, with mode and action guards retained.

    Accept only a plain-prefix ReferenceDefectAction; the other portfolio arm
    uses DirectSegmentedTSA plus eager outer correction via action.owned128.
    Native128 fallback calls this exact compound action, not its plain TSA4.
    """

    def _signature(self):
        try:
            return (super()._signature(), id(self.action), self.harmonics, self.mode,
                    self.base._current_signature(), self.base.cached_signature)
        except (AttributeError, TypeError) as error:
            raise RuntimeError("captured matched action state changed; reconstruct") from error

    def _bind_captured_state(self, action, static_input, static_output):
        if type(action) is not ReferenceDefectAction or action.prefix_backend is not action.base4:
            raise ValueError("whole compound capture requires the plain original TSA4 prefix")
        action.prepare()
        action._validate_field(static_input)
        if (static_input.dtype != torch.complex64 or static_output.dtype != torch.complex64
                or static_output.shape != static_input.shape
                or static_output.device != static_input.device
                or static_output.untyped_storage().data_ptr() == static_input.untyped_storage().data_ptr()):
            raise ValueError("captured input/output require independent matching complex64 storage")
        self.action, self.harmonics, self.mode = action, action.harmonics, action.mode
        super()._bind_captured_state(action, static_input, static_output)
        self.expected_fft_counts = dict(action.application_fft_counts)
        self.info = dict(action.info, capture_api="raw begin/end",
            capture_error_mode="global", graph_pool="private; no pool hint",
            same_explicit_warmup_and_capture_stream=True,
            serial_application_only=True, native128_fallback="exact compound action",
            setup_includes_final_host_binding=True)

    def prepare(self):
        if self._signature() != self.signature:
            raise RuntimeError("captured matched action/mode/cache/layout changed; reconstruct")
        self.action.prepare()
        return self

    def __init__(self, action, prototype):
        if prototype.dtype != torch.complex64 or prototype.device.type != "cuda":
            raise ValueError("production capture requires a CUDA complex64 prototype")
        if type(action) is not ReferenceDefectAction or action.prefix_backend is not action.base4:
            raise ValueError("whole compound capture requires the plain original TSA4 prefix")
        started = time.perf_counter()
        with torch.cuda.device(prototype.device), torch.no_grad():
            action.prepare()
            static_input = prototype.detach().clone()
            caller = torch.cuda.current_stream(prototype.device)
            side = torch.cuda.Stream(device=prototype.device)
            self.capture_stream = side
            self._retained_capture_streams = (side,)
            side.wait_stream(caller)
            with torch.cuda.stream(side):
                for _ in range(3):
                    action(static_input)
            caller.wait_stream(side)
            torch.cuda.synchronize(prototype.device)
            self.graph = torch.cuda.CUDAGraph()
            self.capture_fft_counts = dict(fftn=0, ifftn=0, fft=0, ifft=0)
            originals = {name: getattr(torch.fft, name) for name in self.capture_fft_counts}

            def counted(name):
                def apply(*args, **kwargs):
                    self.capture_fft_counts[name] += 1
                    return originals[name](*args, **kwargs)
                return apply

            with patch.object(torch.fft, "fftn", counted("fftn")), \
                    patch.object(torch.fft, "ifftn", counted("ifftn")), \
                    patch.object(torch.fft, "fft", counted("fft")), \
                    patch.object(torch.fft, "ifft", counted("ifft")):
                static_output = _capture_full_action(self.graph, action, static_input, side)
            caller.wait_stream(side)
            torch.cuda.synchronize(prototype.device)
            self._bind_captured_state(action, static_input, static_output)
        if self.capture_fft_counts != self.expected_fft_counts:
            raise RuntimeError("whole compound capture changed expected spatial/angular FFT counts")
        self.setup_seconds = time.perf_counter() - started
        self.info["capture_fft_counts"] = dict(self.capture_fft_counts)
