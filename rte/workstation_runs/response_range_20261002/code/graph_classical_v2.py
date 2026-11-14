"""Capture the unchanged prepared classical action with strict storage guards.

This wrapper is serial: callers must not overlap applications to its shared
static buffers. A graph result is always a newly owned complex128 tensor, as in
the existing FGMRES deployment wrapper. Native complex128 inputs use the
original eager action. Captured CUDA timings do not establish a learned gain.
"""

import time
from unittest.mock import patch

import torch


class GraphClassicalV2:
    """CUDA graph for an unchanged, prepared complex64 classical action.

    Operator settings, tensor object identities, mutation versions and complete
    layouts are checked before either graph replay or the native128 fallback.
    The captured operators/tensors are held strongly for the graph's lifetime.
    A changed cache is rejected; validation never triggers cache rebuilding.
    """

    _OP_TENSORS = (
        "sigma_t", "sigma_s", "directions", "kx", "ky", "phase_matrix", "weights",
    )
    _OP_SETTINGS = (
        "shape", "dx", "n_angles", "float_dtype", "complex_dtype",
        "anisotropy_g", "setup_generation", "eta_scale", "eta_floor",
        "diffusion_floor",
    )

    def __init__(self, base, prototype):
        if prototype.dtype != torch.complex64 or prototype.device.type != "cuda":
            raise ValueError("capture requires a CUDA complex64 prototype")
        start = time.perf_counter()
        base.prepare()
        static_input = prototype.detach().clone()
        side = torch.cuda.Stream(device=prototype.device)
        side.wait_stream(torch.cuda.current_stream(prototype.device))
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                base(static_input)
        torch.cuda.current_stream(prototype.device).wait_stream(side)
        torch.cuda.synchronize(prototype.device)
        self.graph = torch.cuda.CUDAGraph()
        self.capture_fft_counts = {"fftn": 0, "ifftn": 0}
        original_fftn, original_ifftn = torch.fft.fftn, torch.fft.ifftn

        def counted_fftn(*args, **kwargs):
            self.capture_fft_counts["fftn"] += 1
            return original_fftn(*args, **kwargs)

        def counted_ifftn(*args, **kwargs):
            self.capture_fft_counts["ifftn"] += 1
            return original_ifftn(*args, **kwargs)

        with torch.no_grad(), patch.object(torch.fft, "fftn", counted_fftn), patch.object(
            torch.fft, "ifftn", counted_ifftn
        ):
            with torch.cuda.graph(self.graph):
                static_output = base(static_input)
        torch.cuda.synchronize(prototype.device)
        self.setup_seconds = time.perf_counter() - start
        self._bind_captured_state(base, static_input, static_output)

    @staticmethod
    def _tensor_signature(tensor, include_version=True):
        if tensor is None:
            return None
        if not isinstance(tensor, torch.Tensor):
            return ("not_tensor", id(tensor))
        return (
            id(tensor), tensor.data_ptr(),
            tensor._version if include_version else None,
            tensor.dtype, tensor.device, tuple(tensor.shape),
            tuple(tensor.stride()), tensor.storage_offset(),
        )

    def _operator_tensors(self):
        tensors = []
        for operator in (self.base.rte, self.base.reference):
            tensors.extend(getattr(operator, name, None) for name in self._OP_TENSORS)
        tensors.extend((
            getattr(self.base.reference, "sigma0", None),
            getattr(self.base.reference, "_reference_transport_inverse", None),
        ))
        return tuple(tensors)

    def _signature(self):
        settings = (
            id(self.base), id(self.base.rte), id(self.base.reference),
            id(self.rte), id(self.reference), self.depth,
            self.base.kind, self.base.depth, self.base.eta_scale, self.base.relaxation,
        )
        operator_settings = tuple(
            (name, getattr(operator, name, None))
            for operator in (self.base.rte, self.base.reference)
            for name in self._OP_SETTINGS
        )
        tensors = tuple(self._tensor_signature(tensor) for tensor in self._operator_tensors())
        # copy_ intentionally increments static_input's mutation version on each
        # replay. Its object and complete layout still must remain captured.
        io = tuple(self._tensor_signature(tensor, False) for tensor in (
            self.static_input, self.static_output,
        ))
        return settings, operator_settings, tensors, io

    def _bind_captured_state(self, base, static_input, static_output):
        """Bind one completed capture; also permits CPU tests with a fake replay.

        Production construction calls this only after capture. It does not
        build a CUDA graph or permit re-binding an existing graph safely.
        """
        self.base = base
        self.reference, self.rte, self.depth = base.reference, base.rte, base.depth
        self.static_input, self.static_output = static_input, static_output
        self._captured_operators = (base, base.rte, base.reference)
        self._captured_tensors = self._operator_tensors()
        self._captured_io = (static_input, static_output)
        if self._captured_tensors[-1] is None:
            raise ValueError("the reference inverse must be prepared before capture")
        self.signature = self._signature()

    @torch.no_grad()
    def __call__(self, rhs):
        if self._signature() != self.signature:
            raise RuntimeError("captured classical operator, cache, settings or layout changed")
        if rhs.shape != self.static_input.shape or rhs.device != self.static_input.device:
            raise ValueError("graph input shape and device must match the capture")
        if rhs.dtype == torch.complex128:
            return self.base(rhs)
        if rhs.dtype != torch.complex64:
            raise ValueError("graph action accepts complex64 or native complex128 input")
        self.static_input.copy_(rhs)
        self.graph.replay()
        # A cast from complex64 allocates independent storage on every call.
        # Retained FGMRES basis vectors therefore cannot alias graph buffers.
        return self.static_output.to(torch.complex128)


# The benchmark's historical import name remains convenient in isolated v2.
GraphClassical = GraphClassicalV2
