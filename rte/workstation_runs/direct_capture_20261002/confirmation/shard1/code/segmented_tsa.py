"""Segmented replay of the unchanged original FP32 TSA recurrence.

Primary contract: chunk=1, capture_first=False. Apply the original first stage
eagerly, including its zero addition; replay ONE captured subsequent stage
d-1 times. Every state transition retains the original operations/order and
rounding. Copies introduce exact same-dtype transport between fixed buffers;
they do not replace a physical defect or change the TSA algebra. Optional
capture_first=True captures that same first stage separately. No chunk4 or
other arithmetic variant is implemented in this frozen revision.

This is a serial fixed-material classical implementation control. It neither
implements the failed94-transform identity nor claims lower setup/runtime
before the coordinated GPU experiment measures those quantities.
"""

import time
from unittest.mock import patch

import torch

from rte.transport_baselines import ClassicalTransportStack


class SegmentedTSA:
    """Original TSA with frozen coefficients and mutable, owned working buffers.

    __call__ accepts the captured CUDA complex64 field shape/device or a native
    complex128 field of that same shape/device. Wide input uses original eager
    base(rhs); complex64 returns fresh owned complex128 storage after replay.
    reference/rte/depth aliases support the unchanged benchmark solver. The
    base's depth is never reassigned to capture an individual stage.

    Original coefficient/cache tensor identity, version and complete layout
    are guarded. Working-buffer content/versions change intentionally; their
    identities, storage/layouts and strong references remain fixed. Every
    application overwrites the input and state before subsequent replay, so
    prior working contents are irrelevant. Concurrent calls are unsupported.
    """

    _OP_TENSORS = (
        "sigma_t", "sigma_s", "directions", "kx", "ky", "phase_matrix", "weights",
    )
    _OP_SETTINGS = (
        "shape", "dx", "n_angles", "float_dtype", "complex_dtype",
        "anisotropy_g", "setup_generation", "eta_scale", "eta_floor",
        "diffusion_floor",
    )

    @staticmethod
    def _validate_base(base, chunk, capture_first):
        if type(base) is not ClassicalTransportStack or base.kind != "tsa":
            raise ValueError("supply the original ClassicalTransportStack with kind='tsa'")
        if type(base.depth) is not int or base.depth < 1:
            raise ValueError("positive original TSA depth is required")
        if type(chunk) is not int or chunk != 1:
            raise ValueError("this revision supports only explicit chunk=1")
        if type(capture_first) is not bool:
            raise ValueError("capture_first must be a boolean")

    @staticmethod
    def _first_stage(base, rhs):
        """Original stage0, retaining zeros_like, relaxation and the zero add."""
        z = torch.zeros_like(rhs)
        defect = rhs
        return z + base.relaxation * base.single(defect)

    @staticmethod
    def _subsequent_stage(base, rhs, z):
        """One original later stage; uses the unchanged original physical A."""
        defect = rhs - base.rte.A_op(z)
        return z + base.relaxation * base.single(defect)

    def __init__(self, base, prototype, chunk=1, capture_first=False):
        self._validate_base(base, chunk, capture_first)
        if prototype.dtype != torch.complex64 or prototype.device.type != "cuda":
            raise ValueError("production capture requires a CUDA complex64 prototype")
        start = time.perf_counter()
        base.prepare()
        static_r = prototype.detach().clone()
        static_z = torch.zeros_like(static_r)
        self.first_graph = self.step_graph = None
        first_output = step_output = None
        actual = {"first": {"fftn": 0, "ifftn": 0}, "step": {"fftn": 0, "ifftn": 0}}
        side = torch.cuda.Stream(device=prototype.device)
        side.wait_stream(torch.cuda.current_stream(prototype.device))
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                static_z.copy_(self._first_stage(base, static_r))
                if base.depth > 1:
                    static_z.copy_(self._subsequent_stage(base, static_r, static_z))
        torch.cuda.current_stream(prototype.device).wait_stream(side)
        torch.cuda.synchronize(prototype.device)

        def capture(which, function):
            fft0, ifft0 = torch.fft.fftn, torch.fft.ifftn

            def fft(*args, **kwargs):
                actual[which]["fftn"] += 1
                return fft0(*args, **kwargs)

            def ifft(*args, **kwargs):
                actual[which]["ifftn"] += 1
                return ifft0(*args, **kwargs)

            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), patch.object(torch.fft, "fftn", fft), patch.object(torch.fft, "ifftn", ifft):
                with torch.cuda.graph(graph):
                    output = function()
            return graph, output

        if capture_first:
            self.first_graph, first_output = capture("first", lambda: self._first_stage(base, static_r))
        if base.depth > 1:
            self.step_graph, step_output = capture("step", lambda: self._subsequent_stage(base, static_r, static_z))
        torch.cuda.synchronize(prototype.device)
        self._bind_captured_state(base, static_r, static_z, step_output, first_output,
                                  chunk=chunk, capture_first=capture_first)
        if actual != self.capture_fft_counts_by_graph:
            raise RuntimeError("capture did not retain the original TSA FFT count")
        self.setup_seconds = time.perf_counter() - start
        self.info["setup_includes_final_host_binding"] = True

    @staticmethod
    def _tensor_signature(value, include_version=True):
        if value is None:
            return None
        if not isinstance(value, torch.Tensor):
            return ("not_tensor", id(value))
        return (id(value), value.data_ptr(), value._version if include_version else None,
                value.dtype, value.device, tuple(value.shape), tuple(value.stride()),
                value.storage_offset(), value.untyped_storage().data_ptr())

    def _operator_tensors(self):
        tensors = []
        for operator in (self.base.rte, self.base.reference):
            tensors.extend(getattr(operator, name, None) for name in self._OP_TENSORS)
        tensors.extend((getattr(self.base.reference, "sigma0", None),
                        getattr(self.base.reference, "_reference_transport_inverse", None)))
        return tuple(tensors)

    def _working_tensors(self):
        return (self.static_r, self.static_z, self.step_output, self.first_output)

    def _signature(self):
        b = self.base
        settings = (id(b), id(b.rte), id(b.reference), id(self.rte), id(self.reference),
                    self.depth, self.chunk, self.capture_first,
                    b.kind, b.depth, b.eta_scale, b.relaxation)
        operator_settings = tuple((name, getattr(operator, name, None))
                                  for operator in (b.rte, b.reference) for name in self._OP_SETTINGS)
        original = tuple(self._tensor_signature(value) for value in self._operator_tensors())
        working = tuple(self._tensor_signature(value, False) for value in self._working_tensors())
        return settings, operator_settings, original, working

    def _bind_captured_state(self, base, static_r, static_z, step_output, first_output=None,
                             chunk=1, capture_first=False):
        """Bind finished capture; CPU tests may bind buffers to a fake replay.

        This does not create graphs or safely recapture an existing instance.
        It never calls the inverse property or refreshes the original setup.
        """
        self._validate_base(base, chunk, capture_first)
        self.base, self.reference, self.rte = base, base.reference, base.rte
        self.depth, self.chunk, self.capture_first = base.depth, chunk, capture_first
        self.static_r, self.static_z = static_r, static_z
        self.step_output, self.first_output = step_output, first_output
        required = [static_r, static_z]
        if self.depth > 1:
            required.append(step_output)
        elif step_output is not None:
            raise ValueError("depth1 must not have a subsequent-stage output")
        if capture_first:
            required.append(first_output)
        elif first_output is not None:
            raise ValueError("eager first-stage mode must not have a captured first output")
        for value in required:
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.complex64
                    or value.shape != static_r.shape or value.device != static_r.device):
                raise ValueError("working buffers must have matching complex64 shape/device")
        if (static_r.ndim != 4 or static_r.shape[1] != base.rte.n_angles
                or tuple(static_r.shape[-2:]) != tuple(base.rte.shape)
                or static_r.shape[0] < 1 or (base.rte.sigma_t.shape[0] != 1
                and base.rte.sigma_t.shape[0] != static_r.shape[0])):
            raise ValueError("prototype must match physical field and material batch")
        storage = [value.untyped_storage().data_ptr() for value in required]
        if len(storage) != len(set(storage)):
            raise ValueError("working inputs/state/outputs must not share storage")
        self._captured_operators = (base, base.rte, base.reference)
        self._captured_tensors = self._operator_tensors()
        self._captured_working = self._working_tensors()
        if self._captured_tensors[-1] is None:
            raise ValueError("prepare the original reference inverse before capture")
        self.signature = self._signature()
        first = 3 if capture_first else 0
        step = 4 if self.depth > 1 else 0
        self.capture_fft_counts_by_graph = {
            "first": {"fftn": first, "ifftn": first},
            "step": {"fftn": step, "ifftn": step},
        }
        self.capture_fft_counts = {"fftn": first + step, "ifftn": first + step}
        self.application_fft_counts = {"fftn": 4 * self.depth - 1, "ifftn": 4 * self.depth - 1}
        self.replay_weighted_fft_counts = dict(self.application_fft_counts)
        self.info = dict(
            chunk=1, capture_first=capture_first, first_stage_eager=not capture_first,
            first_graph_replays_per_application=int(capture_first),
            subsequent_graph_replays_per_application=self.depth - 1,
            original_depth=self.depth, original_depth_reassigned=False,
            first_stage_algorithm_fft_counts={"fftn": 3, "ifftn": 3},
            subsequent_stage_algorithm_fft_counts={"fftn": 4, "ifftn": 4},
            capture_fft_counts_by_graph=self.capture_fft_counts_by_graph,
            capture_fft_counts=self.capture_fft_counts,
            application_fft_counts=self.application_fft_counts,
            same_dtype_copy_calls_per_application=self.depth + 1,
            extra_state_copy_calls_vs_full_graph=self.depth,
            same_dtype_copy_payload_bytes=static_r.numel()*static_r.element_size(),
            serial_application_only=True, original_arithmetic_preserved=True,
            side_stream_warmup_rounds=3,
        )

    def prepare(self):
        """Validate the frozen operator/cache/layout; never silently rebuild."""
        try:
            signature = self._signature()
        except (AttributeError, TypeError) as error:
            raise RuntimeError("captured TSA state changed; reconstruct") from error
        if signature != self.signature:
            raise RuntimeError("captured TSA operator/cache/settings/layout changed; reconstruct")
        return self

    @torch.no_grad()
    def __call__(self, rhs):
        self.prepare()
        if rhs.shape != self.static_r.shape or rhs.device != self.static_r.device:
            raise ValueError("input shape/device must match the capture")
        if rhs.dtype == torch.complex128:
            return self.base(rhs)
        if rhs.dtype != torch.complex64:
            raise ValueError("segmented TSA accepts complex64 or native complex128 input")
        self.static_r.copy_(rhs)
        if self.capture_first:
            self.first_graph.replay()
            first = self.first_output
        else:
            first = self._first_stage(self.base, self.static_r)
        self.static_z.copy_(first)
        for _ in range(self.depth - 1):
            self.step_graph.replay()
            self.static_z.copy_(self.step_output)
        # This conversion allocates fresh storage every time. A retained Z
        # vector cannot alias the mutable state, captured output or later result.
        return self.static_z.to(torch.complex128)


SegmentedClassicalTSA = SegmentedTSA
