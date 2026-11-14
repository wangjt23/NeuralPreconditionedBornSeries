"""One fixed gain-two classical reference-defect control.

The literal action is g + R0(2*d - Pi3*d), where g is original TSA4 and
d = r - low.A_op(g), entirely in the input dtype. Pi3 uses the frozen v2
capture-safe original angular FFT mask. There is one full reference solve;
the two reference images are never separately formed or combined. No gain
parameter, search, model or reconstructed inverse is introduced.

Stage frozen full_angle_v2/matched_reference.py as matched_reference_v2.py
beside this helper and its unchanged direct-capture dependencies.
"""

import time
from unittest.mock import patch

import torch

from direct_full_graph import _capture_full_action
from graph_classical_v2 import GraphClassicalV2
from matched_reference_v2 import ReferenceDefectAction


class GainTwoReferenceAction(ReferenceDefectAction):
    """Fixed beta=2 with original or same-base direct-segmented TSA4 prefix.

    __call__ returns input dtype. Inherited owned128 creates independent
    solver-owned storage. The original FP32 inverse and mask are promoted,
    not rebuilt, for native complex128 input. beta/mode and all inherited
    material/cache/mask/backend identities, versions and layouts are frozen.
    """

    def __init__(self, base4, prefix_backend):
        started = time.perf_counter()
        self.beta = 2
        super().__init__(base4, prefix_backend, harmonics=1)
        self.kind = "tsa4_gain_two_reference_defect"
        self.mode = "gain_two"
        self.cached_signature = self._current_signature()
        self.info.update(
            mode=self.mode, beta=2, gain_tunable=False,
            formula="g + R0(2*d - Pi3*d)",
            weighting_before_one_reference_solve=True,
            q3_filters_input_only=False, full_angle_input_retained=True,
            input_angular_weighting="2I-Pi3; identity on q3 and gain2 on its complement",
        )
        self.setup_seconds = time.perf_counter() - started

    def _current_signature(self):
        return super()._current_signature(), type(self.beta), self.beta

    @torch.no_grad()
    def __call__(self, rhs):
        self.prepare()
        self._validate_field(rhs)
        g = self.prefix_backend(rhs).to(dtype=rhs.dtype)
        defect = rhs - self.rte.A_op(g)
        projected = self._project_q3_unchecked(defect)
        # The gain is a literal, not a tunable runtime value. Preserve this
        # multiply/subtract order before the single original cached solve.
        weighted_defect = 2 * defect - projected
        correction = self.reference.reference_transport_solve(weighted_defect, harmonics=None)
        return g + correction


class RawGainTwoReferenceGraph(GraphClassicalV2):
    """Whole fixed-gain compound capture with serial, owned output semantics.

    The plain original TSA4 prefix is captured as part of this graph. The
    alternative direct-segmented prefix uses GainTwoReferenceAction.owned128
    outside whole capture. Native128 fallback is the exact gain-two compound.
    """

    def _signature(self):
        try:
            return (super()._signature(), id(self.action), type(self.beta), self.beta,
                    self.harmonics, self.mode, self.base._current_signature(),
                    self.base.cached_signature)
        except (AttributeError, TypeError) as error:
            raise RuntimeError("captured gain-two state changed; reconstruct") from error

    def _bind_captured_state(self, action, static_input, static_output):
        if type(action) is not GainTwoReferenceAction or action.prefix_backend is not action.base4:
            raise ValueError("whole gain-two capture requires the plain original TSA4 prefix")
        action.prepare()
        action._validate_field(static_input)
        if (static_input.dtype != torch.complex64 or static_output.dtype != torch.complex64
                or static_output.shape != static_input.shape
                or static_output.device != static_input.device
                or static_output.untyped_storage().data_ptr() == static_input.untyped_storage().data_ptr()):
            raise ValueError("captured input/output require independent matching complex64 storage")
        self.action, self.beta = action, 2
        self.harmonics, self.mode = action.harmonics, action.mode
        super()._bind_captured_state(action, static_input, static_output)
        self.expected_fft_counts = dict(action.application_fft_counts)
        self.info = dict(action.info, capture_api="raw begin/end",
            capture_error_mode="global", graph_pool="private; no pool hint",
            same_explicit_warmup_and_capture_stream=True,
            serial_application_only=True, native128_fallback="exact gain-two compound",
            setup_includes_final_host_binding=True)

    def prepare(self):
        if self._signature() != self.signature:
            raise RuntimeError("captured gain-two mode/cache/layout changed; reconstruct")
        self.action.prepare()
        return self

    def __init__(self, action, prototype):
        if prototype.dtype != torch.complex64 or prototype.device.type != "cuda":
            raise ValueError("production capture requires a CUDA complex64 prototype")
        if type(action) is not GainTwoReferenceAction or action.prefix_backend is not action.base4:
            raise ValueError("whole gain-two capture requires the plain original TSA4 prefix")
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
            raise RuntimeError("gain-two capture changed expected spatial/angular FFT counts")
        self.setup_seconds = time.perf_counter() - started
        self.info["capture_fft_counts"] = dict(self.capture_fft_counts)

