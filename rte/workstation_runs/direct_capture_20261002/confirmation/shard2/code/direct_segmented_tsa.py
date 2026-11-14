"""Direct CUDA capture of the frozen original segmented TSA recurrence.

Stage this file beside the *unchanged* segmented_tsa.py dependency. The primary
API is DirectSegmentedTSA(base, prototype, chunk=1, capture_first=False).
Only capture setup changes: one explicitly warmed nondefault stream captures
one subsequent stage through CUDAGraph.capture_begin/capture_end. Application
arithmetic, copies, guards, native-complex128 fallback and output ownership are
inherited from the executed segmented implementation.

No garbage-collector, allocator, package, CUDA setting or original TSA state
is modified. A capture exception is fatal to construction and must abort the
owned experiment process; there is no fallback from uncertain capture state.
"""

import time
from unittest.mock import patch

import torch

from segmented_tsa import SegmentedTSA


def _capture_subsequent_stage(graph, base, static_r, static_z, stream):
    """Capture the unchanged stage on the supplied warmed nondefault stream.

    Begin/end execute inside the same stream context. No pool hint is passed,
    giving this graph its own private pool. Keep the default global error mode.
    End is attempted after a body exception only if begin succeeded; neither
    path attempts to recover or replay a partially constructed graph.
    """
    with torch.cuda.stream(stream), torch.no_grad():
        graph.capture_begin(capture_error_mode="global")
        try:
            output = SegmentedTSA._subsequent_stage(base, static_r, static_z)
        except BaseException as body_error:
            try:
                graph.capture_end()
            except BaseException as end_error:
                raise body_error from end_error
            raise
        else:
            graph.capture_end()
    return output


class DirectSegmentedTSA(SegmentedTSA):
    """One raw captured stage, with the original first stage kept eager.

    Production capture requires a CUDA complex64 prototype. Calls are serial,
    use the original frozen material and cache, and return fresh complex128
    storage. Native complex128 input invokes the original eager base after all
    inherited frozen-state guards. No stream or pool is shared with other
    graph alternatives. Capture memory and operator/output references remain
    live for this wrapper's lifetime.
    """

    @staticmethod
    def _validate_base(base, chunk, capture_first):
        SegmentedTSA._validate_base(base, chunk, capture_first)
        if capture_first:
            raise ValueError("direct capture supports only capture_first=False")

    def _bind_captured_state(self, base, static_r, static_z, step_output,
                             first_output=None, chunk=1, capture_first=False):
        super()._bind_captured_state(
            base, static_r, static_z, step_output, first_output,
            chunk=chunk, capture_first=capture_first,
        )
        self.info.update(
            capture_api="CUDAGraph.capture_begin/capture_end",
            capture_error_mode="global",
            graph_pool="private; no pool hint supplied",
            same_explicit_warmup_and_capture_stream=True,
            context_manager_capture_used=False,
            garbage_collector_modified=False,
            explicit_gc_collect_calls=0,
            explicit_empty_cache_calls=0,
            persistent_cuda_settings_modified=False,
            capture_failure_fallback=False,
        )

    def __init__(self, base, prototype, chunk=1, capture_first=False):
        self._validate_base(base, chunk, capture_first)
        if prototype.dtype != torch.complex64 or prototype.device.type != "cuda":
            raise ValueError("production capture requires a CUDA complex64 prototype")
        start = time.perf_counter()
        self.first_graph = self.step_graph = None
        actual = {"first": {"fftn": 0, "ifftn": 0},
                  "step": {"fftn": 0, "ifftn": 0}}
        step_output = None
        with torch.cuda.device(prototype.device), torch.no_grad():
            base.prepare()
            static_r = prototype.detach().clone()
            static_z = torch.zeros_like(static_r)
            caller = torch.cuda.current_stream(prototype.device)
            side = torch.cuda.Stream(device=prototype.device)
            # Retain the stream, in addition to inherited graph/output/cache
            # references. Warmup and capture use this exact stream object.
            self.capture_stream = side
            self._retained_capture_streams = (side,)
            side.wait_stream(caller)
            with torch.cuda.stream(side):
                for _ in range(3):
                    static_z.copy_(self._first_stage(base, static_r))
                    if base.depth > 1:
                        static_z.copy_(self._subsequent_stage(base, static_r, static_z))
            caller.wait_stream(side)
            # Finish warmup and dependencies outside capture. Retain an
            # explicit synchronization rather than relying on context-entry
            # GC/cache-clearing side effects for completion.
            torch.cuda.synchronize(prototype.device)

            if base.depth > 1:
                fft0, ifft0 = torch.fft.fftn, torch.fft.ifftn

                def fft(*args, **kwargs):
                    actual["step"]["fftn"] += 1
                    return fft0(*args, **kwargs)

                def ifft(*args, **kwargs):
                    actual["step"]["ifftn"] += 1
                    return ifft0(*args, **kwargs)

                self.step_graph = torch.cuda.CUDAGraph()
                with patch.object(torch.fft, "fftn", fft), patch.object(torch.fft, "ifftn", ifft):
                    step_output = _capture_subsequent_stage(
                        self.step_graph, base, static_r, static_z, side,
                    )
            caller.wait_stream(side)
            torch.cuda.synchronize(prototype.device)
            self._bind_captured_state(
                base, static_r, static_z, step_output,
                chunk=chunk, capture_first=capture_first,
            )
        if actual != self.capture_fft_counts_by_graph:
            raise RuntimeError("direct capture did not retain the original TSA FFT count")
        self.setup_seconds = time.perf_counter() - start
        self.info["setup_includes_final_host_binding"] = True
        self.info["explicit_setup_synchronizations"] = 2


DirectSegmentedClassicalTSA = DirectSegmentedTSA
