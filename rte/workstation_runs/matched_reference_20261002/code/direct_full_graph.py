"""Unchanged full classical action with explicit warmed raw CUDA capture.

Private graph pool, default global capture checking and retained static buffers
are unchanged. This version avoids the context's process-wide collection/cache
flush; it never changes the allocator configuration or patches installed code.
"""
import time
from unittest.mock import patch
import torch
from graph_classical_v2 import GraphClassicalV2

def _capture_full_action(graph,base,static_input,stream):
    with torch.cuda.stream(stream),torch.no_grad():
        graph.capture_begin(capture_error_mode='global')
        try:
            static_output=base(static_input)
        except BaseException as body_error:
            try:
                graph.capture_end()
            except BaseException as end_error:
                raise body_error from end_error
            raise
        else:
            graph.capture_end()
    return static_output

class DirectFullGraph(GraphClassicalV2):
    def __init__(self,base,prototype):
        if prototype.dtype!=torch.complex64 or prototype.device.type!='cuda':
            raise ValueError('CUDA complex64 prototype required')
        began=time.perf_counter();base.prepare();static_input=prototype.detach().clone()
        side=torch.cuda.Stream(device=prototype.device);side.wait_stream(torch.cuda.current_stream(prototype.device))
        with torch.cuda.stream(side),torch.no_grad():
            for _ in range(3):base(static_input)
        torch.cuda.current_stream(prototype.device).wait_stream(side);torch.cuda.synchronize(prototype.device)
        self.graph=torch.cuda.CUDAGraph();self.capture_fft_counts={'fftn':0,'ifftn':0}
        fft0,ifft0=torch.fft.fftn,torch.fft.ifftn
        def fft(*args,**kwargs):
            self.capture_fft_counts['fftn']+=1;return fft0(*args,**kwargs)
        def ifft(*args,**kwargs):
            self.capture_fft_counts['ifftn']+=1;return ifft0(*args,**kwargs)
        with patch.object(torch.fft,'fftn',fft),patch.object(torch.fft,'ifftn',ifft):
            static_output=_capture_full_action(self.graph,base,static_input,side)
        torch.cuda.current_stream(prototype.device).wait_stream(side);torch.cuda.synchronize(prototype.device)
        self._bind_captured_state(base,static_input,static_output)
        self.capture_stream=side
        self.setup_seconds=time.perf_counter()-began
        self.direct_capture=True
