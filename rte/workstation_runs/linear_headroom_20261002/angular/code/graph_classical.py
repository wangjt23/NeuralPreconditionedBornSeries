"""Exact original prepared classical action; implementation control, not ML."""
import time
from unittest.mock import patch
import torch

class GraphClassical:
    def __init__(self,base,prototype):
        if prototype.dtype!=torch.complex64 or prototype.device.type!='cuda':
            raise ValueError('prototype must be CUDA complex64')
        start=time.perf_counter();base.prepare();self.base=base
        self.reference,self.rte,self.depth=base.reference,base.rte,base.depth
        self.static_input=prototype.detach().clone()
        side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):base(self.static_input)
        torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph();self.capture_fft_counts={'fftn':0,'ifftn':0}
        fft0,ifft0=torch.fft.fftn,torch.fft.ifftn
        def fft(*a,**k):self.capture_fft_counts['fftn']+=1;return fft0(*a,**k)
        def ifft(*a,**k):self.capture_fft_counts['ifftn']+=1;return ifft0(*a,**k)
        with patch('torch.fft.fftn',fft),patch('torch.fft.ifftn',ifft):
            with torch.cuda.graph(self.graph):self.static_output=base(self.static_input)
        torch.cuda.synchronize();self.setup_seconds=time.perf_counter()-start
        self.signature=self._signature()

    def _signature(self):
        b=self.base;objects=(b.rte,b.reference);tensors=[]
        for op in objects:
            tensors.extend(getattr(op,n) for n in ('sigma_t','sigma_s','directions','kx','ky','phase_matrix','weights'))
        tensors.extend((b.reference.sigma0,b.reference.reference_transport_inverse))
        return ((b.kind,b.depth,b.eta_scale,b.relaxation),
                tuple((x.shape,x.dx,x.n_angles,x.float_dtype,x.anisotropy_g,x.setup_generation) for x in objects),
                tuple((x.data_ptr(),x._version,x.dtype,str(x.device),tuple(x.shape)) for x in tensors))

    @torch.no_grad()
    def __call__(self,rhs):
        if self._signature()!=self.signature:raise RuntimeError('operator/cache changed; recapture required')
        if rhs.shape!=self.static_input.shape or rhs.device!=self.static_input.device:raise ValueError('capture shape/device differs')
        if rhs.dtype!=torch.complex64:return self.base(rhs)
        self.static_input.copy_(rhs);self.graph.replay()
        # Fresh owned storage, exactly the existing deployment output cast.
        return self.static_output.to(torch.complex128)
