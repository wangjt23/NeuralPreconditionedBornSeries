"""One isolated precision intervention, retaining 94 spatial FFT calls at depth16.

G, local defect, scatter, cached R, and outer physical defect retain input dtype.
Only g+R(defect) and outer sums use complex128. Outer defects see the accumulator
cast to the original input dtype. Output is complex128 (also the deployed B
wrapper's existing output dtype). No original source, cache or solver is edited.
"""
import torch
from algebraic_tsa import AlgebraicTSA

class SumStableAlgebraicTSA(AlgebraicTSA):
    def single(self, residual):
        self._check_stored_physics()
        g=self.reference.G(residual)
        gamma=self.reference.sigma0.to(dtype=g.dtype,device=g.device)
        t=self.rte.sigma_t.to(dtype=g.dtype,device=g.device)
        defect=(gamma-t)*g+self.rte.scattering_op(g)
        correction=self.reference.reference_transport_solve(defect)
        return g.to(torch.complex128)+correction.to(torch.complex128)

    def forward(self,residual):
        self.prepare()
        z=torch.zeros_like(residual,dtype=torch.complex128)
        for stage in range(self.depth):
            defect=residual if stage==0 else residual-self.rte.A_op(z.to(residual.dtype))
            z=z+self.relaxation*self.single(defect)
        return z
