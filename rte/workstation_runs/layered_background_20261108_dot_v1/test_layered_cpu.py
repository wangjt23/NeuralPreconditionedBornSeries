"""Cheap deterministic CPU algebra tests. Run from the original FWI repository."""
import json, math, time, sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from layered_inverse import CyclicBlockPCR, LayeredInverse
from rte.models import PeriodicRTE2D
from rte.mixed_precision_transport import high_precision_copy

def main():
    torch.set_num_threads(1);torch.manual_seed(81108);began=time.perf_counter()
    rows=[]
    for n in [2,4,8,16]:
        l,k=3,2
        a=4*torch.eye(l,dtype=torch.complex128)[None,None].expand(k,n,l,l).clone()
        a=a+.05*torch.randn_like(a)
        b=.1*torch.randn_like(a);c=.1*torch.randn_like(a)
        solver=CyclicBlockPCR(a,b,c)
        rhs=torch.randn(k,n,l,2,dtype=torch.complex128)
        value=solver(rhs)
        dense=torch.zeros(k,n*l,n*l,dtype=torch.complex128)
        for i in range(n):
            dense[:,i*l:(i+1)*l,i*l:(i+1)*l]+=a[:,i]
            j=(i-1)%n;dense[:,i*l:(i+1)*l,j*l:(j+1)*l]+=b[:,i]
            j=(i+1)%n;dense[:,i*l:(i+1)*l,j*l:(j+1)*l]+=c[:,i]
        exact=torch.linalg.solve(dense,rhs.reshape(k,n*l,2)).reshape_as(rhs)
        error=float((value-exact).norm()/exact.norm())
        assert error<1e-12,(n,error)
        rows.append(dict(test="PCR_vs_dense",n=n,error=error))
    for axis in [0,1]:
        for scheme in ["compact4","upwind"]:
            for nyquist in [False,True]:
                n,l=16,12
                op=PeriodicRTE2D(shape=(n,n),dx=1/n,n_angles=l,anisotropy_g=.65,fdtype=torch.float32)
                x=torch.arange(n)/n
                t=torch.where(x<.375,torch.tensor(4.),torch.tensor(90.))
                s=t-torch.where(x<.375,torch.tensor(.04),torch.tensor(.02))
                tt=t[None,None,:,None].expand(1,1,n,n).clone()
                ss=s[None,None,:,None].expand_as(tt).clone()
                if axis:tt,ss=tt.transpose(-1,-2).contiguous(),ss.transpose(-1,-2).contiguous()
                r=torch.randn(1,l,n,n,dtype=torch.complex128)
                op.setup(tt,ss,r)
                hi=high_precision_copy(op)
                inv=LayeredInverse(op,scheme=scheme,nyquist=nyquist,dtype=torch.complex128,
                                    background=(axis,tt,ss))
                value=inv(r)
                surrogate_error=float((inv.surrogate_action(value)-r).norm()/r.norm())
                assert surrogate_error<2e-10,(axis,scheme,nyquist,surrogate_error)
                physical_defect=float((hi.A_op(value)-r).norm()/r.norm())
                item=dict(test="layered_inverse",axis=axis,scheme=scheme,nyquist=nyquist,
                    surrogate_error=surrogate_error,physical_random_rhs_defect=physical_defect)
                if nyquist:
                    signs=(1-2*(torch.arange(n)%2))[None,None,:,None]
                    special=signs*torch.randn(1,l,1,1,dtype=torch.complex128).expand(1,l,n,n)
                    if axis:special=special.transpose(-1,-2).contiguous()
                    forward_diff=float((hi.A_op(special)-inv.surrogate_action(special)).norm()/hi.A_op(special).norm())
                    back=float((inv(hi.A_op(special))-special).norm()/special.norm())
                    assert forward_diff<2e-12,(axis,scheme,forward_diff)
                    assert back<2e-10,(axis,scheme,back)
                    item.update(nyquist_forward_difference=forward_diff,nyquist_inverse_error=back)
                rows.append(item)
    print(json.dumps(dict(complete=True,cpu_only=True,tests=len(rows),wall_seconds=time.perf_counter()-began,rows=rows),indent=2))
if __name__=="__main__":main()
