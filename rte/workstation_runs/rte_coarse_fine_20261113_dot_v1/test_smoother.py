import json,time
from pathlib import Path
import prior_permutation_driver as p
import torch
@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);start=time.perf_counter()
 bank=torch.load(p.ROOT/'rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt',map_location='cpu',weights_only=True,mmap=True)
 q=bank['bank']['q'][0:1].to(torch.complex128);low,high,base,margin,common=p.construct(q)
 actual=p.build_base(low,'tsa8');actual.prepare();assert actual.depth==8 and base.depth==64
 gen=torch.Generator().manual_seed(2026111302)
 r=torch.complex(torch.randn(q.shape,dtype=torch.float32,generator=gen),torch.randn(q.shape,dtype=torch.float32,generator=gen))
 z=torch.zeros_like(r)
 for j in range(8):z=z+base.single(r if j==0 else r-low.A_op(z))
 ref=actual(r);rel=float((z-ref).norm()/ref.norm());assert rel<=1e-6,rel
 result=dict(passed=True,actual_TSA8_depth=actual.depth,unchanged_outer_depth=base.depth,relative_difference=rel,bitwise=torch.equal(z,ref),seconds=time.perf_counter()-start)
 Path(__file__).with_name('SMOOTHER_BRIDGE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
