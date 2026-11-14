import os,sys,time,json,hashlib
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=Path('/data/wjt/NeuralPreconditionedBornSeries');sys.path[:0]=[str(ROOT),str(ROOT/'rte/workstation_runs/rte_predictability_20261108_dot_v1'),str(ROOT/'rte/workstation_runs/layered_background_20261108_dot_v1')]
from angular_modes import AngularBandMode
from screen_structured import medium
from layered_inverse import background_from_material
from rte.models import PeriodicRTE2D
from rte.mixed_precision_transport import high_precision_copy
@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261110);began=time.perf_counter();out=HERE/'zero_columns_01';out.mkdir(exist_ok=False);n=128;l=12;t,s,_=medium(n,1,10/3,'cpu');low=PeriodicRTE2D(shape=(n,n),dx=1/n,n_angles=l,anisotropy_g=.65,eta_scale=0.,eta_floor=0.,device='cpu');low.setup(t,s,torch.zeros(1,l,n,n,dtype=torch.complex64));high=high_precision_copy(low);axis,t,s,_=background_from_material(low.sigma_t,low.sigma_s);assert axis==0;records=[]
 for k in [-8,0,8]:
  block=AngularBandMode(high,t[0,0,:,0].double(),s[0,0,:,0].double(),k);zero=torch.zeros(l*n,2,dtype=torch.complex128);z,st=block.repaired(zero);assert torch.equal(z,zero) and st['success'] and st['corrections']==0
  q=torch.complex(torch.randn(l*n,1,dtype=torch.float64),torch.randn(l*n,1,dtype=torch.float64));mix=torch.cat([zero[:,:1],q,1e-200*q,zero[:,:1]],dim=1);x,ms=block.repaired(mix);assert ms['success'] and ms['zero_columns']==2 and torch.equal(x[:,[0,3]],zero)
  for col in [1,2]:
   scale=mix[:,col].abs().max();res=float(((mix[:,col]-block.original(x[:,col:col+1])[:,0])/scale).norm()/(mix[:,col]/scale).norm());assert res<=1e-10
  records.append(dict(k=k,all_zero=st,mixed=ms,zero_output_exact=True,tiny_nonzero_scale=1e-200,tiny_nonzero_not_discarded=bool(x[:,2].abs().max()>0)))
 r=dict(complete=True,pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0)),device='cpu',records=records,wall_seconds=time.perf_counter()-began,source_hashes={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ['qualify_zero_columns.py','angular_modes.py']},scope='Strict zero columns and mixed zero/nonzero/tiny-nonzero columns; nonzero relative target remains1e-10 without absolute floors.')
 (out/'results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
