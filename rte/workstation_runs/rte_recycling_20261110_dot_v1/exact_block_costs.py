"""Separate calibration of assembly and LU costs; not free candidate setup."""
import os,sys,time,json,hashlib
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=Path('/data/wjt/NeuralPreconditionedBornSeries')
sys.path[:0]=[str(ROOT),str(ROOT/'rte/workstation_runs/rte_predictability_20261108_dot_v1'),str(ROOT/'rte/workstation_runs/layered_background_20261108_dot_v1')]
from screen_structured import medium
from layered_inverse import background_from_material
from rte.models import PeriodicRTE2D
from rte.mixed_precision_transport import high_precision_copy
@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);out=HERE/'block_costs_01';out.mkdir(exist_ok=False);start=time.perf_counter()
 total,s,info=medium(128,1,10/3,'cpu');low=PeriodicRTE2D(shape=(128,128),dx=1/128,n_angles=12,anisotropy_g=.65,eta_scale=0.,eta_floor=0.,device='cpu');low.setup(total,s,torch.ones((1,12,128,128),dtype=torch.complex64));high=high_precision_copy(low)
 t0=time.perf_counter();axis,t,s,_=background_from_material(low.sigma_t,low.sigma_s);t=t[0,0,:,0].double();s=s[0,0,:,0].double();assert axis==0
 phase=(high.phase_matrix*high.weights[None]).to(torch.complex128);normal=high.directions[:,0].to(torch.complex128);tangent=high.directions[:,1].to(torch.complex128);nk=high.kx.flatten();tk=high.ky.flatten();I=torch.eye(128,dtype=torch.complex128);D=torch.fft.ifft(1j*nk[:,None]*torch.fft.fft(I,dim=0),dim=0)
 base=torch.kron(torch.diag(normal),D.contiguous())+torch.kron(torch.eye(12,dtype=torch.complex128),torch.diag(t).to(torch.complex128))-torch.kron(phase.contiguous(),torch.diag(s).to(torch.complex128));common=time.perf_counter()-t0;rows=[];factors={}
 for k in range(-8,9):
  t0=time.perf_counter();M=base+torch.kron(torch.diag(1j*tk[k%128]*tangent),I);assembly=time.perf_counter()-t0;t0=time.perf_counter();lu,piv=torch.linalg.lu_factor(M);factor=time.perf_counter()-t0;factors[k]=(lu,piv)
  rows.append(dict(k=k,assembly_seconds=assembly,LU_seconds=factor,total_seconds=assembly+factor,cross_device_copy_seconds=0.,factor_bytes=lu.numel()*lu.element_size()+piv.numel()*piv.element_size()))
 report=dict(complete=True,pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0)),common_assembly_seconds=common,total_assembly_seconds=common+sum(r['assembly_seconds'] for r in rows),total_LU_seconds=sum(r['LU_seconds'] for r in rows),cross_device_copy_seconds=0.,factor_bytes=sum(r['factor_bytes'] for r in rows),wall_seconds=time.perf_counter()-start,rows=rows,source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Separate same-material cost decomposition. Main benchmark retains its actually paid total setup; this calibration is additional research cost, not subtracted from or freely substituted into benchmark.')
 (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
