import sys,json,time,hashlib,importlib.util,os
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=Path('/data/wjt/NeuralPreconditionedBornSeries');OLD=ROOT/'rte/workstation_runs/rte_recycling_20261110_dot_v1/exact_mode_probe.py'
sys.path[:0]=[str(ROOT),str(ROOT/'rte/workstation_runs/layered_background_20261108_dot_v1'),str(ROOT/'rte/workstation_runs/rte_predictability_20261108_dot_v1')]
from qualified_exact_modes import ExactModes
spec=importlib.util.spec_from_file_location('sealed_exact_modes',OLD);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
from screen_structured import medium
from rte.models import PeriodicRTE2D
from rte.mixed_precision_transport import high_precision_copy
@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261110);began=time.perf_counter();out=HERE/'cpu_qualification_01';out.mkdir(exist_ok=False)
 n=32;b=torch.complex(torch.randn(1,12,n,n,dtype=torch.float64),torch.randn(1,12,n,n,dtype=torch.float64));t,s,_=medium(n,1,10/3,'cpu');low=PeriodicRTE2D(shape=(n,n),dx=1/n,n_angles=12,anisotropy_g=.65,eta_scale=0.,eta_floor=0.,device='cpu');low.setup(t,s,b.to(torch.complex64));high=high_precision_copy(low)
 old=mod.ExactModes(low,high,list(range(-8,9)));new=ExactModes(low,high,list(range(-8,9)));u=old.apply(b);v=new.apply(b);error=float((u-v).norm()/u.norm());assert error==0.
 target=new.project(b,list(range(-8,9)));res=float((target-high.A_op(v)+new.perturbation(v,high)).norm()/target.norm());assert res<1e-10
 saved=v.clone();new.apply(b*2);assert torch.equal(v,saved)
 stages=[]
 class Stop(Exception):pass
 def stop(f,k):
  stages.append(k)
  if len(f.factors)==1:raise Stop()
 stopped=False
 try:ExactModes(low,high,[-1,0,1],checkpoint=stop)
 except Stop:stopped=True
 assert stopped and stages==[None,-1]
 report=dict(complete=True,device='cpu',affinity=sorted(os.sched_getaffinity(0)),grid=n,mode_count=17,bitwise_equal_to_sealed_implementation=True,relative_difference=error,original_background_residual=res,output_ownership_pass=True,partial_setup_stop_pass=True,partial_stop_stages=stages,wall_seconds=time.perf_counter()-began,source_hashes={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ['qualify_cpu.py','qualified_exact_modes.py']},sealed_source_sha256=hashlib.sha256(OLD.read_bytes()).hexdigest(),scope='Small CPU algebra/output ownership and early-stop qualification only; no GPU initialization or performance inference.')
 (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
