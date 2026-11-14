"""Three paid, counterbalanced CPU batch comparisons. Frozen prior parameters."""
import os,sys,time,json,hashlib,gc,subprocess
from pathlib import Path
import torch
from exact_mode_probe import ExactModes
HERE=Path(__file__).resolve().parent;ROOT=Path('/data/wjt/NeuralPreconditionedBornSeries')
sys.path[:0]=[str(ROOT),str(ROOT/'rte/workstation_runs/rte_predictability_20261108_dot_v1'),str(ROOT/'rte/workstation_runs/layered_background_20261108_dot_v1')]
from screen_structured import medium
from layered_inverse import LayeredInverse
from rte.models import PeriodicRTE2D
from rte.mixed_precision_transport import high_precision_copy
from rte.train_source_predictor import build_base
from rte.source_initialization import solve_from_initial
from rte.augmented_fgmres import augmented_fgmres

def construct(b):
 t=time.perf_counter();total,scatter,info=medium(128,1,10/3,'cpu');low=PeriodicRTE2D(shape=(128,128),dx=1/128,n_angles=12,anisotropy_g=.65,eta_scale=0.,eta_floor=0.,device='cpu');low.setup(total,scatter,b.to(torch.complex64));high=high_precision_copy(low)
 phase=high.phase_matrix.double()*high.weights.double()[None];mu=torch.linalg.eigvalsh((phase+phase.T)/2).max();margin=float((high.sigma_t-high.sigma_s*mu).min());assert margin>0
 physical=time.perf_counter()-t;t=time.perf_counter();base=build_base(low,'tsa64');base.prepare();tsa=time.perf_counter()-t
 return low,high,base,margin,physical,tsa

@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);began=time.perf_counter();out=HERE/'confirmation_cpu_01';out.mkdir(exist_ok=False);rows=[];batches=[];refs=[]
 ids=[0,4,7,14];orders=[['compact4_inject','exact8_inner'],['exact8_inner','compact4_inject'],['compact4_inject','exact8_inner']]
 settings=dict(complete=False,pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0)),threads=1,device='cpu',orders=orders,source_ids=ids,grid=128,L=12,g=.65,prefix=1,tangent_cut=8,inner_tolerance=1e-4,inner_maxiter=8,outer_tolerance=1e-5,outer_maxiter=100,heldout9626=False,neural_training=False,new_accurate_teachers=False,wall_limit=450,source_hashes={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ['confirm_coupled_cpu.py','exact_mode_probe.py']},scope='Same-instance CPU confirmation only: every paired batch rebuilds and pays all setup. No source/workload expansion, material generalization or GPU result.',production_hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['rte/models.py','rte/source_initialization.py','rte/augmented_fgmres.py','rte/train_source_predictor.py']})
 def save():
  for k,v in [('settings',settings),('rows',rows),('batches',batches),('reference_checks',refs)]:(out/(k+'.json')).write_text(json.dumps(v,indent=2)+'\n')
 save()
 try:
  saved=torch.load(ROOT/'rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt',map_location='cpu',weights_only=True)
  rhs=[saved['bank']['q'][i:i+1].to(torch.complex128) for i in ids]
  settings['rhs_sha256']={str(i):hashlib.sha256(saved['bank']['q'][i:i+1].numpy().tobytes()).hexdigest() for i in ids};del saved
  refbeg=time.perf_counter();low,high,base,margin,physical,tsa=construct(rhs[0]);pre=lambda r:base(r.to(torch.complex64)).to(torch.complex128);reference={}
  for sid,b in zip(ids,rhs):
   t=time.perf_counter();u,st=solve_from_initial(high.A_op,b,pre,None,maxiter=100);assert not st['failed'];reference[sid]=u
   refs.append(dict(source_id=sid,seconds=time.perf_counter()-t,status=st))
  settings['reference_research_seconds']=time.perf_counter()-refbeg
  del pre,base,low,high;gc.collect();save()
  for rep,order in enumerate(orders):
   for position,method in enumerate(order):
    if time.perf_counter()-began>450:raise RuntimeError('450s wall budget')
    batch_start=time.perf_counter();low,high,base,margin,physical,tsa=construct(rhs[0]);pre=lambda r:base(r.to(torch.complex64)).to(torch.complex128)
    t=time.perf_counter()
    if method=='compact4_inject':factor=LayeredInverse(low,scheme='compact4',nyquist=True,dtype=torch.complex64)
    else:factor=ExactModes(low,high,list(range(-8,9)));factor.matrices.clear()
    factor_seconds=time.perf_counter()-t;setup=physical+tsa+factor_seconds;batchrows=[]
    for sid,b in zip(ids,rhs):
     bn=float(b.norm());inner_records=[];t=time.perf_counter()
     if method=='compact4_inject':inject=lambda x,r:factor(r).to(torch.complex128)
     else:
      def inject(x,r):
       project=lambda v:factor.project(v,list(range(-8,9)))
       ia=lambda v:project(high.A_op(v));ip=lambda v:project(factor.apply(project(v)))
       ti=time.perf_counter();z,st=solve_from_initial(ia,project(r),ip,None,tolerance=1e-4,maxiter=8)
       inner_records.append(dict(**st,seconds=time.perf_counter()-ti));return z
     u,st=augmented_fgmres(high.A_op,b,pre,None,inject=inject,inject_after=1,maxiter=100);solve=time.perf_counter()-t;t=time.perf_counter()
     res=float((b-high.A_op(u)).norm()/bn);delta=res*bn/margin;bound=delta/(float(u.norm())-delta);diff=float((u-reference[sid]).norm()/reference[sid].norm())
     accepted=bool(torch.isfinite(u).all()) and not st['failed'] and res<=1e-5 and bound<=1e-3 and diff<=1e-4
     check=time.perf_counter()-t;st.pop('trace',None)
     row=dict(repeat=rep,method=method,method_order=position,source_id=sid,solve_seconds=solve,check_seconds=check,accepted_seconds=solve+check,shared_setup_seconds=setup,status=st,inner_records=inner_records,accepted=accepted,true_residual=res,evaluated_relative_bound=bound,relative_field_difference=diff)
     rows.append(row);batchrows.append(row)
    elapsed=time.perf_counter()-batch_start
    batch=dict(repeat=rep,method=method,method_order=position,physical_setup_seconds=physical,tsa_setup_seconds=tsa,factor_setup_seconds=factor_seconds,setup_seconds=setup,online_four_rhs_seconds=sum(x['accepted_seconds'] for x in batchrows),batch_complete_seconds=setup+sum(x['accepted_seconds'] for x in batchrows),observed_batch_wall_seconds=elapsed,calls=[x['status']['classical_calls'] for x in batchrows],inner_calls=[v['iterations'] for x in batchrows for v in x['inner_records']],accepted=all(x['accepted'] for x in batchrows))
    batches.append(batch);save();print(json.dumps(batch),flush=True)
    del inject,pre,factor,base,low,high,u;gc.collect()
  settings['complete']=True
 except BaseException as exc:settings['error']=repr(exc);raise
 finally:
  settings.update(wall_seconds=time.perf_counter()-began,attempts=len(rows),failures=sum(not r['accepted'] for r in rows),production_hashes_after={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in settings['production_hashes']});save()
if __name__=='__main__':main()
