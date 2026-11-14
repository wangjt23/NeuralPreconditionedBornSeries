"""CPU-only coupled P8 A P8 solve with bounded8 inner iterations, same4RHS.
Not a frozen performance confirmation. Exact stored normal Fourier derivative
replaces the compact derivative; every material factor is charged to the batch.
"""
import os,sys,time,json,hashlib,math,random
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
from rte.augmented_fgmres import augmented_fgmres,projected_direction_loss

@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);began=time.perf_counter();out=HERE/'nested_background_01';out.mkdir(exist_ok=False);rows=[];contexts=[]
 ids=[0,4,7,14];settings=dict(complete=False,pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0)),device='cpu',threads=1,source_ids=ids,grid=128,L=12,g=.65,methods=['native','compact4_inject','compact8_inner','exact8_inner'],repeats=1,interpretation='One-pass development mechanism/cost screen, not timing confirmation. Other research jobs may run on disjoint cores. No GPU timing claim.',neural_training=False,new_teachers=False,heldout9626=False,source_hashes={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ['nested_background_gate.py','exact_mode_probe.py']})
 def save():
  for k,v in [('settings',settings),('rows',rows),('contexts',contexts)]:(out/(k+'.json')).write_text(json.dumps(v,indent=2)+'\n')
 save()
 try:
  saved=torch.load(ROOT/'rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt',map_location='cpu',weights_only=True)
  rhs=[saved['bank']['q'][i:i+1].to(torch.complex128) for i in ids];del saved
  t=time.perf_counter();total,scatter,info=medium(128,1,10/3,'cpu');low=PeriodicRTE2D(shape=(128,128),dx=1/128,n_angles=12,anisotropy_g=.65,eta_scale=0.,eta_floor=0.,device='cpu');low.setup(total,scatter,rhs[0].to(torch.complex64));high=high_precision_copy(low)
  phase=high.phase_matrix.double()*high.weights.double()[None];mu=torch.linalg.eigvalsh((phase+phase.T)/2).max();margin=float((high.sigma_t-high.sigma_s*mu).min());assert margin>0
  physical=time.perf_counter()-t;t=time.perf_counter();base=build_base(low,'tsa64');base.prepare();tsa=time.perf_counter()-t;pre=lambda r:base(r.to(torch.complex64)).to(torch.complex128)
  t=time.perf_counter();compact=LayeredInverse(low,scheme='compact4',nyquist=True,dtype=torch.complex64);compactsetup=time.perf_counter()-t
  t=time.perf_counter();exact=ExactModes(low,high,list(range(-8,9)));exact.matrices.clear();exactsetup=time.perf_counter()-t
  settings.update(physical_setup_seconds=physical,tsa_setup_seconds=tsa,compact_setup_seconds=compactsetup,exact_setup_seconds=exactsetup,exact_factor_bytes=sum(a.numel()*a.element_size()+p.numel()*p.element_size() for a,p in exact.factors.values()),coercivity_margin=margin)
  setups={'native':physical+tsa,'compact4_inject':physical+tsa+compactsetup,'compact8_inner':physical+tsa+compactsetup,'exact8_inner':physical+tsa+exactsetup}
  settings['method_shared_setup_seconds']=setups;save();references={}
  for method in settings['methods']:
   for sid,b in zip(ids,rhs):
    if time.perf_counter()-began>400:raise RuntimeError('400s wall budget')
    bn=float(b.norm());inner_records=[];t=time.perf_counter()
    if method=='native':u,st=solve_from_initial(high.A_op,b,pre,None,maxiter=100);st['classical_calls']=st['iterations']
    else:
     if method=='compact4_inject':inject=lambda x,r:compact(r).to(torch.complex128)
     else:
      def inject(x,r):
       project=lambda v:exact.project(v,list(range(-8,9)))
       ia=lambda v:project(high.A_op(v))
       if method=='exact8_inner':ip=lambda v:project(exact.apply(project(v)))
       else:ip=lambda v:project(compact(project(v)).to(torch.complex128))
       tinner=time.perf_counter()
       z,ist=solve_from_initial(ia,project(r),ip,None,tolerance=1e-4,maxiter=8)
       inner_records.append(dict(**ist,seconds=time.perf_counter()-tinner,requested_tolerance=1e-4,maxiter=8))
       return z
     u,st=augmented_fgmres(high.A_op,b,pre,None,inject=inject,inject_after=1,maxiter=100)
    solve=time.perf_counter()-t;t=time.perf_counter();res=float((b-high.A_op(u)).norm()/bn);d=res*bn/margin;den=float(u.norm())-d;bound=d/den if den>0 else math.inf
    if method=='native':references[sid]=u.clone()
    diff=float((u-references[sid]).norm()/references[sid].norm());accepted=bool(torch.isfinite(u).all()) and not st['failed'] and res<=1e-5 and bound<=1e-3 and diff<=1e-4
    check=time.perf_counter()-t;st.pop('trace',None)
    row=dict(source_id=sid,method=method,inner_records=inner_records,solve_seconds=solve,check_seconds=check,accepted_seconds=solve+check,shared_setup_seconds=setups[method],Q1_seconds=setups[method]+solve+check,status=st,accepted=accepted,true_residual=res,evaluated_relative_bound=bound,relative_field_difference=diff)
    rows.append(row);save();print(json.dumps(dict(method=method,source_id=sid,calls=st['classical_calls'],accepted=accepted,seconds=solve+check)),flush=True)
  settings['complete']=True
 except BaseException as exc:settings['error']=repr(exc);raise
 finally:settings.update(wall_seconds=time.perf_counter()-began,attempts=len(rows),failures=sum(not r['accepted'] for r in rows));save()
if __name__=='__main__':main()
