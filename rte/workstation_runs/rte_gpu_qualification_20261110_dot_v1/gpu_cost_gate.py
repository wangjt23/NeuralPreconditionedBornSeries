"""Bounded same-machine GPU setup/cost gate. Launch only in coordinated window."""
import os,sys,time,json,hashlib,subprocess,gc,math
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=Path('/data/wjt/NeuralPreconditionedBornSeries')
sys.path[:0]=[str(ROOT),str(ROOT/'rte/workstation_runs/layered_background_20261108_dot_v1'),str(ROOT/'rte/workstation_runs/rte_predictability_20261108_dot_v1'),str(ROOT/'rte/workstation_runs/direct_capture_20261002/confirmation/shard0/code')]
from qualified_exact_modes import ExactModes
from screen_structured import medium
from layered_inverse import LayeredInverse,CapturedAction
from direct_segmented_tsa import DirectSegmentedTSA
from rte.models import PeriodicRTE2D
from rte.mixed_precision_transport import high_precision_copy
from rte.train_source_predictor import build_base
from rte.source_initialization import solve_from_initial
from rte.augmented_fgmres import augmented_fgmres

def sync():torch.cuda.synchronize()
def guard():
 s=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
 other=[int(x.strip()) for x in s.splitlines() if x.strip().isdigit() and int(x.strip())!=os.getpid()]
 if other:raise RuntimeError('GPU competitor detected; stopping own qualification: '+str(other))
class SetupGateStop(Exception):pass

def construct(raw):
 sync();t=time.perf_counter();rhs=[x.cuda().to(torch.complex128) for x in raw];sync();copy=time.perf_counter()-t
 t=time.perf_counter();total,scatter,_=medium(128,1,10/3,'cuda');low=PeriodicRTE2D(shape=(128,128),dx=1/128,n_angles=12,anisotropy_g=.65,eta_scale=0.,eta_floor=0.,device='cuda');low.setup(total,scatter,rhs[0].to(torch.complex64));high=high_precision_copy(low)
 phase=high.phase_matrix.double()*high.weights.double()[None];mu=torch.linalg.eigvalsh((phase+phase.T)/2).max();margin=float((high.sigma_t-high.sigma_s*mu).min());assert margin>0;sync();physical=time.perf_counter()-t
 t=time.perf_counter();base=build_base(low,'tsa64');tsa=DirectSegmentedTSA(base,low.source,chunk=1,capture_first=False);sync();presec=time.perf_counter()-t
 pre=lambda r:tsa(r.to(torch.complex64)).to(torch.complex128)
 return rhs,low,high,base,tsa,pre,margin,dict(source_H2D_seconds=copy,source_H2D_bytes=sum(x.numel()*x.element_size() for x in raw),physical_acceptance_seconds=physical,TSA64_setup_seconds=presec,base_setup_seconds=copy+physical+presec)

@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);guard();out=HERE/'gpu_gate_01';out.mkdir(exist_ok=False);began=time.perf_counter();rows=[];refs=[];settings=dict(complete=False,pid=os.getpid(),device='cuda',threads=1,CPU_affinity=sorted(os.sched_getaffinity(0)),source_ids=[0,4,7,14],grid=128,L=12,g=.65,prefix=1,tangent_cut=8,inner_tolerance=1e-4,inner_maxiter=8,outer_tolerance=1e-5,outer_maxiter=100,neural_training=False,new_accurate_teachers=False,heldout9626=False,wall_limit=180,source_hashes={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ['gpu_cost_gate.py','qualified_exact_modes.py']},scope='Single bounded GPU qualification against actual same-machine captured compact4+DirectSegmentedTSA64. Full setup charged; partial setup early-stop is a lower bound only.')
 def save():
  for k,v in [('settings',settings),('rows',rows),('reference_checks',refs)]:(out/(k+'.json')).write_text(json.dumps(v,indent=2)+'\n')
 save()
 try:
  assert json.loads((HERE/'cpu_qualification_01/results.json').read_text())['complete']
  saved=torch.load(ROOT/'rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt',map_location='cpu',weights_only=True);raw=[saved['bank']['q'][i:i+1].clone() for i in settings['source_ids']];del saved
  settings['rhs_sha256']={str(i):hashlib.sha256(x.numpy().tobytes()).hexdigest() for i,x in zip(settings['source_ids'],raw)}
  old=json.loads((ROOT/'rte/workstation_runs/rte_recycling_20261110_dot_v1/confirmation_cpu_01/settings.json').read_text());assert settings['rhs_sha256']==old['rhs_sha256']
  settings['production_hashes']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in old['production_hashes']};assert settings['production_hashes']==old['production_hashes']
  t=time.perf_counter();torch.cuda.init();sync();settings['CUDA_context_initialization_seconds']=time.perf_counter()-t;settings['GPU']=torch.cuda.get_device_name();torch.cuda.reset_peak_memory_stats()
  rhs,low,high,base,tsa,pre,margin,common=construct(raw);material=(low.sigma_t.clone(),low.sigma_s.clone());reference={}
  t=time.perf_counter()
  for sid,b in zip(settings['source_ids'],rhs):
   u,st=solve_from_initial(high.A_op,b,pre,None,maxiter=100);assert not st['failed'];reference[sid]=u;refs.append(dict(source_id=sid,status=st))
  sync();settings['reference_research_seconds']=time.perf_counter()-t;guard()
  t=time.perf_counter();factor=LayeredInverse(low,scheme='compact4',nyquist=True,dtype=torch.complex64);sync();factorsec=time.perf_counter()-t
  t=time.perf_counter();compact=CapturedAction(lambda r:factor(r).to(torch.complex128),rhs[0]);sync();capturesec=time.perf_counter()-t;assert compact.relative_capture_difference<1e-12
  compactsetup=common['base_setup_seconds']+factorsec+capturesec;settings['compact4_setup']=dict(**common,factor_seconds=factorsec,capture_seconds=capturesec,total_setup_seconds=compactsetup,capture_error=compact.relative_capture_difference)
  def solve_batch(name,action,setup):
   records=[]
   for sid,b in zip(settings['source_ids'],rhs):
    if time.perf_counter()-began>180:raise RuntimeError('wall budget')
    inner=[];bn=float(b.norm());sync();t=time.perf_counter()
    if name=='compact4_inject':inject=lambda x,r:action(r)
    else:
     def inject(x,r):
      project=lambda v:exact.project(v,list(range(-8,9)))
      ia=lambda v:project(high.A_op(v))
      sync();ti=time.perf_counter();z,ist=solve_from_initial(ia,project(r),action,None,tolerance=1e-4,maxiter=8);sync();inner.append(dict(**ist,seconds=time.perf_counter()-ti));return z
    u,st=augmented_fgmres(high.A_op,b,pre,None,inject=inject,inject_after=1,maxiter=100);sync();solve=time.perf_counter()-t;t=time.perf_counter()
    res=float((b-high.A_op(u)).norm()/bn);delta=res*bn/margin;bound=delta/(float(u.norm())-delta);diff=float((u-reference[sid]).norm()/reference[sid].norm());ok=bool(torch.isfinite(u).all()) and not st['failed'] and res<=1e-5 and bound<=1e-3 and diff<=1e-4;sync();check=time.perf_counter()-t;st.pop('trace',None)
    row=dict(method=name,source_id=sid,solve_seconds=solve,check_seconds=check,accepted_seconds=solve+check,setup_seconds=setup,Q1_seconds=setup+solve+check,status=st,inner_records=inner,accepted=ok,true_residual=res,evaluated_relative_bound=bound,relative_field_difference=diff);rows.append(row);records.append(row);save();print(json.dumps(dict(method=name,source_id=sid,calls=st['classical_calls'],accepted=ok,seconds=solve+check)),flush=True)
   return dict(setup_seconds=setup,online_four_rhs_seconds=sum(r['accepted_seconds'] for r in records),complete_four_rhs_seconds=setup+sum(r['accepted_seconds'] for r in records),calls=[r['status']['classical_calls'] for r in records],accepted=all(r['accepted'] for r in records))
  settings['compact4_batch']=solve_batch('compact4_inject',compact,compactsetup);assert settings['compact4_batch']['accepted'];save();limit=settings['compact4_batch']['complete_four_rhs_seconds'];guard()
  del compact,factor,pre,tsa,base,low,high,rhs;gc.collect()
  rhs,low,high,base,tsa,pre,margin,ec=construct(raw);assert torch.equal(low.sigma_t,material[0]) and torch.equal(low.sigma_s,material[1]);settings['exact_base_setup']=ec
  sync();tb=time.perf_counter()
  def checkpoint(f,k):
   sync();elapsed=time.perf_counter()-tb;bound=ec['base_setup_seconds']+elapsed
   settings['exact_partial_setup']=dict(completed_modes=list(f.factors),common_assembly_seconds=f.common_assembly_seconds,profiles=f.profiles,exact_factor_elapsed_seconds=elapsed,total_setup_lower_bound_seconds=bound,compact_complete_seconds=limit)
   if bound>=limit:raise SetupGateStop('exact setup lower bound exceeds complete compact4 four-RHS cost')
  exact=ExactModes(low,high,list(range(-8,9)),checkpoint=checkpoint,synchronize=sync);sync();exactfactor=time.perf_counter()-tb
  t=time.perf_counter();project=lambda v:exact.project(v,list(range(-8,9)));eagerip=lambda v:project(exact.apply(project(v)));ip=CapturedAction(eagerip,rhs[0]);sync();assert ip.relative_capture_difference<1e-12
  first=ip(rhs[0]);copy=first.clone();ip(rhs[1]);assert torch.equal(first,copy)
  v=exact.apply(rhs[0]);target=project(rhs[0]);defect=float((target-high.A_op(v)+exact.perturbation(v,high)).norm()/target.norm());assert defect<1e-10;sync();capturequal=time.perf_counter()-t
  exactsetup=ec['base_setup_seconds']+exactfactor+capturequal;settings['exact_setup']=dict(**ec,factor_seconds=exactfactor,capture_qualification_seconds=capturequal,total_setup_seconds=exactsetup,capture_error=ip.relative_capture_difference,background_residual=defect,output_ownership_pass=True,factor_bytes=sum(a.numel()*a.element_size()+p.numel()*p.element_size() for a,p in exact.factors.values()))
  if exactsetup>=limit:raise SetupGateStop('qualified exact setup exceeds complete compact4 four-RHS cost')
  guard();settings['exact_batch']=solve_batch('exact8_inner',ip,exactsetup);settings['result']='COMPLETE_PAIR';settings['complete']=True
 except SetupGateStop as exc:settings.update(complete=True,result='NO_GO_SETUP_LOWER_BOUND',stop_reason=str(exc));print(json.dumps(settings.get('exact_partial_setup',{})),flush=True)
 except BaseException as exc:settings['error']=repr(exc);raise
 finally:
  settings.update(wall_seconds=time.perf_counter()-began,attempts=len(rows),failures=sum(not r['accepted'] for r in rows))
  if torch.cuda.is_initialized():sync();settings.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
  if 'production_hashes' in settings:settings['production_hashes_after']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in settings['production_hashes']}
  save()
if __name__=='__main__':main()
