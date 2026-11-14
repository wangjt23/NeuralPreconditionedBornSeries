"""Paid17-mode CPU screening, then conditional3-pair confirmation; frozen workload."""
import os,sys,time,json,hashlib,gc,subprocess,math,signal
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=Path('/data/wjt/NeuralPreconditionedBornSeries')
sys.path[:0]=[str(ROOT),str(ROOT/'rte/workstation_runs/rte_predictability_20261108_dot_v1'),str(ROOT/'rte/workstation_runs/layered_background_20261108_dot_v1'),str(ROOT/'rte/workstation_runs/rte_recycling_20261110_dot_v1')]
from angular_modes import FullAngularModes
from exact_mode_probe import ExactModes
from screen_structured import medium
from layered_inverse import LayeredInverse,background_from_material
from rte.models import PeriodicRTE2D
from rte.mixed_precision_transport import high_precision_copy
from rte.train_source_predictor import build_base
from rte.source_initialization import solve_from_initial
from rte.augmented_fgmres import augmented_fgmres

MODES=list(range(-8,9))
class ModeQualificationFailure(RuntimeError):
 def __init__(self,records):
  self.records=records;super().__init__('Independent original-A0 qualification failed: '+json.dumps(records[-1]))
def terminate_owned_run(signum,frame):raise KeyboardInterrupt('Owned RTE run terminated by signal '+str(signum))
def project8(r):
 f=torch.fft.fft(r,dim=-1,norm='ortho');out=torch.zeros_like(f)
 for k in MODES:out[...,k]=f[...,k]
 return torch.fft.ifft(out,dim=-1,norm='ortho')
def construct(b):
 t=time.perf_counter();total,scatter,_=medium(128,1,10/3,'cpu');low=PeriodicRTE2D(shape=(128,128),dx=1/128,n_angles=12,anisotropy_g=.65,eta_scale=0.,eta_floor=0.,device='cpu');low.setup(total,scatter,b.to(torch.complex64));high=high_precision_copy(low)
 phase=high.phase_matrix.double()*high.weights.double()[None];mu=torch.linalg.eigvalsh((phase+phase.T)/2).max();margin=float((high.sigma_t-high.sigma_s*mu).min());assert margin>0;physical=time.perf_counter()-t;t=time.perf_counter();base=build_base(low,'tsa64');base.prepare();tsa=time.perf_counter()-t
 return low,high,base,margin,physical,tsa

def validate_all_modes(factor,method,low,high):
 generator=torch.Generator().manual_seed(2026111004);axis,tbg,sbg,_=background_from_material(low.sigma_t,low.sigma_s);assert axis==0;phase=(high.phase_matrix*high.weights[None]).to(torch.complex128);records=[]
 for k in MODES:
  q=torch.complex(torch.randn(1536,1,dtype=torch.float64,generator=generator),torch.randn(1536,1,dtype=torch.float64,generator=generator))
  if method=='angular8_inner':z,st=factor.blocks[k].repaired(q,tolerance=1e-10,max_corrections=3)
  else:
   lu,piv=factor.factors[k];z=torch.linalg.lu_solve(lu,piv,q);st=dict(success=True,corrections=0)
  f=torch.zeros_like(high.source);f[...,k]=z.reshape(1,12,128);field=torch.fft.ifft(f,dim=-1,norm='ortho');target=torch.zeros_like(f);target[...,k]=q.reshape(1,12,128);target=torch.fft.ifft(target,dim=-1,norm='ortho')
  # Independent original physical operator, not the candidate's band/check action.
  actual=high.A_op(field)-(high.sigma_t-tbg.double())*field+(high.sigma_s-sbg.double())*torch.einsum('lm,bmxy->blxy',phase,field)
  residual=float((target-actual).norm()/target.norm());record=dict(k=k,original_A0_residual=residual,repair_status=st,RHS_sha256=hashlib.sha256(q.numpy().tobytes()).hexdigest());records.append(record)
  if not st['success'] or not residual<=1e-10:raise ModeQualificationFailure(records)
 return records

@torch.no_grad()
def main():
 signal.signal(signal.SIGTERM,terminate_owned_run)
 torch.set_num_threads(1);torch.set_num_interop_threads(1);assert set(os.sched_getaffinity(0))=={0};assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert json.loads((HERE/'zero_columns_01/results.json').read_text())['complete']
 out=HERE/'full_01';out.mkdir(exist_ok=False);began=time.perf_counter();rows=[];batches=[];refs=[];qualifications=[];profiles=[]
 settings=dict(complete=False,pid=os.getpid(),affinity=[0],device='cpu',threads=1,shared_host='GB training on CPU32/34 and GPU; not strictly isolated CPU timing. Short Schur timing allowed to finish before admission.',source_ids=[0,4,7,14],grid=128,L=12,g=.65,inclusion_amplitude=3.,prefix=1,tangent_cut=8,inner_tolerance=1e-4,inner_maxiter=8,outer_tolerance=1e-5,outer_maxiter=100,block_target=1e-10,block_max_corrections=3,heldout9626=False,neural_training=False,new_accurate_teachers=False,wall_limit=720,screen_orders=[['compact4_inject','dense8_inner','angular8_inner'],['angular8_inner','dense8_inner','compact4_inject']],confirmation_orders=[['compact4_inject','angular8_inner'],['angular8_inner','compact4_inject'],['compact4_inject','angular8_inner']],source_hashes={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ['run_full_cpu.py','angular_modes.py']},dense_source_sha256=hashlib.sha256((ROOT/'rte/workstation_runs/rte_recycling_20261110_dot_v1/exact_mode_probe.py').read_bytes()).hexdigest(),scope='Full17-mode same-instance CPU qualification only. Every batch rebuilds and pays setup; dense and angular candidates also pay independent original-A0 validation of every mode.')
 old=json.loads((ROOT/'rte/workstation_runs/rte_recycling_20261110_dot_v1/confirmation_cpu_01/settings.json').read_text());settings['production_hashes']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in old['production_hashes']};assert settings['production_hashes']==old['production_hashes']
 settings['dependency_hashes']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['rte/workstation_runs/rte_predictability_20261108_dot_v1/screen_structured.py','rte/workstation_runs/layered_background_20261108_dot_v1/layered_inverse.py','rte/mixed_precision_transport.py']}
 settings['admission_process_snapshot']=subprocess.check_output(['ps','-eo','pid,psr,pcpu,args','--sort=-pcpu'],text=True).splitlines()[:14]
 def save():
  for k,v in [('settings',settings),('rows',rows),('batches',batches),('reference_checks',refs),('mode_qualifications',qualifications),('repair_profiles',profiles)]:
   target=out/(k+'.json');temp=out/(k+'.json.tmp');temp.write_text(json.dumps(v,indent=2)+'\n');os.replace(temp,target)
 save()
 try:
  saved=torch.load(ROOT/'rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt',map_location='cpu',weights_only=True);rhs=[saved['bank']['q'][i:i+1].to(torch.complex128) for i in settings['source_ids']]
  settings['rhs_sha256']={str(i):hashlib.sha256(saved['bank']['q'][i:i+1].numpy().tobytes()).hexdigest() for i in settings['source_ids']};assert settings['rhs_sha256']==old['rhs_sha256'];del saved
  t=time.perf_counter();low,high,base,margin,physical,tsa=construct(rhs[0]);pre=lambda r:base(r.to(torch.complex64)).to(torch.complex128);reference={}
  for sid,b in zip(settings['source_ids'],rhs):
   t0=time.perf_counter();u,st=solve_from_initial(high.A_op,b,pre,None,maxiter=100);assert not st['failed'];reference[sid]=u;refs.append(dict(source_id=sid,seconds=time.perf_counter()-t0,status=st))
  settings['reference_research_seconds']=time.perf_counter()-t;del pre,base,low,high;gc.collect();save()
  def run_batch(method,stage,rep,position):
   if time.perf_counter()-began>720:raise RuntimeError('wall budget')
   tick=time.perf_counter();low,high,base,margin,physical,tsa=construct(rhs[0]);pre=lambda r:base(r.to(torch.complex64)).to(torch.complex128);t=time.perf_counter()
   if method=='compact4_inject':factor=LayeredInverse(low,scheme='compact4',nyquist=True,dtype=torch.complex64);factor_bytes=factor.factor_bytes
   elif method=='dense8_inner':factor=ExactModes(low,high,MODES);factor.matrices.clear();factor_bytes=sum(a.numel()*a.element_size()+p.numel()*p.element_size() for a,p in factor.factors.values())
   else:factor=FullAngularModes(low,high);factor_bytes=factor.bytes
   factorsec=time.perf_counter()-t;t=time.perf_counter();qualification=[]
   if method!='compact4_inject':
    try:qualification=validate_all_modes(factor,method,low,high)
    except ModeQualificationFailure as exc:
     qualifications.append(dict(stage=stage,repeat=rep,method=method,records=exc.records,failed=True));save();raise
   validation=time.perf_counter()-t;setup=physical+tsa+factorsec+validation;qualifications.append(dict(stage=stage,repeat=rep,method=method,records=qualification));batchrows=[]
   for sid,b in zip(settings['source_ids'],rhs):
    inner=[];bn=float(b.norm());before=len(factor.profiles) if method=='angular8_inner' else 0;t=time.perf_counter()
    if method=='compact4_inject':inject=lambda x,r:factor(r).to(torch.complex128)
    else:
     def inject(x,r):
      ia=lambda v:project8(high.A_op(v));ip=lambda v:project8(factor.apply(project8(v)));ti=time.perf_counter();z,st=solve_from_initial(ia,project8(r),ip,None,tolerance=1e-4,maxiter=8);inner.append(dict(**st,seconds=time.perf_counter()-ti));return z
    try:u,st=augmented_fgmres(high.A_op,b,pre,None,inject=inject,inject_after=1,maxiter=100)
    except BaseException as exc:
     rows.append(dict(stage=stage,repeat=rep,method=method,source_id=sid,accepted=False,error=repr(exc)));save();raise
    solve=time.perf_counter()-t;t=time.perf_counter();res=float((b-high.A_op(u)).norm()/bn);delta=res*bn/margin;bound=delta/(float(u.norm())-delta);diff=float((u-reference[sid]).norm()/reference[sid].norm());ok=bool(torch.isfinite(u).all()) and not st['failed'] and res<=1e-5 and bound<=1e-3 and diff<=1e-4;check=time.perf_counter()-t;st.pop('trace',None)
    row=dict(stage=stage,repeat=rep,method=method,method_order=position,source_id=sid,solve_seconds=solve,check_seconds=check,accepted_seconds=solve+check,shared_setup_seconds=setup,Q1_seconds=setup+solve+check,status=st,inner_records=inner,accepted=ok,true_residual=res,evaluated_relative_bound=bound,relative_field_difference=diff)
    if method=='angular8_inner':
     calls=factor.profiles[before:];row['mode_inverse_calls']=len(calls);row['mode_repair_max_residual']=max((max(z['residual_trace'][-1]) for call in calls for z in call),default=0.);row['mode_repair_corrections']=sum(z['corrections'] for call in calls for z in call)
    rows.append(row);batchrows.append(row)
    if not ok:save();raise RuntimeError('Original outer acceptance failed: '+json.dumps(row))
   elapsed=time.perf_counter()-tick
   if method=='angular8_inner':profiles.append(dict(stage=stage,repeat=rep,method=method,applications=factor.profiles))
   result=dict(stage=stage,repeat=rep,method=method,method_order=position,physical_setup_seconds=physical,TSA_setup_seconds=tsa,factor_setup_seconds=factorsec,independent_validation_seconds=validation,setup_seconds=setup,online_four_rhs_seconds=sum(r['accepted_seconds'] for r in batchrows),complete_four_rhs_seconds=setup+sum(r['accepted_seconds'] for r in batchrows),observed_batch_wall_seconds=elapsed,factor_or_logical_stored_bytes=factor_bytes,calls=[r['status']['classical_calls'] for r in batchrows],inner_calls=[v['iterations'] for r in batchrows for v in r['inner_records']],accepted=all(r['accepted'] for r in batchrows))
   batches.append(result);save();print(json.dumps(result),flush=True)
   del inject,pre,factor,base,low,high,u;gc.collect();return result
  screens=[]
  for rep,order in enumerate(settings['screen_orders']):screens.append({m:run_batch(m,'screen',rep,pos) for pos,m in enumerate(order)})
  passed=all(d['angular8_inner']['setup_seconds']<d['dense8_inner']['setup_seconds'] and d['angular8_inner']['complete_four_rhs_seconds']<min(d['compact4_inject']['complete_four_rhs_seconds'],d['dense8_inner']['complete_four_rhs_seconds']) for d in screens)
  settings['screen_passed']=passed;save()
  if not passed:settings.update(complete=True,result='STOP_SCREEN_NO_COMPLETE_COST_ADVANTAGE');return
  for rep,order in enumerate(settings['confirmation_orders']):
   for pos,m in enumerate(order):run_batch(m,'confirmation',rep,pos)
  settings.update(complete=True,result='SCREEN_AND_THREE_PAIRS_COMPLETE')
 except BaseException as exc:settings['error']=repr(exc);raise
 finally:
  settings.update(wall_seconds=time.perf_counter()-began,attempts=len(rows),failures=sum(not r['accepted'] for r in rows),production_hashes_after={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in settings['production_hashes']});save()
if __name__=='__main__':main()
