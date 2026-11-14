"""Same-factor second-injection study on the preserved K material and four sources."""
import os,sys,time,json,gc,hashlib,subprocess,resource,signal,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
import prior_permutation_driver as p
import torch
from scheduled_fgmres import augmented_fgmres as scheduled
from rte.augmented_fgmres import augmented_fgmres as original
from rte.source_initialization import solve_from_initial
ROOT=p.ROOT;E=p.E

@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 out=HERE/'results';out.mkdir(exist_ok=False)
 started=time.perf_counter();prior=json.loads((HERE/'PRIOR_K_RESULTS.json').read_text());audit=json.loads((HERE/'SOURCE_AUDIT.json').read_text())
 report=dict(complete=False,pid=os.getpid(),threads=1,affinity=sorted(os.sched_getaffinity(0)),device='cpu',scope='K unchanged N128 material, original four RHS, one vs two fixed P8 correction attempts. Classical control, not a learned-method claim.',orders=['tsa64_zero','compact4','tsa16_zero','E_once','E_twice'],timing_scope='fresh operator and applicable factors plus qualification, full solves, checks, observers and per-source logging; resident interpreter and shared source bank',sources={},batches=[],rows=[],resource_at_start=p.admission(),current_stage='source_verification',tests=json.loads((HERE/'UNIT_TESTS.json').read_text()))
 def save():
  temp=out/'results.json.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');os.replace(temp,out/'results.json')
 def check(cond,message):
  if not cond:raise RuntimeError(message)
 def emit(event,**data):print(json.dumps(dict(event=event,**data)),flush=True)
 def observe_factory(log):
  def observer(event):
   tick=time.perf_counter();r=event.pop('residual');pr=p.project8(r)
   event.update(norm=float(r.norm()),P8_norm=float(pr.norm()),complement_norm=float((r-pr).norm()))
   event['observer_seconds']=time.perf_counter()-tick;log.append(event)
  return observer
 def prepare(method):
  depth=16 if method=='tsa16_zero' else 64
  low,high,base,margin,common=p.construct(rhs[0],depth)
  check(base.kind=='tsa' and base.depth==depth and common['depth']==depth,'TSA depth binding failure')
  check(p.thash(low.sigma_t)==prior['material_identity']['total_sha256'] and p.thash(low.sigma_s)==prior['material_identity']['scatter_sha256'],'K material identity changed')
  bg,_,_=p.background(low)
  check(bg==prior['background_identity'],'K background changed')
  return low,high,base,margin,common,bg
 def factor_for(method,low,high,rec):
  tick=time.perf_counter()
  if method=='compact4':
   f=p.LayeredInverse(low,scheme='compact4',nyquist=True,dtype=torch.complex64);rec.update(factor_seconds=time.perf_counter()-tick,factor_bytes=f.factor_bytes)
  else:
   f=p.FullAngularModes(low,high);rec.update(factor_seconds=time.perf_counter()-tick,factor_bytes=f.bytes,qualification={})
   p.qualify(f,low,high,rec['qualification'],save)
  return f
 try:
  check(report['tests']['passed'],'unit tests did not pass')
  for f,h in prior['source_hashes'].items():check(p.digest(ROOT/f)==h,'Changed production dependency '+f)
  for f,h in prior['E_source_hashes'].items():check(p.digest(E/f)==h,'Changed sealed E dependency '+f)
  report['sources']={str(f):p.digest(f) for f in HERE.glob('*.py')}
  bankpath=ROOT/'rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt'
  bank=torch.load(bankpath,map_location='cpu',weights_only=True,mmap=True);rhs=[]
  for record in audit['source_records']:
   q=bank['bank']['q'][record['id']:record['id']+1].to(torch.complex128)
   check(p.thash(q)==record['source128_sha256'],'Original RHS identity changed');rhs.append(q.clone())
  del bank,q;gc.collect();report['source_ids']=p.IDS;report['input_seconds']=time.perf_counter()-started;save()
  # One real K source bridge: no fields or factor are reused by timed methods.
  report['current_stage']='one_shot_equivalence_bridge';bridge=dict();report['bridge']=bridge
  bt=time.perf_counter();low,high,base,margin,common,bg=prepare('E_once');f=factor_for('E_once',low,high,bridge)
  pre=lambda r:base(r.to(torch.complex64)).to(torch.complex128)
  innerA=lambda v:p.project8(high.A_op(v))
  def inj(x,r):return solve_from_initial(innerA,p.project8(r),lambda v:p.project8(f.apply(p.project8(v))),None,tolerance=1e-4,maxiter=8)[0]
  u,a=original(high.A_op,rhs[0],pre,inject=inj,inject_after=1,tolerance=1e-5,maxiter=100)
  v,b=scheduled(high.A_op,rhs[0],pre,inject=inj,inject_after=1,inject_schedule=(1,),tolerance=1e-5,maxiter=100)
  rel=float((u-v).norm()/u.norm());keys=['iterations','classical_calls','injection_calls','injection_accepted','injection_rejected','failed']
  check(rel<=1e-12 and all(a[k]==b[k] for k in keys) and not a['failed'],'Original/scheduled one-shot bridge failed')
  bridge.update(passed=True,relative_field_difference=rel,original=a,scheduled=b,seconds=time.perf_counter()-bt)
  del u,v,pre,inj,innerA,f,low,high,base;gc.collect();save();emit('bridge_passed',difference=rel,seconds=bridge['seconds'])
  refs={};first_inputs={};batch_no=0
  def run(method):
   nonlocal batch_no
   batch_no+=1;label=f'{batch_no:02d}_{method}';report['current_stage']=label
   tick=time.perf_counter();rec=dict(method=method,label=label,status='running',resource_admission=p.admission());report['batches'].append(rec)
   low,high,base,margin,common,bg=prepare(method);rec.update(common=common,background=bg,actual_TSA_depth=int(base.depth))
   pre=lambda r:base(r.to(torch.complex64)).to(torch.complex128);f=None
   if method in ['compact4','E_once','E_twice']:f=factor_for(method,low,high,rec)
   rec['setup_seconds']=time.perf_counter()-tick
   counters=dict(outer_A=0,inner_A=0,final_A=0)
   def A(v):counters['outer_A']+=1;return high.A_op(v)
   def innerA(v):counters['inner_A']+=1;return p.project8(high.A_op(v))
   rows=[]
   for j,(sid,q) in enumerate(zip(p.IDS,rhs)):
    inner=[];events=[];input_states=[];ti=time.perf_counter()
    if method.startswith('tsa'):
     u,st=solve_from_initial(A,q,pre,None,tolerance=1e-5,maxiter=100);st.update(classical_calls=st['iterations'],injection_calls=0)
    else:
     if method=='compact4':inject=lambda x,r:f(r).to(torch.complex128)
     else:
      def inject(x,r):
       it=time.perf_counter();input_states.append((x.clone(),r.clone()))
       z,ist=solve_from_initial(innerA,p.project8(r),lambda v:p.project8(f.apply(p.project8(v))),None,tolerance=1e-4,maxiter=8)
       inner.append(dict(**ist,seconds=time.perf_counter()-it,input_norm=float(r.norm())));return z
     if method=='E_twice':
      u,st=scheduled(A,q,pre,inject=inject,inject_after=1,inject_schedule=(1,2),observer=observe_factory(events),tolerance=1e-5,maxiter=100)
     elif method=='E_once':
      u,st=scheduled(A,q,pre,inject=inject,inject_after=1,inject_schedule=(1,),observer=observe_factory(events),tolerance=1e-5,maxiter=100)
     else:u,st=original(A,q,pre,inject=inject,inject_after=1,tolerance=1e-5,maxiter=100)
    solve=time.perf_counter()-ti;ti=time.perf_counter();counters['final_A']+=1
    bn=float(q.norm());res=float((q-high.A_op(u)).norm()/bn);delta=res*bn/margin;bound=delta/(float(u.norm())-delta)
    if method=='tsa64_zero':refs[sid]=u.clone()
    diff=float((u-refs[sid]).norm()/refs[sid].norm())
    accepted=bool(torch.isfinite(u).all()) and not st['failed'] and res<=1e-5 and 0<=bound<=1e-3 and diff<=1e-4
    check(accepted and all(not e['failed'] and e['true_residual']<=1e-4 for e in inner),'Original equation or inner convergence failure')
    prefix_comparison=None
    if input_states:
     x0,r0=input_states[0]
     if sid not in first_inputs:first_inputs[sid]=(x0.clone(),r0.clone())
     xr,rr=first_inputs[sid];prefix_comparison=dict(state=float((x0-xr).norm()/xr.norm()),residual=float((r0-rr).norm()/rr.norm()))
     check(max(prefix_comparison.values())<=1e-12,'First-injection paths no longer match')
    check_time=time.perf_counter()-ti
    row=dict(method=method,label=label,source_id=sid,solve_seconds=solve,check_seconds=check_time,status=st,inner=inner,observer_events=events,first_input_comparison=prefix_comparison,true_residual=res,evaluated_relative_bound=bound,relative_field_difference=diff,accepted=accepted)
    report['rows'].append(row);rows.append(row)
    if j==0:rec['first_RHS_seconds']=time.perf_counter()-tick
    save();emit('source_done',label=label,source_id=sid,calls=st['classical_calls'],attempts=st['injection_calls'],inner_iterations=[e['iterations'] for e in inner],residual=res)
   rec.update(status='completed',complete_four_RHS_seconds=time.perf_counter()-tick,classical_calls=[r['status']['classical_calls'] for r in rows],injection_calls=[r['status']['injection_calls'] for r in rows],inner_iterations=[[x['iterations'] for x in r['inner']] for r in rows],inner_seconds=sum(x['seconds'] for r in rows for x in r['inner']),A_counts=counters,peak_process_RSS_bytes=p.rss())
   if method.startswith('E_'):rec['inverse_profiles']=f.profiles
   save();emit('batch_done',method=method,label=label,seconds=rec['complete_four_RHS_seconds'],calls=rec['classical_calls'],attempts=rec['injection_calls'])
   del low,high,base,pre,A,innerA,f,u
   if method in ['compact4','E_once','E_twice']:del inject
   gc.collect();return rec
  controls=[run(m) for m in ['tsa64_zero','compact4','tsa16_zero']]
  once=run('E_once');twice=run('E_twice')
  reduced=any(a<b for a,b in zip(twice['classical_calls'],once['classical_calls']))
  report['mechanism_removed_classical_work']=reduced
  if reduced:
   # Predetermined opposite order, regardless of the first timing sign.
   report['reverse_pair']=[run('E_twice')['label'],run('E_once')['label']]
  report['complete']=True;report['result']='SECOND_INJECTION_CALL_REDUCTION' if reduced else 'NO_CLASSICAL_WORK_REMOVED'
 except BaseException as exc:
  report.update(complete=False,error=repr(exc),traceback=traceback.format_exc());raise
 finally:
  report.update(research_wall_seconds=time.perf_counter()-started,peak_process_RSS_bytes=p.rss(),resource_at_end=p.admission(),production_unchanged=all(p.digest(ROOT/f)==h for f,h in prior['source_hashes'].items()),E_unchanged=all(p.digest(E/f)==h for f,h in prior['E_source_hashes'].items()))
  save();emit('finished',complete=report['complete'],result=report.get('result'),error=report.get('error'),seconds=report['research_wall_seconds'])
if __name__=='__main__':main()
