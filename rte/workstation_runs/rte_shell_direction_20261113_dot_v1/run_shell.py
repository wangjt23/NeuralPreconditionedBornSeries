"""Fixed second-direction support control: repeated P8 versus |kt|=9..16."""
import time
PROCESS_START=time.perf_counter()
import os,json,gc,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
import prior_permutation_driver as p
import torch
from angular_modes import AngularBandMode,FullAngularModes
from scheduled_fgmres import augmented_fgmres as scheduled
from rte.source_initialization import solve_from_initial
SHELL=list(range(-16,-8))+list(range(9,17))
class ShellModes(FullAngularModes):
 def __init__(self,low,high):
  from layered_inverse import background_from_material
  self.shape=low.source.shape;self.axis,self.t,self.s,_=background_from_material(low.sigma_t,low.sigma_s)
  if self.axis!=0:raise ValueError('Only original normal axis0')
  t=self.t[0,0,:,0].double();s=self.s[0,0,:,0].double()
  self.blocks={k:AngularBandMode(high,t,s,k) for k in SHELL}
  self.profiles=[];self.bytes=sum(b.bytes for b in self.blocks.values())

def project_shell(r):
 f=torch.fft.fft(r,dim=-1,norm='ortho');z=torch.zeros_like(f)
 for k in SHELL:z[...,k]=f[...,k]
 return torch.fft.ifft(z,dim=-1,norm='ortho')

def qualify_shell(factor,low,high,rec,save):
 start=time.perf_counter();gen=torch.Generator().manual_seed(2026111309);_,t,s=p.background(low);rec['modes']=[];rec['boundary_cases']=[];qs={}
 for k in SHELL:
  q=torch.complex(torch.randn(12*p.N,1,dtype=torch.float64,generator=gen),torch.randn(12*p.N,1,dtype=torch.float64,generator=gen));qs[k]=q
  x,st=factor.blocks[k].repaired(q,tolerance=1e-10,max_corrections=3);r=p.A0_check(high,t,s,k,x,q)
  rec['modes'].append(dict(k=k,RHS_sha256=p.thash(q),repair=st,original_A0_relative=r.tolist()))
  if not st['success'] or not bool((r<=1e-10).all()) or not bool(torch.isfinite(x).all()):raise RuntimeError('Shell original-A0 qualification failed')
 for k in [-16,-9,9,16]:
  q=qs[k];zero=torch.zeros(12*p.N,2,dtype=torch.complex128);mix=torch.cat([zero[:,:1],q,1e-200*q,zero[:,:1]],dim=1)
  for name,rhs in [('zero',zero),('mixed_zero_ordinary_tiny',mix)]:
   x,st=factor.blocks[k].repaired(rhs,tolerance=1e-10,max_corrections=3);r=p.A0_check(high,t,s,k,x,rhs);inactive=~(rhs!=0).any(dim=0);strict=bool((x[:,inactive]==0).all());tiny=bool(x[:,2].abs().max()>0) if name.startswith('mixed') else None
   rec['boundary_cases'].append(dict(k=k,kind=name,repair=st,original_A0_relative=r.tolist(),strict_zero=strict,tiny_nonzero=tiny))
   if not st['success'] or not bool((r<=1e-10).all()) or not strict or tiny is False:raise RuntimeError('Shell zero/tiny qualification failed')
 rec.update(passed=True,qualification_seconds=time.perf_counter()-start);save()

@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and set(os.sched_getaffinity(0))=={0}
 out=HERE/'results';out.mkdir(exist_ok=False)
 report=dict(complete=False,pid=os.getpid(),source_ids=p.IDS,discovery_sources=[0,4],not_used_in_this_selection=[7,14],heldout_used=False,training=False,device='cpu',threads=1,affinity=[0],orders=['one_shot_P8','repeat_P8','shell_second'],shell_modes=SHELL,
  scope='Isolates the support of the second direction after classical call2. Original first P8 correction and TSA64 are unchanged. Shell is selected using paid source0/4 diagnostic; not claimed unseen-source generalization. No oracle response vectors used.',
  timing_scope='Each method freshly pays operators, TSA64, all applicable factor setup, qualification, solves, checks, observers, and per-source logging. Entire process wall separately recorded. Four original sources, not new material or GPU evidence.',batches=[],rows=[],resource_at_start=p.admission(),current_stage='inputs')
 def save():
  temp=out/'results.json.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');os.replace(temp,out/'results.json')
 def check(ok,msg):
  if not ok:raise RuntimeError(msg)
 def emit(event,**data):print(json.dumps(dict(event=event,**data)),flush=True)
 try:
  prior=json.loads((HERE/'PRIOR_K_RESULTS.json').read_text());audit=json.loads((HERE/'SOURCE_AUDIT.json').read_text())
  for f,h in prior['source_hashes'].items():check(p.digest(p.ROOT/f)==h,'Changed production source '+f)
  for f,h in prior['E_source_hashes'].items():check(p.digest(p.E/f)==h,'Changed E source '+f)
  report['source_hashes']={f.name:p.digest(f) for f in HERE.glob('*.py')};report['plan_sha256']=p.digest(HERE/'SHELL_PLAN.md')
  report['selection_evidence_sha256']=p.digest(p.ROOT/'rte/workstation_runs/rte_error_response_20261113_dot_v1/results/results.json')
  gen=torch.Generator().manual_seed(2026111309);v=torch.complex(torch.randn(1,12,p.N,p.N,dtype=torch.float64,generator=gen),torch.randn(1,12,p.N,p.N,dtype=torch.float64,generator=gen));sv=project_shell(v);pv=p.project8(v)
  tests=dict(shell_idempotence=float((project_shell(sv)-sv).norm()/sv.norm()),disjointness=float(p.project8(sv).norm()/sv.norm()),P8_orthogonality=float((pv.conj()*sv).sum().abs()/(pv.norm()*sv.norm())))
  check(max(tests.values())<=1e-12,'Projector unit test');report['projector_tests']=tests;del v,sv,pv
  bank=torch.load(p.ROOT/'rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt',map_location='cpu',weights_only=True,mmap=True);rhs={}
  for record in audit['source_records']:
   sid=record['id'];q=bank['bank']['q'][sid:sid+1].to(torch.complex128);check(p.thash(q)==record['source128_sha256'],'Source identity');rhs[sid]=q.clone()
  del bank,q;gc.collect();report['input_and_test_seconds']=time.perf_counter()-PROCESS_START;save()
  refs={};first_inputs={};second_states={};batch_no=0
  def run(method):
   nonlocal batch_no
   batch_no+=1;label=f'{batch_no:02d}_{method}';report['current_stage']=label;t=time.perf_counter();rec=dict(method=method,label=label,status='running',resource_admission=p.admission());report['batches'].append(rec)
   low,high,base,margin,common=p.construct(rhs[0],64);check(base.kind=='tsa' and base.depth==64,'TSA64 binding')
   check(p.thash(low.sigma_t)==prior['material_identity']['total_sha256'] and p.thash(low.sigma_s)==prior['material_identity']['scatter_sha256'],'Material identity')
   bg,_,_=p.background(low);check(bg==prior['background_identity'],'Background identity');rec.update(common=common,background=bg,actual_TSA_depth=int(base.depth))
   ti=time.perf_counter();f=p.FullAngularModes(low,high);rec['P8_factor_seconds']=time.perf_counter()-ti;rec['P8_factor_bytes']=f.bytes;rec['P8_qualification']={};p.qualify(f,low,high,rec['P8_qualification'],save)
   shell=None
   if method=='shell_second':
    ti=time.perf_counter();shell=ShellModes(low,high);rec['shell_factor_seconds']=time.perf_counter()-ti;rec['shell_factor_bytes']=shell.bytes;rec['shell_qualification']={};qualify_shell(shell,low,high,rec['shell_qualification'],save)
   rec['setup_seconds']=time.perf_counter()-t;rows=[];counters=dict(outer_A=0,inner_A=0,final_A=0);pre_times=[]
   def pre(r):
    tp=time.perf_counter();z=base(r.to(torch.complex64)).to(torch.complex128);pre_times.append(time.perf_counter()-tp);return z
   def A(v):counters['outer_A']+=1;return high.A_op(v)
   for j,sid in enumerate(p.IDS):
    inner=[];events=[];inputs=[];pre_times.clear();ti=time.perf_counter()
    def observer(event):
     to=time.perf_counter();r=event.pop('residual');pr=p.project8(r);sr=project_shell(r)
     event.update(norm=float(r.norm()),P8_norm=float(pr.norm()),shell_norm=float(sr.norm()),rest_norm=float((r-pr-sr).norm()))
     if method=='one_shot_P8' and event['event']=='classical_accepted' and event['classical_calls']==2 and sid not in second_states:second_states[sid]=r.clone()
     event['observer_seconds']=time.perf_counter()-to;events.append(event)
    def inject(x,r):
     idx=len(inner)+1;inputs.append((x.clone(),r.clone()));ti=time.perf_counter();current=shell if method=='shell_second' and idx==2 else f;proj=current.project
     def innerA(v):counters['inner_A']+=1;return proj(high.A_op(v))
     z,st=solve_from_initial(innerA,proj(r),lambda v:proj(current.apply(proj(v))),None,tolerance=1e-4,maxiter=8)
     inner.append(dict(**st,seconds=time.perf_counter()-ti,injection_index=idx,support='shell9to16' if current is shell else 'P8',input_norm=float(r.norm())))
     return z
    schedule=(1,) if method=='one_shot_P8' else (1,2)
    u,st=scheduled(A,rhs[sid],pre,inject=inject,inject_after=1,inject_schedule=schedule,observer=observer,tolerance=1e-5,maxiter=100)
    solve=time.perf_counter()-ti;ti=time.perf_counter();counters['final_A']+=1;bn=float(rhs[sid].norm());rn=float((rhs[sid]-high.A_op(u)).norm());delta=rn/margin;bound=delta/(float(u.norm())-delta)
    if method=='one_shot_P8' and sid not in refs:refs[sid]=u.clone()
    diff=float((u-refs[sid]).norm()/refs[sid].norm());accepted=not st['failed'] and rn/bn<=1e-5 and 0<=bound<=1e-3 and diff<=1e-4 and bool(torch.isfinite(u).all())
    check(accepted and all(not v['failed'] and v['true_residual']<=1e-4 for v in inner),'Original equation or inner convergence')
    x0,r0=inputs[0]
    if sid not in first_inputs:first_inputs[sid]=(x0.clone(),r0.clone())
    xr,rr=first_inputs[sid];first=dict(state=float((x0-xr).norm()/xr.norm()),residual=float((r0-rr).norm()/rr.norm()));check(max(first.values())<=1e-12,'First coarse path changed')
    second=None
    if len(inputs)>1:
     r2=inputs[1][1];rr=second_states[sid];second=dict(relative_residual_difference=float((r2-rr).norm()/rr.norm()),source_normalized_residual_difference=float((r2-rr).norm()/bn))
    check(len(pre_times)==st['classical_calls'],'Classical counter mismatch')
    check_seconds=time.perf_counter()-ti
    row=dict(method=method,label=label,source_id=sid,selection_role='discovery' if sid in [0,4] else 'not_used_for_support_selection',solve_seconds=solve,check_seconds=check_seconds,status=st,inner=inner,observer_events=events,preconditioner_seconds=sum(pre_times),preconditioner_calls=len(pre_times),true_residual=rn/bn,evaluated_relative_bound=bound,relative_field_difference=diff,first_input_comparison=first,second_input_comparison=second,accepted=accepted)
    rows.append(row);report['rows'].append(row)
    if j==0:rec['first_RHS_seconds']=time.perf_counter()-t
    save();emit('source_done',method=method,source_id=sid,calls=st['classical_calls'],injections=st['injection_calls'],inner_iterations=[v['iterations'] for v in inner],residual=rn/bn)
   rec.update(status='completed',complete_four_RHS_seconds=time.perf_counter()-t,classical_calls=[r['status']['classical_calls'] for r in rows],injection_calls=[r['status']['injection_calls'] for r in rows],inner_seconds=sum(v['seconds'] for r in rows for v in r['inner']),preconditioner_seconds=sum(r['preconditioner_seconds'] for r in rows),A_counts=counters,P8_inverse_profiles=f.profiles,peak_RSS_bytes=p.rss())
   if shell is not None:rec['shell_inverse_profiles']=shell.profiles
   rec['remaining_seconds']=rec['complete_four_RHS_seconds']-rec['setup_seconds']-rec['inner_seconds']-rec['preconditioner_seconds']
   save();emit('batch_done',method=method,seconds=rec['complete_four_RHS_seconds'],calls=rec['classical_calls'])
   del low,high,base,f,shell,u,inject,pre,A;gc.collect();return rec
  baseline=run('one_shot_P8');null=run('repeat_P8');candidate=run('shell_second')
  call_reduction=any(a<b for a,b in zip(candidate['classical_calls'],baseline['classical_calls']))
  cost_win=candidate['complete_four_RHS_seconds']<min(baseline['complete_four_RHS_seconds'],null['complete_four_RHS_seconds'])
  report.update(candidate_call_reduction=call_reduction,candidate_cost_win=cost_win)
  if call_reduction and cost_win:
   # Confirm only the same winner/incumbent pair in reverse order, without tuning.
   reverse_candidate=run('shell_second');reverse_base=run('one_shot_P8');report['reverse_cost_win']=reverse_candidate['complete_four_RHS_seconds']<reverse_base['complete_four_RHS_seconds'];report['result']='SHELL_DIRECTION_COST_PASS' if report['reverse_cost_win'] else 'SHELL_DIRECTION_FORWARD_ONLY'
  else:report['result']='SHELL_DIRECTION_NO_GO'
  report['complete']=True
 except BaseException as exc:
  report.update(error=repr(exc),traceback=traceback.format_exc());raise
 finally:
  report.update(total_process_seconds=time.perf_counter()-PROCESS_START,peak_process_RSS_bytes=p.rss(),resource_at_end=p.admission())
  if 'prior' in locals():report.update(production_unchanged=all(p.digest(p.ROOT/f)==h for f,h in prior['source_hashes'].items()),E_unchanged=all(p.digest(p.E/f)==h for f,h in prior['E_source_hashes'].items()))
  save();emit('finished',complete=report['complete'],result=report.get('result'),seconds=report['total_process_seconds'],error=report.get('error'))
if __name__=='__main__':main()
