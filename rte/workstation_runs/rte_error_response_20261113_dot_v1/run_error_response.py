"""Paid two-source original-equation error-response diagnostic; no learned method."""
import time
PROCESS_START=time.perf_counter()
import os,sys,json,gc,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
import prior_permutation_driver as p
import torch
from diagnostic_fgmres import augmented_fgmres as scheduled
from rte.source_initialization import solve_from_initial
IDS=[0,4]

@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and set(os.sched_getaffinity(0))=={0}
 out=HERE/'results';out.mkdir(exist_ok=False)
 report=dict(complete=False,pid=os.getpid(),source_ids=IDS,device='cpu',affinity=[0],threads=1,
  scope='Fixed K sources 0,4. Diagnose A^{-1}r at two real one-shot coarse-corrected states. No performance candidate, source scan, training, heldout use, or free offline teachers.',
  sampling_states=['after_coarse','after_classical_2'],residual_tolerance=1e-5,response_tolerance=1e-10,response_bound_limit=1e-6,
  P8_definition='Last spatial axis tangent Fourier |k|<=8; all angular components retained.',
  grazing_definition='Two explicitly reported groups, abs(stored omega_tangent)<=1e-6 and abs(stored omega_normal)<=1e-6; no assumed ideal-angle zeros.',
  ledger={},incumbents=[],responses=[],qualification={},resource_at_start=p.admission(),current_stage='input')
 def save():
  t=time.perf_counter();temp=out/'results.json.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');os.replace(temp,out/'results.json')
  report['json_write_seconds']=report.get('json_write_seconds',0.)+time.perf_counter()-t
 def check(ok,msg):
  if not ok:raise RuntimeError(msg)
 def emit(event,**data):print(json.dumps(dict(event=event,**data)),flush=True)
 def bound(resnorm,fieldnorm,margin):
  delta=resnorm/margin
  return delta,delta/(fieldnorm-delta) if fieldnorm>delta else float('inf')
 def spectrum(v):
  spec=torch.fft.fft(v,dim=-1,norm='ortho');power=spec.abs().square();total=float(power.sum())
  kval=torch.fft.fftfreq(p.N)*p.N;ka=kval.abs();bands=[(0,8),(9,16),(17,32),(33,64)]
  fractions={f'{a}_{b}':float(power[..., (ka>=a)&(ka<=b)].sum())/total for a,b in bands}
  qa=power[...,ka>8].sum(dim=(0,2,3));pa=power[...,ka<=8].sum(dim=(0,2,3));qtotal=float(qa.sum())
  tangential=(high.directions[:,1].abs()<=1e-6);normal=(high.directions[:,0].abs()<=1e-6)
  check(abs(sum(fractions.values())-1)<1e-12,'Non-partitioning Fourier bands')
  return dict(norm=float(v.norm()),bands_energy_fraction=fractions,Q8_energy_fraction=qtotal/total,
   Q8_angle_energy_fraction_of_Q8=(qa/qtotal).tolist(),P8_angle_energy_fraction_of_total=(pa/total).tolist(),
   Q8_small_tangential_velocity_fraction=float(qa[tangential].sum())/qtotal,
   Q8_small_normal_velocity_fraction=float(qa[normal].sum())/qtotal)
 try:
  t=time.perf_counter();prior=json.loads((HERE/'PRIOR_K_RESULTS.json').read_text());audit=json.loads((HERE/'SOURCE_AUDIT.json').read_text())
  for f,h in prior['source_hashes'].items():check(p.digest(p.ROOT/f)==h,'Changed production dependency '+f)
  for f,h in prior['E_source_hashes'].items():check(p.digest(p.E/f)==h,'Changed E dependency '+f)
  report['source_hashes']={f.name:p.digest(f) for f in HERE.glob('*.py')};report['plan_sha256']=p.digest(HERE/'DIAGNOSTIC_PLAN.md')
  bank=torch.load(p.ROOT/'rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt',map_location='cpu',weights_only=True,mmap=True)
  audited={r['id']:r for r in audit['source_records']};rhs={}
  for sid in IDS:
   q=bank['bank']['q'][sid:sid+1].to(torch.complex128)
   check(p.thash(q)==audited[sid]['source128_sha256'],'Original source identity changed');rhs[sid]=q.clone()
  del bank,q;gc.collect();report['ledger']['input_verification_seconds']=time.perf_counter()-t
  t=time.perf_counter();low,high,base,margin,common=p.construct(rhs[0],64)
  check(base.kind=='tsa' and base.depth==64,'TSA64 identity')
  check(p.thash(low.sigma_t)==prior['material_identity']['total_sha256'] and p.thash(low.sigma_s)==prior['material_identity']['scatter_sha256'],'K identity')
  bg,_,_=p.background(low);check(bg==prior['background_identity'],'Background identity')
  report.update(common=common,background=bg,coercivity_margin=margin,directions=high.directions.tolist(),weights=high.weights.tolist())
  report['ledger']['operator_and_TSA_setup_seconds']=time.perf_counter()-t
  t=time.perf_counter();factor=p.FullAngularModes(low,high);report['factor_bytes']=factor.bytes;report['ledger']['factor_setup_seconds']=time.perf_counter()-t
  t=time.perf_counter();p.qualify(factor,low,high,report['qualification'],save);report['ledger']['factor_qualification_seconds']=time.perf_counter()-t
  snapshots=[];pre_profile=[];A_profile=[];inner_profile=[]
  def pre(r):
   t=time.perf_counter();z=base(r.to(torch.complex64)).to(torch.complex128);pre_profile.append(time.perf_counter()-t);return z
  def A(r):
   t=time.perf_counter();z=high.A_op(r);A_profile.append(time.perf_counter()-t);return z
  def innerA(r):return p.project8(high.A_op(r))
  for sid in IDS:
   report['current_stage']=f'incumbent_source_{sid}';events=[];observed=[];pre_profile.clear();A_profile.clear();inner_profile.clear()
   t=time.perf_counter()
   def observer(event):
    start=time.perf_counter();r=event.pop('residual');x=event.pop('solution',None);event['norm']=float(r.norm());events.append(event)
    name='after_coarse' if event['event']=='injection_accepted' else ('after_classical_2' if event['event']=='classical_accepted' and event['classical_calls']==2 else None)
    if name:
     check(x is not None,'Missing actual iterate');observed.append(dict(source_id=sid,state=name,residual=r,solution=x,event=dict(event)))
    event['observer_callback_seconds']=time.perf_counter()-start
   def inject(x,r):
    ti=time.perf_counter();z,ist=solve_from_initial(innerA,p.project8(r),lambda v:p.project8(factor.apply(p.project8(v))),None,tolerance=1e-4,maxiter=8)
    inner_profile.append(dict(**ist,seconds=time.perf_counter()-ti));return z
   u,st=scheduled(A,rhs[sid],pre,inject=inject,inject_after=1,inject_schedule=(1,),observer=observer,tolerance=1e-5,maxiter=100)
   solve=time.perf_counter()-t;t=time.perf_counter();bn=float(rhs[sid].norm());rn=float((rhs[sid]-high.A_op(u)).norm());delta,rb=bound(rn,float(u.norm()),margin)
   check(not st['failed'] and rn/bn<=1e-5 and 0<=rb<=1e-3,'Incumbent original-equation acceptance')
   check(st['injection_accepted']==1 and len(observed)==2 and [x['state'] for x in observed]==report['sampling_states'],'Unexpected sampling path')
   check(all(not v['failed'] and v['true_residual']<=1e-4 for v in inner_profile),'Inner convergence')
   for snap in observed:
    r,x=snap['residual'],snap.pop('solution');actual=rhs[sid]-high.A_op(x)
    err=float((actual-r).norm()/r.norm());check(err<=1e-10,'Snapshot is not actual original-equation residual')
    snap['snapshot_residual_identity_relative']=err;snap['source_relative_residual']=float(r.norm()/bn);snap['residual_sha256']=p.thash(r)
   snapshots.extend(observed);checks=time.perf_counter()-t
   report['incumbents'].append(dict(source_id=sid,solve_seconds=solve,check_seconds=checks,status=st,inner=list(inner_profile),preconditioner_calls=len(pre_profile),preconditioner_seconds=sum(pre_profile),outer_A_calls=len(A_profile),outer_A_seconds=sum(A_profile),observer_events=events,true_residual=rn/bn,evaluated_relative_bound=rb))
   save();emit('incumbent_done',source_id=sid,calls=st['classical_calls'],residual=rn/bn,states=[x['state'] for x in observed])
  report['ledger']['incumbent_solves_seconds']=sum(r['solve_seconds'] for r in report['incumbents'])
  report['ledger']['incumbent_checks_seconds']=sum(r['check_seconds'] for r in report['incumbents'])
  # All precision reference responses are paid offline diagnostic work, never free deployment corrections.
  for snap in snapshots:
   sid,state=snap['source_id'],snap['state'];report['current_stage']=f'response_{sid}_{state}';r=snap.pop('residual');pre_profile.clear();A_profile.clear();t=time.perf_counter()
   e,st=solve_from_initial(A,r,pre,None,tolerance=1e-10,maxiter=100);solve=time.perf_counter()-t;t=time.perf_counter()
   rr=r-high.A_op(e);rn,en=float(r.norm()),float(e.norm());res=float(rr.norm()/rn);delta,rb=bound(float(rr.norm()),en,margin)
   check(not st['failed'] and res<=1e-10 and 0<=rb<=1e-6 and bool(torch.isfinite(e).all()),'Response not accurate in original equation')
   check_seconds=time.perf_counter()-t;t=time.perf_counter();rs,es=spectrum(r),spectrum(e)
   ep=p.project8(e);eq=e-ep;ap=high.A_op(ep);aq=high.A_op(eq);pap=p.project8(ap);paq=p.project8(aq);qap=ap-pap;qaq=aq-paq;rp=p.project8(r);rq=r-rp
   def n(v):return float(v.norm())
   def cos(a,b):return float((a.conj()*b).sum().real/(a.norm()*b.norm()))
   coupling=dict(P_A_Pe_norm=n(pap),P_A_Qe_norm=n(paq),Q_A_Pe_norm=n(qap),Q_A_Qe_norm=n(qaq),P_r_norm=n(rp),Q_r_norm=n(rq),P_cross_to_residual_norm=n(paq)/n(rp),Q_cross_to_residual_norm=n(qap)/n(rq),P_block_cosine=cos(pap,paq),Q_block_cosine=cos(qap,qaq),split_A_relative_error=n(r-ap-aq)/rn)
   check(coupling['split_A_relative_error']<=1e-10,'Block action split did not recover original residual')
   # A posteriori conservative interval for the exact response Q8 energy fraction.
   f=es['Q8_energy_fraction'];d=delta/en;interval=[max(0.,f**.5-d)**2/(1+d)**2,min(1.,(f**.5+d)**2/(1-d)**2)]
   diagnostics=time.perf_counter()-t;t=time.perf_counter();path=out/f'response_source{sid}_{state}.pt'
   torch.save(dict(residual=r,response=e,source_id=sid,state=state,original_operator_sources=prior['source_hashes'],directions=high.directions,margin=margin),path)
   artifact=dict(file=path.name,bytes=path.stat().st_size,sha256=p.digest(path));write_seconds=time.perf_counter()-t
   row=dict(**snap,status=st,response_solve_seconds=solve,response_check_seconds=check_seconds,spectral_coupling_seconds=diagnostics,tensor_save_and_hash_seconds=write_seconds,
    preconditioner_calls=len(pre_profile),preconditioner_seconds=sum(pre_profile),response_A_calls=len(A_profile),response_A_seconds=sum(A_profile),response_relative_residual=res,response_evaluated_relative_error_bound=rb,response_Q8_exact_energy_interval=interval,
    residual_spectrum=rs,error_response_spectrum=es,coupling=coupling,artifact=artifact)
   report['responses'].append(row);save();emit('response_done',source_id=sid,state=state,iterations=st['iterations'],seconds=solve,residual_Q8=rs['Q8_energy_fraction'],error_Q8=es['Q8_energy_fraction'],error_bound=rb)
  for field in ['response_solve_seconds','response_check_seconds','spectral_coupling_seconds','tensor_save_and_hash_seconds']:report['ledger'][field]=sum(r[field] for r in report['responses'])
  report['complete']=True;report['result']='PAID_TWO_SOURCE_ERROR_RESPONSE_DIAGNOSTIC_COMPLETE'
 except BaseException as exc:
  report.update(error=repr(exc),traceback=traceback.format_exc());raise
 finally:
  report['current_stage']='complete' if report['complete'] else 'failed'
  report.update(total_process_seconds=time.perf_counter()-PROCESS_START,peak_process_RSS_bytes=p.rss(),resource_at_end=p.admission())
  if 'prior' in locals():report.update(production_unchanged=all(p.digest(p.ROOT/f)==h for f,h in prior['source_hashes'].items()),E_unchanged=all(p.digest(p.E/f)==h for f,h in prior['E_source_hashes'].items()))
  # This is an exclusive top-level ledger. Nested profiles must not be added again.
  report['ledger']['import_startup_logging_and_other_seconds']=report['total_process_seconds']-sum(report['ledger'].values())
  check(report['ledger']['import_startup_logging_and_other_seconds']>=0,'Overlapping top-level cost ledger')
  save();emit('finished',complete=report['complete'],seconds=report['total_process_seconds'],error=report.get('error'))
if __name__=='__main__':main()
