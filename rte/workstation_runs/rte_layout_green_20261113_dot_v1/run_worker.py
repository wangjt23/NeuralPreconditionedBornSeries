import time
START=time.perf_counter();CPUSTART=time.process_time()
import traceback,json,os,sys
from layout_common import *

@torch.no_grad()
def main():
 method,label=sys.argv[1:];assert method in ['original','packed','angular'];out=HERE/label;out.mkdir(exist_ok=False)
 report=dict(complete=False,pid=os.getpid(),method=method,label=label,n=128,k=0,cost={},applications=[],no_physical_sources=True,resource_at_start=snapshot(),runtime=configure())
 def save():
  temp=out/'results.json.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');os.replace(temp,out/'results.json')
 try:
  prior,prev,sealed=seal();report.update(sealed);bridge=json.loads((HERE/'bridge/results.json').read_text());assert bridge['complete'];assert sealed['source_hashes']==bridge['source_hashes'] and sealed['plan_sha256']==bridge['plan_sha256'];report['bridge_sha256']=p.digest(HERE/'bridge/results.json')
  tick=time.perf_counter();cpu=time.process_time()
  if method!='angular':
   ctx,rec=nb.construct();assert rec['Dhat_sha256']==prev['construction']['Dhat_sha256'];assert rec['background']==prior['background_identity'];assert rec['model_original_right_defect_bound']<=1e-3;report['construction']=rec;report['cost'].update(rec['times'])
  else:
   ctx=physical();assert ctx['background']==prior['background_identity'];report['cost']['physical_setup_seconds']=time.perf_counter()-tick
  V,original,approximate=operators(ctx);report['cost']['physical_and_basis_setup_total_wall_seconds']=time.perf_counter()-tick;report['cost']['physical_and_basis_setup_total_CPU_seconds']=time.process_time()-cpu
  report['RSS_before_factor']=rss();tick=time.perf_counter();cpu=time.process_time()
  if method=='angular':
   fac=old.AngularBandMode(ctx['high'],ctx['tv'],ctx['sv'],0)
   def raw(q):return fac.raw(q.reshape(128,12,-1).permute(1,0,2).reshape(1536,-1)).reshape(12,128,-1).permute(1,0,2).reshape(1536,-1)
   factorbytes=fac.bytes
  else:
   factory=nf.NestedFactor if method=='original' else pf.PackedFactor;fac=factory(ctx['tree'],V,ctx['normal']);raw=fac.apply;factorbytes=old.size_bytes(fac.tensors())
  report['cost']['factor_wall_seconds']=time.perf_counter()-tick;report['cost']['factor_CPU_seconds']=time.process_time()-cpu;report['factor_bytes']=factorbytes
  if method!='angular':report['factor_records']=fac.records;assert factorbytes==rec['predicted_cost']['persistent_tensor_bytes']
  else:report['angular_factor_breakdown']=fac.setup
  gc.collect();report['RSS_after_factor']=rss();report['factor_phase_peak_RSS_bytes']=p.rss()
  tick=time.perf_counter();report['operator_identity']=identities(ctx);assert report['operator_identity']==bridge['operator_identity']
  if method!='angular':report['basis_identity']=basis_identity(ctx['tree']);assert report['basis_identity']==bridge['basis_identity']
  report['cost']['identity_validation_seconds']=time.perf_counter()-tick;tick=time.perf_counter();refs=torch.load(HERE/'bridge/accepted_reference.pt',weights_only=True);report['cost']['reference_read_seconds']=time.perf_counter()-tick
  q,inputs=probes();report['probe_sha256']=p.thash(q);assert report['probe_sha256']==bridge['probe_sha256'];save()
  for repeat in range(3):
   for key,values in inputs.items():
    tick=time.perf_counter();cpu=time.process_time();x,status=old.repaired(original,raw,values,tolerance=1e-10,max_corrections=3);seconds=time.perf_counter()-tick;cpusec=time.process_time()-cpu
    tick=time.perf_counter();check=validate(values,x,original,refs[key]);checksec=time.perf_counter()-tick;row=dict(repeat=repeat,input=key,input_columns=values.shape[1],checked_inverse_wall_seconds=seconds,checked_inverse_CPU_seconds=cpusec,independent_check_seconds=checksec,status=status,**check);report['applications'].append(row);assert check['passed'] and status['success'],row
  report['application_phase_peak_RSS_bytes']=p.rss();report['raw_diagnostics']=[]
  for key,values in inputs.items():
   tick=time.perf_counter();cpu=time.process_time();z=raw(values);wall=time.perf_counter()-tick;cpusec=time.process_time()-cpu;tick=time.perf_counter();rel=p.relative((z-refs['raw_'+key]).T,refs['raw_'+key].T);orig=p.relative((values-original(z)).T,values.T);row=dict(input=key,raw_apply_wall_seconds=wall,raw_apply_CPU_seconds=cpusec,raw_difference_from_original_nested=rel.tolist(),raw_original_residual=orig.tolist())
   if method!='angular':row['raw_approximate_residual']=p.relative((values-approximate(z)).T,values.T).tolist();assert bool((rel<=1e-10).all()),row
   row['diagnostic_check_seconds']=time.perf_counter()-tick;report['raw_diagnostics'].append(row)
  report['complete']=True
 except BaseException as exc:report.update(error=repr(exc),traceback=traceback.format_exc());raise
 finally:
  report.update(process_wall_seconds=time.perf_counter()-START,process_CPU_seconds=time.process_time()-CPUSTART,peak_process_RSS_bytes=p.rss(),RSS_end=rss(),resource_at_end=snapshot())
  if 'prior' in locals():report.update(unchanged(prior))
  save();print(json.dumps(dict(method=method,label=label,pid=report['pid'],complete=report['complete'],factor_seconds=report['cost'].get('factor_wall_seconds'),applications=[r['checked_inverse_wall_seconds'] for r in report['applications']],process_seconds=report['process_wall_seconds'],error=report.get('error'))),flush=True)
if __name__=='__main__':main()
