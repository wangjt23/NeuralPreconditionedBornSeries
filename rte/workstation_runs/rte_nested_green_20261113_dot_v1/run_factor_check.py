import time
START=time.perf_counter()
import os,sys,json,traceback,gc
from pathlib import Path
import nested_factor as nf
nb=nf.nb;p=nb.p;torch=nb.torch;old=nb.old;DT=nb.DT;HERE=Path(__file__).resolve().parent

def snapshot():
 # Aggregate proc counters avoid a whole-host process scan in the timed worker.
 info=dict(loadavg=Path('/proc/loadavg').read_text().strip(),uptime=Path('/proc/uptime').read_text().strip())
 info['memory']=[x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemAvailable:','MemFree:','SwapFree:'))]
 info['cpu0']=next(x for x in Path('/proc/stat').read_text().splitlines() if x.startswith('cpu0 '));return info

def rss():
 return int(next(x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')).split()[1])*1024

@torch.no_grad()
def main():
 method=sys.argv[1];assert method in ['nested','angular'];torch.set_num_threads(1);torch.set_num_interop_threads(1);assert set(os.sched_getaffinity(0))=={0} and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 out=HERE/method;out.mkdir(exist_ok=False);report=dict(complete=False,pid=os.getpid(),method=method,n=128,k=0,threads=1,cpu_affinity=[0],CUDA_disabled=True,no_physical_RHS=True,no_dense_inverse=True,cost={},resource_at_start=snapshot())
 def save():
  temp=out/'results.json.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');os.replace(temp,out/'results.json')
 try:
  prior=json.loads((HERE/'PRIOR_K_RESULTS.json').read_text());pre=json.loads((HERE/'preflight/results.json').read_text());oldrec=json.loads((HERE/'PRIOR_OPERATOR_RESULTS.json').read_text());assert pre['complete'] and pre['feasible_for_one_factor_check']
  for f,h in prior['source_hashes'].items():assert p.digest(p.ROOT/f)==h,f
  for f,h in prior['E_source_hashes'].items():assert p.digest(p.E/f)==h,f
  assert p.digest(HERE/'nested_basis.py')==pre['source_hashes']['nested_basis.py'];report.update(source_hashes={x.name:p.digest(x) for x in HERE.glob('*.py')},plan_sha256=p.digest(HERE/'NESTED_PLAN.md'),preflight_sha256=p.digest(HERE/'preflight/results.json'))
  n=128;l=12;dim=n*l
  if method=='nested':
   report['unit']=nf.unit();ctx,rec=nb.construct();assert rec['background']==prior['background_identity'];assert rec['predicted_cost']['persistent_tensor_bytes']==pre['construction']['predicted_cost']['persistent_tensor_bytes'];assert rec['model_original_right_defect_bound']<=1e-3
   report['construction']=rec;report['cost'].update(rec['times']);tv=ctx['tv'];sv=ctx['sv'];K=ctx['K'];kn=ctx['kn'];normal=ctx['normal'];tangent=ctx['tangent'];high=ctx['high'];kt=high.ky.flatten()[0];V=tv[:,None,None]*torch.eye(l,dtype=DT)-sv[:,None,None]*K[None]+1j*kt*torch.diag(tangent.to(DT))[None]
   report['RSS_before_factor']=rss();tick=time.perf_counter();fac=nf.NestedFactor(ctx['tree'],V,normal);report['cost']['factor_seconds']=time.perf_counter()-tick;report['factor_bytes']=old.size_bytes(fac.tensors());assert report['factor_bytes']==rec['predicted_cost']['persistent_tensor_bytes'];report['factor_records']=fac.records;raw=fac.apply;Dh=ctx['Dhat']
   def approximate(q):
    x=q.reshape(n,l,-1);der=torch.einsum('ij,jlr->ilr',Dh,x);return (der*normal[None,:,None]+torch.einsum('nij,njr->nir',V,x)).reshape_as(q)
  else:
   assert json.loads((HERE/'nested/results.json').read_text())['complete'];tick=time.perf_counter();total,scatter,_=p.medium(n,1,10/3,'cpu');low=p.PeriodicRTE2D(shape=(n,n),dx=1/n,n_angles=l,anisotropy_g=.65,eta_scale=0.,eta_floor=0.,device='cpu');low.setup(total,scatter,torch.zeros((1,l,n,n),dtype=torch.complex64));high=p.high_precision_copy(low);bg,tb,sb=p.background(low);assert bg==prior['background_identity'];tv=tb[0,0,:,0].double();sv=sb[0,0,:,0].double();K=(high.phase_matrix*high.weights[None]).to(DT);kn=high.kx.flatten();normal=high.directions[:,0];tangent=high.directions[:,1];kt=high.ky.flatten()[0];V=tv[:,None,None]*torch.eye(l,dtype=DT)-sv[:,None,None]*K[None]+1j*kt*torch.diag(tangent.to(DT))[None];report['cost']['physical_setup_seconds']=time.perf_counter()-tick;report['RSS_before_factor']=rss();tick=time.perf_counter();fac=old.AngularBandMode(high,tv,sv,0);report['cost']['factor_seconds']=time.perf_counter()-tick;report['factor_bytes']=fac.bytes
   def raw(q):return fac.raw(q.reshape(n,l,-1).permute(1,0,2).reshape(dim,-1)).reshape(l,n,-1).permute(1,0,2).reshape(dim,-1)
  def original(q):
   x=q.reshape(n,l,-1);der=torch.fft.ifft(1j*kn[:,None,None]*torch.fft.fft(x,dim=0),dim=0);return (der*normal[None,:,None]+torch.einsum('nij,njr->nir',V,x)).reshape_as(q)
  gc.collect();report['RSS_after_factor']=rss();report['factor_phase_peak_RSS_bytes']=p.rss();save()
  gen=torch.Generator().manual_seed(2026111312);q=torch.complex(torch.randn(dim,3,dtype=torch.float64,generator=gen),torch.randn(dim,3,dtype=torch.float64,generator=gen));q=q/q.norm(dim=0);mixed=torch.cat([torch.zeros(dim,1,dtype=DT),q,1e-200*q[:,:1]],1);report['probe_sha256']=p.thash(q);assert report['probe_sha256']==oldrec['rows'][0]['probe_sha256'];ref=None
  if method=='angular':
   tick=time.perf_counter();ref=torch.load(HERE/'nested/accepted_reference.pt',weights_only=True);report['cost']['reference_read_seconds']=time.perf_counter()-tick
  report['applications']=[]
  for repeat in range(3):
   tick=time.perf_counter();x,status=old.repaired(original,raw,mixed,tolerance=1e-10,max_corrections=3);elapsed=time.perf_counter()-tick;tick=time.perf_counter();rel=p.relative((mixed-original(x)).T,mixed.T);zero=bool((x[:,0]==0).all());tiny=bool(x[:,-1].abs().max()>0)
   if ref is None:ref=x.clone()
   diff=p.relative((x-ref).T,ref.T);passed=bool((rel<=1e-10).all()) and bool((diff<=1e-9).all()) and zero and tiny and bool(torch.isfinite(x).all());report['applications'].append(dict(repeat=repeat,checked_inverse_seconds=elapsed,independent_check_seconds=time.perf_counter()-tick,status=status,original_relative=rel.tolist(),reference_difference=diff.tolist(),zero=zero,tiny=tiny,passed=passed));assert passed,report['applications'][-1]
  report['application_phase_peak_RSS_bytes']=p.rss()
  if method=='nested':
   tick=time.perf_counter();torch.save(ref,out/'accepted_reference.pt');report['cost']['reference_save_seconds']=time.perf_counter()-tick;tick=time.perf_counter();z=raw(q);report['raw_original_relative']=p.relative((q-original(z)).T,q.T).tolist();report['raw_approximate_relative']=p.relative((q-approximate(z)).T,q.T).tolist();report['cost']['raw_defect_diagnostic_seconds']=time.perf_counter()-tick;z2,counts=nf.count_application(fac,q);report['call_count']=counts;report['instrumentation_bridge']=float((z-z2).norm()/z.norm());assert report['instrumentation_bridge']<1e-12 and counts['upward_nodes']==31 and counts['downward_nodes']==31
  report['complete']=True
 except BaseException as exc:report.update(error=repr(exc),traceback=traceback.format_exc());raise
 finally:
  report.update(process_seconds=time.perf_counter()-START,peak_process_RSS_bytes=p.rss(),RSS_end=rss(),resource_at_end=snapshot())
  if 'prior' in locals():report.update(production_unchanged=all(p.digest(p.ROOT/f)==h for f,h in prior['source_hashes'].items()),E_unchanged=all(p.digest(p.E/f)==h for f,h in prior['E_source_hashes'].items()))
  save();print(json.dumps(dict(method=method,complete=report['complete'],factor_bytes=report.get('factor_bytes'),factor_seconds=report['cost'].get('factor_seconds'),application_seconds=[r['checked_inverse_seconds'] for r in report.get('applications',[])],process_seconds=report['process_seconds'],error=report.get('error'))),flush=True)
if __name__=='__main__':main()
