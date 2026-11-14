import time
START=time.perf_counter();CPUSTART=time.process_time()
import sys,os,json,traceback,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'rte_layout_green_20261113_dot_v1'
sys.path.insert(0,str(OLD))
import layout_common as c
p=c.p;torch=c.torch;DT=c.DT

@torch.no_grad()
def main():
 out=HERE/'results';out.mkdir(exist_ok=False);report=dict(complete=False,pid=os.getpid(),runtime=c.configure(),resource_at_start=c.snapshot(),cost={},scope='One fixed witness for scalar closure; no full solve or speed comparison')
 def save():
  tmp=out/'results.json.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');os.replace(tmp,out/'results.json')
 try:
  bridge=json.loads((OLD/'bridge/results.json').read_text());prior=json.loads((OLD/'PRIOR_K_RESULTS.json').read_text());assert bridge['complete'];report['reference_bridge_sha256']=p.digest(OLD/'bridge/results.json');report['code_sha256']=p.digest(HERE/'run_closure.py');report['plan_sha256']=p.digest(HERE/'CLOSURE_PLAN.md')
  for f,h in bridge['source_hashes'].items():assert p.digest(OLD/f)==h,f
  for f,h in prior['source_hashes'].items():assert p.digest(p.ROOT/f)==h,f
  for f,h in prior['E_source_hashes'].items():assert p.digest(p.E/f)==h,f
  t=time.perf_counter();cpu=time.process_time();ctx=c.physical();assert ctx['background']==prior['background_identity'];identity=c.identities(ctx);assert identity==bridge['operator_identity'];report['operator_identity']=identity;report['cost']['physical_and_identity_wall_seconds']=time.perf_counter()-t;report['cost']['physical_and_identity_CPU_seconds']=time.process_time()-cpu
  n=128;l=12;K=ctx['K'];w=ctx['high'].weights;kn=ctx['kn'];normal=ctx['normal'];tangent=ctx['tangent'];kt=ctx['high'].ky.flatten()[0];tv=ctx['tv'];sv=ctx['sv'];R=w.to(DT);one=torch.ones(l,dtype=DT);P=one[:,None]*R[None,:]/R.sum();Q=torch.eye(l,dtype=DT)-P
  t=time.perf_counter();h=torch.zeros(l,dtype=DT);h[0]=1/torch.sqrt(torch.tensor(2.,dtype=torch.float64));h[1]=-h[0];f=torch.exp(2j*torch.pi*torch.arange(n,dtype=torch.float64)/n);psi=h[:,None]*f[None,:];source=sv[None,:]*(K@psi);input_average=R@psi;isotropic_null=one[:,None]*(R@psi)[None,:]
  assert bool((input_average==0).all()) and bool((isotropic_null==0).all());singular=torch.linalg.svdvals(K);report['algebra']=dict(weights=w.tolist(),normal=normal.tolist(),tangent=tangent.tolist(),witness_h=h.real.tolist(),spatial_mode=1,tangent_mode=0,input_average_norm=float(input_average.norm()),phase_singular_values=singular.tolist(),phase_rank=int(torch.linalg.matrix_rank(K)),K_on_mean_zero_spectral_norm=float(torch.linalg.matrix_norm(K@Q,ord=2)),collision_source_relative_to_input=float(source.norm()/psi.norm()),isotropic_null_norm=float(isotropic_null.norm()),psi_sha256=p.thash(psi),source_sha256=p.thash(source),K_mean_zero_tolerance=1e-12);report['cost']['witness_and_small_algebra_seconds']=time.perf_counter()-t
  t=time.perf_counter();D=torch.fft.ifft(1j*kn[:,None]*torch.fft.fft(torch.eye(n,dtype=DT),dim=0),dim=0);T=normal[:,None,None]*D[None]+torch.diag_embed(tv[None,:].expand(l,n).to(DT)+1j*kt*tangent[:,None]);report['cost']['directional_matrix_assembly_seconds']=time.perf_counter()-t;report['RSS_before_factor']=c.rss();t=time.perf_counter();cpu=time.process_time();lu,piv=torch.linalg.lu_factor(T);Y=torch.linalg.lu_solve(lu,piv,source[:,:,None]).squeeze(-1);report['cost']['twelve_directional_LU_and_solve_wall_seconds']=time.perf_counter()-t;report['cost']['twelve_directional_LU_and_solve_CPU_seconds']=time.process_time()-cpu;report['factor_phase_peak_RSS_bytes']=p.rss()
  t=time.perf_counter()
  def Lop(v):return normal[:,None]*torch.fft.ifft(1j*kn[None,:]*torch.fft.fft(v,dim=1),dim=1)+(tv[None,:]+1j*kt*tangent[:,None])*v
  matrix_bridge=float(((T@psi[:,:,None]).squeeze(-1)-Lop(psi)).norm()/Lop(psi).norm());res=p.relative(Lop(Y)-source,source);output_average=R@Y;relative=float(output_average.norm()/(R.norm()*Y.norm()));absolute=float(output_average.norm());report['closure']=dict(dense_FFT_action_relative=matrix_bridge,directional_original_relative_residual=res.tolist(),output_average_norm=absolute,output_average_normalized=relative,output_average_relative_to_input=float(output_average.norm()/psi.norm()),transport_update_norm=float(Y.norm()),source_norm=float(source.norm()),same_scalar_input=True,closure_falsified=bool(relative>1e-10 and absolute>1e-10),Y_sha256=p.thash(Y),mean_output_sha256=p.thash(output_average));assert matrix_bridge<1e-12 and bool((res<1e-12).all()),report['closure'];report['cost']['closure_and_original_checks_seconds']=time.perf_counter()-t
  report['logical_bytes']=dict(D=D.numel()*D.element_size(),directional_matrices=T.numel()*T.element_size(),LU=lu.numel()*lu.element_size(),pivots=piv.numel()*piv.element_size(),witness=psi.numel()*psi.element_size(),source=source.numel()*source.element_size(),response=Y.numel()*Y.element_size());report['complete']=True
 except BaseException as exc:report.update(error=repr(exc),traceback=traceback.format_exc());raise
 finally:
  report.update(process_wall_seconds=time.perf_counter()-START,process_CPU_seconds=time.process_time()-CPUSTART,peak_process_RSS_bytes=p.rss(),RSS_end=c.rss(),resource_at_end=c.snapshot())
  if 'prior' in locals():report.update(c.unchanged(prior))
  save();print(json.dumps(dict(complete=report['complete'],pid=report['pid'],closure=report.get('closure'),cost=report['cost'],process_seconds=report['process_wall_seconds'],error=report.get('error'))),flush=True)
if __name__=='__main__':main()
