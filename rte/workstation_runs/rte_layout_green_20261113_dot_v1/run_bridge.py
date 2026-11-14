import time
START=time.perf_counter();CPUSTART=time.process_time()
import traceback,json,os
from layout_common import *

@torch.no_grad()
def main():
 out=HERE/'bridge';out.mkdir(exist_ok=False);report=dict(complete=False,pid=os.getpid(),cost={},resource_at_start=snapshot(),runtime=configure())
 def save():
  temp=out/'results.json.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');os.replace(temp,out/'results.json')
 try:
  prior,prev,sealed=seal();report.update(sealed);report['old_small_unit']=nf.unit();tick=time.perf_counter();ctx,rec=nb.construct();report['cost']['construction_with_checks_seconds']=time.perf_counter()-tick;report['construction']=rec
  assert rec['Dhat_sha256']==prev['construction']['Dhat_sha256'] and rec['background']==prior['background_identity'] and rec['model_original_right_defect_bound']<=1e-3
  report['basis_identity']=basis_identity(ctx['tree']);report['operator_identity']=identities(ctx);V,original,approximate=operators(ctx)
  report['rotation_bridge']=[];rotate=pf.rotate
  def checked_rotate(G,Q,m,l):
   candidate=rotate(G,Q,m,l);left=torch.einsum('ip,pajb->iajb',Q.conj().T,G.reshape(m,l,m,l));ref=torch.einsum('iajb,jk->iakb',left,Q).reshape(m*l,m*l).contiguous();error=float((candidate-ref).norm()/ref.norm())
   row=dict(m=m,l=l,G_stride=list(G.stride()),Q_stride=list(Q.stride()),relative_error=error,explicit_pack_bytes=2*G.numel()*G.element_size(),arithmetic_same=True);report['rotation_bridge'].append(row);assert error<1e-12,row;return candidate
  tick=time.perf_counter();oldfac=nf.NestedFactor(ctx['tree'],V,ctx['normal']);report['cost']['old_factor_seconds']=time.perf_counter()-tick
  pf.rotate=checked_rotate
  try:
   tick=time.perf_counter();newfac=pf.PackedFactor(ctx['tree'],V,ctx['normal']);report['cost']['packed_factor_and_transform_checks_seconds']=time.perf_counter()-tick
  finally:pf.rotate=rotate
  assert len(report['rotation_bridge'])==14
  report['factor_bytes_old']=old.size_bytes(oldfac.tensors());report['factor_bytes_packed']=old.size_bytes(newfac.tensors());assert report['factor_bytes_old']==report['factor_bytes_packed']==rec['predicted_cost']['persistent_tensor_bytes']
  q,inputs=probes();prior_op=json.loads((HERE/'PRIOR_OPERATOR_RESULTS.json').read_text());report['probe_sha256']=p.thash(q);assert report['probe_sha256']==prior_op['rows'][0]['probe_sha256']
  refs={};report['action_bridge']=[];tick=time.perf_counter()
  for key,values in inputs.items():
   a=oldfac.apply(values);b=newfac.apply(values);rawdiff=p.relative((a-b).T,a.T);apx=p.relative((values-approximate(b)).T,values.T)
   xa,sta=old.repaired(original,oldfac.apply,values,tolerance=1e-10,max_corrections=3);xb,stb=old.repaired(original,newfac.apply,values,tolerance=1e-10,max_corrections=3)
   ca=validate(values,xa,original);cb=validate(values,xb,original,xa);row=dict(input=key,raw_difference=rawdiff.tolist(),raw_approximate_residual=apx.tolist(),old_status=sta,packed_status=stb,old_check=ca,packed_check=cb);report['action_bridge'].append(row)
   assert bool((rawdiff<=1e-10).all()) and ca['passed'] and cb['passed'],row
   refs[key]=xa;refs['raw_'+key]=a
  report['cost']['all_action_checks_seconds']=time.perf_counter()-tick;tick=time.perf_counter();torch.save(refs,out/'accepted_reference.pt');report['cost']['reference_save_seconds']=time.perf_counter()-tick
  report['complete']=True
 except BaseException as exc:report.update(error=repr(exc),traceback=traceback.format_exc());raise
 finally:
  report.update(process_wall_seconds=time.perf_counter()-START,process_CPU_seconds=time.process_time()-CPUSTART,peak_process_RSS_bytes=p.rss(),RSS_end=rss(),resource_at_end=snapshot())
  if 'prior' in locals():report.update(unchanged(prior))
  save();print(json.dumps(dict(complete=report['complete'],pid=report['pid'],seconds=report['process_wall_seconds'],error=report.get('error'))),flush=True)
if __name__=='__main__':main()
