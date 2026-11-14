"""Independent bounded apply-only directions; no solver or training calls."""
import time
START=time.perf_counter()
import argparse,json,pathlib,resource,sys,socket,subprocess
import numpy as np
import torch
from batch_gpu_common import sha,aid,write,timing,load_case,Snapshot,best_real_line,sync
from rte.train_source_predictor import build_base

ROOT=pathlib.Path(__file__).resolve().parent.parent
def prolong(field,n):
 m=field.shape[-1];assert field.shape[-2]==m and n%m==0
 f=torch.fft.fftn(field,dim=(-2,-1))
 f=f.clone();f[...,m//2,:]=0;f[...,:,m//2]=0
 out=torch.zeros((*field.shape[:-2],n,n),dtype=torch.complex128,device=field.device)
 k=torch.cat((torch.arange(m//2,device=field.device),torch.arange(n-m//2,n,device=field.device)))
 out[...,k[:,None],k[None,:]]=f*((n/m)**2)
 return torch.fft.ifftn(out,dim=(-2,-1))

@torch.no_grad()
def run(mode):
 policy=json.loads((ROOT/'policy.json').read_text());job=ROOT/('jobC'if mode=='local'else'jobD')
 assert socket.gethostname()=='FWI-Standard-PC-i440FX-PIIX-1996'
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==policy['head']
 for name,h in policy['code_sha256'].items():assert sha(ROOT/name)==h,name
 torch.set_num_threads(1);deadline=START+policy['gpu_job_soft_seconds']
 job.mkdir(exist_ok=False);snap=Snapshot(job/'candidates',policy['max_gpu_job_snapshot_bytes'])
 ledger=json.loads((ROOT/'jobA/snapshot_ledger.json').read_text());assert ledger['complete']
 cases=[f for f in ledger['fixtures']if f['size']in(128,256)]
 teacher_parent=json.loads((ROOT/'jobA/policy.json').read_text())['teacher_banks']['64']
 rows=[];setup=[];identities=[];counts=dict(H=0,M=0,S=0,full_solves=0,training_steps=0,new_seed_generation=0)
 def limit():
  assert time.perf_counter()<deadline,'GPU direction soft deadline exceeded'
  assert torch.cuda.max_memory_reserved()<=policy['max_gpu_reserved_bytes'],'GPU reserved cap exceeded'
 def H(high,x):
  limit();y,seconds=timing(lambda:high.A_op(x));counts['H']+=1
  assert counts['H']<=policy['max_gpu_job_H_actions'];return y,seconds
 sys.path.insert(0,policy['frozen_capture_code'])
 from direct_segmented_tsa import DirectSegmentedTSA
 try:
  for item in cases:
   limit();n,i=item['size'],item['index'];tag=f'n{n}_id{i}'
   (low,high,a),load_seconds=timing(lambda:load_case(ROOT,item))
   depth=8 if n==128 else 16
   began=time.perf_counter();base=build_base(low,'tsa'+str(depth));wrapped=DirectSegmentedTSA(base,low.source,chunk=1,capture_first=False);sync()
   base_setup=time.perf_counter()-began
   def M(r):return wrapped(r.to(torch.complex64)).to(torch.complex128)
   q=high.source
   b,base_seconds=timing(lambda:M(q));counts['M']+=1
   began=time.perf_counter();b=snap.save(tag+'_strong_source',b);persist_base=time.perf_counter()-began
   Hb,hb_seconds=H(high,b);frontier=q-Hb
   if mode=='local':
    from mloc import LocalCollisionReplacement
    candidate,construction_seconds=timing(lambda:LocalCollisionReplacement(high))
    maps={'strong':M,'reference':candidate.reference_apply,'material_local':candidate}
    candidate_diagnostics=candidate.diagnostics
   else:
    began=time.perf_counter();assert sha(teacher_parent['path'])==teacher_parent['sha256']
    coarse_payload=torch.load(teacher_parent['path'],map_location='cpu',weights_only=True);bank=coarse_payload['bank']
    target_parent=item['parent'];assert sha(target_parent['path'])==target_parent['sha256']
    target_payload=torch.load(target_parent['path'],map_location='cpu',weights_only=True)
    target_data=target_payload['data'] if n==128 else target_payload
    source_parameter_identity={}
    for key in ('centers','phases','amplitudes','angular_codes','geometry_base64'):
     assert torch.equal(coarse_payload['data'][key][i:i+1],target_data[key][i:i+1]),'Crossgrid source/geometry mismatch: '+key
     source_parameter_identity[key]=True
    assert bank['target'].dtype==torch.complex128
    teacher=bank['target'][i:i+1].to('cuda');ref=bank['reference'][i:i+1].to(device='cuda',dtype=torch.complex128)
    phi_teacher=prolong(teacher,n);phi_reference=prolong(ref,n);sync()
    construction_seconds=time.perf_counter()-began
    # Exact chosen Fourier interpolation retains the old grid samples; no
    # source generalization or continuous-geometry claim is made.
    coarse_spectrum=torch.fft.fftn(teacher,dim=(-2,-1));coarse_spectrum[...,32,:]=0;coarse_spectrum[...,:,32]=0
    filtered_teacher=torch.fft.ifftn(coarse_spectrum,dim=(-2,-1))
    interpolation=float(torch.linalg.vector_norm(phi_teacher[...,::n//64,::n//64]-filtered_teacher)/torch.linalg.vector_norm(filtered_teacher))
    assert interpolation<=1e-12
    maps={'teacher_column':lambda r:phi_teacher,'reference_column':lambda r:phi_reference}
    candidate_diagnostics=dict(coarse_size=64,full_angles=12,interpolation_relative_error=interpolation,
      reused_same_development_source_teacher=True,old_teacher_acquisition_seconds=None,
      acquisition_cost_unknown_not_zero=True,no_new_teacher_solve=True,crossgrid_source_geometry_parameter_identity=source_parameter_identity,
      frontier_role='Fixed trial column for original-source response, not a frontier response teacher',
      Nyquist_rule='Identically remove coarse negative-Nyquist rows/columns from both teacher and analytic response before normalized Fourier embedding',
      filtered_teacher_relative_difference=float(torch.linalg.vector_norm(filtered_teacher-teacher)/torch.linalg.vector_norm(teacher)))
    del bank,coarse_payload,target_payload,target_data
   setup.append(dict(size=n,index=i,input_readback_and_physical_setup_seconds=load_seconds,strong_base_setup_seconds=base_setup,
     strong_source_apply_seconds=base_seconds,strong_source_persistence_seconds=persist_base,strong_source_H_seconds=hb_seconds,
     candidate_setup_seconds=construction_seconds,candidate_diagnostics=candidate_diagnostics))
   for context,r in [('source',q),('one_strong_action_defect',frontier)]:
    assert float(torch.linalg.vector_norm(r))>0
    fields={};images={}
    for name,apply in maps.items():
     limit();times=[];x=None
     for rep in range(3):
      x,elapsed=timing(lambda:apply(r));times.append(elapsed)
      if name=='strong':counts['M']+=1
     began=time.perf_counter();x=snap.save(tag+'_'+context+'_'+name,x.to(torch.complex128));persist=time.perf_counter()-began
     y,hseconds=H(high,x);alpha,ratio=best_real_line(r,y)
     fields[name]=x;images[name]=y
     began=time.perf_counter()
     proposed_field=alpha*x if context=='source' else b+alpha*x
     reconstruction=snap.save(tag+'_'+context+'_'+name+'_line',proposed_field)
     line_save=time.perf_counter()-began
     actual,actual_seconds=H(high,reconstruction)
     actual_ratio=float(torch.linalg.vector_norm(q-actual)/torch.linalg.vector_norm(r))
     expected_image=alpha*y if context=='source' else Hb+alpha*y
     image_linear_gap=float(torch.linalg.vector_norm(actual-expected_image)/torch.linalg.vector_norm(r))
     assert image_linear_gap<=1e-10,'Constructed proposal image consistency failed'
     assert bool(torch.isfinite(actual).all())
     rows.append(dict(size=n,index=i,context=context,map=name,apply_seconds=times,median_apply_seconds=sorted(times)[1],
       candidate_persistence_readback_seconds=persist,original_H_seconds=hseconds,line_coefficient=alpha,
       measured_image_best_real_line_ratio=ratio,constructed_field_actual_H_line_ratio=actual_ratio,
       constructed_line_persistence_seconds=line_save,constructed_line_H_seconds=actual_seconds,
       constructed_image_linearity_gap=image_linear_gap,ordinary_fp64_diagnostic=True,
       certification_established=False,no_fullsolver_result=True))
     write(job/'results.json',rows);write(job/'setup.json',setup);limit()
    if mode=='local':
     began=time.perf_counter();v=fields['reference'];z=fields['material_local']
     diff=snap.save(tag+'_'+context+'_commutator_input',z-v)
     def stream(field):
      counts['S']+=1
      return torch.fft.ifftn(candidate.symbol*torch.fft.fftn(field,dim=(-2,-1)),dim=(-2,-1))
     def C(field,t,s):
      return t*field-s*torch.einsum('ij,bjhw->bihw',high.phase_matrix.to(torch.complex128),high.weights[None,:,None,None]*field)
     cref=C(v,candidate.tref,candidate.sref);e_ref=r-stream(v)-cref
     e_local=C(z,high.sigma_t,high.sigma_s)-cref
     penalty=stream(diff);actual_residual=r-images['material_local']
     relative=lambda x:float(torch.linalg.vector_norm(x)/torch.linalg.vector_norm(r))
     gap=relative(actual_residual-(e_ref-penalty-e_local))
     assert gap<=1e-10,'Actual-H commutator diagnostic identity failed'
     identities.append(dict(size=n,index=i,context=context,reference_closure_relative=relative(e_ref),
       local_collision_closure_relative=relative(e_local),streaming_material_penalty_relative=relative(penalty),
       actual_residual_relative=relative(actual_residual),identity_discrepancy_relative=gap,
       diagnostic_seconds=time.perf_counter()-began,ordinary_fp64_diagnostic_not_proof=True))
     write(job/'commutator.json',identities);limit()
    del fields,images
   del low,high,a,base,wrapped,maps
   if mode=='local':del candidate
  snap.finish();limit()
  write(job/'settings.json',dict(complete=True,mode=mode,counts=counts,elapsed_seconds=time.perf_counter()-START,
   cpu_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
   gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),warm_apply_is_component_only=True,
   entire_job_includes_startup_setup_readback_transfers_snapshots_and_H_diagnostics=True,
   no_solution_residual_accuracy_or_pair_claim=True,full_solves=0,training_steps=0))
 except BaseException as exc:
  write(job/'settings.json',dict(complete=False,mode=mode,counts=counts,elapsed_seconds=time.perf_counter()-START,
    error=dict(type=type(exc).__name__,message=str(exc))))
  raise
 print(json.dumps(dict(complete=True,mode=mode,rows=len(rows),counts=counts)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['local','coarse']);run(p.parse_args().mode)
