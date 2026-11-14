"""FWI-only frozen snapshot construction and separately started certification."""
import time
START=time.perf_counter()
import argparse, hashlib, json, os, pathlib, resource, socket, subprocess, sys
from fractions import Fraction as F
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parent.parent
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def identity(a):
 a=np.ascontiguousarray(a)
 h=hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
 return dict(dtype=str(a.dtype),shape=list(a.shape),bytes=a.nbytes,sha256=h)
def write(path,obj):
 path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def validation():
 p=json.loads((ROOT/'policy.json').read_text())
 assert socket.gethostname()=='FWI-Standard-PC-i440FX-PIIX-1996'
 assert subprocess.check_output(['id','-un'],text=True).strip()=='wjt'
 assert pathlib.Path.cwd()==pathlib.Path('/data/wjt/NeuralPreconditionedBornSeries')
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==p['head']
 assert p['max_full_solves']==0 and p['development_ids']==[0,3]
 assert p['tau_rational']==[1,100000] and p['reserve_rational']==[1,100]
 for n,h in p['code_sha256'].items():assert sha(ROOT/n)==h,n
 return p
def freeze():
 p=validation(); deadline=START+p['freeze_soft_seconds']
 import torch
 from rte.mixed_precision_transport import high_precision_copy
 from rte.run_large_angular_spectral_experiments import ExperimentConfig, make_operator
 from rte.train_source_predictor import build_base, material_operator
 torch.set_num_threads(1)
 destination=ROOT/'fixtures';destination.mkdir(exist_ok=False)
 ledger=[];costs={k:0. for k in ('parent_hash_load','physical_setup','capture_setup','M','H','transfers','snapshot_save','snapshot_hash')}
 counts=dict(full_solves=0,training_steps=0,new_seed_generation=0,M=0,H=0)
 def limit():
  assert time.perf_counter()<deadline,'Freeze soft deadline exceeded'
  assert torch.cuda.max_memory_reserved()<=p['max_reserved_bytes'],'GPU bound exceeded'
 def array(t):return t.detach().cpu().numpy().copy()
 def snapshot(n,i,kind,parent,high,b,q):
  limit();torch.cuda.synchronize();began=time.perf_counter()
  # Perturb actual represented GPU bits. These are checker controls, not solves.
  a=(b*(1.+2.**-20)).clone();c=(b*(1.+2.**-8)).clone()
  arrays=dict(source=array(q),baseline=array(b),positive=array(a),negative=array(c),
    original_source=array(high.source),stream_symbol=array(high._stream_symbol(1,torch.complex128,'cuda')),
    sigma_t=array(high.sigma_t),sigma_s=array(high.sigma_s),phase=array(high.phase_matrix),weights=array(high.weights))
  costs['transfers']+=time.perf_counter()-began
  name=f'n{n}_id{i}.npz';path=destination/name
  began=time.perf_counter()
  with path.open('xb') as f:np.savez(f,**arrays)
  costs['snapshot_save']+=time.perf_counter()-began
  began=time.perf_counter();digest=sha(path);ids={k:identity(v) for k,v in arrays.items()}
  costs['snapshot_hash']+=time.perf_counter()-began
  ledger.append(dict(size=n,index=i,kind=kind,path='fixtures/'+name,sha256=digest,bytes=path.stat().st_size,
    arrays=ids,parent=parent,provenance='Saved development teacher' if n<256 else 'New M(original stored source), q=rounded GPU H(b); exact newly persisted bits; no old transient replay claim',
    paired_role='Manufactured engineering perturbations only; no independent baseline/candidate solver comparison',
    positive_scale_rational=str(F(1)+F(1,2**20)),negative_scale_rational=str(F(1)+F(1,2**8))))
  # Release all in-memory snapshots before any certifier is started.
  del arrays,a,c
 for n,parent in p['teacher_banks'].items():
  n=int(n);limit();began=time.perf_counter();assert sha(parent['path'])==parent['sha256']
  data=torch.load(parent['path'],map_location='cpu',weights_only=True);bank=data['bank']
  assert bank['target'].dtype==torch.complex128 and bank['q'].dtype==torch.complex64
  assert bank['q'].shape==(16,12,n,n)
  costs['parent_hash_load']+=time.perf_counter()-began
  cfg=ExperimentConfig(size=n,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes='mixed')
  for i in p['development_ids']:
   torch.cuda.synchronize();began=time.perf_counter()
   t,s,q,b=[bank[k][i:i+1].to('cuda')for k in ('t','s','q','target')]
   low=make_operator(cfg,t,s,q,'cuda');high=high_precision_copy(low)
   assert torch.equal(high.source,q.to(torch.complex128));torch.cuda.synchronize()
   costs['physical_setup']+=time.perf_counter()-began
   snapshot(n,i,'saved_teacher',parent,high,b,high.source)
   del t,s,q,b,low,high
  del bank,data
 parent=p['n256_material_parent'];limit();began=time.perf_counter()
 assert sha(parent['path'])==parent['sha256']
 data=torch.load(parent['path'],map_location='cpu',weights_only=True)
 costs['parent_hash_load']+=time.perf_counter()-began
 cfg=ExperimentConfig(size=256,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes='mixed')
 sys.path.insert(0,p['frozen_capture_code'])
 from direct_segmented_tsa import DirectSegmentedTSA
 for i in p['development_ids']:
  limit();torch.cuda.synchronize();began=time.perf_counter()
  low,high=material_operator(cfg,data,torch.tensor([i]),.1,'cuda');torch.cuda.synchronize()
  costs['physical_setup']+=time.perf_counter()-began
  began=time.perf_counter();base=build_base(low,'tsa16');wrapper=DirectSegmentedTSA(base,low.source,chunk=1,capture_first=False)
  torch.cuda.synchronize();costs['capture_setup']+=time.perf_counter()-began
  began=time.perf_counter();b=wrapper(low.source).clone();torch.cuda.synchronize()
  costs['M']+=time.perf_counter()-began;counts['M']+=1
  assert b.dtype==torch.complex128
  began=time.perf_counter();q=high.A_op(b).clone();torch.cuda.synchronize()
  costs['H']+=time.perf_counter()-began;counts['H']+=1
  snapshot(256,i,'new_manufactured_M_source',parent,high,b,q)
  del low,high,base,wrapper,b,q
 limit();assert len(ledger)==6 and counts['M']==counts['H']==2
 receipt=dict(complete=True,all_snapshots_persisted_before_checker=True,fixtures=ledger,costs=costs,counts=counts,
   elapsed_seconds=time.perf_counter()-START,cpu_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
   gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
   source_arrays_exported=False,checker_calls=0,torch=str(torch.__version__),numpy=str(np.__version__),cuda=str(torch.version.cuda))
 write(ROOT/'snapshot_ledger.json',receipt)
 print(json.dumps(dict(complete=True,fixtures=len(ledger),counts=counts,elapsed_seconds=receipt['elapsed_seconds'])))
def check():
 p=validation();deadline=START+p['check_soft_seconds']
 from stagewise_fft import certify_residual,paired_field_check,load_native_cpu_backend
 assert sha(ROOT/'code/native_cpu.so')==json.loads((ROOT.parent/'preflight_receipt.json').read_text())['native_sha256']
 guard=load_native_cpu_backend(ROOT/'code/native_cpu.so')
 result=ROOT/'results';result.mkdir(exist_ok=False)
 began=time.perf_counter();ledger_path=ROOT/'snapshot_ledger.json';ledger=json.loads(ledger_path.read_text())
 assert ledger['complete'] and ledger['all_snapshots_persisted_before_checker'] and ledger['checker_calls']==0
 ledger_hash=sha(ledger_path);ledger_load_seconds=time.perf_counter()-began
 rows=[];pairs=[];costs=dict(readback_hash_seconds=0.,certificates_seconds=0.,direct_pair_controls_seconds=0.,final_snapshot_hash_seconds=0.,json_persistence_seconds=0.)
 first_ready_seconds=time.perf_counter()-START
 def save(complete,error=None):
  began=time.perf_counter();write(result/'certificates.json',rows);write(result/'pairs.json',pairs)
  costs['json_persistence_seconds']+=time.perf_counter()-began
  write(result/'settings.json',dict(complete=complete,error=error,certificates=len(rows),paired_controls=len(pairs),costs=costs,
   elapsed_seconds=time.perf_counter()-START,first_checker_ready_seconds=first_ready_seconds,ledger_load_seconds=ledger_load_seconds,
   snapshot_ledger_sha256=ledger_hash,native_guard=guard,native_sha256=sha(ROOT/'code/native_cpu.so'),
   cpu_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,full_solves=0,training_steps=0,
   source_arrays_exported=False,paired_solver_comparison=False,global_gpu_contract_established=False,
   cost_scope='Entire fresh CPU checker process including imports/startup, readback/hash, both N256 certificates, direct paired gates and JSON; generation/transfers/snapshot costs separately recorded and included by orchestrator'))
 try:
  for fixture in ledger['fixtures']:
   assert time.perf_counter()<deadline,'Checker soft deadline exceeded'
   began=time.perf_counter();path=ROOT/fixture['path'];assert sha(path)==fixture['sha256']
   with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
   assert {k:identity(v)for k,v in a.items()}==fixture['arrays']
   read_seconds=time.perf_counter()-began;costs['readback_hash_seconds']+=read_seconds
   operands={k:a[k] for k in ('source','stream_symbol','sigma_t','sigma_s','phase','weights')}
   targets=['baseline','positive'] if fixture['size']==256 else ['baseline']
   for role in targets:
    assert time.perf_counter()<deadline,'Checker soft deadline exceeded'
    began=time.perf_counter();cert=certify_residual(candidate=a[role],baseline=a['baseline'],**operands)
    elapsed=time.perf_counter()-began;costs['certificates_seconds']+=elapsed
    row=dict(size=fixture['size'],index=fixture['index'],kind=fixture['kind'],role=role,
      snapshot_path=fixture['path'],snapshot_sha256=fixture['sha256'],certificate=cert,certificate_call_seconds=elapsed,
      paired_role='Self-baseline engineering gate' if role=='baseline' else 'Manufactured represented perturbation engineering gate')
    rows.append(row);save(False)
    assert time.perf_counter()<deadline,'Checker soft deadline exceeded after certificate'
   for role in ('positive','negative'):
    assert time.perf_counter()<deadline,'Checker soft deadline exceeded before paired control'
    began=time.perf_counter();pair=paired_field_check(a[role],a['baseline'])
    elapsed=time.perf_counter()-began;costs['direct_pair_controls_seconds']+=elapsed
    pairs.append(dict(size=fixture['size'],index=fixture['index'],role=role,seconds=elapsed,**pair))
    if role=='positive':assert pair['paired_field_accepted']
    else:assert pair['paired_field_rigorous_rejection']
    assert time.perf_counter()<deadline,'Checker soft deadline exceeded after paired control'
   began=time.perf_counter();assert sha(path)==fixture['sha256']
   assert {k:identity(v)for k,v in a.items()}==fixture['arrays'],'Input mutation'
   costs['final_snapshot_hash_seconds']+=time.perf_counter()-began
   del a,operands;save(False)
  assert len(rows)==8 and len(pairs)==12
  assert sha(ledger_path)==ledger_hash
  assert time.perf_counter()<deadline,'Checker soft deadline exceeded before completion'
  save(True)
 except BaseException as exc:
  save(False,dict(type=type(exc).__name__,message=str(exc)));raise
 print(json.dumps(dict(complete=True,certificates=len(rows),paired_controls=len(pairs),elapsed_seconds=time.perf_counter()-START)))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','check']);args=parser.parse_args()
 (freeze if args.mode=='freeze' else check)()
