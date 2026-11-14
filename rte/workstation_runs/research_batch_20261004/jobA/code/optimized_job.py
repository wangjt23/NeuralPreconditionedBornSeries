"""Fresh upper-only checker process on the SAME frozen N256 operand bits."""
import time
START=time.perf_counter()
import json,pathlib,resource,socket,subprocess
import numpy as np
from stagewise_driver import sha,identity,write
from optimized_stagewise import certify_residual,load_native_cpu_backend
ROOT=pathlib.Path(__file__).resolve().parent.parent
def run():
 p=json.loads((ROOT/'policy.json').read_text());deadline=START+p['optimized_soft_seconds']
 assert socket.gethostname()=='FWI-Standard-PC-i440FX-PIIX-1996'
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==p['head']
 for name,h in p['code_sha256'].items():assert sha(ROOT/name)==h,name
 job=ROOT/'jobB';job.mkdir(exist_ok=False)
 assert sha(ROOT/'jobA/code/native_cpu.so')==json.loads((ROOT/'preflight_receipt.json').read_text())['native_sha256']
 guard=load_native_cpu_backend(ROOT/'jobA/code/native_cpu.so')
 ledgerpath=ROOT/'jobA/snapshot_ledger.json';ledgerhash=sha(ledgerpath)
 ledger=json.loads(ledgerpath.read_text());assert ledger['complete'] and ledger['checker_calls']==0
 ready=time.perf_counter()-START;rows=[];costs=dict(readback_and_hash_seconds=0.,certificate_seconds=0.,post_hash_seconds=0.)
 def save(complete,error=None):
  write(job/'certificates.json',rows)
  write(job/'settings.json',dict(complete=complete,error=error,costs=costs,certificates=len(rows),elapsed_seconds=time.perf_counter()-START,
    first_checker_ready_seconds=ready,cpu_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    ledger_sha256=ledgerhash,native_guard=guard,native_sha256=sha(ROOT/'jobA/code/native_cpu.so'),
    full_solves=0,training_steps=0,H_actions=0,M_actions=0,input_bits_shared_with_jobA=True,
    cost_scope='Entire cold process/imports, common frozen fixture readback/hashes, all four complete certificates including paired gates and serialization'))
 try:
  for item in ledger['fixtures']:
   if item['size']!=256:continue
   assert time.perf_counter()<deadline
   began=time.perf_counter();path=ROOT/'jobA'/item['path'];assert sha(path)==item['sha256']
   with np.load(path,allow_pickle=False)as z:a={k:z[k].copy()for k in z.files}
   assert {k:identity(v)for k,v in a.items()}==item['arrays']
   costs['readback_and_hash_seconds']+=time.perf_counter()-began
   args={k:a[k]for k in ('source','stream_symbol','sigma_t','sigma_s','phase','weights')}
   for role in ('baseline','positive'):
    assert time.perf_counter()<deadline
    began=time.perf_counter();c=certify_residual(candidate=a[role],baseline=a['baseline'],**args)
    elapsed=time.perf_counter()-began;costs['certificate_seconds']+=elapsed
    rows.append(dict(size=256,index=item['index'],role=role,snapshot_sha256=item['sha256'],certificate=c,certificate_call_seconds=elapsed))
    save(False);assert time.perf_counter()<deadline,'Optimized soft deadline exceeded after certificate'
   began=time.perf_counter();assert sha(path)==item['sha256']
   assert {k:identity(v)for k,v in a.items()}==item['arrays']
   costs['post_hash_seconds']+=time.perf_counter()-began
  assert len(rows)==4 and sha(ledgerpath)==ledgerhash
  assert time.perf_counter()<deadline;save(True)
 except BaseException as e:
  save(False,dict(type=type(e).__name__,message=str(e)));raise
 print(json.dumps(dict(complete=True,certificates=len(rows),elapsed_seconds=time.perf_counter()-START)))
if __name__=='__main__':run()
