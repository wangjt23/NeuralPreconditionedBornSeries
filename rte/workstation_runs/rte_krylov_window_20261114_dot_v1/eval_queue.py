"""Independent native FGMRES checkpoint evaluation queue. No optimizer or selection."""
import os,math
from trainer_skeleton import *
from rte.npbs_unet_ablation import fgmres_real_stacked_with_trace
HERE=Path(__file__).parent
RHS=ROOT/'rte/workstation_runs/rte_rhs_holdout_20261114_dot_v1/inputs.pt'
TARGETS=[128,512,1024,2048]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));os.replace(tmp,p)
def evaluate(cp,target):
 ck=torch.load(cp,weights_only=True);assert ck['update']==target
 r=Runner(146999);r.model.load_state_dict(ck['model']);r.model.eval();base=torch.load(OLD/'data.pt',weights_only=True)
 panels=[('TRAIN8',base['train'],list(range(8)),None)]
 if target==2048:
  panels.append(('DEV32',base['validation'],list(range(32)),None))
  h=torch.load(RHS,weights_only=True);panels.append(('RHS_holdout',dict(sigma_t=h['sigma_t'],sigma_s=h['sigma_s']),list(range(8)),h['sources']))
 p=cp.parent/f'eval_native_{target}.json'
 report=dict(complete=False,checkpoint=str(cp),checkpoint_sha256=sha(cp),update=target,script_sha256=sha(__file__),scope='fixed endpoint native FP64 physical residual; no training/selection',rhs_sha256=sha(RHS) if target==2048 else None,timing='first cache build then 3 warm repeats; simultaneous jobs may affect walltime; no exclusive GPU lock',rows=[])
 with torch.no_grad():
  for name,data,ids,sources in panels:
   for i in ids:
    for sid in range(2 if sources is not None else 1):
     if sources is None:q=source_from_data(r.cfg,data,torch.tensor([i]),'cuda').to(torch.complex128)
     else:q=sources[sid:sid+1].cuda()
     torch.cuda.synchronize();t=time.perf_counter()
     r.op.setup(sigma_t=data['sigma_t'][i:i+1].cuda(),sigma_s=data['sigma_s'][i:i+1].cuda(),source=q.to(torch.complex64))
     r.op64.setup(sigma_t=data['sigma_t'][i:i+1].double().cuda(),sigma_s=data['sigma_s'][i:i+1].double().cuda(),source=q);clear(r.model)
     torch.cuda.synchronize();setup=time.perf_counter()-t
     for rep in [-1,0,1,2]:
      counts={'A':0,'M':0}
      def A(x):counts['A']+=1;return r.op64.A_op(x)
      def M(x):counts['M']+=1;return r.model(x.to(torch.complex64)).to(torch.complex128)
      torch.cuda.synchronize();t=time.perf_counter()
      try:
       sol,hist,_,calls=fgmres_real_stacked_with_trace(A,q,M,tol=1e-5,maxiter=100,restart=20,max_restarts=5)
       residual=float((q-A(sol)).norm()/q.norm());torch.cuda.synchronize()
       row=dict(panel=name,index=i,rhs=sid,repeat=rep,seconds=time.perf_counter()-t,setup_seconds=setup,residual=residual,accepted=math.isfinite(residual) and residual<=1e-5,calls=counts,native_calls=calls[-1],history=hist)
      except Exception as e:
       torch.cuda.synchronize();row=dict(panel=name,index=i,rhs=sid,repeat=rep,seconds=time.perf_counter()-t,setup_seconds=setup,accepted=False,error=repr(e),calls=counts)
      report['rows'].append(row)
     save(p,report)
 report['complete']=True;save(p,report);print('EVALUATED',cp,flush=True)
def main():
 torch.set_num_threads(4);queue=HERE/'evaluation_queue_status.json';start=time.time()
 pending=[(HERE/f'v2_{arm}_{seed}',t) for t in TARGETS for seed in [146501,146502] for arm in ['O','K']]
 done=[];failed=[]
 while pending:
  progressed=False
  for folder,t in list(pending):
   result=folder/f'eval_native_{t}.json'
   if result.exists() and json.loads(result.read_text()).get('complete'):
    pending.remove((folder,t));done.append(str(result));continue
   cp=folder/f'checkpoint_{t}.pt'
   if cp.exists():
    try:evaluate(cp,t)
    except (EOFError,RuntimeError) as e:
     # If a save is still in progress retain retry evidence; no training restart.
     with (HERE/'evaluation_load_errors.jsonl').open('a') as f:f.write(json.dumps(dict(checkpoint=str(cp),error=repr(e),time=time.time()))+'\n')
     if cp.stat().st_mtime>time.time()-30:continue
     failed.append(dict(checkpoint=str(cp),error=repr(e)));pending.remove((folder,t));continue
    pending.remove((folder,t));done.append(str(result));progressed=True
   elif (folder/'failure.json').exists():
    failed.append(dict(checkpoint=str(cp),reason='training_failed_before_checkpoint',failure=json.loads((folder/'failure.json').read_text())));pending.remove((folder,t))
  save(queue,dict(pid=os.getpid(),complete=not pending,done=done,failed=failed,pending=[(str(a),b) for a,b in pending],elapsed=time.time()-start))
  if pending and not progressed:time.sleep(20)
if __name__=='__main__':main()
