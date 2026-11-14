import sys,time,json,hashlib,argparse
from pathlib import Path
import torch
ROOT=Path('/data/wjt/NeuralPreconditionedBornSeries')
OLD=ROOT/'rte/workstation_runs/rte_mcmg_born_20261114_dot_v1'
sys.path.insert(0,str(OLD))
from train import Cycle,ExperimentConfig,source_from_data,make_operator
from rte.models import PeriodicRTE2D
from window import real,complex,start_cycle,detach_state,step,window,k_loss,o_loss
def clear(m):m.cache=None;m.key=None
def digest(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
class Counted:
 def __init__(self,f):self.f=f;self.calls=0;self.seconds=0.
 def __call__(self,x):
  torch.cuda.synchronize();t=time.perf_counter();y=self.f(x);torch.cuda.synchronize();self.calls+=1;self.seconds+=time.perf_counter()-t;return y
 def counts(self):return dict(calls=self.calls,seconds=self.seconds)
def actual_trajectory(A,M,b,restart=20,maxiter=100,tol=1e-5):
 states=[];rows=[];costs=[];t=time.perf_counter()
 with torch.no_grad():
  s=start_cycle(A,b,torch.zeros_like(b));states.append(detach_state(s));costs.append(time.perf_counter()-t)
  if float(b.norm())==0:return dict(states=states,rows=rows,costs=costs,reason='initial_converged')
  for _ in range(maxiter):
   s,r=step(A,M,b,s,restart);states.append(detach_state(s));rows.append(r);costs.append(time.perf_counter()-t)
   if float(r['relative'])<=tol:reason='converged';break
   if s['breakdown']:reason='breakdown';break
  else:reason='maxiter'
 return dict(states=states,rows=rows,costs=costs,reason=reason)
def select_start(trace,gen,length=8,early_policy='unresolved',source_only=False):
 n=len(trace['rows']);eligible=list(range(n-length+1))
 if not eligible:raise RuntimeError(f'EARLY_POLICY_REQUIRED n={n} reason={trace["reason"]} policy={early_policy}')
 if source_only:return 0,'source_dryrun',eligible
 groups=[g.tolist() for g in torch.tensor(eligible).tensor_split(3)]
 if any(not g for g in groups):raise RuntimeError('SHORT_STRATA_POLICY_REQUIRED')
 band=int(torch.randint(3,(1,),generator=gen));g=groups[band];start=g[int(torch.randint(len(g),(1,),generator=gen))]
 return start,['early','middle','late'][band],eligible
class Runner:
 def __init__(self,seed):
  torch.manual_seed(seed);self.gen=torch.Generator().manual_seed(seed);self.data=torch.load(OLD/'data.pt',weights_only=True)['train']
  self.cfg=ExperimentConfig(size=64,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes='mixed')
  q=source_from_data(self.cfg,self.data,torch.tensor([0]),'cuda')
  self.op=make_operator(self.cfg,self.data['sigma_t'][:1],self.data['sigma_s'][:1],q,'cuda')
  self.op64=PeriodicRTE2D(shape=(64,64),dx=1/64,n_angles=12,anisotropy_g=.65,eta_scale=0.,fdtype=torch.float64,device='cuda')
  self.model=Cycle(self.op).cuda();self.model.load_state_dict(torch.load(OLD/'physical_2048.pt',weights_only=True)['model'])
  self.optimizer=torch.optim.Adam(self.model.parameters(),lr=3e-4);self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,2048,eta_min=1e-5);self.generation=0
 def setup(self,idx):
  ids=torch.tensor([idx]);q=source_from_data(self.cfg,self.data,ids,'cuda');t=self.data['sigma_t'][ids].cuda();s=self.data['sigma_s'][ids].cuda()
  self.op.setup(sigma_t=t,sigma_s=s,source=q);self.op64.setup(sigma_t=t.double(),sigma_s=s.double(),source=q.to(torch.complex128));self.generation+=1;clear(self.model)
  return Counted(lambda x:real(self.op64.A_op(complex(x)))),Counted(lambda x:real(self.model(complex(x).to(torch.complex64)).to(torch.complex128))),real(self.op64.source)
 def contribution(self,idx,arm,source_only=False,early_policy='unresolved'):
  A,M,b=self.setup(idx);t=time.perf_counter();trace=actual_trajectory(A,M,b);trajectory=dict(A=A.counts(),M=M.counts(),wall=time.perf_counter()-t)
  start,band,eligible=select_start(trace,self.gen,early_policy=early_policy,source_only=source_only)
  ctx=trace['states'][start];actual=trace['rows'][start:start+8];hashes=[digest(r['v']) for r in actual]
  clear(self.model);rstart=(b-A(ctx['x'])).detach();before=(A.calls,M.calls);t=time.perf_counter()
  if arm=='O':loss=o_loss(A,M,actual)
  else:
   _,rows=window(A,M,b,ctx);loss=k_loss(rows,rstart)
  if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
  (loss/2).backward();torch.cuda.synchronize()
  return dict(material=idx,arm=arm,start=start,band=band,n=len(trace['rows']),reason=trace['reason'],eligible_count=len(eligible),input_hashes=hashes,prefix_seconds=trace['costs'][start],trajectory=trajectory,gradient_seconds=time.perf_counter()-t,loss=float(loss),A_total=A.counts(),M_total=M.counts(),gradient_A_calls=A.calls-before[0],gradient_M_calls=M.calls-before[1],setup_generation=self.generation)
 def save(self,path,update,protocol):
  torch.save(dict(model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),scheduler=self.scheduler.state_dict(),update=update,protocol=protocol,rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),draw=self.gen.get_state())),path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');p.add_argument('--arm',choices=['O','K'],default='K');p.add_argument('--seed',type=int,default=146501);a=p.parse_args()
 if not a.dry_run:raise RuntimeError('UNFROZEN: long training intentionally not enabled')
 torch.set_num_threads(4);r=Runner(a.seed);r.optimizer.zero_grad(set_to_none=True);records=[r.contribution(i,a.arm,source_only=True) for i in [0,1]]
 out=dict(scope='qualification only; source8; effectivebatch2; no optimizer step',arm=a.arm,records=records,gradnorm=float(torch.nn.utils.clip_grad_norm_(r.model.parameters(),float('inf'))),peak_memory=torch.cuda.max_memory_allocated())
 (Path(__file__).parent/('skeleton_dryrun_'+a.arm+'.json')).write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)


def train_frozen(arm,seed,outdir,protocol):
 """Coordinator-only entry. Explicit frozen protocol required; no implicit CLI."""
 if not protocol.get('frozen',False):raise RuntimeError('protocol not frozen')
 if protocol.get('early_policy')!='halt_unresolved':
  raise RuntimeError('early policy implementation awaits Pro; cannot silently infer')
 outdir=Path(outdir)
 outdir.mkdir(parents=True,exist_ok=False)
 (outdir/'protocol.json').write_text(json.dumps(protocol,indent=2))
 r=Runner(seed);starttime=time.perf_counter();counts={}
 try:
  for update in range(1,2049):
   r.optimizer.zero_grad(set_to_none=True);records=[]
   # Independent one-material Krylov bases. Two gradients averaged, never joint H.
   ids=torch.randint(8,(2,),generator=r.gen).tolist()
   for idx in ids:
    row=r.contribution(idx,arm,early_policy='unresolved');records.append(row)
    key=str(idx)+':'+row['band'];counts[key]=counts.get(key,0)+1
   gn=float(torch.nn.utils.clip_grad_norm_(r.model.parameters(),float('inf')))
   if not torch.isfinite(torch.tensor(gn)):raise RuntimeError('nonfinite gradient')
   r.optimizer.step();r.scheduler.step();clear(r.model)
   row=dict(update=update,materials=ids,contributions=records,gradnorm=gn,lr=r.scheduler.get_last_lr()[0],wall=time.perf_counter()-starttime,counts=counts.copy())
   with (outdir/'draws.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   if update%128==0:
    r.save(outdir/f'checkpoint_{update}.pt',update,protocol)
    (outdir/'status.json').write_text(json.dumps(dict(complete=update==2048,**row),indent=2))
 except Exception as e:
  (outdir/'failure.json').write_text(json.dumps(dict(update=locals().get('update'),materials=locals().get('ids'),error=repr(e),wall=time.perf_counter()-starttime),indent=2))
  raise

if __name__=='__main__':main()
