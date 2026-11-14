"""Pro V2 frozen current-trajectory O/K experiment; no DEV selection."""
import os,math
from trainer_skeleton import *
HERE=Path(__file__).parent
SEEDS=[146501,146502]
def sample(T,c,U):
 if T==0:return 0,0
 s=0 if c==0 or T==1 else 1+math.floor((U if c==1 else 1+U)*(T-1)/2)
 return s,min(8,T-s)
def loss_k(rows,b):
 return torch.stack([torch.log(torch.clamp(rows[min(j,len(rows))-1]['residual'].square().sum()/(1e-10*b.square().sum()).clamp_min(1e-300),min=1.)) for j in [2,4,8]]).mean()
def validate_row(state,row):
 if not torch.isfinite(row['residual']).all() or not torch.isfinite(row['relative']):raise RuntimeError('nonfinite trajectory')
 if state['breakdown'] and float(row['relative'])>1e-5:raise RuntimeError('unconverged breakdown: no reliable endpoint')
def trajectory(A,M,b):
 with torch.no_grad():
  s=start_cycle(A,b,torch.zeros_like(b));states=[detach_state(s)];rows=[];costs=[0.];t=time.perf_counter()
  if float(b.norm())==0:return dict(states=states,rows=rows,costs=costs,reason='initial_converged')
  for _ in range(100):
   s,row=step(A,M,b,s,20);validate_row(s,row);states.append(detach_state(s));rows.append(row);costs.append(time.perf_counter()-t)
   if float(row['relative'])<=1e-5:return dict(states=states,rows=rows,costs=costs,reason='converged')
  return dict(states=states,rows=rows,costs=costs,reason='budget')
def suffix(A,M,b,ctx,q):
 s=detach_state(ctx);rows=[]
 for _ in range(q):
  s,row=step(A,M,b,s,20);validate_row(s,row);rows.append(row)
  if float(row['relative'])<=1e-5:break
 return rows
def contribute(r,idx,arm,c,U):
 A,M,b=r.setup(idx);t=time.perf_counter();tr=trajectory(A,M,b);trajectory_cost=dict(A=A.counts(),M=M.counts(),wall=time.perf_counter()-t)
 T=len(tr['rows']);s,q=sample(T,c,U)
 row=dict(material=idx,c=c,U=U,T=T,s=s,q=q,reason=tr['reason'],trajectory=trajectory_cost,prefix_seconds=tr['costs'][s],setup_generation=r.generation)
 if T==0:return dict(**row,loss=0.,actual_calls=0,logical_fill=3,input_hashes=[],A_total=A.counts(),M_total=M.counts())
 actual=tr['rows'][s:s+q];row['input_hashes']=[digest(v['v']) for v in actual];clear(r.model);t=time.perf_counter();before=(A.calls,M.calls)
 if arm=='O':loss=o_loss(A,M,actual);used=q
 else:
  rows=suffix(A,M,b,tr['states'][s],q);loss=loss_k(rows,b);used=len(rows)
 if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
 (loss/2).backward();torch.cuda.synchronize()
 return dict(**row,loss=float(loss),actual_calls=used,logical_fill=sum(j>used for j in [2,4,8]),gradient_seconds=time.perf_counter()-t,gradient_A_calls=A.calls-before[0],gradient_M_calls=M.calls-before[1],A_total=A.counts(),M_total=M.counts())
def qualify():
 # Successful one-call breakdown is valid, short-budget endpoints remain positive;
 # nonconverged breakdown/nonfinite explicitly fail. No extra calls for padding.
 b=torch.tensor([1.,2.],device='cuda',dtype=torch.float64)
 A=lambda x:x
 tr=trajectory(A,A,b);assert len(tr['rows'])==1
 rows=suffix(A,A,b,tr['states'][0],1);assert len(rows)==1 and float(loss_k(rows,b))==0.
 rr=[dict(residual=b*.01) for _ in range(3)]
 assert float(loss_k(rr,b))>0
 assert sample(0,2,.9)==(0,0)
 assert all(sample(1,c,.9)==(0,1) for c in range(3))
 for T in range(1,101):
  for c in range(3):
   for U in [0.,.2,.999999]:
    s,q=sample(T,c,U);assert 0<=s<T and q==min(8,T-s) and q>=1
 for residual in [b,b*float('nan')]:
  try:validate_row(dict(breakdown=True),dict(residual=residual,relative=residual.norm()/b.norm()))
  except RuntimeError:pass
  else:raise AssertionError('failure not caught')
 out=dict(pass_qualification=True,success_calls=1,success_K=0.,budget_q3_K=float(loss_k(rr,b)),T0='occupies slot; no M',T1='s0 q1',failure='raises; never resamples',all_sampler_checks=True)
 (HERE/'v2_qualification.json').write_text(json.dumps(out,indent=2));print(out)
def prepare():
 for seed in SEEDS:
  g=torch.Generator().manual_seed(seed);ids=torch.randint(8,(2048,2),generator=g);u=torch.rand((2048,2),generator=g,dtype=torch.float64);c=torch.arange(4096).reshape(2048,2)%3
  p=HERE/f'draw_seed{seed}.pt'
  if p.exists():
   old=torch.load(p,weights_only=True);assert torch.equal(old['ids'],ids) and torch.equal(old['u'],u)
  else:torch.save(dict(ids=ids,u=u,c=c),p)
def train_run(arm,seed):
 torch.set_num_threads(4);out=HERE/f'v2_{arm}_{seed}';out.mkdir(exist_ok=False)
 protocol=dict(version='ProV2',arm=arm,seed=seed,updates=2048,batch=2,checkpoint_every=128,initial='physical_2048.pt',optimizer='AdamW',weight_decay=1e-5,lr=3e-4,eta_min=1e-5,clip='none; finite grad checked',restart=20,maxiter=100,physical_tolerance=1e-5,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),draw_sha256=hashlib.sha256((HERE/f'draw_seed{seed}.pt').read_bytes()).hexdigest(),scope='stopgrad full prefix; independent material bases; K full suffix Arnoldi LS gradient',selection='fixed2048; DEV endpoint only',early='T0 retained; short q retained; j beyond q reuses terminal; no retries')
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2));r=Runner(seed);r.optimizer=torch.optim.AdamW(r.model.parameters(),lr=3e-4,weight_decay=1e-5);r.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(r.optimizer,2048,eta_min=1e-5)
 draws=torch.load(HERE/f'draw_seed{seed}.pt',weights_only=True);t=time.perf_counter();counts={}
 try:
  for update in range(1,2049):
   r.optimizer.zero_grad(set_to_none=True);records=[]
   for slot in range(2):
    idx=int(draws['ids'][update-1,slot]);c=int(draws['c'][update-1,slot]);U=float(draws['u'][update-1,slot]);records.append(contribute(r,idx,arm,c,U))
    key=f'{idx}:{c}';counts[key]=counts.get(key,0)+1
   active=any(v['T']>0 for v in records)
   gn=float(torch.nn.utils.clip_grad_norm_(r.model.parameters(),float('inf'))) if active else 0.
   if not math.isfinite(gn):raise RuntimeError('nonfinite gradient')
   if active:r.optimizer.step()
   # Explicit schedule by predetermined batch, including all-T0 batches.
   lr=1e-5+(3e-4-1e-5)*(1+math.cos(math.pi*update/2048))/2
   for group in r.optimizer.param_groups:group['lr']=lr
   r.scheduler.last_epoch=update;r.scheduler._last_lr=[lr];clear(r.model)
   row=dict(update=update,records=records,gradnorm=gn,clip_applied=False,optimizer_step=active,lr=lr,wall=time.perf_counter()-t,counts=counts.copy(),peak_memory=torch.cuda.max_memory_allocated())
   with (out/'draws.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   (out/'status.json').write_text(json.dumps(dict(pid=os.getpid(),complete=update==2048,**row),indent=2))
   if update%128==0:r.save(out/f'checkpoint_{update}.pt',update,protocol)
   if update%16==0 or update==1:print(arm,seed,update,row['wall'],flush=True)
 except Exception as e:
  (out/'failure.json').write_text(json.dumps(dict(update=locals().get('update'),slot=locals().get('slot'),error=repr(e),partial_records=locals().get('records'),wall=time.perf_counter()-t),indent=2));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--qualify',action='store_true');p.add_argument('--arm',choices=['O','K']);p.add_argument('--seed',type=int);a=p.parse_args()
 if a.qualify:qualify();prepare()
 else:train_run(a.arm,a.seed)
