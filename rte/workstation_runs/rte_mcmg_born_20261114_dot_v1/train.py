import sys,time,json,hashlib,os,copy
from pathlib import Path
sys.path.insert(0,'/data/wjt/NeuralPreconditionedBornSeries')
import torch
from torch import nn
import torch.nn.functional as F
from rte.run_large_angular_spectral_experiments import ExperimentConfig,make_operator,setup_batch,source_from_data,dataset_fingerprint
from rte.train_collision_transport_ensemble import material_data
from rte.train_neural_transport import residual_sample
P=Path(__file__).parent

def conv(a,b,k=3,stride=1): return nn.Conv2d(a,b,k,stride=stride,padding=k//2,padding_mode='circular',bias=False)
class Cycle(nn.Module):
 def __init__(self,op):
  super().__init__(); self.op=op; self.widths=[24,32,48]; self.cache=None; self.key=None
  self.coeff=nn.ModuleList([nn.Sequential(nn.Conv2d(3,w,3,padding=1,padding_mode='circular'),nn.SiLU(),nn.Conv2d(w,2*w,1)) for w in self.widths])
  self.A=nn.ModuleList([conv(w,w) for w in self.widths]);self.B=nn.ModuleList([conv(w,w) for w in self.widths]);self.R=nn.ModuleList([conv(a,b,3,2) for a,b in zip(self.widths[:-1],self.widths[1:])]);self.P=nn.ModuleList([conv(b,a) for a,b in zip(self.widths[:-1],self.widths[1:])]);self.lift=conv(24,24,1);self.head=conv(24,24,1);nn.init.zeros_(self.head.weight)
 def fields(self):
  key=tuple((x.data_ptr(),x._version,tuple(x.shape)) for x in [self.op.sigma_t,self.op.sigma_s,*self.parameters()])
  if not torch.is_grad_enabled() and self.cache is not None and key==self.key:return self.cache
  t=self.op.sigma_t;s=self.op.sigma_s;c=torch.cat([t.log(),(t-s).clamp_min(1e-8).log(),s/t],1).float(); fs=[]
  for j,m in enumerate(self.coeff):
   if j:c=F.avg_pool2d(c,2)
   a,b=m(c).chunk(2,1);fs.append((a.tanh(),1+.5*b.tanh()))
  if not torch.is_grad_enabled():self.cache=fs;self.key=key
  return fs
 def cycle(self,b,fs,j=0):
  a,g=fs[j]; apply=lambda u:u+.1*a*self.A[j](u)
  smooth=lambda r:.1*g*self.B[j](r)
  u=smooth(b)
  if j<len(fs)-1:
   e=self.cycle(self.R[j](b-apply(u)),fs,j+1);u=u+self.P[j](e.repeat_interleave(2,-2).repeat_interleave(2,-1))
  return u+smooth(b-apply(u))
 def delta(self,z,fs):
  x=torch.cat([z.real,z.imag],1).float();y=self.head(self.cycle(self.lift(x),fs));re,im=y.chunk(2,1);return torch.complex(re,im).to(z.dtype)
 def forward(self,r):
  z=self.op.G(r);return z+self.delta(z,self.fields())
def ratio(a,b):return (a.abs().square().flatten(1).sum(1)/b.abs().square().flatten(1).sum(1).clamp_min(1e-25)).mean()
def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-25))
def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.manual_seed(141401)
 cfg=ExperimentConfig(size=64,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes='mixed',batch_size=8)
 train=material_data(cfg,256,141401);dev=material_data(cfg,32,142401)
 ids=torch.arange(8);q=source_from_data(cfg,train,ids,'cuda');op=make_operator(cfg,train['sigma_t'][ids],train['sigma_s'][ids],q,'cuda');model=Cycle(op).cuda();initial=copy.deepcopy(model.state_dict())
 protocol=dict(train_count=256,validation_count=32,train_seed=141401,validation_seed=142401,updates=2048,batch=8,initialization_seed=141401,draw_seed=143401,size=64,angles=12,g=.65,G0='op.G: identical input and metric map',arms=['physical','matched'],train_fp=dataset_fingerprint(train),dev_fp=dataset_fingerprint(dev),parameters=sum(p.numel() for p in model.parameters()),source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),purpose='new full-channel residual-linear learned V-cycle; loss pairing is control, not novel claim',test_access=False,selection='fixed final endpoint; no best selection',physical_stop='subsequent original-equation relative residual <=1e-5 with FP64 acceptance',known_scope='periodic 2D L12 g=.65 mixed materials; no angular or physical OOD claim')
 with torch.no_grad():
  r=torch.randn_like(q);s=torch.randn_like(q);zero=rel(model(r),op.G(r));nn.init.normal_(model.head.weight,std=.01);lin=rel(model(.4*r-.7*s),.4*model(r)-.7*model(s)); model.load_state_dict(initial)
 protocol['qualification']={'analytic_restore':zero,'real_linearity':lin};assert zero<1e-7 and lin<1e-4
 (P/'protocol.json').write_text(json.dumps(protocol,indent=2));torch.save({'train':train,'validation':dev},P/'data.pt')
 start=time.time();status=dict(pid=os.getpid(),complete=False,protocol=protocol,arms=[]);(P/'status.json').write_text(json.dumps(status,indent=2))
 for arm in ['physical','matched']:
  model.load_state_dict(initial);model.train();gen=torch.Generator().manual_seed(143401);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,2048,eta_min=5e-5);hist=[];armstart=time.time()
  for step in range(1,2049):
   ids=torch.randperm(256,generator=gen)[:8];q=setup_batch(cfg,op,train,ids,'cuda');r=residual_sample(op,q,gen,.2);opt.zero_grad(set_to_none=True);pred=model(r);e=r-op.A_op(pred);loss=ratio(e,r) if arm=='physical' else ratio(op.G(e),op.G(r));assert torch.isfinite(loss), (arm,step,float(loss));loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),10);opt.step();sch.step()
   if step%128==0 or step==1:
    row=dict(step=step,loss=float(loss),physical=float(ratio(e.detach(),r)),gradnorm=float(gn),seconds=time.time()-armstart);hist.append(row);print(arm,json.dumps(row),flush=True);status['active']=dict(arm=arm,**row);(P/'status.json').write_text(json.dumps(status,indent=2))
  torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'step':2048,'protocol':protocol},P/(arm+'_2048.pt'));status['arms'].append(dict(arm=arm,complete=True,history=hist,seconds=time.time()-armstart));(P/'status.json').write_text(json.dumps(status,indent=2))
 status.update(complete=True,total_seconds=time.time()-start,peak_allocated=torch.cuda.max_memory_allocated());(P/'status.json').write_text(json.dumps(status,indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':main()
