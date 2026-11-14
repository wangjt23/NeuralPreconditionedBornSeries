import os,sys,time,json,hashlib,traceback,subprocess
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/"rte/workstation_runs/rte_predictability_20261108_dot_v1"
sys.path[:0]=[str(ROOT),str(OLD)]
from fit_and_evaluate import convert_items

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
 p=HERE/(name+".tmp");p.write_text(json.dumps(obj,indent=2));os.replace(p,HERE/name)
def sync():torch.cuda.synchronize()
class Block(nn.Module):
 def __init__(self,w):
  super().__init__();self.local=nn.Conv2d(w,w,3,padding=1,padding_mode="circular",groups=w)
  self.mix=nn.Conv2d(w,w,1);self.spec=nn.Parameter(torch.randn(w,65,1,2)*.02)
 def forward(self,x):
  z=torch.fft.rfft(x,dim=-2);z=z*torch.view_as_complex(self.spec.contiguous())[None]
  return x+F.gelu(self.mix(self.local(x)+torch.fft.irfft(z,n=128,dim=-2)))
class Model(nn.Module):
 def __init__(self):
  super().__init__();self.stem=nn.Conv2d(26,32,1);self.blocks=nn.Sequential(*[Block(32) for _ in range(3)]);self.head=nn.Conv2d(32,24,1)
 def forward(self,x):
  y=self.head(self.blocks(F.gelu(self.stem(x))));a,b=y.chunk(2,1)
  z=torch.fft.fft(torch.complex(a,b),dim=-1,norm="ortho")
  z=torch.cat([z[...,:9],z[...,-8:]],-1).flatten(1)
  return z/z.norm(dim=1,keepdim=True).clamp_min(1e-12)
@torch.no_grad()
def evaluate(m,X,Y,ids):
 vals=[]
 for b in ids.split(8):vals.extend((m(X[b])-Y[b]).abs().square().sum(1).cpu().tolist())
 return dict(mean=sum(vals)/len(vals),max=max(vals),values=vals)
def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(2)
 assert torch.cuda.is_available()
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 start=time.perf_counter()
 report=dict(complete=False,pid=os.getpid(),start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),device=torch.cuda.get_device_name(),cuda=torch.version.cuda,torch=torch.__version__,arms=[],source_hash=digest(__file__),plan_hash=digest(HERE/"PLAN.json"),heldout9626=False,DEV_training=False,resource=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.used,memory.total,utilization.gpu","--format=csv,noheader"],text=True))
 save("status.json",report)
 try:
  files=[OLD/"prepare_01/features.pt",OLD/"prepare_01/cases.json",ROOT/"rte/workstation_runs/tail_multiresolution_bank_20260926/train.pt"]
  report["input_hashes"]={str(p):digest(p) for p in files}
  raw=torch.load(files[0],map_location="cpu",weights_only=True);records={(r["split"],r["index"]):r for r in json.loads(files[1].read_text())}
  items=convert_items([r for r in raw["items"] if r["split"]=="train"],records)
  parent=torch.load(files[2],map_location="cpu",weights_only=True)["bank"]
  xs=[];ys=[]
  for i in range(128):
   it=items[("train",i)];frame=records[("train",i)]["frame"]
   material=torch.cat([parent["t"][i:i+1].log(),(parent["t"][i:i+1]-parent["s"][i:i+1]).log()],1)
   if frame["axis"]:material=material.transpose(-1,-2)
   material=torch.roll(material,(-frame["shift_n"],-frame["shift_t"]),(-2,-1))
   material=F.avg_pool2d(material,(1,2))/4
   c=it["features"].reshape(1,12,128,33)
   spec=torch.zeros(1,12,128,64,dtype=torch.complex64);spec[...,:17]=c[...,:17];spec[...,-16:]=c[...,-16:]
   q=torch.fft.ifft(spec,dim=-1,norm="ortho");q=q/q.abs().square().mean().sqrt().clamp_min(1e-12)
   xs.append(torch.cat([material,q.real,q.imag],1));ys.append(it["target"])
  X=torch.cat(xs).cuda();Y=torch.stack(ys).cuda()
  fit=torch.tensor(raw["fit_ids"],device="cuda");internal=torch.tensor(raw["internal_ids"],device="cuda")
  assert len(fit)==96 and len(internal)==32 and set(fit.tolist()).isdisjoint(internal.tolist())
  assert torch.isfinite(X).all() and torch.isfinite(Y).all()
  report.update(fit_ids=fit.tolist(),internal_ids=internal.tolist(),X_shape=list(X.shape),Y_shape=list(Y.shape),data_preparation_seconds=time.perf_counter()-start)
  save("status.json",report)
  for seed in [2026111401,2026111402]:
   torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);g=torch.Generator().manual_seed(seed)
   m=Model().cuda();opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-5)
   schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,8192,eta_min=5e-5)
   arm=dict(seed=seed,complete=False,updates=0,parameters=sum(p.numel() for p in m.parameters()),initial_fit=evaluate(m,X,Y,fit),initial_internal=evaluate(m,X,Y,internal),curve=[])
   report["arms"].append(arm);save("status.json",report);tick=time.perf_counter();sync()
   for step in range(1,8193):
    ids=fit[torch.randint(96,(8,),generator=g).cuda()]
    m.train();opt.zero_grad(set_to_none=True);pred=m(X[ids]);loss=(pred-Y[ids]).abs().square().sum(1).mean()
    assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(m.parameters(),10);opt.step();schedule.step()
    if step%256==0:
     sync();m.eval();row=dict(step=step,loss=float(loss),gradient_norm=float(grad),elapsed=time.perf_counter()-tick,fit=evaluate(m,X,Y,fit),internal=evaluate(m,X,Y,internal))
     arm["curve"].append(row);arm["updates"]=step;save("status.json",report);print(seed,step,row["fit"]["mean"],row["internal"]["mean"],flush=True)
   sync();arm.update(complete=True,training_seconds=time.perf_counter()-tick,final_fit=evaluate(m,X,Y,fit),final_internal=evaluate(m,X,Y,internal))
   m.eval();times=[]
   for k in range(10):
    sync();t=time.perf_counter()
    with torch.no_grad():z=m(X[fit[:1]])
    sync();times.append(time.perf_counter()-t)
   arm["model_only_single_seconds_including_first"]=times
   torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),scheduler=schedule.state_dict(),seed=seed,step=8192,fit_ids=fit.cpu(),internal_ids=internal.cpu(),source_hash=report["source_hash"],input_hashes=report["input_hashes"]),HERE/f"seed{seed}_8192.pt")
   arm["checkpoint_sha256"]=digest(HERE/f"seed{seed}_8192.pt");save("status.json",report)
   del m,opt,schedule;torch.cuda.empty_cache()
  report["complete"]=True
 except BaseException as e:report.update(error=repr(e),traceback=traceback.format_exc());raise
 finally:
  report.update(wall_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());save("status.json",report)
if __name__=="__main__":main()

