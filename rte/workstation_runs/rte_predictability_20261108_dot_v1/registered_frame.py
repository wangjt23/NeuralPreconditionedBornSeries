"""Observable residual registration preflight. No new solves or teachers."""
import os,sys,time,json,hashlib,math,gc
from pathlib import Path
from types import SimpleNamespace
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(HERE))
from prepare_predictability import ObservableFrame,competitors,sync,safe
from fit_and_evaluate import Frame as MaterialFrame,dist_real,dist_complex

class ResidualFrame(ObservableFrame):
 def visible(self,low,r):
  # Remove the arbitrary inclusion frame before residual-only tangent alignment.
  self.shift_t=0
  c0=self.encode(r,16)
  z=torch.zeros((*c0.shape[:-1],self.n),dtype=c0.dtype,device=c0.device);z[...,self.ix[16]]=c0
  band=torch.fft.ifft(z,dim=-1,norm="ortho")
  energy=band.abs().square().sum(tuple(range(band.ndim-1)))
  self.shift_t=int(energy.argmax())
  c=self.encode(r,16)
  flat=c.flatten();largest=flat[flat.abs().argmax()]
  phase=largest/largest.abs() if bool(largest.abs()>1e-30) else torch.ones_like(largest)
  scale=c.abs().square().mean().sqrt().clamp_min(1e-30)
  return self.material_features(low).float(),(c/(phase*scale)).to(torch.complex64).flatten(),phase,scale

class NoTangentFrame(ResidualFrame):
 def visible(self,low,r):
  self.shift_t=0;c=self.encode(r,16);flat=c.flatten();largest=flat[flat.abs().argmax()]
  phase=largest/largest.abs() if bool(largest.abs()>1e-30) else torch.ones_like(largest)
  scale=c.abs().square().mean().sqrt().clamp_min(1e-30)
  return self.material_features(low).float(),(c/(phase*scale)).to(torch.complex64).flatten(),phase,scale

@torch.no_grad()
def main():
 torch.set_num_threads(1)
 if competitors():raise RuntimeError("GPU competitor before admission")
 out=HERE/"registered_01";out.mkdir(exist_ok=False);start=time.perf_counter();rows=[]
 settings=dict(complete=False,pid=os.getpid(),source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  new_teachers=False,new_solves=False,training=False,heldout9626=False,variants=["material","no_tangent","residual"],
  design="Same96fit/32internal/16dev. Normal frame fixed from material. Residual frame tangent argmax from IFFT C16r energy, all angles/normal, integer shift. Gauge from C16r only. Same real POD and unit target norm.",
  wall_limit=300)
 def save():
  (out/"settings.json").write_text(json.dumps(safe(settings),indent=2)+"\n")
  (out/"contexts.json").write_text(json.dumps(safe(rows),indent=2)+"\n")
 save()
 raw=torch.load(HERE/"prepare_01/features.pt",map_location="cpu",weights_only=True)
 records={(r["split"],r["index"]):r for r in json.loads((HERE/"prepare_01/cases.json").read_text())}
 variants={v:{} for v in settings["variants"]}
 frames=dict(material=MaterialFrame,no_tangent=NoTangentFrame,residual=ResidualFrame)
 try:
  for split in ["train","validation"]:
   if competitors():raise RuntimeError("New GPU competitor")
   bank=torch.load(ROOT/f"rte/workstation_runs/tail_multiresolution_bank_20260926/{split}.pt",map_location="cpu",weights_only=True)["bank"]
   for old in [x for x in raw["items"] if x["split"]==split]:
    if time.perf_counter()-start>300:raise RuntimeError("Registration budget")
    i=old["index"];key=(split,i);rec=records[key]
    low=SimpleNamespace(shape=(128,128),sigma_t=bank["t"][i:i+1].cuda(),sigma_s=bank["s"][i:i+1].cuda())
    oldframe=ObservableFrame(low);assert oldframe.info()==rec["frame"]
    phase0=complex(*rec["phase"])
    c16=old["residual_features"].cuda().to(torch.complex128).reshape(1,12,128,33)*phase0*rec["residual_rms"]
    rband=oldframe.decode(c16,16)
    eunit=oldframe.decode((old["target"].cuda().to(torch.complex128)*phase0).reshape(1,12,128,17),8)
    info=dict(split=split,index=i,frames={})
    for name,cls in frames.items():
     f=cls(low);m,x,phi,scale=f.visible(low,rband)
     y=f.encode(eunit/phi,8).flatten();y=y/y.norm()
     f2=cls(low);m2,x2,phi2,scale2=f2.visible(low,oldframe.project(rband,16))
     check=float((x-x2).norm()/x.norm());assert check<2e-6
     reconstruction=f.decode((y*phi).reshape(1,12,128,17),8)
     err=float((reconstruction-eunit).norm()/eunit.norm());assert err<2e-6
     info["frames"][name]=dict(frame=f.info(),visible_idempotence_error=check,physical_target_reconstruction_error=err)
     variants[name][key]=dict(material=m.cpu(),features=x.cpu(),target=y.to(torch.complex64).cpu())
    rows.append(info)
   del bank;gc.collect()
  for name,items in variants.items():
   if competitors():raise RuntimeError("New GPU competitor")
   fit=raw["fit_ids"];Y=torch.stack([items[("train",i)]["target"] for i in fit]).cuda().to(torch.complex128)
   gram=(Y@Y.conj().T).real;ev,V=torch.linalg.eigh((gram+gram.T)/2);ix=torch.argsort(ev,descending=True);ev=ev[ix];V=V[:,ix]
   keep=ev>1e-12*ev.max();ev=ev[keep];V=V[:,keep];U=(V.T.to(Y.dtype)@Y)/ev.sqrt()[:,None]
   errors={}
   for split,ids in [("internal",raw["internal_ids"]),("development",list(range(16)))]:
    keysplit="train" if split=="internal" else "validation"
    y=torch.stack([items[(keysplit,i)]["target"] for i in ids]).cuda().to(torch.complex128)
    coeff=(y@U.conj().T).real
    errors[split]={str(k):(y-coeff[:,:k].to(U.dtype)@U[:k]).norm(dim=1).cpu().tolist() for k in [8,16,32,96]}
   settings.setdefault("results",{})[name]=dict(POD_eigenvalues=ev.cpu().tolist(),errors=errors,
    orthogonality_error=float(((U@U.conj().T).real-torch.eye(len(U),device="cuda")).abs().max()))
   if name!="material":
    torch.save(dict(items=items,fit_ids=fit,internal_ids=raw["internal_ids"],variant=name),out/f"features_{name}.pt")
   save();print("REGISTER",name,{sp:{k:sum(v)/len(v) for k,v in rs.items()} for sp,rs in errors.items()},flush=True)
  settings["complete"]=True
 except BaseException as exc:settings["error"]=repr(exc);raise
 finally:
  settings["wall_seconds"]=time.perf_counter()-start;save()
if __name__=="__main__":main()
