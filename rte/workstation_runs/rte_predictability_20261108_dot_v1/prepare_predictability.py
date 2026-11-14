"""Prepare observable RTE prefix1 learning data and truncated-input counterfactuals.
No new physical-medium teachers. Auxiliary truncated-rhs solves are diagnostic
and explicitly counted. Existing true teachers supply all learning labels.
"""
import sys,os,time,json,hashlib,subprocess,gc,math
from pathlib import Path
import torch
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"rte/workstation_runs/direct_capture_20261002/confirmation/shard0/code"))
sys.path.insert(0,str(ROOT/"rte/workstation_runs/layered_background_20261108_dot_v1"))
from direct_segmented_tsa import DirectSegmentedTSA
from layered_inverse import background_from_material
from rte.run_large_angular_spectral_experiments import ExperimentConfig
from rte.train_source_predictor import material_operator,build_base
from rte.augmented_fgmres import augmented_fgmres,projected_direction_loss
from rte.source_initialization import solve_from_initial

def sync():torch.cuda.synchronize()
def competitors():
 s=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True)
 return [int(v.strip()) for v in s.splitlines() if v.strip().isdigit() and int(v.strip())!=os.getpid()]
def safe(x):
 if isinstance(x,float) and not math.isfinite(x):return str(x)
 if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
 if isinstance(x,list):return [safe(v) for v in x]
 return x

class ObservableFrame:
 def __init__(self,low):
  self.n=low.shape[0]
  axis,tbg,sbg,_=background_from_material(low.sigma_t,low.sigma_s)
  self.axis=axis
  t=low.sigma_t.transpose(-1,-2) if axis else low.sigma_t
  bg=tbg.transpose(-1,-2) if axis else tbg
  profile=bg[0,0,:,0]
  self.shift_n=int(torch.argmax(profile-torch.roll(profile,1)))
  inclusion=(t-bg).sum(-2).flatten()
  self.shift_t=int(torch.argmax(inclusion)) if bool(inclusion.max()>0) else 0
  self.ix={}
  for k in [8,16]:
   f=torch.fft.fftfreq(self.n,device=t.device)*self.n
   self.ix[k]=torch.nonzero(f.abs()<=k).flatten()
 def canonical(self,x):
  if self.axis:x=x.transpose(-1,-2)
  return torch.roll(x,(-self.shift_n,-self.shift_t),(-2,-1))
 def original(self,x):
  x=torch.roll(x,(self.shift_n,self.shift_t),(-2,-1))
  return x.transpose(-1,-2) if self.axis else x
 def encode(self,x,k=8):
  return torch.fft.fft(self.canonical(x),dim=-1,norm="ortho")[...,self.ix[k]]
 def decode(self,c,k=8):
  z=torch.zeros((*c.shape[:-1],self.n),dtype=c.dtype,device=c.device)
  z[...,self.ix[k]]=c
  return self.original(torch.fft.ifft(z,dim=-1,norm="ortho"))
 def project(self,x,k=8):return self.decode(self.encode(x,k),k)
 def material_features(self,low):
  t=self.canonical(low.sigma_t);a=self.canonical(low.sigma_t-low.sigma_s)
  v=torch.cat((t.clamp_min(1e-30).log(),a.clamp_min(1e-30).log()),1)
  pooled=F.adaptive_avg_pool2d(v,(8,8)).flatten()
  profile=v.mean(-1).flatten()
  return torch.cat((pooled,profile,torch.tensor([float(self.axis)],device=t.device)))
 def visible(self,low,r):
  c=self.canonical(r);flat=c.flatten();phase=flat[flat.abs().argmax()];phase=phase/phase.abs().clamp_min(1e-30)
  scale=c.abs().square().mean().sqrt().clamp_min(1e-30)
  features=self.encode(r/(phase*scale),16).to(torch.complex64).flatten()
  return self.material_features(low).float(),features,phase,scale
 def info(self):return dict(axis=self.axis,shift_n=self.shift_n,shift_t=self.shift_t)

@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.manual_seed(20261109)
 if competitors():raise RuntimeError("GPU competitor before admission")
 out=HERE/"prepare_01";out.mkdir(exist_ok=False);began=time.perf_counter()
 rows=[];counter=[];items=[]
 settings=dict(complete=False,pid=os.getpid(),training=False,neural_training=False,new_medium_teachers=False,
  auxiliary_truncated_rhs_solves=0,heldout9626=False,grid=128,seed=20261109,
  split="First96 of seeded permutation of128 training physical IDs for fit; remaining32 internal validation. Existing16 development separate.",
  inputs="Only sigma_t,sigma_s and actual prefix1 residual; no source parameters, generating geometry labels, teacher features or x1 input.",
  target="Unit-normalized complex coefficients of teacher-x1 in material-observable frame; fixed real-span POD possible. Source phase normalized from largest residual entry.",
  source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),wall_limit=600)
 def save():
  for k,v in [("settings",settings),("cases",rows),("counterfactuals",counter)]:(out/(k+".json")).write_text(json.dumps(safe(v),indent=2)+"\n")
 save();torch.cuda.reset_peak_memory_stats()
 try:
  for split in ["train","validation"]:
   saved=torch.load(ROOT/f"rte/workstation_runs/tail_multiresolution_bank_20260926/{split}.pt",map_location="cpu",weights_only=True)
   data=saved["data"];bank=saved["bank"];count=len(bank["t"])
   cfg=ExperimentConfig(size=128,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes="mixed")
   for index in range(count):
    if competitors():raise RuntimeError("New GPU competitor")
    if time.perf_counter()-began>settings["wall_limit"]:raise RuntimeError("Preparation wall limit")
    sync();t=time.perf_counter();low,high=material_operator(cfg,data,torch.tensor([index]),.1,"cuda")
    for key,v in [("t",low.sigma_t),("s",low.sigma_s),("q",low.source)]:assert torch.equal(bank[key][index:index+1].cuda(),v)
    teacher=bank["target"][index:index+1].cuda()
    teacher_res=float((high.A_op(teacher)-high.source).norm()/high.source.norm());assert teacher_res<2e-9
    base=build_base(low,"tsa16");tsa=DirectSegmentedTSA(base,low.source,chunk=1,capture_first=False)
    pre=lambda r:tsa(r.to(torch.complex64)).to(torch.complex128)
    prefixes=[];augmented_fgmres(high.A_op,high.source,pre,None,maxiter=1,inject_after=1,capture_prefix=prefixes.append)
    assert len(prefixes)==1;prefix=prefixes[0];r=prefix["residual"];e=teacher-prefix["solution"]
    frame=ObservableFrame(low);mat,feat,phase,scale=frame.visible(low,r)
    c=frame.encode(e/phase,8);ynorm=c.norm().clamp_min(1e-30);y=(c/ynorm).to(torch.complex64).flatten()
    reconstruction=frame.decode((y.reshape(1,12,128,17)*phase).to(torch.complex128),8)
    quotient=float(projected_direction_loss(high.A_op,reconstruction,r,prefix["image_basis"].transpose(0,1))[0])
    record=dict(split=split,index=index,teacher_residual=teacher_res,prefix_residual=prefix["relative_residual"],
     frame=frame.info(),unit_target_norm=float(y.norm()),encoded_target_scale=float(ynorm),phase=[float(phase.real),float(phase.imag)],
     residual_rms=float(scale),free_target_quotient_ratio=quotient)
    if split=="validation":
     ce=frame.encode(e,8)
     for cut in [8,16]:
      rhs=frame.project(r,cut)
      sync();start=time.perf_counter();et,st=solve_from_initial(high.A_op,rhs,pre,None,tolerance=1e-9,maxiter=180);sync()
      aux_time=time.perf_counter()-start;settings["auxiliary_truncated_rhs_solves"]+=1
      ctr=frame.encode(et,8);difference=float((ce-ctr).norm()/ce.norm())
      raw=frame.decode(ctr,8)
      optimal=float(projected_direction_loss(high.A_op,raw,r,prefix["image_basis"].transpose(0,1))[0])
      counter.append(dict(index=index,cut=cut,visible_residual_identity_error=float((frame.encode(r,cut)-frame.encode(rhs,cut)).norm()/frame.encode(r,cut).norm()),
       removed_input_relative_norm=float((r-rhs).norm()/r.norm()),retained_output_relative_difference=difference,
       actual_rhs_quotient_ratio=optimal,auxiliary_seconds=aux_time,auxiliary_status=st,
       auxiliary_explicit_residual=float((rhs-high.A_op(et)).norm()/rhs.norm()),
       scope="Fixed same A and same visible projected r; conditional map test. Not a claim both residuals arise from identical physical-source prefix trajectories."))
    sync();record["case_seconds"]=time.perf_counter()-t;rows.append(record)
    items.append(dict(split=split,index=index,material=mat.cpu(),residual_features=feat.cpu(),target=y.cpu(),frame=frame.info()))
    save()
    if (index+1)%16==0:print("PREPARED",split,index+1,"counter",len(counter),flush=True)
    del low,high,teacher,base,tsa,prefix,prefixes,r,e,frame,mat,feat,c,y,reconstruction;gc.collect();sync();torch.cuda.empty_cache()
   del saved,data,bank;gc.collect()
  permutation=torch.randperm(128,generator=torch.Generator().manual_seed(20261109)).tolist()
  torch.save(dict(items=items,fit_ids=permutation[:96],internal_ids=permutation[96:],settings=settings),out/"features.pt")
  settings["feature_bytes"]=(out/"features.pt").stat().st_size;settings["fit_ids"]=permutation[:96];settings["internal_ids"]=permutation[96:]
  settings["complete"]=True
 except BaseException as exc:settings["error"]=repr(exc);raise
 finally:
  settings.update(wall_seconds=time.perf_counter()-began,cases=len(rows),counterfactual_cases=len(counter),
   peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());save()
if __name__=="__main__":main()
