"""Paid, online same-operator recycling. Isolated experiment; production unchanged.
No teachers are used to construct any candidate. Native solve_from_initial remains
in charge of the real-stacked recurrence and original-RHS stopping threshold.
"""
import os,sys,time,json,hashlib,subprocess,argparse,random,math
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
ROOT=Path('/data/wjt/NeuralPreconditionedBornSeries')
sys.path[:0]=[str(ROOT),str(ROOT/'rte/workstation_runs/rte_predictability_20261108_dot_v1'),str(ROOT/'rte/workstation_runs/direct_capture_20261002/confirmation/shard0/code'),str(ROOT/'rte/workstation_runs/layered_background_20261108_dot_v1')]
from screen_structured import medium
from rte.models import PeriodicRTE2D
from rte.mixed_precision_transport import high_precision_copy
from rte.train_source_predictor import build_base
from rte.source_initialization import solve_from_initial
from rte.augmented_fgmres import augmented_fgmres
from rte.npbs_unet_ablation import complex_to_real_channels,real_channels_to_complex

def safe(x):
 if isinstance(x,float) and not math.isfinite(x): return str(x)
 if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [safe(v) for v in x]
 return x

def pack(x):return complex_to_real_channels(x).flatten()
def unpack(x,shape):return real_channels_to_complex(x.reshape(shape[0],2*shape[1],*shape[2:]),shape[1],torch.complex128)

class Recycle:
 def __init__(self,shape,device,cap=32):
  self.shape=shape;self.cap=cap
  self.U=torch.zeros((0,2*math.prod(shape)),device=device,dtype=torch.float64)
  self.C=self.U.clone();self.rejected=0
 def initial(self,b):
  if not len(self.C):return None
  return unpack((self.C@pack(b))@self.U,self.shape)
 def project(self,z,w):
  if len(self.C):
   coeff=self.C@pack(w)
   z=z-unpack(coeff@self.U,self.shape)
   w=w-unpack(coeff@self.C,self.shape)
  return z,w
 def update(self,zs,ws):
  # Two-pass MGS uses only paid actual directions/images. U and C change by
  # the identical real coefficients, preserving A U = C in exact arithmetic.
  for z,w in zip(zs,ws):
   if len(self.C)>=self.cap:break
   z,w=pack(z),pack(w);norm0=w.norm()
   for _ in range(2):
    if len(self.C):
     coeff=self.C@w;w=w-coeff@self.C;z=z-coeff@self.U
   norm=w.norm()
   if not bool(norm>1e-10*norm0.clamp_min(1e-30)):
    self.rejected+=1;continue
   self.C=torch.cat((self.C,(w/norm)[None]),0)
   self.U=torch.cat((self.U,(z/norm)[None]),0)
 def audit(self,A):
  orth=float((self.C@self.C.T-torch.eye(len(self.C),device=self.C.device)).norm()) if len(self.C) else 0.
  defects=[float((pack(A(unpack(u,self.shape)))-c).norm()) for u,c in zip(self.U,self.C)]
  return dict(rank=len(self.C),rejected=self.rejected,orthogonality_defect=orth,AU_C_max=max(defects,default=0.),bytes=(self.U.numel()+self.C.numel())*8)

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--out',default='cpu_probe_01');p.add_argument('--repeats',type=int,default=1);p.add_argument('--with-layered',action='store_true');p.add_argument('--wall-limit',type=float,default=900);p.add_argument('--source-order',default='0,4,7,14');args=p.parse_args()
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 device=args.device
 def sync():
  if device=='cuda':torch.cuda.synchronize()
 def gpu_guard():
  if device!='cuda':return
  out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
  rivals=[int(v.strip()) for v in out.splitlines() if v.strip().isdigit() and int(v.strip())!=os.getpid()]
  if rivals:raise RuntimeError('GPU competitor: '+str(rivals))
 gpu_guard();out=HERE/args.out;out.mkdir(exist_ok=False);began=time.perf_counter();rows=[];audits=[]
 ids=[int(x) for x in args.source_order.split(',')]
 settings=dict(complete=False,pid=os.getpid(),device=device,threads=1,source_ids=ids,grid=128,L=12,g=.65,heldout9626=False,neural_training=False,new_teachers=False,source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),methods=['native','capture_control','solution_init','krylov_init','gcro'],contract='Original stored A64: residual<=1e-5, evaluated coercivity relative-error bound<=1e-3, field difference<=1e-4 vs paired native.',cost='First RHS, all captured A images, cloning, basis construction/update, initial projection, projected preconditioning, native solver and final acceptance charged. No test-source teachers. Shared physical/TSA setup charged once per batch. CPU timings are not comparable to Nov8 GPU results.',repeats=args.repeats,software=dict(torch=torch.__version__,python=sys.version),git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
 if args.with_layered:settings['methods'].append('compact4_inject')
 settings['production_hashes']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['rte/models.py','rte/source_initialization.py','rte/augmented_fgmres.py','rte/train_source_predictor.py']}
 def save():
  for k,v in [('settings',settings),('rows',rows),('audits',audits)]:(out/(k+'.json')).write_text(json.dumps(safe(v),indent=2)+'\n')
 save()
 try:
  saved=torch.load(ROOT/'rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt',map_location='cpu',weights_only=True)
  rhs=[saved['bank']['q'][i:i+1].to(device).to(torch.complex128) for i in ids]
  settings['rhs_sha256']={str(i):hashlib.sha256(saved['bank']['q'][i:i+1].numpy().tobytes()).hexdigest() for i in ids};del saved
  sync();t=time.perf_counter();total,scatter,info=medium(128,1,10/3,device)
  low=PeriodicRTE2D(shape=(128,128),dx=1/128,n_angles=12,anisotropy_g=.65,eta_scale=0.,eta_floor=0.,device=device)
  low.setup(total,scatter,rhs[0].to(torch.complex64));high=high_precision_copy(low)
  phase=high.phase_matrix.double()*high.weights.double()[None];mu=torch.linalg.eigvalsh((phase+phase.T)/2).max();margin=float((high.sigma_t-high.sigma_s*mu).min());assert margin>0
  sync();physical=time.perf_counter()-t;t=time.perf_counter();base=build_base(low,'tsa64')
  if device=='cuda':
   from direct_segmented_tsa import DirectSegmentedTSA
   base=DirectSegmentedTSA(base,low.source,chunk=1,capture_first=False)
  else:base.prepare()
  pre=lambda r:base(r.to(torch.complex64)).to(torch.complex128)
  sync();tsa=time.perf_counter()-t;settings.update(physical_setup_seconds=physical,tsa_setup_seconds=tsa,coercivity_margin=margin,medium=info)
  g=None;gsetup=0.
  if args.with_layered:
   from layered_inverse import LayeredInverse,CapturedAction
   sync();t=time.perf_counter();factor=LayeredInverse(low,scheme='compact4',nyquist=True,dtype=torch.complex64)
   g0=lambda r:factor(r).to(torch.complex128)
   g=CapturedAction(g0,high.source) if device=='cuda' else g0
   sync();gsetup=time.perf_counter()-t;settings['layered_setup_seconds']=gsetup
  save();references={}
  for rep in range(-1,args.repeats):
   methods=settings['methods'].copy();random.Random(20261110+rep).shuffle(methods)
   if rep<0:methods.remove('native');methods.insert(0,'native')
   for method in methods:
    gpu_guard();rec=Recycle(rhs[0].shape,device);batchrows=[];sync();batch_start=time.perf_counter()
    for pos,(sid,b) in enumerate(zip(ids,rhs)):
     if time.perf_counter()-began>args.wall_limit:raise RuntimeError('wall budget')
     zs=[];ws=[];bn=float(b.norm());sync();t=time.perf_counter()
     initial=rec.initial(b) if method in ['solution_init','krylov_init','gcro'] else None
     initial_residual=float((b-high.A_op(initial)).norm()/bn) if initial is not None else 1.
     def wrapped(r):
      z=pre(r);w=high.A_op(z)
      if method=='gcro':z,w=rec.project(z,w)
      zs.append(z.clone());ws.append(w.clone());return z
     if method=='compact4_inject':u,st=augmented_fgmres(high.A_op,b,pre,None,inject=lambda x,r:g(r),inject_after=1,maxiter=100)
     else:
      action=wrapped if method in ['capture_control','krylov_init','gcro'] else pre
      u,st=solve_from_initial(high.A_op,b,action,initial,maxiter=100)
     sync();solve=time.perf_counter()-t;t=time.perf_counter()
     residual=float((b-high.A_op(u)).norm()/bn);d=residual*bn/margin;den=float(u.norm())-d;bound=d/den if den>0 else float('inf')
     if method=='native' and rep<0:references[sid]=u.clone()
     diff=float((u-references[sid]).norm()/references[sid].norm())
     accepted=bool(torch.isfinite(u).all()) and not st['failed'] and residual<=1e-5 and bound<=1e-3 and diff<=1e-4
     sync();check=time.perf_counter()-t;t=time.perf_counter()
     if pos<len(ids)-1:
      if method=='solution_init':rec.update([u],[high.A_op(u)])
      elif method in ['krylov_init','gcro']:rec.update(zs,ws)
     sync();update=time.perf_counter()-t
     st.pop('trace',None)
     row=dict(repeat=rep,method=method,position=pos,source_id=sid,solve_seconds=solve,check_seconds=check,recycle_update_seconds=update,accepted_seconds=solve+check+update,initial_residual=initial_residual,accepted=accepted,true_residual=residual,evaluated_relative_bound=bound,relative_field_difference=diff,status=st,recycle_rank_after=len(rec.C),captured_directions=len(zs))
     rows.append(row);batchrows.append(row);save()
     print(json.dumps(dict(event='RHS',method=method,repeat=rep,source_id=sid,iterations=st['iterations'],accepted=accepted,seconds=row['accepted_seconds'],rank=len(rec.C))),flush=True)
    sync();wall=time.perf_counter()-batch_start
    # Audits are diagnostics, excluded from the deployment time, but disclosed.
    sync();t=time.perf_counter();audit=rec.audit(high.A_op);sync();auditsec=time.perf_counter()-t
    audits.append(dict(repeat=rep,method=method,batch_accepted_seconds=sum(r['accepted_seconds'] for r in batchrows),batch_elapsed_seconds_including_receipt_io=wall,setup_seconds=physical+tsa+(gsetup if method=='compact4_inject' else 0.),batch_complete_seconds=physical+tsa+(gsetup if method=='compact4_inject' else 0.)+sum(r['accepted_seconds'] for r in batchrows),audit_seconds=auditsec,**audit));save()
  settings['complete']=True
 except BaseException as exc:settings['error']=repr(exc);raise
 finally:
  settings.update(wall_seconds=time.perf_counter()-began,attempts=len(rows),failures=sum(not r['accepted'] for r in rows))
  settings['production_hashes_after']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in settings['production_hashes']}
  save()
if __name__=='__main__':main()
