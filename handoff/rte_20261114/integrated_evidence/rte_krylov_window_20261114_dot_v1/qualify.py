import sys,time,json,hashlib
from pathlib import Path
ROOT=Path('/data/wjt/NeuralPreconditionedBornSeries');OLD=ROOT/'rte/workstation_runs/rte_mcmg_born_20261114_dot_v1';OUT=Path(__file__).parent;sys.path.insert(0,str(OLD));from train import *
from window import real,complex,prefix,window,k_loss,o_loss
from rte.models import PeriodicRTE2D
from rte.npbs_unet_ablation import fgmres_real_stacked_with_trace

def main():
 torch.set_num_threads(4);torch.manual_seed(146401);data=torch.load(OLD/'data.pt',weights_only=True)['train'];cfg=ExperimentConfig(size=64,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes='mixed');ids=torch.tensor([0]);q=source_from_data(cfg,data,ids,'cuda');op=make_operator(cfg,data['sigma_t'][:1],data['sigma_s'][:1],q,'cuda');a64=PeriodicRTE2D(shape=(64,64),dx=1/64,n_angles=12,anisotropy_g=.65,eta_scale=0.,fdtype=torch.float64,device='cuda').setup(sigma_t=data['sigma_t'][:1].double().cuda(),sigma_s=data['sigma_s'][:1].double().cuda(),source=q.to(torch.complex128));m=Cycle(op).cuda().eval();m.load_state_dict(torch.load(OLD/'physical_2048.pt',weights_only=True)['model']);A=lambda x:real(a64.A_op(complex(x)));M=lambda x:real(m(complex(x).to(torch.complex64)).to(torch.complex128));b=real(a64.source);out=dict(forward=[],finite_difference=[],timings=[],source_hash=hashlib.sha256((OUT/'window.py').read_bytes()).hexdigest(),scope='fixed-length prefix only for equivalence; training windows must be selected from actual unconverged trajectory',steps=8,restart=20)
 for start in [0,8,16,20]:
  m.cache=None;m.key=None;ctx,_=prefix(A,M,b,start)
  with torch.no_grad():
   state,rows=window(A,M,b,ctx);native,h,_,calls=fgmres_real_stacked_with_trace(a64.A_op,a64.source,lambda r:m(r.to(torch.complex64)).to(torch.complex128),tol=0.,maxiter=start+8,restart=20,max_restarts=5);xr=real(native);err=float((state['x']-xr).norm()/xr.norm());hist_err=abs(float(rows[-1]['relative'])-h[-1]);out['forward'].append(dict(start=start,solution_relative_error=err,residual_absolute_difference=hist_err,native_calls=calls[-1],window_final=float(rows[-1]['relative'])));print('forward',out['forward'][-1],flush=True)
 ctx,_=prefix(A,M,b,8);rstart=b-A(ctx['x']);alpha=torch.tensor(1.,device='cuda',dtype=torch.float64,requires_grad=True)
 def Mscale(x):
  r=complex(x).to(torch.complex64);z=m.op.G(r);corr=m.delta(z,m.fields());return real(z.to(torch.complex128)+alpha*corr.to(torch.complex128))
 _,rows=window(A,Mscale,b,ctx);loss=k_loss(rows,rstart);grad=torch.autograd.grad(loss,alpha)[0];base=float(alpha)
 for eps in [1e-2,3e-3,1e-3,3e-4]:
  vals=[]
  for sign in [1,-1]:
   with torch.no_grad():alpha.fill_(base+sign*eps);_,rr=window(A,Mscale,b,ctx);vals.append(float(k_loss(rr,rstart)))
  fd=(vals[0]-vals[1])/(2*eps);error=abs(fd-float(grad))/max(abs(fd),abs(float(grad)),1e-12);out['finite_difference'].append(dict(epsilon=eps,autograd=float(grad),finite_difference=fd,relative_error=error));print('fd',out['finite_difference'][-1],flush=True)
 with torch.no_grad():alpha.fill_(base)
 for arm in ['O','K']:
  m.zero_grad(set_to_none=True);m.cache=None;m.key=None;ctx,_=prefix(A,M,b,8);rstart=b-A(ctx['x']);torch.cuda.synchronize();t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
  if arm=='O':
   with torch.no_grad():_,rows=window(A,M,b,ctx)
   loss=o_loss(A,M,rows)
  else:
   _,rows=window(A,M,b,ctx);loss=k_loss(rows,rstart)
  loss.backward();torch.cuda.synchronize();out['timings'].append(dict(arm=arm,forward_backward_seconds=time.perf_counter()-t,loss=float(loss),gradient_norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),float('inf'))),peak_memory=torch.cuda.max_memory_allocated()));print('time',out['timings'][-1],flush=True)
 out['pass_forward']=max(x['solution_relative_error'] for x in out['forward'])<1e-3;out['pass_scalar_fd']=min(x['relative_error'] for x in out['finite_difference'])<.02;(OUT/'qualification.json').write_text(json.dumps(out,indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':main()
