import time
START=time.perf_counter()
import os,json,math
from pathlib import Path
import run_operator_green as m
p=m.p;torch=m.torch;HERE=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);assert set(os.sched_getaffinity(0))=={0} and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 original=json.loads((HERE/'results/results.json').read_text());check=lambda x:None
 n=original['n'];eta=original['block_absolute_F_budget'];alpha=original['coercivity_alpha'];normal=original['max_normal_velocity']
 low=p.PeriodicRTE2D(shape=(n,n),dx=1/n,n_angles=12,anisotropy_g=.65,eta_scale=0.,eta_floor=0.,device='cpu');actual=low.kx.flatten().double();ideal=2*math.pi*torch.fft.fftfreq(n,d=1/n,dtype=torch.float64)
 eye=torch.eye(n,dtype=torch.complex128)
 def derivative(k):return torch.fft.ifft(1j*k[:,None]*torch.fft.fft(eye,dim=0),dim=0)
 D=derivative(actual);Di=derivative(ideal);z=torch.exp(2j*math.pi*torch.arange(n,dtype=torch.float64)/n);idx=torch.arange(n);sgn=((-1.)**(idx[:,None]-idx[None,:])).double();den=z[:,None]-z[None,:];mask=~torch.eye(n,dtype=torch.bool);Dc=2j*math.pi*sgn*z[None,:]/torch.where(mask,den,torch.ones_like(den));Dc.diagonal().fill_(-1j*math.pi)
 ta=m.ScalarTree((D-D.conj().T)/2,eta,m.LEAF);ti=m.ScalarTree((Di-Di.conj().T)/2,eta,m.LEAF)
 def stats(tree):
  l=12;cost=0
  def node(t):
   nonlocal cost
   if t.leaf:cost+=16*(t.n*l)**2+4*(t.n*l)
   else:
    R=t.r*l;cost+=16*(t.n*l*R+3*R*R)+4*R;node(t.a);node(t.b)
  node(tree);H=tree.assemble();error=D-H;bound=math.sqrt(float(error.abs().sum(0).max())*float(error.abs().sum(1).max()))
  return dict(ranks=[r['rank'] for r in tree.records if r['kind']=='low_rank'],predicted_factor_bytes_including_scalar_tree=cost+m.size_bytes(tree.tensors()),row_column_bound_vs_actual_D=bound,original_operator_q_bound=normal*bound/alpha)
 sa=stats(ta);si=stats(ti);assert sa['predicted_factor_bytes_including_scalar_tree']==original['rows'][0]['candidate_factor_bytes_including_derivative']
 # For the alternative exact-arithmetic model, use a transparent triangle bound too.
 erri=Di-ti.assemble();idealerr=math.sqrt(float(erri.abs().sum(0).max())*float(erri.abs().sum(1).max()));freq=float((actual-ideal).abs().max());tri=normal*(freq+idealerr)/alpha
 out=dict(complete=True,pid=os.getpid(),scope='Scalar-only post-run diagnostic; no ideal Green factor or inverse constructed, no solver speed measured',original_results_sha256=p.digest(HERE/'results/results.json'),plan_sha256=p.digest(HERE/'SCALAR_DIAGNOSTIC_PLAN.md'),source_sha256=p.digest(Path(__file__)),Cauchy_bridge_relative_corrected=float((Dc-Di).norm()/Di.norm()),original_Cauchy_diagnostic_bug='float32 sign tensor caused scalar2pi coefficient rounding before complex128 promotion; does not affect actual-D construction or solve',stored_vs_ideal_frequency_2norm=freq,actual_scalar=sa,ideal_scalar_only=si,triangle_q_bound_including_stored_frequency=tri,triangle_q4=tri**4,model_bound_is_not_uniform_floating_certificate=True,elapsed_process_seconds=time.perf_counter()-START,peak_process_RSS_bytes=p.rss())
 (HERE/'SCALAR_DIAGNOSTIC.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
