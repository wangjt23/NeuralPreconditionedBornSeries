"""CPU-only exact structural audit; isotropic inverse is a labeled counterfactual."""
import os,sys,time,json,hashlib,math
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=Path('/data/wjt/NeuralPreconditionedBornSeries')
sys.path[:0]=[str(ROOT),str(ROOT/'rte/workstation_runs/rte_predictability_20261108_dot_v1'),str(ROOT/'rte/workstation_runs/layered_background_20261108_dot_v1')]
from screen_structured import medium
from layered_inverse import background_from_material
from rte.models import PeriodicRTE2D
from rte.mixed_precision_transport import high_precision_copy

class IsotropicWoodbury:
 def __init__(self,T,s,w):
  start=time.perf_counter();self.n=len(s);self.l=len(w);self.w=w.to(T.dtype)
  self.t_lu,self.t_piv=torch.linalg.lu_factor(T);a=time.perf_counter()
  self.B=torch.linalg.lu_solve(self.t_lu,self.t_piv,torch.diag(s).to(T.dtype).expand(self.l,-1,-1).contiguous())
  H=torch.eye(self.n,dtype=T.dtype,device=T.device)-(self.w[:,None,None]*self.B).sum(0)
  b=time.perf_counter();self.h_lu,self.h_piv=torch.linalg.lu_factor(H)
  self.cost=dict(transport_LU_seconds=a-start,reduced_assembly_seconds=b-a,reduced_LU_seconds=time.perf_counter()-b,total_seconds=time.perf_counter()-start)
  self.bytes=sum(v.numel()*v.element_size() for v in [self.t_lu,self.t_piv,self.B,self.h_lu,self.h_piv,self.w])
 def __call__(self,q):
  shape=q.shape;y=torch.linalg.lu_solve(self.t_lu,self.t_piv,q.reshape(self.l,self.n,-1))
  h=torch.linalg.lu_solve(self.h_lu,self.h_piv,(self.w[:,None,None]*y).sum(0))
  return (y+self.B@h).reshape(shape)

def inverse_certificate(M):
 inv=torch.linalg.inv(M);res=torch.eye(len(M),dtype=M.dtype)-M@inv
 # Conservative floating-point residual enclosure for the stored double matrix.
 # Inputs are ordinary double numbers, and gamma_n bounds dot-product rounding.
 eps=torch.finfo(M.dtype).eps;gamma=(4*len(M)*eps)/(1-4*len(M)*eps)
 computed=float(res.abs().sum(1).max());roundoff=gamma*float((M.abs()@inv.abs()).sum(1).max())
 return dict(computed_inverse_residual_inf=computed,conservative_roundoff_allowance=roundoff,residual_upper_bound=computed+roundoff,invertibility_certified=computed+roundoff<1.)

@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261110);began=time.perf_counter();out=HERE/'audit_01';out.mkdir(exist_ok=False);n=128;l=12
 b=torch.ones((1,l,n,n),dtype=torch.complex64);t0,s0,_=medium(n,1,10/3,'cpu');low=PeriodicRTE2D(shape=(n,n),dx=1/n,n_angles=l,anisotropy_g=.65,eta_scale=0.,eta_floor=0.,device='cpu');low.setup(t0,s0,b);high=high_precision_copy(low)
 axis,tbg,sbg,_=background_from_material(low.sigma_t,low.sigma_s);assert axis==0;t=tbg[0,0,:,0].double();s=sbg[0,0,:,0].double();K=high.phase_matrix*high.weights[None];Kiso=torch.ones_like(K)*high.weights[None]
 assert torch.equal(Kiso,PeriodicRTE2D._make_phase_matrix(l,0.,torch.float32).double()*high.weights[None])
 sv=torch.linalg.svdvals(K);isv=torch.linalg.svdvals(Kiso);sub=K[:6,6:];subsv=torch.linalg.svdvals(sub)
 F=torch.fft.fft(torch.eye(l,dtype=torch.complex128),dim=0,norm='ortho');Kf=F@K.to(torch.complex128)@F.conj().T
 Vn=F@torch.diag(high.directions[:,0].to(torch.complex128))@F.conj().T;Vt=F@torch.diag(high.directions[:,1].to(torch.complex128))@F.conj().T
 circ=torch.stack([torch.roll(K[0],i) for i in range(l)]);nearest=torch.zeros((l,l),dtype=torch.bool)
 for i in range(l):nearest[i,(i-1)%l]=True;nearest[i,(i+1)%l]=True
 structural=dict(g=.65,angular_singular_values=sv.tolist(),angular_rank=int(torch.linalg.matrix_rank(K)),angular_inverse_certificate=inverse_certificate(K),isotropic_singular_values=isv.tolist(),isotropic_rank=int(torch.linalg.matrix_rank(Kiso)),scattering_profile_min=float(s.min()),scattering_profile_nonzeros=int(torch.count_nonzero(s)),collision_rank_product=int(torch.linalg.matrix_rank(K))*int(torch.count_nonzero(s)),required_reduced_dimension_general=12*n,proposed_isotropic_reduced_dimension=n,offdiagonal_submatrix_singular_values=subsv.tolist(),offdiagonal_submatrix_rank=int(torch.linalg.matrix_rank(sub)),offdiagonal_inverse_certificate=inverse_certificate(sub),lower_bound_after_any_angular_diagonal_reassignment=int(torch.linalg.matrix_rank(sub))*n,isotropic_omission_spectral_norm=float(torch.linalg.matrix_norm(K-Kiso,ord=2)),isotropic_omission_relative_frobenius=float((K-Kiso).norm()/K.norm()),stored_circulant_relative_defect=float((K-circ).norm()/K.norm()),angular_fourier_collision_offdiagonal_relative_norm=float((Kf-torch.diag(torch.diag(Kf))).norm()/Kf.norm()),angular_fourier_normal_non_nearest_relative_norm=float(Vn.masked_fill(nearest,0).norm()/Vn.norm()),angular_fourier_tangent_non_nearest_relative_norm=float(Vt.masked_fill(nearest,0).norm()/Vt.norm()),boundary='Periodic Fourier; no special boundary rows.',spatial_coefficient='Positive diagonal sigma_s(y), multiplied after full angular scattering.',docstring_caution='Class docstring says isotropic, but actual g=.65 branch uses full normalized angular kernel.')
 I=torch.eye(n,dtype=torch.complex128);nk=high.kx.flatten();tk=high.ky.flatten();D=torch.fft.ifft(1j*nk[:,None]*torch.fft.fft(I,dim=0),dim=0);normal=high.directions[:,0];tangent=high.directions[:,1];S=torch.diag(s).to(torch.complex128);records=[]
 for k in [0,1,8]:
  stage=time.perf_counter();T=normal[:,None,None]*D[None]+torch.diag(t).to(torch.complex128)[None]+(1j*tk[k]*tangent[:,None,None])*I[None];Tfull=torch.block_diag(*T.unbind(0));G=Tfull-torch.kron(K.to(torch.complex128).contiguous(),S);Gi=Tfull-torch.kron(Kiso.to(torch.complex128).contiguous(),S);assembly=time.perf_counter()-stage
  q=torch.complex(torch.randn(l*n,3,dtype=torch.float64),torch.randn(l*n,3,dtype=torch.float64))
  stage=time.perf_counter();lu,piv=torch.linalg.lu_factor(G);dense_setup=time.perf_counter()-stage;stage=time.perf_counter();il,ip=torch.linalg.lu_factor(Gi);iso_dense_setup=time.perf_counter()-stage
  wb=IsotropicWoodbury(T,s,high.weights);stage=time.perf_counter();actual=torch.linalg.lu_solve(lu,piv,q);dense_apply=time.perf_counter()-stage;stage=time.perf_counter();iso=torch.linalg.lu_solve(il,ip,q);iso_apply=time.perf_counter()-stage;stage=time.perf_counter();wrong=wb(q);wb_apply=time.perf_counter()-stage
  vectors=[]
  for j in range(3):
   v=q[:,j];modes=torch.zeros((1,l,n,n),dtype=torch.complex128);modes[...,k]=v.reshape(1,l,n);field=torch.fft.ifft(modes,dim=-1,norm='ortho');image=high.A_op(field)-(high.sigma_t-tbg.double())*field+(high.sigma_s-sbg.double())*torch.einsum('lm,bmxy->blxy',K.to(torch.complex128),field);mode=torch.fft.fft(image,dim=-1,norm='ortho')[...,k].flatten()
   block_error=float((mode-G@v).norm()/(G@v).norm());iso_res=float((Gi@wrong[:,j]-q[:,j]).norm()/q[:,j].norm());iso_diff=float((wrong[:,j]-iso[:,j]).norm()/iso[:,j].norm());actual_res=float((G@actual[:,j]-q[:,j]).norm()/q[:,j].norm());wrong_res=float((G@wrong[:,j]-q[:,j]).norm()/q[:,j].norm());wrong_diff=float((wrong[:,j]-actual[:,j]).norm()/actual[:,j].norm())
   assert block_error<1e-10 and iso_res<1e-10 and iso_diff<1e-10 and actual_res<1e-10
   vectors.append(dict(vector=j,original_A0_vs_explicit_block_relative_difference=block_error,actual_dense_residual=actual_res,isotropic_Woodbury_residual=iso_res,isotropic_Woodbury_vs_isotropic_dense=iso_diff,rank1_Woodbury_true_anisotropic_residual=wrong_res,rank1_Woodbury_vs_true_dense_relative_difference=wrong_diff))
  records.append(dict(k=k,assembly_seconds=assembly,actual_dense_LU_seconds=dense_setup,isotropic_dense_LU_seconds=iso_dense_setup,isotropic_Woodbury_setup=wb.cost,actual_dense_apply_three_rhs_seconds=dense_apply,isotropic_dense_apply_three_rhs_seconds=iso_apply,isotropic_Woodbury_apply_three_rhs_seconds=wb_apply,actual_dense_factor_bytes=lu.numel()*lu.element_size()+piv.numel()*piv.element_size(),isotropic_Woodbury_bytes=wb.bytes,vectors=vectors))
 report=dict(complete=True,scope='Structural obstruction audit on actual g=.65. Rank-Ny Woodbury is tested only as a g=0 diagnostic counterfactual, not deployed as an approximation.',pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0)),device='cpu',threads=1,grid=n,L=l,modes=[0,1,8],seed=20261110,vectors_per_mode=3,structural=structural,mode_records=records,wall_seconds=time.perf_counter()-began,source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),production_hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['rte/models.py','rte/mixed_precision_transport.py','rte/workstation_runs/layered_background_20261108_dot_v1/layered_inverse.py']},GPU_used=False,new_full_PDE_solves=False,neural_training=False)
 (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
