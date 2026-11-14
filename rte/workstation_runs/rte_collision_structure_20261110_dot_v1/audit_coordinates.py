"""Compare actual angular and normal Fourier structure without replacing A."""
import os,sys,time,json,hashlib,math
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=Path('/data/wjt/NeuralPreconditionedBornSeries')
sys.path[:0]=[str(ROOT),str(ROOT/'rte/workstation_runs/rte_predictability_20261108_dot_v1'),str(ROOT/'rte/workstation_runs/layered_background_20261108_dot_v1')]
from screen_structured import medium
from layered_inverse import background_from_material
from rte.models import PeriodicRTE2D
from rte.mixed_precision_transport import high_precision_copy
@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);start=time.perf_counter();out=HERE/'coordinates_01';out.mkdir(exist_ok=False);n=128;l=12
 t,s,_=medium(n,1,10/3,'cpu');low=PeriodicRTE2D(shape=(n,n),dx=1/n,n_angles=l,anisotropy_g=.65,eta_scale=0.,eta_floor=0.,device='cpu');low.setup(t,s,torch.zeros(1,l,n,n,dtype=torch.complex64));high=high_precision_copy(low);axis,tbg,sbg,_=background_from_material(low.sigma_t,low.sigma_s);assert axis==0
 t=tbg[0,0,:,0].double();s=sbg[0,0,:,0].double();K=high.phase_matrix*high.weights[None];F=torch.fft.fft(torch.eye(l,dtype=torch.complex128),dim=0,norm='ortho');Cn=F@torch.diag(high.directions[:,0].to(torch.complex128))@F.conj().T;Ct=F@torch.diag(high.directions[:,1].to(torch.complex128))@F.conj().T;Kf=F@K.to(torch.complex128)@F.conj().T
 mask=torch.eye(l,dtype=torch.bool)
 for i in range(l):mask[i,(i+1)%l]=True;mask[i,(i-1)%l]=True
 En=Cn.masked_fill(mask,0);Et=Ct.masked_fill(mask,0);Ek=Kf.masked_fill(mask,0)
 alpha=float((t-s*torch.linalg.eigvalsh((K+K.T)/2).max()).min());I=torch.eye(n,dtype=torch.complex128);D=torch.fft.ifft(1j*high.kx.flatten()[:,None]*torch.fft.fft(I,dim=0),dim=0)
 modes=[]
 for k in [0,1,8]:
  G=torch.kron(Cn.contiguous(),D.contiguous())+1j*high.ky.flatten()[k]*torch.kron(Ct.contiguous(),I)+torch.kron(torch.eye(l,dtype=torch.complex128),torch.diag(t).to(torch.complex128))-torch.kron(Kf.contiguous(),torch.diag(s).to(torch.complex128))
  E=torch.kron(En.contiguous(),D.contiguous())+1j*high.ky.flatten()[k]*torch.kron(Et.contiguous(),I)-torch.kron(Ek.contiguous(),torch.diag(s).to(torch.complex128))
  bound=float(torch.linalg.matrix_norm(En,2)*high.kx.abs().max()+high.ky.flatten()[k].abs()*torch.linalg.matrix_norm(Et,2)+torch.linalg.matrix_norm(Ek,2)*s.abs().max())
  modes.append(dict(k=k,discarded_full_block_relative_frobenius=float(E.norm()/G.norm()),discarded_operator_2norm_upper_bound=bound,original_coercivity_margin=alpha,conditional_refinement_contraction_bound=bound/(alpha-bound),conditional_margin_after_band_projection=alpha-bound))
 f=torch.fft.fftfreq(n,dtype=torch.float64)*n;normal={}
 for name,v in [('sigma_t',t),('sigma_s',s)]:
  c=torch.fft.fft(v,norm='ortho');cuts=[]
  for width in [1,2,4,8,16,32,48,63]:
   tail=c[f.abs()>width];rec=torch.fft.ifft(c.masked_fill(f.abs()>width,0),norm='ortho').real
   cuts.append(dict(half_bandwidth=width,relative_coefficient_tail_l2=float(tail.norm()/c.norm()),relative_material_linf=float((v-rec).abs().max()/v.abs().max()),max_absolute_material_error=float((v-rec).abs().max())))
  normal[name]=dict(nonzero_fourier_coefficients=int(torch.count_nonzero(c)),above_1e12_relative_coefficients=int(torch.count_nonzero(c.abs()>1e-12*c.abs().max())),cuts=cuts)
 angle=torch.remainder(torch.atan2(high.directions[:,1],high.directions[:,0]),2*math.pi);ideal=torch.arange(l,dtype=torch.float64)*2*math.pi/l;err=torch.remainder(angle-ideal+math.pi,2*math.pi)-math.pi
 report=dict(complete=True,pid=os.getpid(),device='cpu',affinity=sorted(os.sched_getaffinity(0)),nominal_nodes='arange(L,float32)*2*pi/L',weights_bitwise_constant=bool(torch.equal(high.weights,high.weights[0].expand_as(high.weights))),stored_angle_max_error_radians=float(err.abs().max()),unit_direction_max_norm_error=float((high.directions.norm(dim=1)-1).abs().max()),angular_unitarity_defect=float((F@F.conj().T-torch.eye(l)).norm()),angular_modes=modes,normal_material_spectra=normal,selection='Angular cyclic block-tridiagonal part plus explicitly retained original-operator defect/refinement. Not bitwise exact. Reject narrow normal-Fourier band as exact representation; the actual layer profiles have broad support.',scope='Coordinate/source audit only, no banded solver or performance claim yet.',wall_seconds=time.perf_counter()-start,source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),GPU_used=False)
 (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
