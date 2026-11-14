"""Operator-first normal Green factor, fixed A0-weighted precision; no full inverse."""
import time
PROCESS_START=time.perf_counter()
import os,json,math,gc,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
import prior_permutation_driver as p
import torch
from angular_modes import AngularBandMode
N=128;CASES=[0,8];LEAF=8;Q_DESIGN=1e-3
DT=torch.complex128

def size_bytes(tensors):
 seen=set();total=0
 for t in tensors:
  key=(t.untyped_storage().data_ptr(),t.untyped_storage().nbytes())
  if key not in seen:seen.add(key);total+=key[1]
 return total

class ScalarTree:
 def __init__(self,D,eta,leaf=8,i=0,depth=0,records=None):
  self.i=i;self.n=len(D);self.depth=depth;self.leaf=True;self.records=[] if records is None else records
  n=self.n;h=n//2
  if n>leaf:
   tick=time.perf_counter();u,s,vh=torch.linalg.svd(D[:h,h:],full_matrices=False)
   tail=torch.cat([s.flip(0).square().cumsum(0).flip(0).sqrt(),s.new_zeros(1)])
   r=int(torch.nonzero(tail<=eta)[0,0]);r=max(1,r)
   if r<min(h,n-h):
    self.leaf=False;self.r=r;self.L=(u[:,:r]*s[:r].sqrt()).clone().contiguous();self.R=(vh[:r].conj().T*s[:r].sqrt()).clone().contiguous()
    error=D[:h,h:]-self.L@self.R.conj().T
    self.records.append(dict(i=i,n=n,depth=depth,kind='low_rank',rank=r,angular_rank=12*r,actual_block_F_error=float(error.norm()),budget=eta,svd_seconds=time.perf_counter()-tick,singular_values=s.tolist()))
    self.a=ScalarTree(D[:h,:h],eta,leaf,i,depth+1,self.records);self.b=ScalarTree(D[h:,h:],eta,leaf,i+h,depth+1,self.records)
   else:self.records.append(dict(i=i,n=n,depth=depth,kind='full_rank_dense_leaf',rank=r,svd_seconds=time.perf_counter()-tick,singular_values=s.tolist()))
  else:self.records.append(dict(i=i,n=n,depth=depth,kind='small_dense_leaf'))
  if self.leaf:self.D=D.clone().contiguous()
 def assemble(self):
  if self.leaf:return self.D.clone()
  h=self.a.n;out=torch.empty((self.n,self.n),dtype=DT);out[:h,:h]=self.a.assemble();out[h:,h:]=self.b.assemble();out[:h,h:]=self.L@self.R.conj().T;out[h:,:h]=-out[:h,h:].conj().T;return out
 def tensors(self):return [self.D] if self.leaf else [self.L,self.R]+self.a.tensors()+self.b.tensors()

class GreenFactor:
 def __init__(self,tree,V,normal,records=None):
  self.tree=tree;self.n=tree.n;self.l=len(normal);self.records=[] if records is None else records;t=time.perf_counter();l=self.l
  if tree.leaf:
   matrix=torch.kron(tree.D.contiguous(),torch.diag(normal.to(DT)))+torch.block_diag(*V.unbind())
   self.lu,self.piv=torch.linalg.lu_factor(matrix)
   self.records.append(dict(i=tree.i,n=tree.n,kind='dense_leaf_LU',dimension=l*tree.n,seconds=time.perf_counter()-t));return
  h=tree.a.n;self.a=GreenFactor(tree.a,V[:h],normal,self.records);self.b=GreenFactor(tree.b,V[h:],normal,self.records)
  t=time.perf_counter();self.WL=self.a.apply(torch.kron(tree.L,torch.diag(normal.to(DT))));self.WR=self.b.apply(-torch.kron(tree.R,torch.diag(normal.to(DT))))
  self.ML=self.project(tree.L,self.WL);self.MR=self.project(tree.R,self.WR);S=torch.eye(tree.r*l,dtype=DT)-self.ML@self.MR
  self.lu,self.piv=torch.linalg.lu_factor(S)
  # Numerical local factor backward residual, not a uniform whole inverse certificate.
  self.records.append(dict(i=tree.i,n=tree.n,kind='recursive_Schur',scalar_rank=tree.r,Schur_dimension=tree.r*l,own_seconds=time.perf_counter()-t,Schur_matrix_norm=float(S.norm()),Schur_factor_finite=bool(torch.isfinite(self.lu).all())))
 def project(self,U,q):return torch.einsum('ij,jac->iac',U.conj().T,q.reshape(U.shape[0],self.l,-1)).reshape(U.shape[1]*self.l,-1)
 def apply(self,q):
  if self.tree.leaf:return torch.linalg.lu_solve(self.lu,self.piv,q)
  h=self.a.n*self.l;yL=self.a.apply(q[:h]);yR=self.b.apply(q[h:]);cL=self.project(self.tree.L,yL);cR=self.project(self.tree.R,yR)
  a=torch.linalg.lu_solve(self.lu,self.piv,cL-self.ML@cR);b=cR-self.MR@a
  return torch.cat([yL-self.WL@b,yR-self.WR@a],dim=0)
 def tensors(self):
  result=[self.lu,self.piv]
  if not self.tree.leaf:result +=[self.WL,self.WR,self.ML,self.MR]+self.a.tensors()+self.b.tensors()
  return result

def repaired(action,inverse,q,tolerance=1e-10,max_corrections=3):
 active=(q!=0).any(dim=0);x=torch.zeros_like(q);trace=[]
 if not bool(active.any()):return x,dict(success=True,corrections=0,residual_trace=[[0.]*q.shape[1]])
 scale=q[:,active].abs().amax(dim=0);den=(q[:,active]/scale).norm(dim=0);x[:,active]=inverse(q[:,active])
 for j in range(max_corrections+1):
  r=q-action(x);ratio=torch.zeros(q.shape[1],dtype=torch.float64);ratio[active]=(r[:,active]/scale).norm(dim=0)/den;trace.append(ratio.tolist())
  if bool((ratio<=tolerance).all()):return x,dict(success=True,corrections=j,residual_trace=trace)
  if j<max_corrections:x[:,active]+=inverse(r[:,active])
 return x,dict(success=False,corrections=max_corrections,residual_trace=trace)

def unit():
 t=time.perf_counter();g=torch.Generator().manual_seed(2026111401)
 def rnd(*shape):return torch.complex(torch.randn(*shape,dtype=torch.float64,generator=g),torch.randn(*shape,dtype=torch.float64,generator=g))
 a=rnd(4,4);b=rnd(4,4);u=rnd(4,1);v=rnd(4,1);D=torch.cat([torch.cat([a-a.conj().T,u@v.conj().T],dim=1),torch.cat([-v@u.conj().T,b-b.conj().T],dim=1)],dim=0)
 normal=torch.tensor([-.7,.4,.9],dtype=torch.float64);V=torch.stack([torch.eye(3,dtype=DT)*(2+.1j*j) for j in range(8)])
 tree=ScalarTree(D,1e-12,leaf=2);fac=GreenFactor(tree,V,normal);G=torch.kron(D,torch.diag(normal.to(DT)))+torch.block_diag(*V.unbind());q=rnd(24,3);x=fac.apply(q);ref=torch.linalg.solve(G,q)
 rec=dict(seconds=time.perf_counter()-t,action_residual=float((G@x-q).norm()/q.norm()),relative_solution_difference=float((x-ref).norm()/ref.norm()),root_low_rank=not tree.leaf,records=fac.records)
 if not rec['root_low_rank'] or max(rec['action_residual'],rec['relative_solution_difference'])>1e-12:raise RuntimeError('Recursive algebra unit test failed: '+str(rec))
 return rec

@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and set(os.sched_getaffinity(0))=={0}
 out=HERE/'results';out.mkdir(exist_ok=False)
 report=dict(complete=False,pid=os.getpid(),n=N,cases=CASES,device='cpu',threads=1,affinity=[0],Q_design=Q_DESIGN,leaf_nominal=LEAF,fullrank_leaf_fallback=True,physical_RHS_used=False,heldout_used=False,training=False,full_dense_inverse_constructed=False,scope='Known derivative compression + recursive operator factor; fixed original-A0-weighted absolute error budget. No inverse snapshots. Strictly an approximate inverse before original-G defect repair.',rows=[],cost={},resource_at_start=p.admission())
 def save():
  tmp=out/'results.json.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');os.replace(tmp,out/'results.json')
 def check(v,msg):
  if not v:raise RuntimeError(msg)
 def emit(event,**kw):print(json.dumps(dict(event=event,**kw)),flush=True)
 try:
  prior=json.loads((HERE/'PRIOR_K_RESULTS.json').read_text())
  for f,h in prior['source_hashes'].items():check(p.digest(p.ROOT/f)==h,'Production source changed '+f)
  for f,h in prior['E_source_hashes'].items():check(p.digest(p.E/f)==h,'E source changed '+f)
  report['source_hashes']={f.name:p.digest(f) for f in HERE.glob('*.py')};report['plan_sha256']=p.digest(HERE/'OPERATOR_GREEN_PLAN.md');report['unit']=unit();save();emit('algebra_unit_passed',**{k:v for k,v in report['unit'].items() if k!='records'})
  t=time.perf_counter();total,scatter,_=p.medium(N,1,10/3,'cpu');low=p.PeriodicRTE2D(shape=(N,N),dx=1/N,n_angles=12,anisotropy_g=.65,eta_scale=0.,eta_floor=0.,device='cpu');low.setup(total,scatter,torch.zeros((1,12,N,N),dtype=torch.complex64));high=p.high_precision_copy(low);bg,tb,sb=p.background(low);check(bg==prior['background_identity'],'Background differs from sealed K/E')
  tv=tb[0,0,:,0].double();sv=sb[0,0,:,0].double();l=high.n_angles;d=N*l;K=(high.phase_matrix*high.weights[None]).to(DT);kn=high.kx.flatten();normal=high.directions[:,0];tangent=high.directions[:,1];mu=float(torch.linalg.eigvalsh((K+K.conj().T)/2).max());alpha=float((tv-sv*mu).min());check(alpha>0,'Nonpositive alpha');normalmax=float(normal.abs().max());depth=int(math.log2(N//LEAF));eta=alpha*Q_DESIGN/(2*normalmax*depth)
  report.update(background=bg,directions=high.directions.tolist(),phase_rank=int(torch.linalg.matrix_rank(K)),phase_hermitian_max_eigenvalue=mu,coercivity_alpha=alpha,max_normal_velocity=normalmax,max_actual_normal_frequency=float(kn.abs().max()),derivative_depth_upper_bound=depth,block_absolute_F_budget=eta)
  report['cost']['physical_setup_seconds']=time.perf_counter()-t;t=time.perf_counter();D=torch.fft.ifft(1j*kn[:,None]*torch.fft.fft(torch.eye(N,dtype=DT),dim=0),dim=0);Db=(D-D.conj().T)/2;tree=ScalarTree(Db,eta,LEAF);Dh=tree.assemble();report['cost']['known_derivative_build_and_compression_seconds']=time.perf_counter()-t
  t=time.perf_counter();perdepth={}
  for r in tree.records:
   if r['kind']=='low_rank':perdepth[r['depth']]=max(perdepth.get(r['depth'],0),r['actual_block_F_error'])
  # Conservative exact norm inequalities, evaluated in ordinary floating point.
  hermdrop=float((D-Db).norm());hierbound=hermdrop+sum(perdepth.values());directbound=math.sqrt(float((D-Dh).abs().sum(0).max())*float((D-Dh).abs().sum(1).max()));delta=normalmax*min(hierbound,directbound);qbound=delta/alpha
  # Cauchy bridge is a diagnostic only; actual stored frequency D is compressed.
  z=torch.exp(2j*math.pi*torch.arange(N,dtype=torch.float64)/N);ii=torch.arange(N);sgn=(-1.)**(ii[:,None]-ii[None,:]);den=z[:,None]-z[None,:];mask=~torch.eye(N,dtype=torch.bool);Dc=torch.empty((N,N),dtype=DT);Dc[mask]=(2j*math.pi*sgn*z[None,:]/torch.where(mask,den,torch.ones_like(den)))[mask];Dc.diagonal().fill_(-1j*math.pi)
  ideal=2*math.pi*torch.fft.fftfreq(N,d=1/N,dtype=torch.float64);Di=torch.fft.ifft(1j*ideal[:,None]*torch.fft.fft(torch.eye(N,dtype=DT),dim=0),dim=0)
  report['derivative']=dict(scalar_matrix_bytes=D.numel()*D.element_size(),tree_factor_bytes=size_bytes(tree.tensors()),records=tree.records,per_depth_F_bound=perdepth,skew_projection_error_F=hermdrop,approx_skew_defect_F=float((Dh+Dh.conj().T).norm()),hierarchy_norm_upper_bound=hierbound,row_column_norm_upper_bound=directbound,operator_delta_upper_bound=delta,right_defect_exact_arithmetic_upper_bound=qbound,after3_repairs_exact_arithmetic_bound=qbound**4,ideal_Cauchy_bridge_relative=float((Dc-Di).norm()/Di.norm()),stored_vs_ideal_frequency_norm=float((kn-ideal).abs().max()),actual_D_FFT_bridge_relative=float((D@torch.ones(N,1,dtype=DT)-torch.fft.ifft(1j*kn[:,None]*torch.fft.fft(torch.ones(N,1,dtype=DT),dim=0),dim=0)).norm()),bounds_are_interval_certified=False)
  check(qbound<=Q_DESIGN,'Fixed operator error construction failed budget');check(report['derivative']['approx_skew_defect_F']<1e-12,'Skew assembly failed');report['cost']['derivative_proof_checks_seconds']=time.perf_counter()-t;save();emit('derivative_bound',qbound=qbound,ranks=[r.get('rank') for r in tree.records if r['kind']=='low_rank'],bytes=report['derivative']['tree_factor_bytes'])
  for k in CASES:
   start=time.perf_counter();rec=dict(k=k,status='running',cost={},resource_at_start=p.admission());report['rows'].append(rec);save();kt=high.ky.flatten()[k];V=tv[:,None,None]*torch.eye(l,dtype=DT)-sv[:,None,None]*K[None]+1j*kt*torch.diag(tangent.to(DT))[None]
   def original(q):
    x=q.reshape(N,l,-1);der=torch.fft.ifft(1j*kn[:,None,None]*torch.fft.fft(x,dim=0),dim=0)
    return (der*normal[None,:,None]+torch.einsum('nij,njr->nir',V,x)).reshape_as(q)
   def approximate(q):
    x=q.reshape(N,l,-1);der=torch.einsum('ij,jlr->ilr',Dh,x)
    return (der*normal[None,:,None]+torch.einsum('nij,njr->nir',V,x)).reshape_as(q)
   g=torch.Generator().manual_seed(2026111312);q=torch.complex(torch.randn(d,3,dtype=torch.float64,generator=g),torch.randn(d,3,dtype=torch.float64,generator=g));q=q/q.norm(dim=0);mixed=torch.cat([torch.zeros(d,1,dtype=DT),q,1e-200*q[:,:1]],dim=1)
   t=time.perf_counter();G=torch.kron(D,torch.diag(normal.to(DT)))+torch.block_diag(*V.unbind());rec['cost']['original_matrix_for_control_seconds']=time.perf_counter()-t;t=time.perf_counter();matrix_bridge=float((G@q-original(q)).norm()/original(q).norm());check(matrix_bridge<1e-12,'Original matrix bridge')
   # Full original PDE embedding of three source-independent normal vectors.
   ft=torch.zeros((3,l,N,N),dtype=DT);ft[...,k]=q.T.reshape(3,N,l).permute(0,2,1);field=torch.fft.ifft(ft,dim=-1,norm='ortho');actual=high.A_op(field)-(high.sigma_t-tb.double())*field+(high.sigma_s-sb.double())*torch.einsum('lm,bmxy->blxy',K,field)
   ft.zero_();ft[...,k]=original(q).T.reshape(3,N,l).permute(0,2,1);target=torch.fft.ifft(ft,dim=-1,norm='ortho');embedding=float((actual-target).norm()/target.norm());check(embedding<1e-12,'Full original A0 bridge');del ft,field,actual,target
   rec.update(matrix_bridge=matrix_bridge,original_A0_embedding_bridge=embedding,probe_sha256=p.thash(q),actual_tangent_frequency=float(kt));rec['cost']['operator_bridges_seconds']=time.perf_counter()-t
   t=time.perf_counter();fac=GreenFactor(tree,V,normal);rec['cost']['recursive_operator_factor_seconds']=time.perf_counter()-t;rec['factor_records']=fac.records;rec['candidate_factor_bytes_excluding_shared_derivative']=size_bytes(fac.tensors());rec['candidate_factor_bytes_including_derivative']=size_bytes(fac.tensors()+tree.tensors());rec['candidate_raw_apply_seconds']=[];rec['candidate_approximate_action_residual']=[]
   t=time.perf_counter();raw=fac.apply(q);rec['candidate_raw_apply_three_seconds']=time.perf_counter()-t;rec['raw_original_residual']=p.relative((q-original(raw)).T,q.T).tolist();rec['raw_approximate_operator_residual']=p.relative((q-approximate(raw)).T,q.T).tolist()
   # Build both strong controls without supplying their answers to construction.
   t=time.perf_counter();lu,piv=torch.linalg.lu_factor(G);rec['cost']['dense_original_LU_seconds']=time.perf_counter()-t;rec['dense_LU_factor_bytes']=size_bytes([lu,piv])
   t=time.perf_counter();angular=AngularBandMode(high,tv,sv,k);rec['cost']['existing_angular_band_factor_seconds']=time.perf_counter()-t;rec['angular_band_factor_bytes']=angular.bytes
   def angular_raw(b):
    x=b.reshape(N,l,-1).permute(1,0,2).reshape(d,-1);y=angular.raw(x);return y.reshape(l,N,-1).permute(1,0,2).reshape(d,-1)
   rec['inverse_tests']=[];reference=None;t=time.perf_counter()
   for name,inv in [('dense_LU',lambda b:torch.linalg.lu_solve(lu,piv,b)),('existing_angular_band',angular_raw),('operator_first_recursive_Green',fac.apply)]:
    tt=time.perf_counter();x,st=repaired(original,inv,mixed,tolerance=1e-10,max_corrections=3);seconds=time.perf_counter()-tt
    if reference is None:reference=x.clone()
    ratios=p.relative((mixed-original(x)).T,mixed.T);inactive=~(mixed!=0).any(dim=0);zero=bool((x[:,inactive]==0).all());tiny=bool(x[:,-1].abs().max()>0);fd=p.relative((x-reference).T,reference.T)
    rec['inverse_tests'].append(dict(method=name,seconds=seconds,status=st,original_G_relative=ratios.tolist(),relative_to_dense_solution=fd.tolist(),strict_zero=zero,tiny_nonzero=tiny,passed=bool((ratios<=1e-10).all()) and zero and tiny and bool(torch.isfinite(x).all())))
   rec['cost']['all_probe_repair_and_independent_check_seconds']=time.perf_counter()-t;check(all(v['passed'] for v in rec['inverse_tests'][:2]),'Strong control failed');rec['candidate_passed']=rec['inverse_tests'][-1]['passed'];rec.update(status='completed',complete_case_seconds=time.perf_counter()-start,peak_process_RSS_bytes=p.rss());rec['cost']['case_other_seconds']=rec['complete_case_seconds']-sum(rec['cost'].values());save();emit('case_done',k=k,passed=rec['candidate_passed'],setup=rec['cost']['recursive_operator_factor_seconds'],bytes=rec['candidate_factor_bytes_including_derivative'],rawmax=max(rec['raw_original_residual']),floatingmax=max(rec['raw_approximate_operator_residual']),repairs=rec['inverse_tests'][-1]['status']['corrections'],seconds=rec['complete_case_seconds'])
   del fac,angular,lu,piv,reference,G,V,raw,q,mixed,x;gc.collect()
  report['complete']=True;report['all_cases_passed']=all(r['candidate_passed'] for r in report['rows']);report['result']='OPERATOR_FIRST_WEIGHTED_CONCEPT_QUALIFIED_ON_PROBES' if report['all_cases_passed'] else 'OPERATOR_FIRST_FLOATING_FACTOR_OBSTRUCTION'
 except BaseException as exc:report.update(error=repr(exc),traceback=traceback.format_exc());raise
 finally:
  report.update(total_process_seconds=time.perf_counter()-PROCESS_START,peak_process_RSS_bytes=p.rss(),resource_at_end=p.admission())
  if 'prior' in locals():report.update(production_unchanged=all(p.digest(p.ROOT/f)==h for f,h in prior['source_hashes'].items()),E_unchanged=all(p.digest(p.E/f)==h for f,h in prior['E_source_hashes'].items()))
  save();emit('finished',complete=report['complete'],result=report.get('result'),seconds=report['total_process_seconds'],error=report.get('error'))
if __name__=='__main__':main()
