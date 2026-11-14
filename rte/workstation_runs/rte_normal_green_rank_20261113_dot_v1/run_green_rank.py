"""Source-independent fixed-rank normal-space Green representation diagnostic."""
import time
PROCESS_START=time.perf_counter()
import os,json,gc,traceback,math
from pathlib import Path
HERE=Path(__file__).resolve().parent
import prior_permutation_driver as p
import torch
from angular_modes import AngularBandMode
CASES=[(128,0),(128,8),(256,0),(256,8)]
RANK=12;LEAF_NORMAL_CELLS=8

def periodic_gap(i,ni,j,nj,n):
 def gap(a,na,b,nb):return max(0,b-(a+na),a-(b+nb))
 return min(gap(i,ni,j+shift,nj) for shift in [-n,0,n])

class NormalHMatrix:
 def __init__(self,B,n,l):
  self.n=n;self.l=l;self.blocks=[];self.records=[];self.bytes=0;start=time.perf_counter()
  def visit(i,ni,j,nj):
   rows=slice(i*l,(i+ni)*l);cols=slice(j*l,(j+nj)*l);block=B[rows,cols];gap=periodic_gap(i,ni,j,nj,n)
   if gap>=max(ni,nj) and min(ni*l,nj*l)>RANK:
    tick=time.perf_counter();u,s,vh=torch.linalg.svd(block,full_matrices=False);left=(u[:,:RANK]*s[:RANK]).contiguous();right=vh[:RANK].clone().contiguous();self.blocks.append((rows,cols,left,right));self.bytes+=left.numel()*left.element_size()+right.numel()*right.element_size()
    self.records.append(dict(i=i,ni=ni,j=j,nj=nj,gap=gap,kind='rank12_far',seconds=time.perf_counter()-tick,rank12_relative_F_error=float(s[RANK:].norm()/s.norm()),singular_values=s.tolist()))
   elif max(ni,nj)<=LEAF_NORMAL_CELLS:
    dense=block.clone();self.blocks.append((rows,cols,dense,None));self.bytes+=dense.numel()*dense.element_size();self.records.append(dict(i=i,ni=ni,j=j,nj=nj,gap=gap,kind='dense_near'))
   elif ni==nj:
    hi,hj=ni//2,nj//2
    for a,na in [(i,hi),(i+hi,ni-hi)]:
     for b,nb in [(j,hj),(j+hj,nj-hj)]:visit(a,na,b,nb)
   elif ni>nj:
    h=ni//2;visit(i,h,j,nj);visit(i+h,ni-h,j,nj)
   else:
    h=nj//2;visit(i,ni,j,h);visit(i,ni,j+h,nj-h)
  visit(0,n,0,n);self.build_seconds=time.perf_counter()-start
 def apply(self,q):
  out=torch.zeros_like(q)
  for rows,cols,left,right in self.blocks:out[rows]+=left@q[cols] if right is None else left@(right@q[cols])
  return out
 def assemble_and_cover(self):
  d=self.n*self.l;h=torch.zeros((d,d),dtype=torch.complex128);cover=torch.zeros((d,d),dtype=torch.uint8)
  for rows,cols,left,right in self.blocks:h[rows,cols]=left if right is None else left@right;cover[rows,cols]+=1
  if not bool((cover==1).all()):raise RuntimeError('H-matrix block partition is not a disjoint cover')
  return h

def repaired(action,inverse,q,tolerance=1e-10,max_corrections=3):
 active=(q!=0).any(dim=0);x=torch.zeros_like(q);scale=q[:,active].abs().amax(dim=0);den=(q[:,active]/scale).norm(dim=0);trace=[]
 if not bool(active.any()):return x,dict(success=True,corrections=0,residual_trace=[[0.]*q.shape[1]])
 x[:,active]=inverse(q[:,active])
 for k in range(max_corrections+1):
  r=q-action(x);ratio=torch.zeros(q.shape[1],dtype=torch.float64);ratio[active]=(r[:,active]/scale).norm(dim=0)/den;trace.append(ratio.tolist())
  if bool((ratio<=tolerance).all()):return x,dict(success=True,corrections=k,residual_trace=trace)
  if k<max_corrections:x[:,active]+=inverse(r[:,active])
 return x,dict(success=False,corrections=max_corrections,residual_trace=trace)

def ranks(s):
 norm=s.norm();tail=torch.cat([s.flip(0).square().cumsum(0).flip(0).sqrt(),s.new_zeros(1)])/norm
 return {str(t):int(torch.nonzero(tail<=t)[0,0]) for t in [1e-2,1e-4,1e-6,1e-10]}

@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and set(os.sched_getaffinity(0))=={0}
 out=HERE/'results';out.mkdir(exist_ok=False)
 report=dict(complete=False,pid=os.getpid(),cases=CASES,rank=RANK,leaf_normal_cells=LEAF_NORMAL_CELLS,device='cpu',threads=1,affinity=[0],source_independent=True,physical_RHS_used=False,heldout_used=False,training=False,
  scope='Actual layered A0(k) inverse in normal-physical/full-angle ordering, not heterogeneous 2D parity-Schur hierarchy, global POD, or HSS splitting. Dense inverse/SVD construction is fully paid offline diagnosis, not a deployable low-cost constructor.',
  qualification='Same original-G relative residual1e-10 and max3 repairs as existing angular-band representation, on fixed random/zero/tiny columns. Not a full heterogeneous-RTE solve or continuum theorem.',rows=[],resource_at_start=p.admission())
 def save():
  temp=out/'results.json.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');os.replace(temp,out/'results.json')
 def check(v,msg):
  if not v:raise RuntimeError(msg)
 def emit(event,**kw):print(json.dumps(dict(event=event,**kw)),flush=True)
 try:
  prior=json.loads((HERE/'PRIOR_K_RESULTS.json').read_text())
  for f,h in prior['source_hashes'].items():check(p.digest(p.ROOT/f)==h,'Production source changed '+f)
  for f,h in prior['E_source_hashes'].items():check(p.digest(p.E/f)==h,'Sealed E source changed '+f)
  report['source_hashes']={f.name:p.digest(f) for f in HERE.glob('*.py')};report['plan_sha256']=p.digest(HERE/'GREEN_RANK_PLAN.md')
  check(periodic_gap(0,32,64,32,128)==32 and periodic_gap(0,32,96,32,128)==0 and periodic_gap(0,32,32,32,128)==0,'Periodic geometry unit test')
  for n,k in CASES:
   start=time.perf_counter();rec=dict(n=n,k=k,status='running',resource_admission=p.admission(),cost={});report['rows'].append(rec);report['current_stage']=f'N{n}_k{k}';save();t=time.perf_counter()
   total,scatter,_=p.medium(n,1,10/3,'cpu')
   low=p.PeriodicRTE2D(shape=(n,n),dx=1/n,n_angles=12,anisotropy_g=.65,eta_scale=0.,eta_floor=0.,device='cpu');low.setup(total,scatter,torch.zeros((1,12,n,n),dtype=torch.complex64));high=p.high_precision_copy(low)
   bg,tb,sb=p.background(low);check(bg['axis']==0,'Unexpected normal axis')
   if n==128:check(bg==prior['background_identity'],'N128 background differs from sealed K/E')
   tv=tb[0,0,:,0].double();sv=sb[0,0,:,0].double();l=high.n_angles;d=l*n;K=(high.phase_matrix*high.weights[None]).to(torch.complex128);kn=high.kx.flatten();kt=high.ky.flatten()[k];normal=high.directions[:,0];tangent=high.directions[:,1]
   rec.update(background=bg,directions=high.directions.tolist(),actual_tangent_frequency=float(kt),full_dimension=d,physical_material_sha256=dict(total=p.thash(total),scatter=p.thash(scatter)))
   rec['cost']['physical_setup_seconds']=time.perf_counter()-t;t=time.perf_counter();I=torch.eye(n,dtype=torch.complex128);D=torch.fft.ifft(1j*kn[:,None]*torch.fft.fft(I,dim=0),dim=0);S=torch.diag(sv).to(torch.complex128);ga=torch.empty((l,n,l,n),dtype=torch.complex128)
   for a in range(l):
    for b in range(l):
     block=-K[a,b]*S
     if a==b:block=block+normal[a]*D+torch.diag((tv+1j*kt*tangent[a]).to(torch.complex128))
     ga[a,:,b,:]=block
   ga=ga.reshape(d,d)
   def action(q):
    x=q.reshape(l,n,-1);derivative=torch.fft.ifft(1j*kn[None,:,None]*torch.fft.fft(x,dim=1),dim=1)
    return (normal[:,None,None]*derivative+1j*kt*tangent[:,None,None]*x+tv[None,:,None]*x-sv[None,:,None]*torch.einsum('lm,mnr->lnr',K,x)).reshape_as(q)
   gen=torch.Generator().manual_seed(2026111312);q=torch.complex(torch.randn(d,3,dtype=torch.float64,generator=gen),torch.randn(d,3,dtype=torch.float64,generator=gen));q=q/q.norm(dim=0)
   rec['cost']['dense_original_assembly_seconds']=time.perf_counter()-t;t=time.perf_counter();matrix_bridge=float((ga@q-action(q)).norm()/action(q).norm());check(matrix_bridge<=1e-12,'Dense and original mode action disagree')
   # Independent full original PDE embedding, including actual material subtraction.
   ft=torch.zeros((3,l,n,n),dtype=torch.complex128);ft[...,k]=q.T.reshape(3,l,n);field=torch.fft.ifft(ft,dim=-1,norm='ortho');actual=high.A_op(field)-(high.sigma_t-tb.double())*field+(high.sigma_s-sb.double())*torch.einsum('lm,bmxy->blxy',K,field)
   ft.zero_();ft[...,k]=(ga@q).T.reshape(3,l,n);target=torch.fft.ifft(ft,dim=-1,norm='ortho');embedding_bridge=float((actual-target).norm()/target.norm());check(embedding_bridge<=1e-12,'Dense original A0 embedding disagrees')
   rec.update(matrix_bridge=matrix_bridge,original_A0_embedding_bridge=embedding_bridge,probe_sha256=p.thash(q));rec['cost']['original_operator_bridges_seconds']=time.perf_counter()-t;del ft,field,actual,target
   t=time.perf_counter();lu,piv=torch.linalg.lu_factor(ga);rec['cost']['dense_LU_seconds']=time.perf_counter()-t;t=time.perf_counter();ident=torch.eye(d,dtype=torch.complex128);ba=torch.linalg.lu_solve(lu,piv,ident);rec['cost']['dense_reference_inverse_seconds']=time.perf_counter()-t;t=time.perf_counter();defect=ident-ga@ba;rec['dense_inverse_max_column_residual']=float(defect.norm(dim=0).max());check(rec['dense_inverse_max_column_residual']<=1e-10,'Dense inverse inaccurate');rec['cost']['dense_inverse_qualification_seconds']=time.perf_counter()-t;del defect
   # Space-major ordering retains all12 angles in every normal cell.
   g=ga.reshape(l,n,l,n).permute(1,0,3,2).reshape(d,d).contiguous();b=ba.reshape(l,n,l,n).permute(1,0,3,2).reshape(d,d).contiguous();t=time.perf_counter();hm=NormalHMatrix(b,n,l);h=hm.assemble_and_cover();rec['cost']['rank12_hmatrix_SVD_construction_seconds']=time.perf_counter()-t
   rec.update(dense_inverse_bytes=b.numel()*b.element_size(),H_factor_bytes=hm.bytes,H_far_blocks=sum(v['kind']=='rank12_far' for v in hm.records),H_dense_blocks=sum(v['kind']=='dense_near' for v in hm.records),H_block_records=hm.records)
   t=time.perf_counter();inverse_error=b-h;rd=ident-g@h;rec['global_metrics']=dict(relative_inverse_Frobenius_error=float(inverse_error.norm()/b.norm()),absolute_Frobenius_I_minus_GH=float(rd.norm()),RMS_column_original_residual=float(rd.norm()/math.sqrt(d)),max_column_original_residual=float(rd.norm(dim=0).max()))
   rec['cost']['global_original_defect_seconds']=time.perf_counter()-t;del inverse_error,rd;t=time.perf_counter();rec['representative_blocks']=[]
   for name,j in [('separated_quarters',n//2),('periodic_adjacent_quarters',3*n//4)]:
    ni=n//4;rr=slice(0,ni*l);cc=slice(j*l,(j+ni)*l);z=b[rr,cc];u,ss,vh=torch.linalg.svd(z,full_matrices=False);approx=(u[:,:RANK]*ss[:RANK])@vh[:RANK];error=z-approx;weighted=g[:,rr]@error;exact_action=g[:,rr]@z
    rec['representative_blocks'].append(dict(name=name,i=0,j=j,normal_cells=ni,periodic_gap=periodic_gap(0,ni,j,ni,n),singular_values=ss.tolist(),relative_Frobenius_ranks=ranks(ss),rank12_relative_inverse_Frobenius_error=float(error.norm()/z.norm()),rank12_A_weighted_absolute_Frobenius_error=float(weighted.norm()),rank12_A_weighted_RMS_column_defect=float(weighted.norm()/math.sqrt(ni*l)),rank12_A_weighted_error_relative_to_block_action=float(weighted.norm()/exact_action.norm())))
   rec['cost']['representative_rank_and_A_weighted_diagnostics_seconds']=time.perf_counter()-t;t=time.perf_counter()
   def to_space(q):return q.reshape(l,n,-1).permute(1,0,2).reshape(d,-1)
   def from_space(q):return q.reshape(n,l,-1).permute(1,0,2).reshape(d,-1)
   def hinv(q):return from_space(hm.apply(to_space(q)))
   hb=hinv(q);bridge=float((to_space(hb)-h@to_space(q)).norm()/(h@to_space(q)).norm());check(bridge<=1e-12,'Hierarchical action does not match assembly');rec['hierarchical_action_bridge']=bridge
   mixed=torch.cat([torch.zeros(d,1,dtype=torch.complex128),q,1e-200*q[:,:1]],dim=1);ordinary=~(mixed!=0).any(dim=0);rec['inverse_tests']=[]
   def addtest(name,x,st,seconds):
    ratios=p.relative((mixed-action(x)).T,mixed.T);zero=bool((x[:,ordinary]==0).all());tiny=bool(x[:,-1].abs().max()>0)
    rec['inverse_tests'].append(dict(method=name,seconds=seconds,status=st,original_G_relative=ratios.tolist(),strict_zero=zero,tiny_nonzero=tiny,passed=bool((ratios<=1e-10).all()) and zero and tiny and bool(torch.isfinite(x).all())))
   tt=time.perf_counter();x=torch.linalg.lu_solve(lu,piv,mixed);addtest('dense_LU',x,dict(success=True),time.perf_counter()-tt)
   tt=time.perf_counter();angular=AngularBandMode(high,tv,sv,k);rec['angular_band_factor_seconds']=time.perf_counter()-tt;rec['angular_band_factor_bytes']=angular.bytes;tt=time.perf_counter();x,st=angular.repaired(mixed,tolerance=1e-10,max_corrections=3);addtest('existing_angular_band',x,st,time.perf_counter()-tt)
   tt=time.perf_counter();x,st=repaired(action,hinv,mixed,tolerance=1e-10,max_corrections=3);addtest('fixed_rank12_normal_H',x,st,time.perf_counter()-tt)
   check(all(v['passed'] for v in rec['inverse_tests'][:2]),'Existing inverse baseline failed');rec['normal_H_probe_pass']=rec['inverse_tests'][-1]['passed'];rec['cost']['hierarchy_bridge_and_all_inverse_tests_seconds']=time.perf_counter()-t
   rec.update(status='completed',complete_case_seconds=time.perf_counter()-start,peak_process_RSS_bytes=p.rss());rec['cost']['case_unprofiled_seconds']=rec['complete_case_seconds']-sum(rec['cost'].values());save();emit('case_done',n=n,k=k,passed=rec['normal_H_probe_pass'],inverse_F_error=rec['global_metrics']['relative_inverse_Frobenius_error'],original_defect_RMS=rec['global_metrics']['RMS_column_original_residual'],factor_storage_ratio=hm.bytes/rec['dense_inverse_bytes'],seconds=rec['complete_case_seconds'])
   del low,high,ga,ba,g,b,h,hm,lu,piv,ident,q,x,angular,action,hinv,mixed,hb,D,S,I,ss,u,vh,approx,error,weighted,exact_action,z;gc.collect()
  report['complete']=True;report['all_fixed_rank_cases_pass']=all(r['normal_H_probe_pass'] for r in report['rows']);report['result']='FIXED_NORMAL_GREEN_STRUCTURE_QUALIFIED_ON_PROBES' if report['all_fixed_rank_cases_pass'] else 'FIXED_NORMAL_GREEN_RANK12_NO_GO'
 except BaseException as exc:
  report.update(error=repr(exc),traceback=traceback.format_exc());raise
 finally:
  report.update(total_process_seconds=time.perf_counter()-PROCESS_START,peak_process_RSS_bytes=p.rss(),resource_at_end=p.admission())
  if 'prior' in locals():report.update(production_unchanged=all(p.digest(p.ROOT/f)==h for f,h in prior['source_hashes'].items()),E_unchanged=all(p.digest(p.E/f)==h for f,h in prior['E_source_hashes'].items()))
  save();emit('finished',complete=report['complete'],result=report.get('result'),seconds=report['total_process_seconds'],error=report.get('error'))
if __name__=='__main__':main()
