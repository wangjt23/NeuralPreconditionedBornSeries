import time
START=time.perf_counter()
import os,sys,json,math,gc,traceback,resource
from pathlib import Path
HERE=Path(__file__).resolve().parent
import operator_green as m
p=m.p;torch=m.torch;DT=torch.complex128

class RecomputeFactor:
 def __init__(self,tree,V,normal,records=None):
  self.tree=tree;self.n=tree.n;self.l=len(normal);self.normal=normal;self.records=[] if records is None else records
  if tree.leaf:
   tick=time.perf_counter();G=m.kron(tree.D,torch.diag(normal.to(DT)))+torch.block_diag(*V.unbind());self.lu,self.piv=torch.linalg.lu_factor(G)
   self.records.append(dict(kind='leaf',n=self.n,i=tree.i,seconds=time.perf_counter()-tick));return
  h=tree.a.n;self.a=RecomputeFactor(tree.a,V[:h],normal,self.records);self.b=RecomputeFactor(tree.b,V[h:],normal,self.records)
  tick=time.perf_counter();R=tree.r*self.l;S=torch.empty((R,R),dtype=DT);chunk=self.l
  for j in range(0,R,chunk):
   c=min(chunk,R-j);E=torch.zeros((R,c),dtype=DT);E[j:j+c]=torch.eye(c,dtype=DT)
   mr=self.project(tree.R,self.b.apply(self.lift(tree.R,E,-1)))
   S[:,j:j+c]=E-self.project(tree.L,self.a.apply(self.lift(tree.L,mr,1)))
  self.lu,self.piv=torch.linalg.lu_factor(S)
  self.records.append(dict(kind='streamed_Schur',n=self.n,i=tree.i,rank=tree.r,Schur_dimension=R,column_chunk=chunk,own_seconds=time.perf_counter()-tick,Schur_norm=float(S.norm())))
 def project(self,U,q):return torch.einsum('ij,jac->iac',U.conj().T,q.reshape(U.shape[0],self.l,-1)).reshape(U.shape[1]*self.l,-1)
 def lift(self,U,c,sign=1):return (sign*torch.einsum('ir,rac->iac',U,c.reshape(U.shape[1],self.l,-1))*self.normal[None,:,None]).reshape(U.shape[0]*self.l,-1)
 def apply(self,q):
  if self.tree.leaf:return torch.linalg.lu_solve(self.lu,self.piv,q)
  h=self.a.n*self.l;yR=self.b.apply(q[h:]);zL=self.a.apply(q[:h]-self.lift(self.tree.L,self.project(self.tree.R,yR)))
  a=torch.linalg.lu_solve(self.lu,self.piv,self.project(self.tree.L,zL));tR=self.b.apply(self.lift(self.tree.R,a,-1))
  return torch.cat([zL+self.a.apply(self.lift(self.tree.L,self.project(self.tree.R,tR))),yR-tR],dim=0)
 def tensors(self):
  result=[self.lu,self.piv,self.normal]
  if not self.tree.leaf:result+=self.a.tensors()+self.b.tensors()
  return result

def current_rss():
 for line in Path('/proc/self/status').read_text().splitlines():
  if line.startswith('VmRSS:'):return int(line.split()[1])*1024
 return None

def structure(fac,tree,K,normal):
 t=time.perf_counter();l=len(normal);R=tree.r*l;I=torch.eye(R,dtype=DT);S=I-fac.ML@fac.MR;F=torch.fft.fft(torch.eye(l,dtype=DT),dim=0,norm='ortho');mask=torch.eye(l,dtype=torch.bool)
 for a in range(l):mask[a,(a-1)%l]=True;mask[a,(a+1)%l]=True
 def bandness(M,n,r):
  a=torch.einsum('ai,piqj,jb->paqb',F,M.reshape(n,l,r,l),F.conj().T);error=a*(~mask)[None,:,None,:]
  return dict(relative_F_off_nearest_angular_band=float(error.norm()/a.norm()),max_absolute_off_band=float(error.abs().max()))
 diff=(normal[:,None]-normal[None,:]).abs();diff.diagonal().fill_(float('inf'));Ks=float(K.abs().min());comm=torch.diag(normal.to(DT))@K-K@torch.diag(normal.to(DT))
 rec=dict(normal_values=normal.tolist(),distinct_normal_values=int(torch.unique(normal).numel()),minimum_normal_separation=float(diff.min()),all_phase_entries_nonzero=bool((K!=0).all()),minimum_absolute_phase_entry=Ks,relative_velocity_phase_commutator=float(comm.norm()/(normal.norm()*K.norm())),root_S_dimension=R,root_S_F_norm=float(S.norm()),identity_Neumann_2norm_lower_bound=(float(S.norm())-math.sqrt(R))/math.sqrt(R),root_WL_angular_fourier=bandness(fac.WL,tree.a.n,tree.r),root_WR_angular_fourier=bandness(fac.WR,tree.b.n,tree.r),root_S_minus_I_angular_fourier=bandness(S-I,tree.r,tree.r),quarter_inherited_basis=[])
 pieces=[(tree.L[:32],tree.a.L),(tree.L[32:],tree.a.R),(tree.R[:32],tree.b.L),(tree.R[32:],tree.b.R)]
 for i,(anc,sib) in enumerate(pieces):
  row=dict(quarter=i,inherited_shape=list(anc.shape),sibling_shape=list(sib.shape))
  for name,Z in [('inherited',anc),('combined',torch.cat([anc,sib],dim=1))]:
   vals=torch.linalg.svdvals(Z);tol=torch.finfo(torch.float64).eps*max(Z.shape)*float(vals[0]);row[name]=dict(singular_values=vals.tolist(),standard_FP_rank_threshold=tol,numerical_rank=int((vals>tol).sum()),smallest_singular_value=float(vals[-1]))
  rec['quarter_inherited_basis'].append(row)
 rec['seconds']=time.perf_counter()-t;return rec

def count_apply(fac,q):
 counts=dict(node_visits=0,leaf_visits=0,leaf_rhs_columns=0);restore=[]
 def instrument(node):
  old=node.apply
  def apply(v):
   counts['node_visits']+=1
   if node.tree.leaf:counts['leaf_visits']+=1;counts['leaf_rhs_columns']+=v.shape[1]
   return old(v)
  restore.append((node,old));node.apply=apply
  if not node.tree.leaf:instrument(node.a);instrument(node.b)
 instrument(fac);t=time.perf_counter()
 try:x=fac.apply(q)
 finally:
  for node,old in restore:node.apply=old
 return dict(**counts,seconds=time.perf_counter()-t),x

def unit():
 t=time.perf_counter();g=torch.Generator().manual_seed(2026111401)
 def rnd(*shape):return torch.complex(torch.randn(*shape,dtype=torch.float64,generator=g),torch.randn(*shape,dtype=torch.float64,generator=g))
 a=rnd(4,4);b=rnd(4,4);u=rnd(4,1);v=rnd(4,1);D=torch.cat([torch.cat([a-a.conj().T,u@v.conj().T],1),torch.cat([-v@u.conj().T,b-b.conj().T],1)],0)
 vel=torch.tensor([-.7,.4,.9],dtype=torch.float64);V=torch.stack([torch.eye(3,dtype=DT)*(2+.1j*j) for j in range(8)]);tree=m.ScalarTree(D,1e-12,leaf=2);cached=m.GreenFactor(tree,V,vel);new=RecomputeFactor(tree,V,vel);q=rnd(24,3);G=m.kron(D,torch.diag(vel.to(DT)))+torch.block_diag(*V.unbind());ref=torch.linalg.solve(G,q);x=new.apply(q)
 rec=dict(seconds=time.perf_counter()-t,original_residual=float((G@x-q).norm()/q.norm()),vs_dense=float((x-ref).norm()/ref.norm()),vs_cached=float((x-cached.apply(q)).norm()/ref.norm()))
 assert max(rec[k] for k in ['original_residual','vs_dense','vs_cached'])<1e-12,rec
 return rec

@torch.no_grad()
def main():
 method=sys.argv[1];assert method in ['cached','recompute','angular'];torch.set_num_threads(1);torch.set_num_interop_threads(1);assert set(os.sched_getaffinity(0))=={0} and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 out=HERE/method;out.mkdir(exist_ok=False);report=dict(complete=False,pid=os.getpid(),method=method,n=128,k=0,source_independent=True,original_residual_tolerance=1e-10,max_repairs=3,threads=1,cpu_affinity=[0],CUDA_disabled=True,no_physical_source=True,no_dense_inverse=True,cost={},resource_at_start=p.admission(),rss={})
 def save():
  temp=out/'results.json.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');os.replace(temp,out/'results.json')
 try:
  prior=json.loads((HERE/'PRIOR_K_RESULTS.json').read_text());old=json.loads((HERE/'PRIOR_OPERATOR_RESULTS.json').read_text())
  for f,h in prior['source_hashes'].items():assert p.digest(p.ROOT/f)==h,f
  for f,h in prior['E_source_hashes'].items():assert p.digest(p.E/f)==h,f
  assert p.digest(HERE/'operator_green.py')==old['source_hashes']['run_operator_green.py'];report.update(source_hashes={v.name:p.digest(v) for v in HERE.glob('*.py')},plan_sha256=p.digest(HERE/'RECOMPUTE_PLAN.md'))
  if method=='cached':report['unit']=unit()
  else:assert json.loads((HERE/'cached/results.json').read_text())['complete']
  report['rss']['before_physical']=current_rss();tick=time.perf_counter();n=128;l=12;dim=n*l;total,scatter,_=p.medium(n,1,10/3,'cpu');low=p.PeriodicRTE2D(shape=(n,n),dx=1/n,n_angles=l,anisotropy_g=.65,eta_scale=0.,eta_floor=0.,device='cpu');low.setup(total,scatter,torch.zeros((1,l,n,n),dtype=torch.complex64));high=p.high_precision_copy(low);bg,tb,sb=p.background(low);assert bg==prior['background_identity'];tv=tb[0,0,:,0].double();sv=sb[0,0,:,0].double();K=(high.phase_matrix*high.weights[None]).to(DT);kn=high.kx.flatten();normal=high.directions[:,0];tangent=high.directions[:,1];kt=high.ky.flatten()[0];V=tv[:,None,None]*torch.eye(l,dtype=DT)-sv[:,None,None]*K[None]+1j*kt*torch.diag(tangent.to(DT))[None]
  alpha=float((tv-sv*torch.linalg.eigvalsh((K+K.conj().T)/2).max()).min());assert alpha==old['coercivity_alpha'];report.update(background=bg,alpha=alpha);report['cost']['physical_setup_seconds']=time.perf_counter()-tick;report['rss']['after_physical']=current_rss()
  def original(q):
   x=q.reshape(n,l,-1);der=torch.fft.ifft(1j*kn[:,None,None]*torch.fft.fft(x,dim=0),dim=0)
   return (der*normal[None,:,None]+torch.einsum('nij,njr->nir',V,x)).reshape_as(q)
  if method!='angular':
   tick=time.perf_counter();D=torch.fft.ifft(1j*kn[:,None]*torch.fft.fft(torch.eye(n,dtype=DT),dim=0),dim=0);tree=m.ScalarTree((D-D.conj().T)/2,old['block_absolute_F_budget'],m.LEAF);Dh=tree.assemble();delta=math.sqrt(float((D-Dh).abs().sum(0).max())*float((D-Dh).abs().sum(1).max()));report.update(derivative_sha256=p.thash(Dh),tree_ranks=[v['rank'] for v in tree.records if v['kind']=='low_rank'],model_q_bound=float(normal.abs().max())*delta/alpha);assert report['tree_ranks']==[v['rank'] for v in old['derivative']['records'] if v['kind']=='low_rank'];assert abs(report['model_q_bound']-old['derivative']['right_defect_exact_arithmetic_upper_bound'])<1e-12;del D;report['cost']['derivative_tree_and_bound_seconds']=time.perf_counter()-tick
   tick=time.perf_counter();fac=m.GreenFactor(tree,V,normal) if method=='cached' else RecomputeFactor(tree,V,normal);report['cost']['factor_seconds']=time.perf_counter()-tick;raw=fac.apply;report.update(factor_bytes=m.size_bytes(fac.tensors()+tree.tensors()),factor_records=fac.records)
   def approximate(q):
    x=q.reshape(n,l,-1);der=torch.einsum('ij,jlr->ilr',Dh,x)
    return (der*normal[None,:,None]+torch.einsum('nij,njr->nir',V,x)).reshape_as(q)
  else:
   tick=time.perf_counter();fac=m.AngularBandMode(high,tv,sv,0);report['cost']['factor_seconds']=time.perf_counter()-tick;report['factor_bytes']=fac.bytes
   def raw(q):return fac.raw(q.reshape(n,l,-1).permute(1,0,2).reshape(dim,-1)).reshape(l,n,-1).permute(1,0,2).reshape(dim,-1)
  gc.collect();report['rss']['after_factor']=current_rss();report['factor_phase_peak_RSS_bytes']=p.rss();save()
  gen=torch.Generator().manual_seed(2026111312);q=torch.complex(torch.randn(dim,3,dtype=torch.float64,generator=gen),torch.randn(dim,3,dtype=torch.float64,generator=gen));q=q/q.norm(dim=0);mixed=torch.cat([torch.zeros(dim,1,dtype=DT),q,1e-200*q[:,:1]],1);report['probe_sha256']=p.thash(q);assert report['probe_sha256']==old['rows'][0]['probe_sha256'];report['applications']=[]
  reference=None
  if method!='cached':
   tick=time.perf_counter();reference=torch.load(HERE/'cached/accepted_reference.pt',weights_only=True);report['cost']['reference_read_seconds']=time.perf_counter()-tick
  for rep in range(3):
   tick=time.perf_counter();x,status=m.repaired(original,raw,mixed,tolerance=1e-10,max_corrections=3);seconds=time.perf_counter()-tick;tick=time.perf_counter();rel=p.relative((mixed-original(x)).T,mixed.T);zero=bool((x[:,0]==0).all());tiny=bool(x[:,-1].abs().max()>0)
   if reference is None:reference=x.clone()
   diff=p.relative((x-reference).T,reference.T);accepted=bool((rel<=1e-10).all()) and bool((diff<=1e-9).all()) and zero and tiny and bool(torch.isfinite(x).all())
   report['applications'].append(dict(repeat=rep,checked_inverse_seconds=seconds,independent_check_seconds=time.perf_counter()-tick,status=status,original_relative=rel.tolist(),relative_to_cached=diff.tolist(),zero=zero,tiny=tiny,passed=accepted));assert accepted,report['applications'][-1]
  if method=='cached':
   tick=time.perf_counter();torch.save(reference,HERE/'cached/accepted_reference.pt');report['cost']['reference_save_seconds']=time.perf_counter()-tick
  report['rss']['after_applications']=current_rss();report['application_phase_peak_RSS_bytes']=p.rss()
  if method!='angular':
   tick=time.perf_counter();z=raw(q);report['raw_approximate_relative']=p.relative((q-approximate(z)).T,q.T).tolist();report['raw_original_relative']=p.relative((q-original(z)).T,q.T).tolist();report['cost']['raw_defect_diagnostic_seconds']=time.perf_counter()-tick
   counts,z2=count_apply(fac,q);report['call_count']=counts;report['counted_vs_uninstrumented_relative']=float((z2-z).norm()/z.norm());assert report['counted_vs_uninstrumented_relative']<1e-12
   expected=16 if method=='cached' else 256;assert counts['leaf_visits']==expected
   if method=='cached':report['structure']=structure(fac,tree,K,normal)
  report['complete']=True
 except BaseException as exc:report.update(error=repr(exc),traceback=traceback.format_exc());raise
 finally:
  report.update(process_seconds=time.perf_counter()-START,total_peak_RSS_bytes=p.rss(),current_RSS_end=current_rss(),resource_at_end=p.admission())
  if 'prior' in locals():report.update(production_unchanged=all(p.digest(p.ROOT/f)==h for f,h in prior['source_hashes'].items()),E_unchanged=all(p.digest(p.E/f)==h for f,h in prior['E_source_hashes'].items()))
  save();print(json.dumps(dict(method=method,complete=report['complete'],factor_bytes=report.get('factor_bytes'),factor_seconds=report['cost'].get('factor_seconds'),application_seconds=[r['checked_inverse_seconds'] for r in report.get('applications',[])],leaf_visits=report.get('call_count',{}).get('leaf_visits'),process_seconds=report['process_seconds'],error=report.get('error'))),flush=True)
if __name__=='__main__':main()
