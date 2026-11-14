"""Skew-preserving nested scalar interfaces with an explicit inherited-error ledger."""
import math,time
import operator_green as old
p=old.p;torch=old.torch;DT=torch.complex128

class NestedBasis:
 def __init__(self,D,eta,leaf=8,i=0,n=None,depth=0,root=True,records=None):
  self.i=i;self.n=len(D) if n is None else n;self.depth=depth;self.root=root;self.leaf=self.n<=leaf;self.records=[] if records is None else records;self.Q=None
  rows=torch.arange(i,i+self.n);outside=torch.cat([torch.arange(0,i),torch.arange(i+self.n,len(D))])
  if self.leaf:
   self.D=D[i:i+self.n,i:i+self.n].clone();base=torch.eye(self.n,dtype=DT);self.m=self.n;child_e2=0.
  else:
   h=self.n//2;self.a=NestedBasis(D,eta,leaf,i,h,depth+1,False,self.records);self.b=NestedBasis(D,eta,leaf,i+h,self.n-h,depth+1,False,self.records);self.m=self.a.r+self.b.r
   actual=D[i:i+h,i+h:i+self.n];self.B=self.a.U.conj().T@actual@self.b.U
   self.sibling_F_error=float((actual-self.a.U@self.B@self.b.U.conj().T).norm());base=torch.block_diag(self.a.U,self.b.U);child_e2=self.a.e**2+self.b.e**2
  if root:
   self.r=None;self.U=None;self.e=None
   self.records.append(dict(i=i,n=self.n,depth=depth,kind='root',input_rank=self.m,sibling_F_error=self.sibling_F_error));return
  tick=time.perf_counter();external=D[rows[:,None],outside[None,:]];compressed=base.conj().T@external;u,s,vh=torch.linalg.svd(compressed,full_matrices=False);tail=torch.cat([s.flip(0).square().cumsum(0).flip(0).sqrt(),s.new_zeros(1)]);self.r=int(torch.nonzero(tail<=eta)[0,0]);assert self.r>0
  if self.r==self.m:self.U=base
  else:
   Z=u[:,:self.r];self.U=base@Z;self.Q=torch.cat([u[:,self.r:],Z],dim=1).contiguous()
  self.e=float((external-self.U@(self.U.conj().T@external)).norm());ortho=float((self.U.conj().T@self.U-torch.eye(self.r,dtype=DT)).norm());recurrence=math.sqrt(child_e2+eta**2)
  assert ortho<1e-12,(i,self.n,ortho)
  assert self.e<=recurrence+1e-11,(i,self.n,self.e,recurrence)
  rec=dict(i=i,n=self.n,depth=depth,kind='leaf' if self.leaf else 'internal',input_rank=self.m,interface_rank=self.r,redundant_rank=self.m-self.r,local_discarded_F=float(tail[self.r]),eta=eta,external_projection_F=self.e,recurrence_upper_bound=recurrence,analytic_subtree_bound=eta*math.sqrt(2*(self.n//leaf)-1),global_basis_orthogonality_F=ortho,identity_transfer=self.Q is None,svd_seconds=time.perf_counter()-tick,singular_values=s.tolist())
  if not self.leaf:rec.update(sibling_F_error=self.sibling_F_error,sibling_e_sum_bound=self.a.e+self.b.e)
  self.records.append(rec)
 def assemble(self):
  if self.leaf:return self.D.clone()
  h=self.a.n;out=torch.empty((self.n,self.n),dtype=DT);out[:h,:h]=self.a.assemble();out[h:,h:]=self.b.assemble();out[:h,h:]=self.a.U@self.B@self.b.U.conj().T;out[h:,:h]=-out[:h,h:].conj().T;return out

def predicted_cost(tree,l=12):
 result=dict(LU_complex_entries=0,coupling_complex_entries=0,Q_complex_entries=0,pivot_int32_entries=0,setup_complex_MAC_proxy=0.,application_complex_MAC_proxy=0.,local_fronts=[],root_dimension=tree.m*l)
 def node(t):
  if not t.leaf:node(t.a);node(t.b)
  if t.root:
   d=t.m*l;result['LU_complex_entries']+=d*d;result['pivot_int32_entries']+=d;result['setup_complex_MAC_proxy']+=d**3/3;result['application_complex_MAC_proxy']+=d*d;result['local_fronts'].append(dict(n=t.n,m=t.m,root=True,dense_dimension=d));return
  e=(t.m-t.r)*l;s=t.r*l
  result['local_fronts'].append(dict(n=t.n,m=t.m,r=t.r,e=e,s=s,identity=t.Q is None))
  if e:
   result['LU_complex_entries']+=e*e;result['coupling_complex_entries']+=2*e*s;result['pivot_int32_entries']+=e;result['Q_complex_entries']+=t.m*t.m
   result['setup_complex_MAC_proxy']+=2*t.m**3*l*l+e**3/3+e*e*s+s*s*e
   result['application_complex_MAC_proxy']+=2*t.m*t.m*l+e*e+2*e*s
 node(tree)
 result['persistent_tensor_bytes']=16*sum(result[k] for k in ['LU_complex_entries','coupling_complex_entries','Q_complex_entries'])+4*result['pivot_int32_entries']
 n=tree.n
 # chain: l/3 LU + (l-1) W solve + (l-1) update;
 # corner-response open solve: (2*l+4*(l-1)); corner LU8/3.
 result['angular_baseline_setup_complex_MAC_proxy']=(l/3+2*(l-1)+2*l+4*(l-1)+8/3)*n**3
 result['angular_baseline_application_complex_MAC_proxy']=(l+2*(l-1)+4+2*l)*n*n
 return result

def construct(qdesign=1e-3,leaf=8):
 times={};t=time.perf_counter();n=128;l=12;total,scatter,_=p.medium(n,1,10/3,'cpu');low=p.PeriodicRTE2D(shape=(n,n),dx=1/n,n_angles=l,anisotropy_g=.65,eta_scale=0.,eta_floor=0.,device='cpu');low.setup(total,scatter,torch.zeros((1,l,n,n),dtype=torch.complex64));high=p.high_precision_copy(low);bg,tb,sb=p.background(low);tv=tb[0,0,:,0].double();sv=sb[0,0,:,0].double();K=(high.phase_matrix*high.weights[None]).to(DT);kn=high.kx.flatten();normal=high.directions[:,0];tangent=high.directions[:,1];alpha=float((tv-sv*torch.linalg.eigvalsh((K+K.conj().T)/2).max()).min());v=float(normal.abs().max());times['physical_setup_seconds']=time.perf_counter()-t
 t=time.perf_counter();ideal=2*math.pi*torch.fft.fftfreq(n,d=1/n,dtype=torch.float64);freq=float((kn-ideal).abs().max());C=2*sum(math.sqrt(2*m-1) for m in [8,4,2,1]);remaining=alpha*qdesign/v-freq;assert remaining>0;eta=remaining/C
 ident=torch.eye(n,dtype=DT);D0=torch.fft.ifft(1j*ideal[:,None]*torch.fft.fft(ident,dim=0),dim=0);D0bar=(D0-D0.conj().T)/2;skewdrop=float((D0-D0bar).norm());tree=NestedBasis(D0bar,eta,leaf);Dh=tree.assemble();times['scalar_nested_construction_seconds']=time.perf_counter()-t
 t=time.perf_counter();bydepth={}
 for r in tree.records:
  if 'sibling_F_error' in r:bydepth[r['depth']]=max(bydepth.get(r['depth'],0.),r['sibling_F_error'])
 hierarchy=sum(bydepth.values())+skewdrop;err=D0-Dh;norm1inf=math.sqrt(float(err.abs().sum(0).max())*float(err.abs().sum(1).max()));nested=min(hierarchy,norm1inf);qbound=v*(freq+nested)/alpha;skew=float((Dh+Dh.conj().T).norm());assert skew<1e-12
 cost=predicted_cost(tree,l);times['bound_and_cost_check_seconds']=time.perf_counter()-t
 rec=dict(n=n,l=l,background=bg,alpha=alpha,velocity_norm=v,delta_frequency=freq,desired_q=qdesign,remaining_nested_derivative_budget=remaining,allocation_constant=C,eta=eta,ideal_skew_projection_F=skewdrop,skew_defect_Dhat=skew,per_depth_sibling_F_bound=bydepth,hierarchy_nested_norm_bound=hierarchy,row_column_nested_norm_bound=norm1inf,model_original_right_defect_bound=qbound,model_q4=qbound**4,analytical_nested_budget=C*eta,rank_records=tree.records,predicted_cost=cost,Dhat_sha256=p.thash(Dh),normal_frequency_sha256=p.thash(kn),ideal_frequency_sha256=p.thash(ideal),times=times)
 ctx=dict(low=low,high=high,tb=tb,sb=sb,tv=tv,sv=sv,K=K,kn=kn,normal=normal,tangent=tangent,Dhat=Dh,tree=tree)
 return ctx,rec
