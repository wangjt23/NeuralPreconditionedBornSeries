"""Local redundant/interface elimination for the frozen nested scalar operator."""
import time
import nested_basis as nb
p=nb.p;torch=nb.torch;old=nb.old;DT=nb.DT

class NestedFactor:
 def __init__(self,basis,V,normal,records=None):
  self.root=basis.root;self.leaf=basis.leaf;self.n=basis.n;self.m=basis.m;self.r=basis.r;self.l=len(normal);self.Q=basis.Q;self.records=[] if records is None else records
  if self.leaf:
   tick=time.perf_counter();G=old.kron(basis.D,torch.diag(normal.to(DT)))+torch.block_diag(*V.unbind())
  else:
   h=basis.a.n;self.a=NestedFactor(basis.a,V[:h],normal,self.records);self.b=NestedFactor(basis.b,V[h:],normal,self.records);tick=time.perf_counter();off=old.kron(basis.B,torch.diag(normal.to(DT)));G=torch.cat([torch.cat([self.a.S,off],1),torch.cat([-off.conj().T,self.b.S],1)],0);del self.a.S,self.b.S
  assembled=time.perf_counter()-tick
  if self.root:
   tick=time.perf_counter();self.lu,self.piv=torch.linalg.lu_factor(G);self.records.append(dict(i=basis.i,n=self.n,kind='root',m=self.m,dimension=self.m*self.l,assembly_seconds=assembled,LU_seconds=time.perf_counter()-tick));return
  self.e=(self.m-self.r)*self.l
  if self.e==0:
   assert self.Q is None;self.S=G;self.records.append(dict(i=basis.i,n=self.n,kind='identity_transfer',m=self.m,r=self.r,assembly_seconds=assembled));return
  tick=time.perf_counter();G4=G.reshape(self.m,self.l,self.m,self.l);left=torch.einsum('ip,pajb->iajb',self.Q.conj().T,G4);trans=torch.einsum('iajb,jk->iakb',left,self.Q).reshape(self.m*self.l,self.m*self.l).contiguous();rotation=time.perf_counter()-tick
  tick=time.perf_counter();self.lu,self.piv=torch.linalg.lu_factor(trans[:self.e,:self.e]);lu_time=time.perf_counter()-tick
  tick=time.perf_counter();self.W=torch.linalg.lu_solve(self.lu,self.piv,trans[:self.e,self.e:]);self.Ase=trans[self.e:,:self.e].clone().contiguous();coupling=time.perf_counter()-tick
  tick=time.perf_counter();self.S=trans[self.e:,self.e:]-self.Ase@self.W;schur=time.perf_counter()-tick
  self.records.append(dict(i=basis.i,n=self.n,kind='local_elimination',m=self.m,r=self.r,e=self.e,s=self.r*self.l,assembly_seconds=assembled,rotation_seconds=rotation,LU_seconds=lu_time,W_solve_and_coupling_copy_seconds=coupling,Schur_update_seconds=schur))
 def upward(self,q,state):
  if self.leaf:b=q
  else:
   h=self.a.n*self.l;b=torch.cat([self.a.upward(q[:h],state),self.b.upward(q[h:],state)],0)
  if self.root or self.e==0:return b
  z=torch.einsum('ij,jac->iac',self.Q.conj().T,b.reshape(self.m,self.l,-1)).reshape(self.m*self.l,-1);ye=torch.linalg.lu_solve(self.lu,self.piv,z[:self.e]);state[self]=ye;return z[self.e:]-self.Ase@ye
 def downward(self,z,state):
  if self.root or self.e==0:x=z
  else:
   xe=state[self]-self.W@z;combined=torch.cat([xe,z],0);x=torch.einsum('ij,jac->iac',self.Q,combined.reshape(self.m,self.l,-1)).reshape(self.m*self.l,-1)
  if self.leaf:return x
  h=self.a.r*self.l;return torch.cat([self.a.downward(x[:h],state),self.b.downward(x[h:],state)],0)
 def apply(self,q):
  state={};b=self.upward(q,state);z=torch.linalg.lu_solve(self.lu,self.piv,b);return self.downward(z,state)
 def tensors(self):
  out=[]
  if self.root:out+=[self.lu,self.piv]
  elif self.e:out+=[self.lu,self.piv,self.W,self.Ase,self.Q]
  if not self.leaf:out+=self.a.tensors()+self.b.tensors()
  return out

def count_application(fac,q):
 counts=dict(upward_nodes=0,downward_nodes=0,upward_leaf_visits=0,downward_leaf_visits=0);restore=[]
 def visit(node):
  up=node.upward;down=node.downward
  def uw(v,state):
   counts['upward_nodes']+=1;counts['upward_leaf_visits']+=int(node.leaf);return up(v,state)
  def dw(v,state):
   counts['downward_nodes']+=1;counts['downward_leaf_visits']+=int(node.leaf);return down(v,state)
  restore.append((node,up,down));node.upward=uw;node.downward=dw
  if not node.leaf:visit(node.a);visit(node.b)
 visit(fac);tick=time.perf_counter()
 try:x=fac.apply(q)
 finally:
  for node,up,down in restore:node.upward=up;node.downward=down
 return x,dict(**counts,seconds=time.perf_counter()-tick)

def unit():
 tick=time.perf_counter();n=16;l=3;kn=2*torch.pi*torch.fft.fftfreq(n,d=1/n,dtype=torch.float64);D=torch.fft.ifft(1j*kn[:,None]*torch.fft.fft(torch.eye(n,dtype=DT),dim=0),dim=0);D=(D-D.conj().T)/2;basis=nb.NestedBasis(D,eta=.02,leaf=2);Dh=basis.assemble();vel=torch.tensor([-.7,.4,.9],dtype=torch.float64);V=torch.stack([(2+.1j*j)*torch.eye(l,dtype=DT)-.2*torch.ones(l,l,dtype=DT)/l for j in range(n)]);fac=NestedFactor(basis,V,vel);G=old.kron(Dh,torch.diag(vel.to(DT)))+torch.block_diag(*V.unbind());gen=torch.Generator().manual_seed(2026111401);q=torch.complex(torch.randn(n*l,3,dtype=torch.float64,generator=gen),torch.randn(n*l,3,dtype=torch.float64,generator=gen));x=fac.apply(q);ref=torch.linalg.solve(G,q);rec=dict(seconds=time.perf_counter()-tick,original_approx_operator_residual=float((G@x-q).norm()/q.norm()),dense_reference_difference=float((x-ref).norm()/ref.norm()),redundant_nodes=sum(v['kind']=='local_elimination' for v in fac.records),factor_bytes=old.size_bytes(fac.tensors()),predicted_bytes=nb.predicted_cost(basis,l)['persistent_tensor_bytes'])
 assert rec['redundant_nodes']>0 and max(rec['original_approx_operator_residual'],rec['dense_reference_difference'])<1e-12 and rec['factor_bytes']==rec['predicted_bytes'],rec
 return rec
