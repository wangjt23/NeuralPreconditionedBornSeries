"""Local redundant/interface elimination for the frozen nested scalar operator."""
import time
import nested_basis as nb
p=nb.p;torch=nb.torch;old=nb.old;DT=nb.DT

def rotate(G,Q,m,l):
 # Both explicit packs and both GEMMs are inside the caller's rotation timer.
 left=(Q.conj().T@G.reshape(m,l*m*l)).reshape(m,l,m,l)
 packed=left.permute(0,1,3,2).contiguous().reshape(m*l*l,m)
 right=packed@Q
 return right.reshape(m,l,l,m).permute(0,1,3,2).contiguous().reshape(m*l,m*l)

class PackedFactor:
 def __init__(self,basis,V,normal,records=None):
  self.root=basis.root;self.leaf=basis.leaf;self.n=basis.n;self.m=basis.m;self.r=basis.r;self.l=len(normal);self.Q=basis.Q;self.records=[] if records is None else records
  if self.leaf:
   tick=time.perf_counter();G=old.kron(basis.D,torch.diag(normal.to(DT)))+torch.block_diag(*V.unbind())
  else:
   h=basis.a.n;self.a=PackedFactor(basis.a,V[:h],normal,self.records);self.b=PackedFactor(basis.b,V[h:],normal,self.records);tick=time.perf_counter();off=old.kron(basis.B,torch.diag(normal.to(DT)));G=torch.cat([torch.cat([self.a.S,off],1),torch.cat([-off.conj().T,self.b.S],1)],0);del self.a.S,self.b.S
  assembled=time.perf_counter()-tick
  if self.root:
   tick=time.perf_counter();self.lu,self.piv=torch.linalg.lu_factor(G);self.records.append(dict(i=basis.i,n=self.n,kind='root',m=self.m,dimension=self.m*self.l,assembly_seconds=assembled,LU_seconds=time.perf_counter()-tick));return
  self.e=(self.m-self.r)*self.l
  if self.e==0:
   assert self.Q is None;self.S=G;self.records.append(dict(i=basis.i,n=self.n,kind='identity_transfer',m=self.m,r=self.r,assembly_seconds=assembled));return
  tick=time.perf_counter();trans=rotate(G,self.Q,self.m,self.l);rotation=time.perf_counter()-tick
  tick=time.perf_counter();self.lu,self.piv=torch.linalg.lu_factor(trans[:self.e,:self.e]);lu_time=time.perf_counter()-tick
  tick=time.perf_counter();self.W=torch.linalg.lu_solve(self.lu,self.piv,trans[:self.e,self.e:]);self.Ase=trans[self.e:,:self.e].clone().contiguous();coupling=time.perf_counter()-tick
  tick=time.perf_counter();self.S=trans[self.e:,self.e:]-self.Ase@self.W;schur=time.perf_counter()-tick
  self.records.append(dict(i=basis.i,n=self.n,kind='local_elimination',m=self.m,r=self.r,e=self.e,s=self.r*self.l,assembly_seconds=assembled,rotation_seconds=rotation,LU_seconds=lu_time,W_solve_and_coupling_copy_seconds=coupling,Schur_update_seconds=schur))
 def upward(self,q,state):
  if self.leaf:b=q
  else:
   h=self.a.n*self.l;b=torch.cat([self.a.upward(q[:h],state),self.b.upward(q[h:],state)],0)
  if self.root or self.e==0:return b
  z=(self.Q.conj().T@b.reshape(self.m,-1)).reshape(self.m*self.l,-1);ye=torch.linalg.lu_solve(self.lu,self.piv,z[:self.e]);state[self]=ye;return z[self.e:]-self.Ase@ye
 def downward(self,z,state):
  if self.root or self.e==0:x=z
  else:
   xe=state[self]-self.W@z;combined=torch.cat([xe,z],0);x=(self.Q@combined.reshape(self.m,-1)).reshape(self.m*self.l,-1)
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
