"""Differentiable real-stacked FGMRES windows, retaining restart context.
Prefix is stop-gradient. Within-window Arnoldi, least-squares and restarts
are differentiable. No forced residual restart at arbitrary window start.
"""
import torch,time

def real(x):return torch.cat([x.real,x.imag],1)
def complex(x):
 a,b=x.chunk(2,1);return torch.complex(a,b)
def start_cycle(A,b,x,iteration=0):
 r=b-A(x);beta=r.norm();return dict(xbase=x,x=x,V=[r/beta.clamp_min(1e-30)],Z=[],columns=[],beta=beta,iteration=iteration,cycle_iteration=0,breakdown=False)
def detach_state(s):
 return {k:([v.detach() for v in x] if isinstance(x,list) else x.detach() if torch.is_tensor(x) else x) for k,x in s.items()}
def step(A,M,b,s,restart=20):
 if s['cycle_iteration']==restart:s=start_cycle(A,b,s['x'],s['iteration'])
 j=s['cycle_iteration'];v=s['V'][j];z=M(v);w=A(z);Az=w;hs=[]
 for vi in s['V']:
  h=(w*vi).sum();hs.append(h);w=w-h*vi
 hn=w.norm();col=torch.stack(hs+[hn]);cols=[torch.nn.functional.pad(c,(0,j+2-len(c))) for c in s['columns']]+[col];H=torch.stack(cols,1)
 rhs=torch.cat([s['beta'].reshape(1),b.new_zeros(j+1)]);y=torch.linalg.lstsq(H,rhs).solution;zs=s['Z']+[z];x=s['xbase']+sum(zi*yi for zi,yi in zip(zs,y));r=b-A(x);relative=r.norm()/b.norm().clamp_min(1e-30)
 broken=float(hn.detach())<=1e-14;vs=s['V']+([] if broken else [w/hn]);new=dict(xbase=s['xbase'],x=x,V=vs,Z=zs,columns=[*s['columns'],col],beta=s['beta'],iteration=s['iteration']+1,cycle_iteration=j+1,breakdown=broken)
 info=dict(v=v,z=z,Az=Az,residual=r,relative=relative,predicted=(rhs-H@y).norm()/b.norm().clamp_min(1e-30),iteration=new['iteration'],cycle_iteration=j+1)
 return new,info

def prefix(A,M,b,count,restart=20):
 with torch.no_grad():
  s=start_cycle(A,b,torch.zeros_like(b));history=[]
  for _ in range(count):
   s,t=step(A,M,b,s,restart);history.append(float(t['relative']))
   if s['breakdown']:raise RuntimeError('prefix breakdown at '+str(s['iteration']))
 return detach_state(s),history

def window(A,M,b,state,length=8,restart=20):
 s=detach_state(state);rows=[]
 for _ in range(length):
  s,t=step(A,M,b,s,restart);rows.append(t)
  if s['breakdown']:raise RuntimeError('window breakdown at '+str(s['iteration']))
 return s,rows

def k_loss(rows,start_residual,delta=1e-12):
 den=start_residual.square().sum().clamp_min(1e-30)
 return torch.stack([torch.log(delta+rows[j-1]['residual'].square().sum()/den) for j in (2,4,8)]).mean()
def o_loss(A,M,rows):
 return torch.stack([(v-A(M(v))).square().sum()/v.square().sum().clamp_min(1e-30) for v in [t['v'].detach() for t in rows]]).mean()
