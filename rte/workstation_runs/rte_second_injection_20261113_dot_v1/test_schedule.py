"""Focused schedule tests; no RTE material solves or training."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path('/data/wjt/NeuralPreconditionedBornSeries');sys.path.insert(0,str(ROOT))
import torch
from rte.augmented_fgmres import augmented_fgmres as old
from scheduled_fgmres import augmented_fgmres as new

def main():
 torch.set_num_threads(1);torch.manual_seed(20261113)
 n=9;raw=torch.randn(n,n,dtype=torch.float64);M=(raw.T@raw+torch.diag(torch.linspace(.2,2,n))).to(torch.complex128)
 rhs=torch.randn(1,1,1,n,dtype=torch.float64).to(torch.complex128)
 A=lambda x:(M@x.flatten()).reshape_as(x);B=lambda x:x.clone()
 results=[]
 def compare(label,inject=None,schedule=(1,),after=1):
  u,a=old(A,rhs,B,inject=inject,inject_after=after,tolerance=1e-11)
  v,b=new(A,rhs,B,inject=inject,inject_after=after,inject_schedule=schedule,tolerance=1e-11)
  rel=float((u-v).norm()/u.norm())
  keys=['iterations','classical_calls','injection_calls','injection_accepted','injection_rejected','failed']
  assert rel<=1e-12 and all(a[k]==b[k] for k in keys),(label,rel,a,b)
  results.append(dict(test=label,passed=True,field_difference=rel,counts={k:b[k] for k in keys}))
 compare('none')
 compare('one_shot_zero_rejection',lambda x,r:torch.zeros_like(r))
 compare('one_shot_nonzero',lambda x,r:r/M.diagonal().reshape_as(r))
 compare('legacy_prefix_four',lambda x,r:r/M.diagonal().reshape_as(r),schedule=(4,),after=4)
 events=[];u,s=new(A,rhs,B,inject=lambda x,r:torch.zeros_like(r),inject_schedule=(1,2),observer=events.append,tolerance=1e-11)
 assert s['injection_calls']==2 and s['injection_rejected']==2 and not s['failed'],s
 assert [e['classical_calls'] for e in events if e['event']=='injection_rejected']==[1,2]
 results.append(dict(test='two_rejections_attempt_cap',passed=True,counts=s))
 events=[];u,s=new(A,rhs,B,inject=lambda x,r:r/M.diagonal().reshape_as(r),inject_schedule=(1,2),observer=events.append,tolerance=1e-11)
 assert s['injection_calls']==2 and s['injection_accepted']==2 and not s['failed'],s
 assert [e['classical_calls'] for e in events if e['event']=='injection_accepted']==[1,2]
 results.append(dict(test='two_accepted_fixed_thresholds',passed=True,counts=s))
 calls=[];u,s=new(lambda x:x,rhs,B,inject=lambda x,r:calls.append(1) or r,inject_schedule=(1,2),tolerance=1e-11)
 assert not calls and s['injection_calls']==0 and not s['failed']
 results.append(dict(test='no_injection_after_convergence',passed=True))
 z,s=new(A,torch.zeros_like(rhs),B,inject=lambda x,r:r,inject_schedule=(1,2));assert not z.any() and s['iterations']==0
 results.append(dict(test='zero_rhs',passed=True))
 for bad in [(0,),(2,1),(1,1),(1.,2),(True,2)]:
  try:new(A,rhs,B,inject_schedule=bad)
  except ValueError:pass
  else:raise AssertionError(bad)
 results.append(dict(test='invalid_schedules_rejected',passed=True))
 expected='21ddfb628d35738ca88844d6c8587b0b8809956ed000bc7a859237d12b6bc742'
 assert hashlib.sha256((ROOT/'rte/augmented_fgmres.py').read_bytes()).hexdigest()==expected
 report=dict(passed=True,torch=torch.__version__,tests=results)
 Path(__file__).with_name('UNIT_TESTS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
