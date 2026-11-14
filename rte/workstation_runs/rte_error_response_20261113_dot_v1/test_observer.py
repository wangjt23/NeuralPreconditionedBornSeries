import prior_permutation_driver as p
import torch,json,time
from pathlib import Path
from scheduled_fgmres import augmented_fgmres as old
from diagnostic_fgmres import augmented_fgmres as new
torch.set_num_threads(1);torch.set_num_interop_threads(1)
t=time.perf_counter();q=torch.arange(1,17,dtype=torch.float64).reshape(1,1,4,4).to(torch.complex128);d=torch.arange(2,18,dtype=torch.float64).reshape_as(q.real)
A=lambda x:d*x;B=lambda x:x;inject=lambda x,r:r/(d+1)
a,sa=old(A,q,B,inject=inject,inject_after=1,tolerance=1e-10,maxiter=50)
events=[]
def observe(e):
 if 'solution' in e:
  r=e['residual'];check=float((q-A(e['solution'])-r).norm()/r.norm());assert check<=1e-12
  events.append(dict(event=e['event'],identity_error=check))
b,sb=new(A,q,B,inject=inject,inject_after=1,observer=observe,tolerance=1e-10,maxiter=50)
diff=float((a-b).norm()/a.norm());assert diff<=1e-12 and sa['classical_calls']==sb['classical_calls'] and sa['injection_accepted']==sb['injection_accepted'] and not sb['failed']
report=dict(passed=True,field_difference=diff,old=sa,new=sb,events=events,seconds=time.perf_counter()-t)
Path(__file__).with_name('OBSERVER_TEST.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
