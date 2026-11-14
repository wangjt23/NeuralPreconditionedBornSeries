"""Exact rational rank of the actual stored angular coefficients; CPU only."""
import os,sys,json,hashlib,time
from fractions import Fraction
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=Path('/data/wjt/NeuralPreconditionedBornSeries');sys.path.insert(0,str(ROOT))
from rte.models import PeriodicRTE2D

def det(A):
 a=[[Fraction(float(x)) for x in row] for row in A.tolist()];n=len(a);value=Fraction(1);swaps=0
 for j in range(n):
  pivot=next((i for i in range(j,n) if a[i][j]),None)
  if pivot is None:return Fraction(0),swaps
  if pivot!=j:a[j],a[pivot]=a[pivot],a[j];value=-value;swaps+=1
  p=a[j][j];value*=p
  for i in range(j+1,n):
   q=a[i][j]/p
   for k in range(j+1,n):a[i][k]-=q*a[j][k]
   a[i][j]=Fraction(0)
 return value,swaps

def main():
 torch.set_num_threads(1);start=time.perf_counter();out=HERE/'exact_rank_01';out.mkdir(exist_ok=False)
 phase=PeriodicRTE2D._make_phase_matrix(12,.65,torch.float32).double();weights=torch.full((12,),1/12,dtype=torch.float32).double();K=phase*weights[None]
 # Products of two binary32 values fit exactly in binary64 (at most48 significand bits).
 for i in range(12):
  for j in range(12):assert Fraction(float(K[i,j]))==Fraction(float(phase[i,j]))*Fraction(float(weights[j]))
 result={}
 for name,M in [('full_angular',K),('disjoint_rows0to5_columns6to11',K[:6,6:])]:
  d,swaps=det(M);assert d!=0;result[name]=dict(dimension=len(M),exact_rank=len(M),determinant_numerator=str(d.numerator),determinant_denominator=str(d.denominator),row_swaps=swaps,determinant_nonzero=True)
 r=dict(complete=True,device='cpu',affinity=sorted(os.sched_getaffinity(0)),actual_g=.65,L=12,actual_stored_coefficient_matrix=K.tolist(),stored_matrix_sha256=hashlib.sha256(K.numpy().tobytes()).hexdigest(),products_exact_in_binary64=True,certificates=result,collision_rank_with_positive128_point_profile=1536,rank_lower_bound_after_any_angular_diagonal_reassignment=768,scope='Exact rational certificate for stored coefficient matrix, not a claim that floating-point applications are bitwise linear.',wall_seconds=time.perf_counter()-start,source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),models_sha256=hashlib.sha256((ROOT/'rte/models.py').read_bytes()).hexdigest())
 (out/'results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['actual_stored_coefficient_matrix','certificates']},indent=2))
if __name__=='__main__':main()
