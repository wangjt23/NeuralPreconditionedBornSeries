"""Original-RHS checks and evaluated coercivity estimates for saved neural fields.

No new solves/teachers, no interval-certification claim. All tensors stay on FWI.
"""
import csv,json,pathlib,sys,time
import torch
import rte.benchmark_tail_solver as protocol
from rte.train_source_predictor import material_operator
from rte.run_large_angular_spectral_experiments import ExperimentConfig
from rte.run_residual_subspace_experiment import write_csv

ROOT=pathlib.Path('/data/wjt/NeuralPreconditionedBornSeries');BASE=ROOT/'rte/workstation_runs/training_priority_20261004'

@torch.no_grad()
def main():
 torch.set_num_threads(1);start=time.perf_counter();out=BASE/'accuracy';out.mkdir(exist_ok=False)
 rows=[];teachers={64:torch.load(ROOT/'rte/workstation_runs/source_tail_validation_20260926/validation.pt',map_location='cpu',weights_only=True),
  128:torch.load(ROOT/'rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt',map_location='cpu',weights_only=True)}
 for grid in [64,128,256]:
  data_path=ROOT/f'rte/workstation_runs/tail_solver_benchmark_20260926/validation_n{grid}.pt'
  data=torch.load(data_path,map_location='cpu',weights_only=True)
  cfg=ExperimentConfig(size=grid,n_angles=12,anisotropy_g=.65,eta_scale=0.,regimes='mixed')
  for index in range(16):
   if time.perf_counter()-start>120:raise TimeoutError('saved-field accuracy audit cap')
   p=BASE/'benchmark_joint'/f'neural_field_n{grid}_case{index}.pt';saved=torch.load(p,map_location='cpu',weights_only=True)
   low,high=material_operator(cfg,data,torch.tensor([index]),.1,'cuda')
   field=saved['field'].to('cuda');rhs=high.source
   if not torch.equal(saved['source'].to('cuda'),rhs):raise ValueError('saved original-RHS identity changed')
   residual=rhs-high.A_op(field);rr=float(residual.norm()/rhs.norm())
   P=high.phase_matrix.double().cpu()*high.weights.double().cpu()[None,:]
   H=(P+P.mT)/2;mu=float(torch.linalg.eigvalsh(H).max())
   margin=float((high.sigma_t-high.sigma_s*mu).min());nonnegative=bool((high.sigma_s>=0).all())
   uniform=bool(torch.equal(high.weights,high.weights[0].expand_as(high.weights)))
   bound=None
   if margin>0 and nonnegative and uniform:
    delta=float(residual.norm())/margin;unorm=float(field.norm())
    if unorm>delta:bound=delta/(unorm-delta)
   row=dict(grid=grid,index=index,original_rhs_exact=1,true_residual=rr,failed=int(rr>1e-5),
    stored_status_residual=saved['status']['true_residual'],margin=margin,mu=mu,nonnegative_scattering=int(nonnegative),uniform_angular_weights=int(uniform),
    evaluated_relative_solution_error_bound=bound,teacher_rhs_exact=None,teacher_residual=None,relative_error_vs_saved_teacher=None,
    field_sha256=protocol.digest(p))
   if grid in teachers:
    bank=teachers[grid]['bank'];same=torch.equal(bank['q'][index:index+1].to('cuda').to(torch.complex128),rhs)
    row['teacher_rhs_exact']=int(same)
    if same:
     target=bank['target'][index:index+1].to('cuda');row['teacher_residual']=float((rhs-high.A_op(target)).norm()/rhs.norm())
     row['relative_error_vs_saved_teacher']=float((field-target).norm()/target.norm())
   rows.append(row);write_csv(out/'cases.csv',rows)
 summary={'complete':True,'cases':len(rows),'scope':'48 saved original-RHS extended-neural fields; existing reused development only; no solves or teachers',
  'max_true_residual':max(r['true_residual'] for r in rows),'failures':sum(r['failed'] for r in rows),
  'all_original_rhs_exact':all(r['original_rhs_exact'] for r in rows),
  'coercivity_assumptions_evaluated_all_pass':all(r['margin']>0 and r['nonnegative_scattering'] and r['uniform_angular_weights'] for r in rows),
  'max_evaluated_relative_solution_error_bound':max((r['evaluated_relative_solution_error_bound'] for r in rows if r['evaluated_relative_solution_error_bound'] is not None),default=None),
  'max_relative_error_vs_saved_teacher':max((r['relative_error_vs_saved_teacher'] for r in rows if r['relative_error_vs_saved_teacher'] is not None),default=None),
  'teacher_comparison_count':sum(r['relative_error_vs_saved_teacher'] is not None for r in rows),
  'method':'P is actual rounded phase action including quadrature;mu=maxeig((P+P*)/2);m=min(sigma_t-sigma_s*mu);delta=norm(residual)/m;relative_bound=delta/(norm(field)-delta)',
  'limitations':'exact-arithmetic theorem assumptions, evaluated with ordinary FP64; no directed-rounding enclosure, interval certificate, continuum error or positivity claim',
  'wall_seconds':time.perf_counter()-start,'source_sha256':protocol.digest(__file__)}
 (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
