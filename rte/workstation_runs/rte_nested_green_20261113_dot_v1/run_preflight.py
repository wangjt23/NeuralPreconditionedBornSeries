import time
START=time.perf_counter()
import os,json,traceback
from pathlib import Path
import nested_basis as nb
p=nb.p;torch=nb.torch;HERE=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);assert set(os.sched_getaffinity(0))=={0} and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 out=HERE/'preflight';out.mkdir(exist_ok=False);report=dict(complete=False,pid=os.getpid(),scope='scalar-only fixed-budget nested interface preflight; no Green factor or full inverse',resource_at_start=p.admission())
 try:
  prior=json.loads((HERE/'PRIOR_K_RESULTS.json').read_text());old=json.loads((HERE/'PRIOR_OPERATOR_RESULTS.json').read_text())
  for f,h in prior['source_hashes'].items():assert p.digest(p.ROOT/f)==h,f
  for f,h in prior['E_source_hashes'].items():assert p.digest(p.E/f)==h,f
  report.update(source_hashes={x.name:p.digest(x) for x in HERE.glob('*.py')},plan_sha256=p.digest(HERE/'NESTED_PLAN.md'))
  ctx,rec=nb.construct();assert rec['background']==prior['background_identity'];assert abs(rec['alpha']-old['coercivity_alpha'])<1e-13;report.update(construction=rec)
  baseline=old['rows'][0]['angular_band_factor_bytes'];cost=rec['predicted_cost'];gates=dict(weighted_model_bound=rec['model_original_right_defect_bound']<=rec['desired_q'],persistent_memory=cost['persistent_tensor_bytes']<baseline,application_arithmetic_proxy=cost['application_complex_MAC_proxy']<cost['angular_baseline_application_complex_MAC_proxy'])
  report.update(angular_baseline_factor_bytes=baseline,gates=gates,feasible_for_one_factor_check=all(gates.values()),complete=True);torch.save(ctx['Dhat'],out/'approx_derivative.pt')
 except BaseException as exc:report.update(error=repr(exc),traceback=traceback.format_exc());raise
 finally:
  report.update(process_seconds=time.perf_counter()-START,peak_process_RSS_bytes=p.rss(),resource_at_end=p.admission())
  if 'prior' in locals():report.update(production_unchanged=all(p.digest(p.ROOT/f)==h for f,h in prior['source_hashes'].items()),E_unchanged=all(p.digest(p.E/f)==h for f,h in prior['E_source_hashes'].items()))
  (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(dict(complete=report['complete'],feasible=report.get('feasible_for_one_factor_check'),gates=report.get('gates'),q=report.get('construction',{}).get('model_original_right_defect_bound'),cost=report.get('construction',{}).get('predicted_cost'),seconds=report['process_seconds'],error=report.get('error'))),flush=True)
if __name__=='__main__':main()
