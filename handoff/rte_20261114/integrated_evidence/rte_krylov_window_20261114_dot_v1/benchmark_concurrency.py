"""Prepared only. Run after all training and evaluation queues finish."""
import os,subprocess,math
import trainer_skeleton as ts
def instrumented_call(self,x):
 torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);cpu=time.process_time();t=time.perf_counter();a.record();y=self.f(x);b.record();b.synchronize();self.calls+=1;self.seconds+=time.perf_counter()-t;self.cpu_seconds=getattr(self,'cpu_seconds',0.)+time.process_time()-cpu;self.cuda_event_ms=getattr(self,'cuda_event_ms',0.)+a.elapsed_time(b);return y
def instrumented_counts(self):return dict(calls=self.calls,seconds=self.seconds,cpu_seconds=getattr(self,'cpu_seconds',0.),cuda_event_ms=getattr(self,'cuda_event_ms',0.))
ts.Counted.__call__=instrumented_call
ts.Counted.counts=instrumented_counts
from train_v2 import *
def worker(mode,slot,root):
 torch.set_num_threads(4);r=Runner(146501)
 # Identical initial checkpoint, no optimizer mutation, warm GPU before timing.
 for arm in ['O','K']:
  r.optimizer.zero_grad(set_to_none=True);contribute(r,0,arm,0,.25)
 (root/f'ready_{slot}').write_text(str(os.getpid()))
 while not (root/'go').exists():time.sleep(.01)
 torch.cuda.synchronize();t=time.perf_counter();cpu_start=time.process_time();rows=[]
 groups=range(4) if mode=='serial' else [slot]
 for group in groups:
  for j in range(16):
   idx=j%8;arm='O' if j%2==0 else 'K';c=j%3;U=(j+.5)/16
   r.optimizer.zero_grad(set_to_none=True);row=contribute(r,idx,arm,c,U)
   rows.append(dict(group=group,j=j,**row))
 torch.cuda.synchronize();elapsed=time.perf_counter()-t
 (root/f'worker_{slot}.json').write_text(json.dumps(dict(seconds=elapsed,cpu_seconds=time.process_time()-cpu_start,rows=rows),indent=2))
def run(mode,root):
 root.mkdir(exist_ok=False);n=1 if mode=='serial' else 4
 procs=[subprocess.Popen([sys.executable,__file__,'--worker',mode,'--slot',str(i),'--out',str(root)],stdout=(root/f'worker_{i}.log').open('w'),stderr=subprocess.STDOUT) for i in range(n)]
 while not all((root/f'ready_{i}').exists() for i in range(n)):
  if any(p.poll() is not None for p in procs):raise RuntimeError('worker failed before warm barrier')
  time.sleep(.1)
 t=time.perf_counter();(root/'go').touch()
 for p in procs:
  if p.wait()!=0:raise RuntimeError('benchmark worker failed')
 elapsed=time.perf_counter()-t
 result=dict(mode=mode,total_complete_wall=elapsed,groups=4,windows_per_group=16,windows=64,workers=[json.loads((root/f'worker_{i}.json').read_text()) for i in range(n)])
 (root/'result.json').write_text(json.dumps(result,indent=2));return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',choices=['serial','parallel']);p.add_argument('--slot',type=int);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.out)
 if a.worker:return worker(a.worker,a.slot,root)
 # Do not benchmark while the research jobs or evaluators are active.
 ps=subprocess.check_output(['ps','-eo','args'],text=True)
 blocked=[line for line in ps.splitlines() if ('train_v2.py --arm' in line or 'eval_queue.py' in line or 'watch_panels.py' in line) and 'python' in line]
 if blocked:raise RuntimeError('active research jobs: postpone benchmark')
 gpu_jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
 if gpu_jobs:raise RuntimeError('GPU has existing compute jobs: postpone '+gpu_jobs)
 root.mkdir(exist_ok=False);results=[]
 for k,mode in enumerate(['serial','parallel','parallel','serial','serial','parallel']):results.append(run(mode,root/f'{k}_{mode}'))
 # Exact call counts and initial input hash invariance must hold across modes.
 canonical=None
 for result in results:
  rows=sorted([v for w in result['workers'] for v in w['rows']],key=lambda x:(x['group'],x['j']))
  signature=[(x['group'],x['j'],x['T'],x['s'],x['q'],x['actual_calls'],x['input_hashes']) for x in rows]
  if canonical is None:canonical=signature
  elif signature!=canonical:raise RuntimeError('trajectory mismatch; not comparable')
 (root/'summary.json').write_text(json.dumps(dict(equal_work_and_input_hashes=True,scope='64 identical frozen-weight windows per trial, no parameter updates; warmup excluded; S-P-P-S-S-P order, 3 repetitions each; CUDA event spans may include scheduling delays, not isolated kernel busy time',results=[dict(mode=x['mode'],seconds=x['total_complete_wall']) for x in results]),indent=2))
if __name__=='__main__':main()
