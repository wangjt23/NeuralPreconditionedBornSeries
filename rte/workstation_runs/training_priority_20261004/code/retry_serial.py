"""Finish corrected alignment and a joint timing window after phase1 timing."""
import json,pathlib,subprocess,sys,time
root=pathlib.Path('/data/wjt/NeuralPreconditionedBornSeries');base=root/'rte/workstation_runs/training_priority_20261004'
start=time.time();status={'complete':False,'started':start,'phase':'waiting_for_benchmark'}
def save():
 p=base/'retry_pipeline_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(p)
save()
try:
 while True:
  p=base/'benchmark_status.json'
  if p.exists():
   current=json.loads(p.read_text())
   if 'finished' in current:break
  if time.time()-start>1000:raise TimeoutError('previous benchmark completion receipt absent')
  time.sleep(10)
 if current.get('returncode')!=0:raise RuntimeError('previous benchmark failed; preserve before further work')
 status['phase']='alignment_retry';save()
 subprocess.run([sys.executable,str(base/'code/run_phase.py'),'alignment_retry'],cwd=root)
 aligned=json.loads((base/'alignment_retry_status.json').read_text())
 if aligned.get('returncode')!=0:raise RuntimeError('corrected alignment failed')
 if not json.loads((base/'alignment_retry/settings.json').read_text()).get('complete'):raise RuntimeError('alignment wall cap reached before both arms completed')
 status['phase']='benchmark_joint';save()
 subprocess.run([sys.executable,str(base/'code/run_phase.py'),'benchmark_joint'],cwd=root)
 timed=json.loads((base/'benchmark_joint_status.json').read_text())
 if timed.get('returncode')!=0:raise RuntimeError('joint benchmark failed')
 status['complete']=json.loads((base/'benchmark_joint/settings.json').read_text()).get('complete',False)
 status['phase']='finished'
except BaseException as e:status['error']={'type':type(e).__name__,'message':str(e)}
finally:status['finished']=time.time();status['seconds']=status['finished']-start;save()
