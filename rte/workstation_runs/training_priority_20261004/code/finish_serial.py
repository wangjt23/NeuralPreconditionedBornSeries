"""Finish the approved priority serially after the existing training phase."""
import json,pathlib,subprocess,sys,time
root=pathlib.Path('/data/wjt/NeuralPreconditionedBornSeries')
base=root/'rte/workstation_runs/training_priority_20261004'
start=time.time();status={'complete':False,'started':start,'phase':'waiting_for_training'}
def save():
 p=base/'pipeline_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(p)
save()
try:
 while True:
  p=base/'training_status.json'
  if p.exists():
   training=json.loads(p.read_text())
   if 'finished' in training:break
  if time.time()-start>1000:raise TimeoutError('training completion receipt absent after its declared cap')
  time.sleep(10)
 if training.get('returncode')!=0:raise RuntimeError('training phase failed; preserve failure before further work')
 settings=json.loads((base/'training/settings.json').read_text())
 if not settings.get('complete'):raise RuntimeError('training phase reached its cap before both budgets finished')
 status['phase']='alignment';save()
 subprocess.run([sys.executable,str(base/'code/run_phase.py'),'alignment'],cwd=root)
 alignment=json.loads((base/'alignment_status.json').read_text())
 status['alignment_returncode']=alignment.get('returncode')
 status['phase']='benchmark';save()
 subprocess.run([sys.executable,str(base/'code/run_phase.py'),'benchmark'],cwd=root)
 benchmark=json.loads((base/'benchmark_status.json').read_text())
 status['benchmark_returncode']=benchmark.get('returncode')
 status['complete']=alignment.get('returncode')==0 and benchmark.get('returncode')==0
 status['phase']='finished'
except BaseException as e:
 status['error']={'type':type(e).__name__,'message':str(e)}
finally:
 status['finished']=time.time();status['seconds']=status['finished']-start;save()
