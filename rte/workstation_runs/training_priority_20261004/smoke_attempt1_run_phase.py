"""Record completion for a single bounded FWI job; no automatic retraining."""
import json,os,pathlib,subprocess,sys,time
root=pathlib.Path('/data/wjt/NeuralPreconditionedBornSeries')
base=root/'rte/workstation_runs/training_priority_20261004'
mode=sys.argv[1]
assert mode in {'smoke','training'}
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(root))
status={'mode':mode,'pid':os.getpid(),'started':time.time(),'complete':False}
def save():
 p=base/f'{mode}_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(p)
save()
try:
 if mode=='smoke':
  tests=subprocess.run([sys.executable,'-m','unittest','rte.tests.test_tail_lookahead'],cwd=root,env=env)
  status['tests_returncode']=tests.returncode
  if tests.returncode:raise RuntimeError('existing continuation-loss tests failed')
 cmd=[sys.executable,str(base/'code/train_adequacy.py'),'--output',str(base/('smoke' if mode=='smoke' else 'training'))]
 if mode=='smoke':cmd += ['--smoke','--wall-seconds','240']
 status['command']=cmd;save()
 result=subprocess.run(cmd,cwd=root,env=env)
 status['returncode']=result.returncode
 status['complete']=result.returncode==0
except BaseException as e:
 status['error']={'type':type(e).__name__,'message':str(e)}
 status['returncode']=1
finally:
 status['finished']=time.time();status['seconds']=status['finished']-status['started'];save()
