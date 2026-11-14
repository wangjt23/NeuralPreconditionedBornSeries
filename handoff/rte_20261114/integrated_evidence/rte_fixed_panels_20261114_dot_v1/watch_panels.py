import time,json,traceback,os
from panels import *
TRAIN=ROOT/"rte/workstation_runs/rte_krylov_window_20261114_dot_v1"
torch.set_num_threads(2);todo=[(arm,seed,step) for step in [128,512,1024,2048] for seed in [146501,146502] for arm in ["O","K"]];status=dict(pid=os.getpid(),complete=False,done=[],failed=[]);start=time.time()
while todo and time.time()-start<18*3600:
 for item in todo[:]:
  arm,seed,step=item;name=f"{arm}_{seed}_{step}";folder=TRAIN/f"v2_{arm}_{seed}";checkpoint=folder/f"checkpoint_{step}.pt"
  if (OUT/(name+".json")).exists():status["done"].append(name);todo.remove(item);continue
  if not checkpoint.exists():
   if (folder/"failure.json").exists():status["failed"].append(dict(name=name,reason="training failed before checkpoint",file=str(folder/"failure.json")));todo.remove(item)
   continue
  try:
   first=checkpoint.stat().st_size;time.sleep(2)
   if first!=checkpoint.stat().st_size:continue
   torch.load(checkpoint,map_location="cpu",weights_only=True)
  except Exception:continue
  try:evaluate(checkpoint,name,"cycle");status["done"].append(name)
  except Exception as e:status["failed"].append(dict(name=name,error=repr(e),trace=traceback.format_exc()))
  todo.remove(item);(OUT/"queue_status.json").write_text(json.dumps(status,indent=2))
 status["pending"]=todo;status["elapsed"]=time.time()-start;(OUT/"queue_status.json").write_text(json.dumps(status,indent=2))
 if todo:time.sleep(30)
status.update(complete=not todo,pending=todo,elapsed=time.time()-start);(OUT/"queue_status.json").write_text(json.dumps(status,indent=2))
