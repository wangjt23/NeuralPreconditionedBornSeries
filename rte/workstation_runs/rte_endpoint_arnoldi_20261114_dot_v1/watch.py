import os,time,json,traceback
from evaluate import *
torch.set_num_threads(2);TRAIN=ROOT/"rte/workstation_runs/rte_krylov_window_20261114_dot_v1";FIT=ROOT/"rte/workstation_runs/rte_mcmg_fit8_20261114_dot_v1"
todo=[("before",OLD/"physical_2048.pt","cycle"),("after_FIT8",FIT/"fit_4096.pt","cycle"),("CT4",ROOT/"rte/workstation_runs/ct4_structure_control_n128_seed402_20260913/best.pt","ct")]+[(f"{a}_{s}_2048",TRAIN/f"v2_{a}_{s}/checkpoint_2048.pt","cycle") for s in [146501,146502] for a in ["O","K"]];status=dict(pid=os.getpid(),complete=False,done=[],failed=[]);start=time.time()
while todo and time.time()-start<18*3600:
 for name,cp,kind in todo[:]:
  if (OUT/(name+".json")).exists():status["done"].append(name);todo.remove((name,cp,kind));continue
  if not cp.exists():
   if (cp.parent/"failure.json").exists():status["failed"].append(dict(name=name,reason="training failed before endpoint",file=str(cp.parent/"failure.json")));todo.remove((name,cp,kind))
   continue
  try:
   sz=cp.stat().st_size;time.sleep(2)
   if cp.stat().st_size!=sz:continue
   torch.load(cp,map_location="cpu",weights_only=True)
  except Exception:continue
  try:evaluate(cp,name,kind);status["done"].append(name)
  except Exception as ex:status["failed"].append(dict(name=name,error=repr(ex),trace=traceback.format_exc()))
  todo.remove((name,cp,kind));(OUT/"status.json").write_text(json.dumps(status,indent=2))
 status["pending"]=[n for n,_,_ in todo];status["elapsed"]=time.time()-start;(OUT/"status.json").write_text(json.dumps(status,indent=2))
 if todo:time.sleep(45)
status.update(complete=not todo,elapsed=time.time()-start);(OUT/"status.json").write_text(json.dumps(status,indent=2))
