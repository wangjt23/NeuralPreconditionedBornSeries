"""Owned bounded batch: two CPU jobs plus one serial GPU lane."""
import concurrent.futures,datetime,json,os,pathlib,signal,subprocess,sys,time
START=time.perf_counter();ROOT=pathlib.Path(__file__).resolve().parent.parent
PYTHON='/data1/software/miniconda3/bin/python'
def write(path,x):path.write_text(json.dumps(x,indent=2)+'\n')
def run(name,script,args,budget,path):
 env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='.:'+str(path),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 start=time.perf_counter();command=[PYTHON,str(script),*args]
 with (ROOT/(name+'.stdout.txt')).open('xb')as out,(ROOT/(name+'.stderr.txt')).open('xb')as err:
  proc=subprocess.Popen(command,env=env,stdout=out,stderr=err,start_new_session=True)
  write(ROOT/(name+'.running.json'),dict(name=name,pid=proc.pid,command=command,hard_seconds=budget))
  timeout=False
  try:code=proc.wait(timeout=budget)
  except subprocess.TimeoutExpired:
   timeout=True;os.killpg(proc.pid,signal.SIGTERM)
   try:code=proc.wait(timeout=3)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
 result=dict(name=name,pid=proc.pid,returncode=code,timeout=timeout,external_process_seconds=time.perf_counter()-start,
   hard_seconds=budget,command=command,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 write(ROOT/(name+'.execution.json'),result);return result
def main():
 p=json.loads((ROOT/'policy.json').read_text());results=[]
 freeze=run('freeze',ROOT/'jobA/code/stagewise_driver.py',['freeze'],p['freeze_hard_seconds'],ROOT/'jobA/code');results.append(freeze)
 if freeze['returncode']!=0:
  write(ROOT/'batch_results.json',dict(complete=False,phase='freeze_failed',jobs=results,elapsed_seconds=time.perf_counter()-START));return
 def gpu_lane():
  c=run('C_local',ROOT/'code/gpu_direction_jobs.py',['local'],p['gpu_job_hard_seconds'],ROOT/'code')
  # Different hypotheses retain their own failures. No candidate promotion.
  d=run('D_coarse',ROOT/'code/gpu_direction_jobs.py',['coarse'],p['gpu_job_hard_seconds'],ROOT/'code')
  return[c,d]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3)as pool:
  fa=pool.submit(run,'A_stagewise',ROOT/'jobA/code/stagewise_driver.py',['check'],p['stagewise_hard_seconds'],ROOT/'jobA/code')
  fb=pool.submit(run,'B_optimized',ROOT/'code/optimized_job.py',[],p['optimized_hard_seconds'],ROOT/'code')
  fc=pool.submit(gpu_lane)
  for f in concurrent.futures.as_completed([fa,fb,fc]):
   r=f.result();results.extend(r if isinstance(r,list)else[r])
   write(ROOT/'batch_progress.json',dict(jobs=results,elapsed_seconds=time.perf_counter()-START))
 write(ROOT/'batch_results.json',dict(complete=True,all_processes_successful=all(r['returncode']==0 for r in results),
  jobs=results,elapsed_seconds=time.perf_counter()-START,max_cpu_workers=3,gpu_lane_serial=True,full_solves=0,training_steps=0))
if __name__=='__main__':main()
