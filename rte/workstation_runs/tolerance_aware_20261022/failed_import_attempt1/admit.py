"""One bounded, cooperative window; no waiting, preemption or lock deletion."""
import argparse,datetime,fcntl,json,os,pathlib,signal,socket,subprocess,sys,time

ROOT=pathlib.Path('/data/wjt/NeuralPreconditionedBornSeries')
BASE=ROOT/'rte/workstation_runs/tolerance_aware_20261022'
LOCKS=[pathlib.Path('/home/wjt/codex-schur-sisc-20260925/numerical-transfer.lock'),
       pathlib.Path('/home/wjt/codex-schur-sisc-20260925/directional_linux_20260929/numerical.lock')]

def main(a):
    status=BASE/f'{a.stage}_admission.json'
    if status.exists():raise FileExistsError('stage already admitted or rejected; inspect before any retry')
    record=dict(stage=a.stage,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname(),pid=os.getpid(),wall_cap_seconds=a.wall_seconds,admitted=False)
    handles=[];child=None
    def interrupted(signum,frame):
        raise InterruptedError('own bounded controller received signal '+str(signum))
    for signum in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(signum,interrupted)
    try:
        # Existing Schur lock order; do not create/truncate those files.
        for path in LOCKS:
            if not path.is_file():raise RuntimeError('required existing lock unavailable: '+str(path))
            f=path.open('r');handles.append(f)
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('occupied lock: '+str(path))
        own=(BASE/'numerical.lock').open('a');handles.append(own)
        try:fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('RTE numerical stage already running')
        record['locks']=[str(p) for p in LOCKS]
        import hashlib
        frozen=json.loads((BASE/'FROZEN_MANIFEST.json').read_text())
        for relative,digest in frozen.items():
            if hashlib.sha256((BASE/relative).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('frozen protocol/source changed: '+relative)
        record['frozen_manifest_sha256']=hashlib.sha256((BASE/'FROZEN_MANIFEST.json').read_bytes()).hexdigest()
        processes=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid,ppid,args'],text=True)
        conflicts=[]
        ancestors={os.getpid(),os.getppid()}
        for line in processes.splitlines()[1:]:
            fields=line.strip().split(None,2)
            if len(fields)<3 or int(fields[0]) in ancestors:continue
            command=fields[2].lower()
            if any(s in command for s in ['campaign_schur','campaign_hypre','pressure','run_fwi.py','interleaved_online','controller.py','train_adequacy.py','train_alignment.py','train_tolerance.py']):conflicts.append(line)
        record['conflicting_jobs']=conflicts
        if conflicts:raise RuntimeError('existing controller/numerical job; no window admitted')
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
        record['gpu_before']=gpu
        used,util=(float(x.strip()) for x in gpu.split(','))
        if used>1024 or util>10:raise RuntimeError('GPU busy; no window admitted')
        record.update(admitted=True,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),state='running')
        status.write_text(json.dumps(record,indent=2)+'\n')
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with (BASE/f'{a.stage}.log').open('x') as log:
            child=subprocess.Popen([sys.executable,*a.command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record['child_pid']=child.pid
            status.write_text(json.dumps(record,indent=2)+'\n')
            try:code=child.wait(timeout=a.wall_seconds)
            except subprocess.TimeoutExpired:
                # Only our own bounded child group; never touch another owner's jobs.
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                record['state']='own-stage wall cap';raise TimeoutError('bounded RTE stage exhausted window; partial outputs retained')
        record.update(returncode=code,state='complete' if code==0 else 'failed')
    except Exception as e:
        record.update(error=repr(e),state=record.get('state','not admitted'))
        print(json.dumps(record),flush=True)
        raise
    finally:
        # A detached child must be reaped BEFORE releasing either shared lock.
        # This touches only the process group created by this invocation.
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        record['gpu_after']=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True,capture_output=True).stdout.strip()
        status.write_text(json.dumps(record,indent=2)+'\n')
        for f in reversed(handles):
            try:fcntl.flock(f,fcntl.LOCK_UN)
            finally:f.close()
    if record.get('returncode')!=0:raise SystemExit(record.get('returncode',1))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--wall-seconds',type=float,required=True);p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if not a.command or a.command[0]=='--':a.command=a.command[1:]
    main(a)
