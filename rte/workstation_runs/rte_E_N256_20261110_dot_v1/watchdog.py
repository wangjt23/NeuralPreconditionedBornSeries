"""Owned CPU worker only: wall/RSS ceiling, no imports of numerical packages."""
import os,sys,time,json,signal,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
start=time.monotonic();peak=0;reason=None
p=subprocess.Popen([sys.executable,'-u',str(HERE/'run_n256_gate.py')])
receipt=dict(controller_pid=os.getpid(),worker_pid=p.pid,total_wall_limit_seconds=600,RSS_limit_bytes=6*1024**3,poll_seconds=.25,affinity=sorted(os.sched_getaffinity(0)))
(HERE/'WATCHDOG.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(event='controller_started',**receipt)),flush=True)
try:
    while p.poll() is None:
        try:
            text=Path(f'/proc/{p.pid}/status').read_text()
            current=next(int(x.split()[1])*1024 for x in text.splitlines() if x.startswith('VmRSS:'))
            peak=max(peak,current)
        except (FileNotFoundError,StopIteration):current=0
        if current>receipt['RSS_limit_bytes']:reason='RSS_LIMIT'
        if time.monotonic()-start>600:reason='TOTAL_WALL_LIMIT'
        if reason:
            p.terminate()
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            break
        time.sleep(.25)
finally:
    if p.poll() is None:
        p.terminate()
        try:p.wait(timeout=5)
        except subprocess.TimeoutExpired:p.kill();p.wait()
    receipt.update(returncode=p.returncode,stop_reason=reason,elapsed_seconds=time.monotonic()-start,observed_worker_peak_RSS_bytes=peak,reaped=True)
    (HERE/'WATCHDOG.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(event='controller_finished',**receipt)),flush=True)
sys.exit(0 if p.returncode==0 and reason is None else 1)
