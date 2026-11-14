"""Wait for a real CUDA conflict to end, then execute this one owned study.

No historical Schur-lock gate, process cancellation, or system reconfiguration.
No CUDA context is opened while waiting. This is not a recurring automation.
"""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
STATUS = HERE/'admission.json'


def state(**fields):
    fields.update(controller_pid=os.getpid(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    temporary = HERE/'admission.tmp'
    temporary.write_text(json.dumps(fields, indent=2)+'\n')
    temporary.replace(STATUS)


def main():
    if STATUS.exists():
        raise RuntimeError('inspect existing controller state; never blindly resubmit')
    previous = None
    while True:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, check=True)
        active = [int(x.strip()) for x in result.stdout.splitlines() if x.strip().isdigit()]
        if not active:
            break
        state(status='waiting_for_actual_GPU_conflict', active_gpu_pids=active,
              reason='no concurrent numerical work; existing owner untouched')
        if active != previous:
            print('WAITING_GPU_PIDS', active, flush=True)
            previous = active
        time.sleep(45)
    state(status='running', active_gpu_pids=[], started_after_observed_conflict_ended=True)
    root = Path('/data/wjt/NeuralPreconditionedBornSeries')
    with (HERE/'worker.log').open('x') as log:
        result = subprocess.run(['/data1/software/miniconda3/bin/python', '-u', str(HERE/'measure.py')],
            cwd=root, env=dict(os.environ, PYTHONPATH=str(root), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
            stdout=log, stderr=subprocess.STDOUT)
    state(status='complete' if result.returncode == 0 else 'failed', returncode=result.returncode)


if __name__ == '__main__':
    main()
