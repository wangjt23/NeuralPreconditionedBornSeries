"""Run one frozen development verification with a process-owned hard deadline.

Staging or importing this file does not execute CUDA work. Invoke explicitly
only after the shared FWI GPU run has been coordinated. All writes are inside
the new v2 folder; the runner creates its own fresh result directory.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=float, default=300.)
    args = parser.parse_args()
    if not math.isfinite(args.max_wall_seconds) or not 0 < args.max_wall_seconds <= 300:
        parser.error("max wall budget must be finite, positive and at most 300 seconds")
    root = Path(__file__).resolve().parent
    repo = Path("/data/wjt/NeuralPreconditionedBornSeries").resolve()
    if socket.gethostname() != "FWI-Standard-PC-i440FX-PIIX-1996":
        raise RuntimeError("this launcher is restricted to the authorized FWI host")
    if subprocess.check_output(["id", "-un"], text=True).strip() != "wjt":
        raise RuntimeError("unexpected FWI user")
    if root != repo / "rte/workstation_runs/linear_headroom_20261002/v2":
        raise RuntimeError("unexpected isolated v2 stage path")
    args.policy = args.policy.resolve()
    args.output = args.output.resolve()
    if args.policy.parent != root or args.output.parent != root:
        raise ValueError("policy and fresh output must be direct children of the v2 stage")
    if args.output.exists():
        raise FileExistsError(args.output)
    policy = json.loads(args.policy.read_text())
    for name, expected in policy["code_sha256"].items():
        path = (root / name).resolve()
        if path.parent != root or sha(path) != expected:
            raise RuntimeError("frozen v2 code hash mismatch: " + name)
    for name, expected in policy["original_source_sha256"].items():
        if sha(repo / name) != expected:
            raise RuntimeError("original source hash mismatch: " + name)
    # Refuse overlap with a pre-existing research process. This check never
    # kills, pauses or changes another user's process or an unrelated process.
    processes = subprocess.check_output(
        ["ps", "-u", "wjt", "-o", "pid=,comm="], text=True
    ).splitlines()
    other_research = [line for line in processes if line.split() and
        int(line.split()[0]) != os.getpid() and
        any(word in line.split()[-1].lower() for word in ("python", "train", "benchmark"))]
    if other_research:
        raise RuntimeError("another research process is active: " + repr(other_research))
    gpu_before = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv"], text=True)
    receipt_path = root / "execution_receipt.json"
    log_path = root / "full16_v2.log"
    if receipt_path.exists() or log_path.exists():
        raise FileExistsError("the v2 execution receipt or log already exists")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(repo) + ":" + str(root))
    command = ["/data1/software/miniconda3/bin/python", "-u",
               str(root / "verify_graph_full16_v2.py"), "--policy", str(args.policy),
               "--output", str(args.output), "--max-wall-seconds", str(args.max_wall_seconds)]
    receipt = {
        "complete": False, "started": True, "host": socket.gethostname(),
        "policy_sha256": sha(args.policy), "command": command,
        "max_process_wall_seconds": args.max_wall_seconds,
        "gpu_before": gpu_before, "other_research_processes_before": other_research,
        "training_run": False, "test_touched": False,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    timed_out = False
    with log_path.open("x") as log:
        try:
            # subprocess.run terminates only the direct child it creates on
            # timeout. The runner does not spawn additional research workers.
            completed = subprocess.run(command, cwd=repo, env=env, stdout=log,
                                       stderr=subprocess.STDOUT,
                                       timeout=max(.001, args.max_wall_seconds -
                                                   (time.perf_counter() - start)))
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out, returncode = True, None
    source_unchanged = all(sha(repo / name) == expected for name, expected in
                           policy["original_source_sha256"].items())
    runner_settings = args.output / "settings.json"
    result = json.loads(runner_settings.read_text()) if runner_settings.exists() else {}
    receipt.update({
        "complete": returncode == 0 and bool(result.get("complete")) and not timed_out,
        "exit_code": returncode, "process_timed_out": timed_out,
        "process_wall_seconds": time.perf_counter() - start,
        "source_files_verified": len(policy["original_source_sha256"]),
        "original_sources_unchanged": source_unchanged,
        "gpu_after": subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv"], text=True),
    })
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)
    if not receipt["complete"] or not source_unchanged:
        sys.exit(1)


if __name__ == "__main__":
    run()
