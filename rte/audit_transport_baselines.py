"""Retest learned CT checkpoints against shift/depth/relaxation-tuned transport.

No test source selects settings. Runtime selection uses repeated validation
solves; settings, all candidate scores and final true residuals are retained.
"""

import argparse
import itertools
import json
import math
import statistics
import time
from pathlib import Path

import torch

from rte.collision_transport import CollisionTransportPreconditioner
from rte.residual_subspace import physical_residual_loss
from rte.run_hard_case_experiments import make_case
from rte.run_residual_subspace_experiment import measured_solver, sources, write_csv
from rte.transport_baselines import ClassicalTransportStack


def evaluate_sources(rte, method, qs, args, repeats):
    rows = []
    for q in qs:
        measured_solver(rte, method, q, args)
        samples = [measured_solver(rte, method, q, args) for _ in range(repeats)]
        row = dict(samples[0])
        row["seconds"] = statistics.median(x["seconds"] for x in samples)
        with torch.no_grad():
            row["one_step"] = float(physical_residual_loss(rte.A_op, q, method(q))) ** .5
        rows.append(row)
    return rows


def cost_score(rows, tol):
    failures = sum(not math.isfinite(r["final"]) or r["final"] > tol for r in rows)
    quality = statistics.mean(math.log10(max(tol, r["final"]))
                              if math.isfinite(r["final"]) else 100. for r in rows) if failures else 0.
    return failures, quality, statistics.mean(r["seconds"] for r in rows)


def run(args):
    torch.set_num_threads(args.threads)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    candidates, results = [], []
    for seed in args.seeds:
        rte, _ = make_case(args.case, args.size, args.n_angles, torch.float32, .1)
        validation = sources(args.size, args.n_angles, 3, seed + 2000, torch.device("cpu"))
        tests = sources(args.size, args.n_angles, 3, seed + 3000, torch.device("cpu"))
        methods = [("identity", lambda r: r, {})]
        for kind in args.kinds:
            best = None
            start = time.perf_counter()
            for eta, depth, relaxation in itertools.product(args.shifts, args.depths, args.relaxations):
                if depth == 1 and relaxation != 1.:
                    continue  # A single global scaling changes no exact Krylov space.
                method = ClassicalTransportStack(rte, kind, depth, eta, relaxation).eval()
                rows = evaluate_sources(rte, method, validation, args, args.validation_repeats)
                score = cost_score(rows, args.tol)
                settings = {"eta": eta, "depth": depth, "relaxation": relaxation}
                candidates.append({"seed": seed, "kind": kind, **settings,
                                   "validation_failures": score[0], "validation_log_residual": score[1],
                                   "validation_seconds": score[2],
                                   "validation_iters": statistics.mean(r["iters"] for r in rows)})
                if best is None or score < best[0]:
                    best = score, method, settings
            metadata = {**best[2], "tuning_seconds": time.perf_counter() - start}
            print(f"seed={seed} {kind}: {metadata} score={best[0]}", flush=True)
            methods.append((kind + "_tuned", best[1], metadata))
        if args.checkpoints:
            checkpoint = torch.load(Path(args.checkpoints) / f"seed{seed}_learned_modes.pt",
                                    map_location="cpu", weights_only=False)
            old = checkpoint["args"]
            if (old["case"], old["size"], old["n_angles"]) != (args.case, args.size, args.n_angles):
                raise ValueError("checkpoint problem differs from audit")
            model = CollisionTransportPreconditioner(
                rte, depth=old.get("depth", 2), learned=True, width=old["width"],
                shift_factor=checkpoint["shift_factor"])
            model.load_state_dict(checkpoint["model"])
            model.prepare()
            methods.append(("learned_CT", model, {"depth": model.depth, "selected_step": checkpoint["selected_step"]}))
        for name, method, metadata in methods:
            for i, row in enumerate(evaluate_sources(rte, method, tests, args, args.repeats)):
                results.append({"case": args.case, "seed": seed, "method": name, "test_source": i,
                                **metadata, **row})
        write_csv(out / "candidates.csv", candidates)
        write_csv(out / "results.csv", results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="mixed", choices=["mixed", "anisotropic", "diffusive"])
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--n-angles", type=int, default=12)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--kinds", nargs="+", choices=ClassicalTransportStack.KINDS,
                   default=["reference", "tsa", "dsa", "dsa_transport", "transport"])
    p.add_argument("--shifts", type=float, nargs="+", default=[0., .03, .1, .3])
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--relaxations", type=float, nargs="+", default=[.5, 1.])
    p.add_argument("--solver-steps", type=int, default=30)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--validation-repeats", type=int, default=2)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--checkpoints", default="")
    p.add_argument("--output", default="rte/results/collision_transport_classical_audit")
    args = p.parse_args()
    if min(args.depths) < 1 or min(args.shifts) < 0 or min(args.relaxations) <= 0 or min(args.repeats, args.validation_repeats) < 1:
        p.error("invalid tuning grid or repeats")
    run(args)


if __name__ == "__main__":
    main()
