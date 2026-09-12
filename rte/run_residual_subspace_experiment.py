"""Small falsification experiment for residual-minimized learned corrections.

python -m rte.run_residual_subspace_experiment --steps 300 --seeds 1 2 3

Each model sees the same coefficient field and residual minibatches. Selection
uses validation sources; final test sources never select the checkpoint.
Reports physical losses on full fixed banks and actual A/base calls alongside
timings. This is a per-medium prototype, not coefficient generalization.
"""

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import torch

from rte.npbs_unet_ablation import (
    AngularSpectralDefect2Preconditioner, NPBSUNetPreconditioner,
    collect_residuals_from_sources, inner_base_apply, make_source,
    solve_fgmres, solve_stationary,
)
from rte.residual_subspace import (
    ResidualMinimizedPreconditioner, physical_residual_loss,
    projected_training_loss, residual_minimizing_correction, squared_norm,
)
from rte.run_hard_case_experiments import make_case
from rte.probe_residual_amplification import residual_power_vector


def sources(size, angles, count, seed, device):
    generator = torch.Generator().manual_seed(seed)
    centers = 0.15 + 0.7 * torch.rand(count, 2, generator=generator)
    return [make_source(size, angles, torch.float32, center=tuple(center.tolist()),
                        angular=("beam", "dipole", "isotropic")[i % 3]).to(device)
            for i, center in enumerate(centers)]


def residual_bank(rte, source_list, base_kind, seed):
    torch.manual_seed(seed)
    entries = collect_residuals_from_sources(rte, base_kind, sources=source_list,
                                            n_rollout=6, n_random=16)
    bank = torch.cat(entries)
    return bank / squared_norm(bank).sqrt().clamp_min(1e-20).view(-1, 1, 1, 1)


@torch.no_grad()
def bank_metrics(rte, method, bank):
    ratios = []
    for residual in bank.split(8):
        error = residual - rte.A_op(method(residual))
        ratios.append((squared_norm(error) / squared_norm(residual).clamp_min(1e-30)).sqrt())
    ratio = torch.cat(ratios)
    return {"physical_loss": float(ratio.square().mean()),
            "one_step_mean": float(ratio.mean()), "one_step_max": float(ratio.max())}


def synchronized(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measured_solver(rte, method, source, args, stationary=False):
    original = rte.A_op
    calls = {"operator_calls": 0, "preconditioner_calls": 0}
    def operator(x):
        calls["operator_calls"] += 1
        return original(x)
    def preconditioner(x):
        calls["preconditioner_calls"] += 1
        return method(x)
    synchronized(source.device)
    start = time.perf_counter()
    rte.A_op = operator
    try:
        if stationary:
            result = solve_stationary(rte, source, preconditioner, args.solver_steps, args.tol)
        else:
            result = solve_fgmres(rte, source, preconditioner, args.solver_steps, args.tol,
                                  real_stacked=True)
    finally:
        rte.A_op = original
    synchronized(source.device)
    return {"final": result[1], "iters": result[0],
            "seconds": time.perf_counter() - start, **calls}


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def train(args, rte, kind, train_bank, val_bank, val_sources, seed, out):
    torch.manual_seed(seed)
    legacy = kind in ("legacy_raw", "legacy_replay")
    cls = NPBSUNetPreconditioner if legacy else AngularSpectralDefect2Preconditioner
    kwargs = {"form": "defect2"} if legacy else {"mode_width": 4}
    model = cls(rte, base_kind=args.base_kind, width=16, levels=3, output_init="zero", **kwargs).to(rte.source.device)
    deployed = ResidualMinimizedPreconditioner(model) if kind == "angular_projected" else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.steps), eta_min=args.lr * .01)
    generator = torch.Generator().manual_seed(seed + 900)
    history, best_score, best_state, best_step = [], float("inf"), None, 0
    hard_bank = None
    replay_seconds = 0.0
    start = time.perf_counter()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            tr = bank_metrics(rte, deployed, train_bank)
            va = bank_metrics(rte, deployed, val_bank)
            with torch.no_grad():
                score = sum(solve_fgmres(rte, q, deployed, args.solver_steps, args.tol,
                                        real_stacked=True)[1] for q in val_sources) / len(val_sources)
            if score < best_score:
                best_score, best_state, best_step = score, copy.deepcopy(model.state_dict()), step
            history.append({"step": step, "seed": seed, "kind": kind,
                            **{"train_" + k: v for k, v in tr.items()},
                            **{"validation_" + k: v for k, v in va.items()},
                            "validation_fgmres": score, "selected_step": best_step})
            print(f"{args.case} seed={seed} {kind} step={step} "
                  f"train_loss={tr['physical_loss']:.4g} val_loss={va['physical_loss']:.4g} "
                  f"fgmres={score:.4g} best={best_step}", flush=True)
        if step == args.steps:
            break
        model.train()
        indices = torch.randint(len(train_bank), (args.batch_size,), generator=generator)
        residual = train_bank[indices.to(train_bank.device)]
        if kind == "legacy_replay":
            if step % args.replay_every == 0:
                replay_start = time.perf_counter()
                hard_bank = torch.cat([
                    residual_power_vector(rte.A_op, model, rte.source, seed + 8000 + step + j,
                                          steps=args.replay_power_steps)
                    for j in range(4)
                ])
                replay_seconds += time.perf_counter() - replay_start
            hard_count = max(1, args.batch_size // 4)
            hard_indices = torch.arange(hard_count, device=hard_bank.device) % len(hard_bank)
            residual = torch.cat([residual[:-hard_count], hard_bank[hard_indices]])
        optimizer.zero_grad(set_to_none=True)
        if kind == "angular_projected":
            loss, _ = projected_training_loss(model, residual, raw_weight=args.raw_weight)
        else:
            loss = physical_residual_loss(rte.A_op, residual, model(residual))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite {kind} loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    elapsed = time.perf_counter() - start
    write_csv(out / f"{args.case}_{seed}_{kind}_history.csv", history)
    model.load_state_dict(best_state)
    model.eval()
    torch.save({"model": best_state, "selected_step": best_step, "args": vars(args)},
               out / f"{args.case}_{seed}_{kind}.pt")
    return model, {"selected_step": best_step, "training_seconds": elapsed,
                   "replay_seconds": replay_seconds,
                   "train_final_physical_loss": history[-1]["train_physical_loss"],
                   "validation_final_physical_loss": history[-1]["validation_physical_loss"]}


def run(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {out}; choose a new --output or pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    rows = []
    for seed in args.seeds:
        rte, _ = make_case(args.case, args.size, args.n_angles, torch.float32, .1, device=device)
        train_sources = sources(args.size, args.n_angles, 8, seed + 1000, device)
        val_sources = sources(args.size, args.n_angles, 2, seed + 2000, device)
        test_sources = sources(args.size, args.n_angles, 3, seed + 3000, device)
        train_bank = residual_bank(rte, train_sources, args.base_kind, seed + 4000)
        val_bank = residual_bank(rte, val_sources, args.base_kind, seed + 5000)
        test_bank = residual_bank(rte, test_sources, args.base_kind, seed + 6000)
        def classical(depth):
            return lambda r: inner_base_apply(rte, r, depth=depth, kind=args.base_kind)
        base2, base4 = classical(2), classical(4)
        def mr_classical(r):
            b2 = base2(r)
            # Compute stages 3 and 4 on the remaining residual, for 4 total
            # base calls, matching the learned wrapper's two-direction fit.
            extra = base2(r - rte.A_op(b2))
            return residual_minimizing_correction(rte.A_op, r, b2, [extra, b2])
        def mr_scale(r):
            b2 = base2(r)
            return residual_minimizing_correction(rte.A_op, r, b2, [b2])
        methods = [("base2", base2, 2, {}), ("base4", base4, 4, {}),
                   ("MR_base2", mr_scale, 2, {}), ("MR_classical4", mr_classical, 4, {})]
        for kind in args.models:
            model, metadata = train(args, rte, kind, train_bank, val_bank, val_sources, seed, out)
            methods.extend([(kind, model, 2, metadata),
                            ("MR_" + kind, ResidualMinimizedPreconditioner(model), 2, metadata)])
        for label, method, depth, metadata in methods:
            with torch.no_grad():
                method(test_sources[0])
            banks = {"train": bank_metrics(rte, method, train_bank),
                     "validation": bank_metrics(rte, method, val_bank),
                     "test": bank_metrics(rte, method, test_bank)}
            for idx, source in enumerate(test_sources):
                with torch.no_grad():
                    source_loss = float(physical_residual_loss(rte.A_op, source, method(source)))
                for stationary in (False, True):
                    metrics = measured_solver(rte, method, source, args, stationary)
                    rows.append({"case": args.case, "seed": seed, "method": label,
                                 "test_source": idx, "solver": "stationary" if stationary else "fgmres",
                                 "test_source_one_step": source_loss ** .5,
                                 "base_calls": depth * metrics["preconditioner_calls"],
                                 **metrics, **metadata,
                                 **{split + "_" + k: v for split, m in banks.items() for k, v in m.items()}})
        write_csv(out / "results.csv", rows)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=["mixed", "anisotropic", "heterogeneous"], default="mixed")
    p.add_argument("--base-kind", default="transport_tsa")
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--n-angles", type=int, default=12)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--solver-steps", type=int, default=12)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--models", nargs="+", choices=["legacy_raw", "legacy_replay", "angular_raw", "angular_projected"],
                   default=["legacy_raw", "angular_raw", "angular_projected"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--raw-weight", type=float, default=.25)
    p.add_argument("--replay-every", type=int, default=50)
    p.add_argument("--replay-power-steps", type=int, default=15)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--output", default="rte/results/residual_subspace_mixed")
    p.add_argument("--overwrite", action="store_true", help="Explicitly allow replacing artifacts in --output.")
    args = p.parse_args()
    if args.steps < 0 or args.eval_every < 1 or args.batch_size < 1 or args.replay_every < 1 or args.replay_power_steps < 1:
        p.error("steps must be nonnegative; eval-every and batch-size positive")
    run(args)


if __name__ == "__main__":
    main()
