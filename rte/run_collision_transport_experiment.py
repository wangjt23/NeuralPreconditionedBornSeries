"""Matched per-medium experiment for learned collision/transport factorization.

python -m rte.run_collision_transport_experiment --steps 300 --seeds 1 2 3

Validation, never test, selects classical shifts and learned checkpoints.
All losses are mean squared relative physical residuals on fixed, balanced
banks. This is a source-heldout laptop experiment, not material generalization.
"""

import argparse
import copy
import csv
import itertools
import json
import statistics
import time
from pathlib import Path

import torch

from rte.collision_transport import CollisionTransportPreconditioner
from rte.prepared_defect_unet import PreparedDefectUNet
from rte.npbs_unet_ablation import (
    NPBSUNetPreconditioner, collect_residuals_from_sources, inner_base_apply, solve_fgmres,
)
from rte.residual_subspace import physical_residual_loss, squared_norm
from rte.run_hard_case_experiments import make_case
from rte.run_residual_subspace_experiment import (
    bank_metrics, measured_solver, sources, synchronized, write_csv,
)


def make_bank(rte, source_list, seed):
    torch.manual_seed(seed)
    entries = collect_residuals_from_sources(
        rte, "transport_tsa", sources=source_list, n_rollout=4, n_random=2 * len(source_list))
    bank = torch.cat(entries)
    bank = bank / squared_norm(bank).sqrt().clamp_min(1e-20).view(-1, 1, 1, 1)
    labels = [f"stage{stage}" for _ in source_list for stage in range(4)]
    labels += ["smooth_action"] * (2 * len(source_list))
    return bank, labels


def selection_score(rte, model, source_list, args):
    results = [solve_fgmres(rte, q, model, args.solver_steps, args.tol, real_stacked=True)
               for q in source_list]
    return (sum(result[1] > args.tol for result in results),
            statistics.mean(result[0] for result in results),
            statistics.mean(result[1] for result in results))


def build_model(rte, kind, shift, args):
    if kind == "legacy_raw":
        model = NPBSUNetPreconditioner(rte, form="defect2", base_kind="transport_tsa",
                                      width=16, levels=3, output_init="zero")
    else:
        model = CollisionTransportPreconditioner(
            rte, learned=True, depth=args.depth, width=args.width, shift_factor=shift,
            learn_gains=kind != "learned_shifts", spatial_gains=kind != "learned_global_modes")
    return model.to(rte.source.device)


def train(rte, kind, shift, banks, validation_sources, seed, args, out):
    torch.manual_seed(seed)
    model = build_model(rte, kind, shift, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.steps), eta_min=.01 * args.lr)
    generator = torch.Generator().manual_seed(seed + 900)
    history, best_score, best_state, best_step = [], None, None, 0
    synchronized(rte.source.device)
    start = time.perf_counter()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            tr = bank_metrics(rte, model, banks["train"][0])
            va = bank_metrics(rte, model, banks["validation"][0])
            score = selection_score(rte, model, validation_sources, args)
            if best_score is None or score < best_score:
                best_score, best_state, best_step = score, copy.deepcopy(model.state_dict()), step
            history.append({"step": step, "seed": seed, "method": kind,
                            "train_loss": tr["physical_loss"], "validation_loss": va["physical_loss"],
                            "validation_failures": score[0], "validation_iters": score[1],
                            "validation_final": score[2], "selected_step": best_step})
            print(f"{args.case} seed={seed} {kind} step={step} "
                  f"train={tr['physical_loss']:.4g} val={va['physical_loss']:.4g} "
                  f"val_iters={score[1]:.2f} failures={score[0]} best={best_step}", flush=True)
        if step == args.steps:
            break
        model.train()
        indices = torch.randint(len(banks["train"][0]), (args.batch_size,), generator=generator)
        residual = banks["train"][0][indices.to(rte.source.device)]
        optimizer.zero_grad(set_to_none=True)
        loss = physical_residual_loss(rte.A_op, residual, model(residual))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        scheduler.step()
    synchronized(rte.source.device)
    training_seconds = time.perf_counter() - start
    write_csv(out / f"seed{seed}_{kind}_history.csv", history)
    model.load_state_dict(best_state)
    model.eval()
    torch.save({"model": best_state, "selected_step": best_step, "args": vars(args),
                "shift_factor": shift}, out / f"seed{seed}_{kind}.pt")
    return model, {"selected_step": best_step, "training_seconds": training_seconds,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "train_initial_loss": history[0]["train_loss"],
                   "train_last_loss": history[-1]["train_loss"],
                   "validation_last_loss": history[-1]["validation_loss"]}


def load_trained(rte, kind, seed, args):
    previous = Path(args.evaluate_from)
    # Checkpoints are local artifacts produced by this runner, not downloaded models.
    checkpoint = torch.load(previous / f"seed{seed}_{kind}.pt", map_location=rte.source.device, weights_only=False)
    for key in ("case", "size", "n_angles", "steps", "eval_every", "batch_size", "lr", "width", "tol", "solver_steps"):
        if checkpoint["args"][key] != getattr(args, key):
            raise ValueError(f"evaluation configuration differs from checkpoint: {key}")
    if checkpoint["args"].get("depth", 2) != args.depth:
        raise ValueError("evaluation depth differs from checkpoint")
    model = build_model(rte, kind, checkpoint["shift_factor"], args)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    old_rows = list(csv.DictReader((previous / "results.csv").open()))
    row = next(row for row in old_rows if row["seed"] == str(seed) and row["method"] == kind)
    metadata = {key: row[key] for key in ("selected_step", "training_seconds", "parameters",
                                         "train_initial_loss", "train_last_loss", "validation_last_loss")}
    return model, metadata


def run(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    settings = {**vars(args), "torch_version": str(torch.__version__),
                "schema": 1, "bank_mixture": "four TSA trajectory states + two smooth-action probes per source",
                "selection": "lexicographic validation (failures, iterations, final true residual)",
                "scope": "fixed medium, heldout source centers, periodic spatial FFT / circular angular quadrature"}
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    rows, subgroup_rows, tuning_rows, traces = [], [], [], []
    for seed in args.seeds:
        rte, _ = make_case(args.case, args.size, args.n_angles, torch.float32, .1, device=device)
        source_lists = {split: sources(args.size, args.n_angles, count, seed + offset, device)
                        for split, count, offset in (("train", 9, 1000), ("validation", 3, 2000), ("test", 3, 3000))}
        banks = {split: make_bank(rte, qs, seed + offset)
                 for (split, qs), offset in zip(source_lists.items(), (4000, 5000, 6000))}
        methods = []
        for depth in (2, 4):
            method = lambda r, d=depth: inner_base_apply(rte, r, depth=d, kind="transport_tsa")
            methods.append((f"TSA{depth}", method, depth, 0, {}))
        tuned_shift = None
        for depth in (2, 4):
            best = None
            for factor in args.shift_grid:
                model = CollisionTransportPreconditioner(rte, depth=depth, shift_factor=factor).eval()
                score = selection_score(rte, model, source_lists["validation"], args)
                tuning_rows.append({"seed": seed, "depth": depth, "shift_factor": factor,
                                    "validation_failures": score[0], "validation_iters": score[1],
                                    "validation_final": score[2]})
                if best is None or score < best[0]:
                    best = score, factor, model
            methods.append((f"HSS{depth}_tuned", best[2], 0, depth, {"shift_factor": best[1]}))
            if depth == args.depth:
                tuned_shift = best[1]
        best_sequence = None
        for factors in itertools.product(args.shift_grid, repeat=2):
            model = CollisionTransportPreconditioner(rte, depth=2, stage_factors=factors).eval()
            score = selection_score(rte, model, source_lists["validation"], args)
            tuning_rows.append({"seed": seed, "depth": 2, "stage_factors": str(factors),
                                "validation_failures": score[0], "validation_iters": score[1],
                                "validation_final": score[2]})
            if best_sequence is None or score < best_sequence[0]:
                best_sequence = score, factors, model
        methods.append(("HSS2_sequence", best_sequence[2], 0, 2, {"stage_factors": str(best_sequence[1])}))
        # Stronger nonlearned control: coordinate search over four layer shifts,
        # two sweeps, always scored on validation sources only. Not a global optimum.
        initial4 = next(m[4]["shift_factor"] for m in methods if m[0] == "HSS4_tuned")
        best_factors, sequence_scores = (initial4,) * 4, {}
        def sequence_score(factors):
            if factors not in sequence_scores:
                model = CollisionTransportPreconditioner(rte, depth=4, stage_factors=factors).eval()
                score = selection_score(rte, model, source_lists["validation"], args)
                sequence_scores[factors] = score, model
                tuning_rows.append({"seed": seed, "depth": 4, "stage_factors": str(factors),
                                    "validation_failures": score[0], "validation_iters": score[1],
                                    "validation_final": score[2]})
            return sequence_scores[factors][0]
        for _ in range(2):
            for stage in range(4):
                candidates = [best_factors[:stage] + (factor,) + best_factors[stage+1:] for factor in args.shift_grid]
                best_factors = min(candidates, key=sequence_score)
        methods.append(("HSS4_sequence", sequence_scores[best_factors][1], 0, 4,
                        {"stage_factors": str(best_factors)}))
        if args.macro_control:
            shift2 = next(m[4]["shift_factor"] for m in methods if m[0] == "HSS2_tuned")
            model = CollisionTransportPreconditioner(rte, depth=2, shift_factor=shift2,
                                                       macro_harmonics=1).eval()
            methods.append(("HSS2_macro1", model, 0, 2, {"shift_factor": shift2}))
        for kind in args.models:
            if args.evaluate_from:
                model, metadata = load_trained(rte, kind, seed, args)
            else:
                model, metadata = train(rte, kind, tuned_shift, banks, source_lists["validation"], seed, args, out)
            methods.append((kind, model, 2 if kind == "legacy_raw" else 0,
                            0 if kind == "legacy_raw" else args.depth, metadata))
            if kind == "legacy_raw":
                start = time.perf_counter()
                prepared = PreparedDefectUNet(model)
                synchronized(device)
                methods.append(("legacy_prepared", prepared, 2, 0,
                                {**metadata, "coefficient_preparation_seconds": time.perf_counter() - start}))
        for label, method, base_depth, factor_depth, metadata in methods:
            synchronized(device)
            start = time.perf_counter()
            if isinstance(method, CollisionTransportPreconditioner):
                method._prepared = None
                method.prepare()
            with torch.no_grad():
                method(source_lists["test"][0])
            synchronized(device)
            setup_and_warmup = time.perf_counter() - start
            metrics = {split: bank_metrics(rte, method, bank) for split, (bank, _) in banks.items()}
            for split, (bank, labels) in banks.items():
                for group in sorted(set(labels)):
                    subset = bank[[i for i, name in enumerate(labels) if name == group]]
                    subgroup_rows.append({"seed": seed, "method": label, "split": split,
                                          "group": group, "count": len(subset), **bank_metrics(rte, method, subset)})
            for idx, source in enumerate(source_lists["test"]):
                with torch.no_grad():
                    one_step = float(physical_residual_loss(rte.A_op, source, method(source))) ** .5
                # One untimed solve warms the full Krylov path for every method.
                trace = solve_fgmres(rte, source, method, args.solver_steps, args.tol, real_stacked=True)
                repetitions = [measured_solver(rte, method, source, args) for _ in range(args.repeats)]
                measured = repetitions[0]
                measured["seconds"] = statistics.median(r["seconds"] for r in repetitions)
                traces.extend({"seed": seed, "method": label, "test_source": idx, "iteration": i,
                               "true_residual": value} for i, value in enumerate(trace[4]))
                rows.append({"seed": seed, "case": args.case, "method": label, "test_source": idx,
                             "source_one_step": one_step, "converged": measured["final"] <= args.tol,
                             "base_calls": base_depth * measured["preconditioner_calls"],
                             "factor_stages": factor_depth * measured["preconditioner_calls"],
                             "setup_and_warmup_seconds": setup_and_warmup,
                             **measured, **metadata,
                             **{split + "_" + key: value for split, m in metrics.items() for key, value in m.items()}})
        write_csv(out / "results.csv", rows)
        write_csv(out / "bank_groups.csv", subgroup_rows)
        write_csv(out / "classical_tuning.csv", tuning_rows)
        write_csv(out / "convergence.csv", traces)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=["mixed", "anisotropic", "diffusive"], default="mixed")
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--n-angles", type=int, default=12)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--solver-steps", type=int, default=30)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--models", nargs="*", choices=["learned_shifts", "learned_modes", "learned_global_modes", "legacy_raw"],
                   default=["learned_shifts", "learned_modes", "legacy_raw"])
    p.add_argument("--shift-grid", type=float, nargs="+", default=[.5, 1., 2., 4.])
    p.add_argument("--width", type=int, default=12)
    p.add_argument("--depth", type=int, choices=[2, 4], default=2, help="Depth of learned factorization (not the U-Net).")
    p.add_argument("--lr", type=float, default=.003)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--macro-control", action="store_true", help="Experimental constant-reference P1 augmentation.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--output", default="rte/results/collision_transport_mixed")
    p.add_argument("--evaluate-from", help="Reevaluate trusted checkpoints with additional classical/cache controls; no training.")
    args = p.parse_args()
    if (args.steps < 0 or min(args.eval_every, args.batch_size, args.repeats, args.width, args.threads) < 1
            or not 1 <= args.solver_steps <= 40 or args.size < 8 or args.n_angles < 3
            or min(args.shift_grid) <= 0 or args.lr <= 0 or args.tol <= 0):
        p.error("require nonnegative steps; positive settings; size>=8, angles>=3, solver-steps in [1,40]")
    run(args)


if __name__ == "__main__":
    main()
