"""Train reusable collision/transport weights on coefficient-heldout RTE ensembles.

Uses the existing versioned material/source generator and adds independently
seeded periodic translations and quarter-turns of both coefficient fields.
Each epoch is a full shuffled pass over materials, with fresh residual probes.
Only validation data select weights or classical settings; test data are built
after training and evaluated once. Exact epoch-boundary resume is supported.
"""

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch

from rte.audit_transport_baselines import cost_score
from rte.collision_transport import CollisionTransportPreconditioner
from rte.npbs_unet_ablation import solve_fgmres
from rte.residual_subspace import physical_residual_loss
from rte.run_large_angular_spectral_experiments import (
    ExperimentConfig, capture_rng, cpu_tree, dataset_fingerprint, dataset_metadata, epoch_batches,
    make_medium_dataset, make_operator, restore_rng, save_checkpoint, setup_batch, source_from_data,
)
from rte.run_residual_subspace_experiment import measured_solver, synchronized, write_csv
from rte.transport_baselines import ClassicalTransportStack


def material_data(cfg, count, seed):
    data = make_medium_dataset(cfg, count, seed)
    generator = torch.Generator().manual_seed(seed + 913)
    geometry = torch.stack([torch.randint(4, (count,), generator=generator),
                            torch.randint(cfg.size, (count,), generator=generator),
                            torch.randint(cfg.size, (count,), generator=generator)], 1)
    for key in ("sigma_t", "sigma_s"):
        data[key] = torch.stack([torch.roll(torch.rot90(x, int(code[0]), (-2, -1)),
                                             shifts=(int(code[1]), int(code[2])), dims=(-2, -1))
                                 for x, code in zip(data[key], geometry)])
    data["geometry"] = geometry
    data["metadata"] = {**data["metadata"], "geometry_version": 1}
    return data


def material_split(args, cfg, split):
    count, offset = {"train": (args.train_size, 1000), "validation": (args.val_size, 2000),
                     "test": (args.test_size, 3000)}[split]
    directory = getattr(args, "dataset_dir", "")
    if not directory:
        return material_data(cfg, count, args.seed + offset)
    payload = torch.load(Path(directory) / f"{split}.pt", map_location="cpu", weights_only=True)
    data = payload["data"]
    expected = {**dataset_metadata(cfg, count, args.seed + offset), "geometry_version": 1}
    if payload.get("schema") != 1 or data["metadata"] != expected or payload["g"] != args.g:
        raise ValueError(f"{split} snapshot configuration mismatch")
    if dataset_fingerprint(data) != payload["fingerprint"]:
        raise ValueError(f"{split} snapshot fingerprint mismatch")
    return data


def warm_start(model, path):
    """Transfer weights only; never imply an exact optimizer/RNG continuation."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"], strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "source_epoch": payload.get("selected_epoch", payload.get("epoch")),
            "optimizer_scheduler_rng": "reset"}


def sample_residual(rte, source, generator, probe_fraction=1/3):
    with torch.no_grad():
        z, trajectory = torch.zeros_like(source), []
        for stage in range(4):
            residual = source - rte.A_op(z)
            trajectory.append(residual)
            if stage < 3:
                z = z + rte.apply_preconditioner(residual, kind="transport_tsa")
        stages = torch.randint(4, (len(source),), generator=generator).to(source.device)
        residual = torch.stack(trajectory)[stages, torch.arange(len(source), device=source.device)]
        probe = torch.randn(source.shape, dtype=source.dtype, generator=generator).to(source.device)
        size = source.shape[-1]
        k = torch.fft.fftfreq(size, device=source.device) * size
        mask = (k.abs()[:, None] <= min(5, size//2)) & (k.abs()[None, :] <= min(5, size//2))
        probe = torch.fft.ifftn(torch.fft.fftn(probe, dim=(-2, -1)) * mask, dim=(-2, -1))
        choose = (torch.rand(len(source), generator=generator) < probe_fraction).to(source.device)
        return torch.where(choose[:, None, None, None], rte.A_op(probe), residual).detach()


@torch.no_grad()
def bank_loss(model, rte, data, cfg, device, probe_fraction, views=4):
    model.eval()
    generator = torch.Generator().manual_seed(data["metadata"]["seed"] + 341)
    total, count = 0., 0
    for indices in torch.arange(len(data["sigma_t"])).split(cfg.batch_size):
        source = setup_batch(cfg, rte, data, indices, device)
        for _ in range(views):
            residual = sample_residual(rte, source, generator, probe_fraction)
            total += len(indices) * float(physical_residual_loss(rte.A_op, residual, model(residual)))
            count += len(indices)
    return total / count


def solver_rows(model_factory, rte, data, cfg, args, device, count=None, timing=True, indices=None):
    rows = []
    if indices is None:
        indices = range(len(data["sigma_t"]) if count is None else min(count, len(data["sigma_t"])))
    for index in indices:
        source = setup_batch(cfg, rte, data, torch.tensor([index]), device)
        synchronized(device)
        start = time.perf_counter()
        model = model_factory()
        if hasattr(model, "prepare"):
            model.prepare()
        synchronized(device)
        prep = time.perf_counter() - start
        if timing:
            # Full untimed warmup, then repeated synchronized true-residual solves.
            measured_solver(rte, model, source, args)
            repeats = [measured_solver(rte, model, source, args) for _ in range(args.repeats)]
            row = dict(repeats[0])
            row["seconds"] = statistics.median(r["seconds"] for r in repeats)
        else:
            result = solve_fgmres(rte, source, model, args.solver_steps, args.tol, real_stacked=True)
            row = {"iters": result[0], "final": result[1], "seconds": result[3]}
        with torch.no_grad():
            source_loss = float(physical_residual_loss(rte.A_op, source, model(source)))
        rows.append({"index": index, "regime": int(data["regime_ids"][index]),
                     "prepare_seconds": prep, "source_loss": source_loss,
                     "first_solve_seconds": prep + row["seconds"], **row})
    return rows


def iteration_score(rows, tol):
    return (sum(not math.isfinite(r["final"]) or r["final"] > tol for r in rows),
            statistics.mean(r["iters"] for r in rows),
            statistics.mean(r["final"] if math.isfinite(r["final"]) else 1e100 for r in rows))


def run(args):
    transfer = getattr(args, "warm_start", "")
    if args.resume and transfer:
        raise ValueError("resume and warm-start are mutually exclusive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    cfg = ExperimentConfig(dataset_size=args.train_size, eval_size=args.val_size, test_size=args.test_size,
                           size=args.size, n_angles=args.n_angles, anisotropy_g=args.g, regimes=args.regimes,
                           batch_size=args.batch_size, eta_scale=0.)
    train = material_split(args, cfg, "train")
    validation = material_split(args, cfg, "validation")
    fingerprints = {"train": dataset_fingerprint(train), "validation": dataset_fingerprint(validation)}
    initial_indices = torch.arange(min(args.batch_size, args.train_size))
    q = source_from_data(cfg, train, initial_indices, device)
    rte = make_operator(cfg, train["sigma_t"][initial_indices], train["sigma_s"][initial_indices], q, device)
    model = CollisionTransportPreconditioner(rte, depth=4, learned=True, width=12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * math.ceil(args.train_size / args.batch_size), eta_min=args.lr * .01)
    generator = torch.Generator().manual_seed(args.seed + 3100)
    out = Path(args.output)
    history, best, best_state, best_epoch = [], None, None, 0
    first_epoch, updates, old_elapsed = 0, 0, 0.
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        ignored = {"output", "resume", "stop_after", "warm_start", "dataset_dir"}
        if {k: v for k, v in vars(args).items() if k not in ignored} != {k: v for k, v in payload["args"].items() if k not in ignored}:
            raise ValueError("resume settings differ, including the scheduler horizon")
        if fingerprints != payload["fingerprints"]:
            raise ValueError("resume data fingerprints differ")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        restore_rng(payload["rng"], generator, device)
        history, best, best_state = payload["history"], payload["best"], payload["best_state"]
        best_epoch, first_epoch, updates = payload["best_epoch"], payload["epoch"], payload["updates"]
        old_elapsed = payload["elapsed_seconds"]
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    transfer_info = warm_start(model, transfer) if transfer else None
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(json.dumps({**vars(args), "fingerprints": fingerprints,
                                                  "torch": str(torch.__version__), "schema": 1,
                                                  "cuda": torch.version.cuda,
                                                  "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                                                  "inference_kernel": "cached_fft_multipliers_v3",
                                                  "cache_guard": "setup_generation_and_retained_input_tensors",
                                                  "warm_start_provenance": transfer_info,
                                                  "test_timing": "randomized_method_order_within_each_medium"}, indent=2) + "\n")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronized(device)
    start = time.perf_counter()

    def assess(epoch):
        nonlocal best, best_state, best_epoch
        tr = bank_loss(model, rte, train, cfg, device, args.probe_fraction)
        va = bank_loss(model, rte, validation, cfg, device, args.probe_fraction)
        rows = solver_rows(lambda: model, rte, validation, cfg, args, device, count=args.selection_count, timing=False)
        score = iteration_score(rows, args.tol)
        if best is None or score < best:
            best, best_state, best_epoch = score, cpu_tree(model.state_dict()), epoch
        history.append({"epoch": epoch, "updates": updates, "train_loss": tr, "validation_loss": va,
                        "validation_failures": score[0], "validation_iterations": score[1],
                        "validation_final": score[2], "selected_epoch": best_epoch})
        print(f"epoch={epoch} train={tr:.5g} val={va:.5g} val_iters={score[1]:.3f} "
              f"fails={score[0]} selected={best_epoch}", flush=True)
        write_csv(out / "history.csv", history)

    if first_epoch == 0 and not history:
        assess(0)
    for epoch in range(first_epoch + 1, args.epochs + 1):
        model.train()
        for indices in epoch_batches(cfg, generator):
            source = setup_batch(cfg, rte, train, indices, device)
            residual = sample_residual(rte, source, generator, args.probe_fraction)
            optimizer.zero_grad(set_to_none=True)
            loss = physical_residual_loss(rte.A_op, residual, model(residual))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            updates += 1
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            assess(epoch)
        synchronized(device)
        save_checkpoint(out / "last.pt", {"args": vars(args), "model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
                    "scheduler": scheduler.state_dict(), "rng": capture_rng(generator, device), "epoch": epoch,
                    "history": history, "best": best, "best_state": best_state, "best_epoch": best_epoch,
                    "updates": updates, "fingerprints": fingerprints,
                    "peak_device_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                    "elapsed_seconds": old_elapsed + time.perf_counter() - start})
        if args.stop_after and epoch >= args.stop_after and epoch < args.epochs:
            return {"status": "paused", "epoch": epoch,
                    "peak_device_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    model.load_state_dict(best_state)
    model.eval()
    torch.save({"model": best_state, "args": vars(args), "selected_epoch": best_epoch,
                "fingerprints": fingerprints}, out / "best.pt")

    # Test media are unavailable to checkpoint and classical-setting selection.
    candidates, factories = [], [("learned_CT", lambda: model)]
    for family in ("hss", "tsa", "dsa", "dsa_transport", "reference"):
        selected = None
        for depth in (1, 2, 4):
            for scale in ((.5, 1., 2., 4.) if family == "hss" else (0., .03, .1)):
                if family == "hss":
                    factory = lambda d=depth, s=scale: CollisionTransportPreconditioner(rte, depth=d, shift_factor=s).eval()
                else:
                    factory = lambda f=family, d=depth, s=scale: ClassicalTransportStack(rte, f, d, s).eval()
                rows = solver_rows(factory, rte, validation, cfg, args, device, count=args.selection_count)
                score = cost_score(rows, args.tol)
                candidates.append({"family": family, "depth": depth, "scale": scale,
                                   "validation_failures": score[0], "validation_quality": score[1], "validation_seconds": score[2]})
                if selected is None or score < selected[0]:
                    selected = score, factory, depth, scale
        factories.append((family + "_tuned", selected[1]))
        print(f"classical {family}: depth={selected[2]} scale={selected[3]} score={selected[0]}", flush=True)
    write_csv(out / "classical_tuning.csv", candidates)
    test = material_split(args, cfg, "test")
    results = []
    order_generator = torch.Generator().manual_seed(args.seed + 6100)
    for index in range(len(test["sigma_t"])):
        for j in torch.randperm(len(factories), generator=order_generator).tolist():
            label, factory = factories[j]
            rows = solver_rows(factory, rte, test, cfg, args, device, indices=(index,))
            results.extend({"method": label, **row} for row in rows)
        write_csv(out / "results.csv", results)
    split_losses = {split: bank_loss(model, rte, data, cfg, device, args.probe_fraction)
                    for split, data in (("train", train), ("validation", validation), ("test", test))}
    write_csv(out / "selected_losses.csv", [{"split": split, "physical_loss": loss,
                                            "selected_epoch": best_epoch} for split, loss in split_losses.items()])
    (out / "test_fingerprint.txt").write_text(dataset_fingerprint(test) + "\n")
    return {"status": "complete", "selected_epoch": best_epoch, "losses": split_losses,
            "peak_device_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-size", type=int, default=256)
    p.add_argument("--val-size", type=int, default=32)
    p.add_argument("--test-size", type=int, default=32)
    p.add_argument("--selection-count", type=int, default=8)
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--n-angles", type=int, default=12)
    p.add_argument("--regimes", default="mixed")
    p.add_argument("--g", type=float, default=.65)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--probe-fraction", type=float, default=1/3)
    p.add_argument("--lr", type=float, default=.003)
    p.add_argument("--solver-steps", type=int, default=30)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="rte/results/collision_transport_ensemble_seed1")
    restart = p.add_mutually_exclusive_group()
    restart.add_argument("--resume", default="", help="Exact continuation with unchanged settings/horizon")
    restart.add_argument("--warm-start", default="", help="Weights only; reset optimizer, schedule and RNG")
    p.add_argument("--dataset-dir", default="", help="Optional verified train/validation/test .pt snapshots")
    p.add_argument("--stop-after", type=int, default=0)
    args = p.parse_args()
    if min(args.train_size, args.val_size, args.test_size, args.selection_count, args.batch_size, args.epochs,
           args.eval_every, args.repeats, args.threads) < 1 or not 0 <= args.probe_fraction <= 1:
        p.error("positive sample/batch/epoch counts and probe-fraction in [0,1] required")
    print(run(args))


if __name__ == "__main__":
    main()
