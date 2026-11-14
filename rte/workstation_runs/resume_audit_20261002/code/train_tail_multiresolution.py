"""Paired, bounded N64-only versus alternating N64/N128 tail continuation.

Run from the RTE repository with its package on PYTHONPATH. The network, TSA8
balanced prefix, continuation loss, and teacher coefficient are unchanged.
Only the grid of every second training update differs. Both arms consume one
saved ID/phase schedule and select checkpoints on all 16 validation media at
both grids. No test files are read or generated.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch


ARMS = ("n64_only", "mixed64_128")
SOURCE_PARAMETERS = ("centers", "phases", "amplitudes", "angular_codes", "geometry_base64")
SOURCE_MODULES = (
    "train_tail_lookahead", "tail_lookahead", "train_tail_direction", "augmented_fgmres",
    "audit_tail_injection", "global_source_predictor", "tail_response_network",
    "train_source_predictor", "mixed_precision_transport", "transport_baselines",
    "run_large_angular_spectral_experiments", "RTE_mgmodel", "source_initialization",
)


def file_digest(path):
    """Stream large teacher/context files rather than duplicating them in RAM."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def tensor_digest(*tensors):
    result = hashlib.sha256()
    for value in tensors:
        value = value.detach().cpu().contiguous()
        result.update(str((str(value.dtype), tuple(value.shape))).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_checkpoint(path, value):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_rows(path, rows):
    if not rows:
        return
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


def check_paired_identity(coarse, fine, *, contexts, expected_count=None):
    """Check original case IDs plus exact nested materials and source parameters.

    Tail residuals/teachers differ between grids and are deliberately not
    compared as nested arrays. The saved data describe the common parent draw.
    """
    cb, fb = coarse["bank"], fine["bank"]
    cd, fd = coarse["data"], fine["data"]
    count = len(cb["q"])
    if count == 0 or count != len(fb["q"]):
        raise ValueError("empty or unequal paired bank lengths")
    if expected_count is not None and count != expected_count:
        raise ValueError(f"validation must retain ALL {expected_count} media, found {count}")
    if cb["q"].shape[-2:] != (64, 64) or fb["q"].shape[-2:] != (128, 128):
        raise ValueError("expected exactly N64 and N128 square grids")
    if cb["q"].shape[1] != 12 or fb["q"].shape[1] != 12:
        raise ValueError("the frozen study uses L12")
    if contexts:
        ids = cb["indices"].cpu()
        fine_ids = fb["indices"].cpu()
        if ids.dtype not in (torch.int32, torch.int64) or not torch.equal(ids, fine_ids):
            raise ValueError("eligible original training IDs differ between grids")
        if ids.ndim != 1 or len(ids) != count or len(torch.unique(ids)) != count:
            raise ValueError("eligible IDs must be distinct and cover every saved context")
        if bool((ids < 0).any()) or bool((ids >= len(cd["sigma_t"])).any()):
            raise ValueError("eligible ID outside original parent bank")
    else:
        ids = torch.arange(count)
    for key in ("sigma_t", "sigma_s"):
        if not torch.equal(cd[key], fd[key][..., ::2, ::2]):
            raise ValueError("parent nested material identity failed: " + key)
    for key in SOURCE_PARAMETERS:
        if not torch.equal(cd[key], fd[key]):
            raise ValueError("paired source/geometry identity failed: " + key)
    for key in ("t", "s"):
        if not torch.equal(cb[key], fb[key][..., ::2, ::2]):
            raise ValueError("saved context/operator nested material identity failed: " + key)
    # Check the eligible-row mapping against the original data, not just against
    # another possibly mislabelled context. The acquired bank uses epsilon=.1.
    for saved, data in ((coarse, cd), (fine, fd)):
        expected_t = data["sigma_t"][ids] / .1
        expected_s = expected_t - .1 * (data["sigma_t"][ids] - data["sigma_s"][ids])
        if not torch.equal(saved["bank"]["t"], expected_t):
            raise ValueError("bank t does not match saved original IDs at epsilon=.1")
        if not torch.equal(saved["bank"]["s"], expected_s):
            raise ValueError("bank s does not match saved original IDs at epsilon=.1")
    return dict(count=count, original_parent_count=len(cd["sigma_t"]),
                original_ids=ids.tolist(), original_ids_sha256=tensor_digest(ids),
                nested_material_and_source_parameters_exact=True,
                eligible_original_id_mapping_exact=True)


def build_sampling_plan(original_ids, steps, batch, seed):
    """The same RNG draw order as train_tail_lookahead, once for both arms."""
    generator = torch.Generator().manual_seed(seed + 72)
    ids, angles, states = [], [], [generator.get_state()]
    for _ in range(steps):
        ids.append(torch.randint(len(original_ids), (batch,), generator=generator))
        angles.append(torch.rand(batch, generator=generator, dtype=torch.float64) * (2 * math.pi))
        states.append(generator.get_state())
    ids = torch.stack(ids) if ids else torch.empty((0, batch), dtype=torch.long)
    angles = torch.stack(angles) if angles else torch.empty((0, batch), dtype=torch.float64)
    original = original_ids[ids]
    return dict(ids=ids, original_ids=original, phase_angles=angles,
                generator_states=torch.stack(states),
                identity_phase_sha256=tensor_digest(ids, original, angles))


def sampling_rows(plan, arm, completed=None):
    """Auditable one-row-per-presentation record; phase identity excludes grid."""
    completed = len(plan["ids"]) if completed is None else completed
    result = []
    for update in range(completed):
        grid = 128 if arm == ARMS[1] and update % 2 else 64
        for position in range(plan["ids"].shape[1]):
            result.append(dict(update=update + 1, position=position,
                               eligible_bank_row=int(plan["ids"][update, position]),
                               original_parent_index=int(plan["original_ids"][update, position]),
                               phase_angle=float(plan["phase_angles"][update, position]), grid=grid))
    return result


def selection_score(records):
    """Equal-grid selection on all 32 solves; residual is a tie breaker only."""
    for grid in (64, 128):
        group = [row for row in records if row["grid"] == grid]
        if len(group) != 16 or sorted(row["index"] for row in group) != list(range(16)):
            raise ValueError("selection requires every validation ID on both grids")
    if any(not math.isfinite(float(row[key])) for row in records for key in
           ("classical_calls", "injection_calls", "true_residual")):
        raise ValueError("nonfinite validation status; cannot select a checkpoint")
    return (sum(int(row["failed"]) for row in records),
            sum(row["classical_calls"] + .5 * row["injection_calls"] for row in records) / 32,
            sum(row["true_residual"] for row in records) / 32)


def sync(device):
    torch.cuda.synchronize(device)


def memory_receipt(device):
    free, total = torch.cuda.mem_get_info(device)
    return dict(allocated_bytes=torch.cuda.memory_allocated(device),
                reserved_bytes=torch.cuda.memory_reserved(device), free_bytes=free, total_bytes=total,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device))


def reset_rng(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state(plan, step):
    return dict(python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
                cuda_rng_all=torch.cuda.get_rng_state_all(),
                generator_rng=plan["generator_states"][step], sampling_cursor=step)


def restore_rng(saved):
    random.setstate(saved["python_rng"])
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng_all"])


def normalize_and_cache(saved_banks, args, device):
    from rte.global_source_predictor import FrozenTSA  # import verifies deployed cache consumer
    from rte.run_large_angular_spectral_experiments import ExperimentConfig, make_operator
    from rte.train_source_predictor import reference_cache
    from rte.transport_baselines import ClassicalTransportStack

    del FrozenTSA
    estimated = 0
    for saved in saved_banks.values():
        bank = saved["bank"]
        estimated += sum(value.numel() * value.element_size() for value in bank.values())
        count, angles, n, _ = bank["q"].shape
        estimated += count * n * n * (angles * angles + angles) * 8
    free, _ = torch.cuda.mem_get_info(device)
    mode = args.cache_mode
    if mode == "auto":
        # Leave room for autograd and batchwise inverse construction. On the
        # 49 GB workstation the two modest banks normally fit together.
        mode = "gpu" if estimated + 8 * 1024**3 < free else "cpu"
    if mode == "gpu" and estimated + 4 * 1024**3 > free:
        raise RuntimeError("insufficient free GPU memory for requested resident banks; use --cache-mode cpu")
    storage = device if mode == "gpu" else torch.device("cpu")
    banks, receipts = {}, {}
    for grid, saved in saved_banks.items():
        sync(device)
        started = time.perf_counter()
        bank = {key: value.to(storage) for key, value in saved["bank"].items()}
        scale = bank["rhs_exact"].abs().square().mean((1, 2, 3), keepdim=True).sqrt()
        if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError("training contexts must have finite, nonzero residual RMS")
        for key in ("q", "rhs_exact", "target", "reference"):
            bank[key] = bank[key] / scale.to(bank[key].real.dtype)
        cfg = ExperimentConfig(size=grid, n_angles=12, anisotropy_g=.65, eta_scale=0., regimes="mixed")
        inverses, streams = [], []
        with torch.no_grad():
            for start in range(0, len(bank["q"]), args.batch):
                sl = slice(start, start + args.batch)
                low = make_operator(cfg, *(bank[key][sl].to(device) for key in ("t", "s", "q")), device)
                base = ClassicalTransportStack(low, "tsa", 8, 0.)
                inverse, stream = reference_cache(base)
                inverses.append(inverse.detach().to(storage))
                streams.append(stream.detach().to(storage))
                del base, inverse, stream, low
        bank["inverse"], bank["stream"] = torch.cat(inverses), torch.cat(streams)
        del inverses, streams, scale
        sync(device)
        banks[grid] = bank
        receipts[str(grid)] = dict(seconds=time.perf_counter() - started,
                                    bytes=sum(v.numel() * v.element_size() for v in bank.values()),
                                    memory=memory_receipt(device))
        print(f"cached N{grid}: {receipts[str(grid)]['seconds']:.2f}s, storage={mode}", flush=True)
    return banks, dict(mode=mode, estimated_resident_bytes=estimated, grids=receipts)


def run_arm(args, arm, out, banks, parents, plan, device, deadline, provenance):
    from rte.benchmark_tail_solver import load_model
    from rte.global_source_predictor import FrozenTSA
    from rte.mixed_precision_transport import high_precision_copy
    from rte.run_large_angular_spectral_experiments import ExperimentConfig, make_operator
    from rte.tail_lookahead import continuation_loss
    from rte.train_tail_direction import teacher_direction_loss, validate

    out.mkdir(exist_ok=args.resume)
    reset_rng(args.seed)  # weights-only warm start; optimizer/schedule/RNG reset per arm
    model = load_model(args.warm_start, False, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr * .05)
    history, validations = [], []
    best, best_step, best_state, start, last_validation = None, None, None, 0, None
    timing = dict(training_seconds=0., validation_seconds=0., checkpoint_seconds=0.)
    if args.resume and (out / "last.pt").exists():
        saved = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        if saved["provenance"] != provenance or saved["identity_phase_sha256"] != plan["identity_phase_sha256"]:
            raise ValueError("resume input/source/sampling identity mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved)
        start = saved["step"]
        history, validations = saved["history"], saved["validations"]
        best, best_step, best_state = saved["best_score"], saved["best_step"], saved["best_model"]
        last_validation, timing = saved["last_validation_step"], saved["timing"]
        if saved["complete"]:
            return saved["receipt"]
    operators = {}
    for grid in (64, 128):
        bank = banks[grid]
        cfg = ExperimentConfig(size=grid, n_angles=12, anisotropy_g=.65, eta_scale=0., regimes="mixed")
        low = make_operator(cfg, *(bank[key][:args.batch].to(device) for key in ("t", "s", "q")), device)
        operators[grid] = low, high_precision_copy(low)

    def receipt(step, complete):
        return dict(arm=arm, complete=complete, updates=step, presentations=step * args.batch,
                    presentations_n64=(step if arm == ARMS[0] else (step + 1) // 2) * args.batch,
                    presentations_n128=(0 if arm == ARMS[0] else step // 2) * args.batch,
                    selected_step=best_step, validation_score=best, timing=copy.deepcopy(timing),
                    identity_phase_sha256=tensor_digest(plan["ids"][:step], plan["original_ids"][:step],
                                                       plan["phase_angles"][:step]),
                    memory=memory_receipt(device))

    def checkpoint(step, complete=False, new_best=False):
        sync(device)
        began = time.perf_counter()
        state = dict(args=vars(args), arm=arm, step=step, complete=complete,
                     model=cpu_copy(model.state_dict()), optimizer=cpu_copy(optimizer.state_dict()),
                     scheduler=scheduler.state_dict(), best_model=best_state, best_score=best,
                     best_step=best_step, last_validation_step=last_validation, history=history,
                     validations=validations, timing=copy.deepcopy(timing), provenance=provenance,
                     identity_phase_sha256=plan["identity_phase_sha256"], receipt=receipt(step, complete),
                     **rng_state(plan, step))
        if new_best:
            state["score"] = best  # standard downstream evaluator schema
            save_checkpoint(out / "best.pt", state)
        save_checkpoint(out / "last.pt", state)
        timing["checkpoint_seconds"] += time.perf_counter() - began
        write_rows(out / "history.csv", history)
        write_rows(out / "validation.csv", validations)
        write_rows(out / "sampling_consumed.csv", sampling_rows(plan, arm, step))
        write_json(out / "receipt.json", receipt(step, complete))

    step = start
    try:
        for step in range(start, args.steps + 1):
            if last_validation != step and (step % args.eval_every == 0 or step == args.steps):
                sync(device)
                began = time.perf_counter()
                rows = []
                scores = {}
                for grid in (64, 128):
                    score, records = validate(model, parents[grid], device)
                    scores[str(grid)] = score
                    rows.extend(dict(grid=grid, **row) for row in records)
                sync(device)
                seconds = time.perf_counter() - began
                timing["validation_seconds"] += seconds
                score = selection_score(rows)
                new_best = best is None or score < best
                if new_best:
                    best, best_step, best_state = score, step, cpu_copy(model.state_dict())
                validations.extend(dict(step=step, **row) for row in rows)
                last_validation = step
                checkpoint(step, step == args.steps, new_best)
                print(f"{arm} step {step}: pooled={score}, per_grid={scores}, best={best_step}, validation={seconds:.2f}s", flush=True)
            if step == args.steps:
                return receipt(step, True)
            if deadline is not None and time.perf_counter() >= deadline:
                checkpoint(step)
                return receipt(step, False)
            sync(device)
            began = time.perf_counter()
            grid = 128 if arm == ARMS[1] and step % 2 else 64
            bank = banks[grid]
            ids = plan["ids"][step].to(bank["q"].device)
            t, s, r, ref, target, basis, inverse, stream = [bank[key][ids].to(device) for key in
                ("t", "s", "rhs_exact", "reference", "target", "image_basis", "inverse", "stream")]
            angle = plan["phase_angles"][step]
            phase = torch.polar(torch.ones_like(angle), angle).to(device)
            r, target = r * phase[:, None, None, None], target * phase[:, None, None, None]
            ref = (ref * phase[:, None, None, None]).to(torch.complex64)
            basis = basis * phase[:, None, None, None, None]
            low, high = operators[grid]
            low.setup(t, s, r.to(torch.complex64))
            high.setup(t.double(), s.double(), r)
            preconditioner = FrozenTSA(low, inverse, stream, 8)
            precondition = lambda value: preconditioner(value.to(torch.complex64)).to(torch.complex128)
            model.train()
            proposal = model(t, s, r.to(torch.complex64), ref).to(torch.complex128)
            ratios = continuation_loss(high.A_op, precondition, proposal, r, basis, 2)
            supervised = teacher_direction_loss(proposal, target).mean()
            loss = ratios.clamp_min(1e-12).log().mean() + .05 * supervised
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite lookahead loss; preceding checkpoint retained")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            sync(device)
            seconds = time.perf_counter() - began
            timing["training_seconds"] += seconds
            history.append(dict(step=step + 1, grid=grid, loss=float(loss), teacher_loss=float(supervised),
                                injection_ratio=float(ratios[:, 0].mean()), final_ratio=float(ratios[:, -1].mean()),
                                grad=float(grad), lr=optimizer.param_groups[0]["lr"], seconds=seconds,
                                cumulative_training_seconds=timing["training_seconds"]))
            if (step + 1) % 20 == 0:
                write_rows(out / "history.csv", history)
                print(f"{arm} update {step + 1}/{args.steps}: N{grid} loss={float(loss):.5f}, train={timing['training_seconds']:.2f}s", flush=True)
    finally:
        del optimizer, scheduler, model, operators
        torch.cuda.empty_cache()


def validate_arguments(args):
    if args.steps < 0 or args.batch < 1 or args.eval_every < 1 or args.lr <= 0:
        raise ValueError("steps>=0, batch>=1, eval-every>=1 and lr>0 required")
    if args.max_wall_seconds is not None and args.max_wall_seconds <= 0:
        raise ValueError("positive wall budget required")
    if args.steps and args.batch > 128:
        raise ValueError("batch larger than the original 128 training draws")


def run(args):
    validate_arguments(args)
    torch.set_num_threads(1)
    wall_start = time.perf_counter()
    deadline = wall_start + args.max_wall_seconds if args.max_wall_seconds is not None else None
    out = Path(args.output)
    if args.resume:
        if not (out / "settings.json").is_file():
            raise ValueError("--resume requires an existing study settings.json")
    else:
        out.mkdir(parents=True, exist_ok=False)
    paths = dict(warm_start=args.warm_start, train_context_n64=args.context_n64,
                 train_context_n128=args.context_n128, validation_parent_n64=args.validation_n64,
                 validation_parent_n128=args.validation_n128, train_parent_n64=args.train_parent_n64,
                 train_parent_n128=args.train_parent_n128)
    settings = dict(args=vars(args), complete=False, scope="training and validation only; zero test access",
                    selection="all16 each of N64/N128: pooled failures, mean(classical calls + .5*injection calls), mean true residual",
                    protocol="two lookahead TSA8 continuation steps + .05 teacher loss; AdamW decay1e-5; cosine LR to5%; same warm weights, RNG reset and ID/phase schedule",
                    limitation="contexts and selection use TSA8 with balanced initialization; strongest prior N256 classical candidate was TSA16-zero; this is a controlled training ablation, not a speed claim",
                    planned_presentations_per_arm=args.steps * args.batch,
                    authoritative_full_protocol=(args.steps == 1000 and args.batch == 4 and args.eval_every == 200 and args.lr == .0003 and args.seed == 27002),
                    input_paths=paths, torch_version=str(torch.__version__))
    began = time.perf_counter()
    settings["input_sha256"] = {name: file_digest(path) for name, path in paths.items()}
    import importlib
    settings["source_sha256"] = {str(Path(__file__).resolve()): file_digest(__file__)}
    for module_name in SOURCE_MODULES:
        module = importlib.import_module("rte." + module_name)
        settings["source_sha256"][str(Path(module.__file__).resolve())] = file_digest(module.__file__)
    contexts = {64: torch.load(args.context_n64, map_location="cpu", weights_only=True),
                128: torch.load(args.context_n128, map_location="cpu", weights_only=True)}
    parents = {64: torch.load(args.validation_n64, map_location="cpu", weights_only=True),
               128: torch.load(args.validation_n128, map_location="cpu", weights_only=True)}
    settings["paired_identity"] = dict(train=check_paired_identity(contexts[64], contexts[128], contexts=True),
                                       validation=check_paired_identity(parents[64], parents[128], contexts=False, expected_count=16))
    for grid in (64, 128):
        if contexts[grid]["parent_sha256"] != settings["input_sha256"][f"train_parent_n{grid}"]:
            raise ValueError(f"N{grid} context ancestry SHA256 mismatch with supplied training parent")
    warm = torch.load(args.warm_start, map_location="cpu", weights_only=True)
    for key in ("width", "blocks", "modes"):
        setattr(args, key, warm["args"][key])
    settings["architecture"] = dict(name="TailResponseNetwork", **{key: getattr(args, key) for key in ("width", "blocks", "modes")})
    del warm
    plan = build_sampling_plan(contexts[64]["bank"]["indices"], args.steps, args.batch, args.seed)
    settings["shared_identity_phase_sha256"] = plan["identity_phase_sha256"]
    settings["input_hash_load_identity_seconds"] = time.perf_counter() - began
    provenance = dict(input_sha256=settings["input_sha256"], source_sha256=settings["source_sha256"],
                      architecture=settings["architecture"], steps=args.steps, batch=args.batch,
                      seed=args.seed, lr=args.lr, eval_every=args.eval_every, cache_mode=args.cache_mode)
    if args.resume:
        old = json.loads((out / "settings.json").read_text())
        if old["provenance"] != provenance or old["shared_identity_phase_sha256"] != plan["identity_phase_sha256"]:
            raise ValueError("resume protocol, input or source hash changed")
        if old.get("complete"):
            raise ValueError("study already complete; refusing to overwrite it")
        settings["previous_receipts"] = old.get("arm_receipts", [])
    else:
        save_checkpoint(out / "shared_sampling_plan.pt", plan)
        for arm in ARMS:
            write_rows(out / f"sampling_planned_{arm}.csv", sampling_rows(plan, arm))
    settings["provenance"] = provenance
    settings["sampling_plan_file_sha256"] = file_digest(out / "shared_sampling_plan.pt")
    print("paired training/validation identity verified; shared ID/phase schedule " + plan["identity_phase_sha256"], flush=True)
    write_json(out / "settings.json", settings)
    if args.identity_only:
        settings.update(identity_only=True, wall_seconds=time.perf_counter() - wall_start)
        write_json(out / "settings.json", settings)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; this script is for the authorized FWI GPU, not local CPU training")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    settings.update(gpu=torch.cuda.get_device_name(device), gpu_memory_before=memory_receipt(device))
    write_json(out / "settings.json", settings)
    banks, cache = normalize_and_cache(contexts, args, device)
    del contexts
    settings["reference_cache"] = cache
    settings["arm_receipts"] = []
    write_json(out / "settings.json", settings)
    try:
        for arm in ARMS:
            if deadline is not None and time.perf_counter() >= deadline:
                settings["stop_reason"] = "wall budget reached before next arm"
                break
            result = run_arm(args, arm, out / arm, banks, parents, plan, device, deadline, provenance)
            settings["arm_receipts"].append(result)
            write_json(out / "settings.json", settings)
            if not result["complete"]:
                settings["stop_reason"] = "wall budget reached at an optimizer-step boundary"
                break
        settings["complete"] = len(settings["arm_receipts"]) == 2 and all(row["complete"] for row in settings["arm_receipts"])
        if settings["complete"]:
            a, b = settings["arm_receipts"]
            if a["presentations"] != b["presentations"] or a["identity_phase_sha256"] != b["identity_phase_sha256"]:
                raise RuntimeError("completed arms did not consume identical ID/phase presentations")
            settings["sampling_equality_verified"] = True
            settings["comparison"] = dict(selected_scores={row["arm"]: row["validation_score"] for row in (a, b)},
                                          selected_steps={row["arm"]: row["selected_step"] for row in (a, b)},
                                          interpretation="validation-only controlled resolution ablation; no test generalization or elapsed solve speed claim")
    except BaseException as error:
        settings["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        settings.update(wall_seconds=time.perf_counter() - wall_start, gpu_memory_after=memory_receipt(device))
        write_json(out / "settings.json", settings)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", required=True)
    result.add_argument("--context-n64", default="rte/workstation_runs/tail_injection_gate_20260926/train_context.pt")
    result.add_argument("--context-n128", default="rte/workstation_runs/tail_multiresolution_bank_20260926/train_context.pt")
    result.add_argument("--validation-n64", default="rte/workstation_runs/source_tail_validation_20260926/validation.pt")
    result.add_argument("--validation-n128", default="rte/workstation_runs/tail_multiresolution_bank_20260926/validation.pt")
    result.add_argument("--train-parent-n64", default="rte/workstation_runs/source_predictor_pilot_20260926/train.pt")
    result.add_argument("--train-parent-n128", default="rte/workstation_runs/tail_multiresolution_bank_20260926/train.pt")
    result.add_argument("--warm-start", default="rte/workstation_runs/tail_direction_pilot_20260926/best.pt")
    result.add_argument("--steps", type=int, default=1000)
    result.add_argument("--eval-every", type=int, default=200)
    result.add_argument("--batch", type=int, default=4)
    result.add_argument("--lr", type=float, default=.0003)
    result.add_argument("--seed", type=int, default=27002)
    result.add_argument("--cache-mode", choices=("auto", "gpu", "cpu"), default="auto")
    result.add_argument("--max-wall-seconds", type=float, help="soft whole-run budget; never interrupts a GPU operation/validation sweep")
    result.add_argument("--identity-only", action="store_true", help="CPU hashes, identity and sampling checks only; no training")
    result.add_argument("--resume", action="store_true", help="explicitly resume an incomplete output root with identical hashes/protocol")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
