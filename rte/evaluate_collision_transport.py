"""No-training, paired evaluator for collision/transport checkpoints.

Classical settings are selected on validation media (or loaded from a saved
tuning CSV).  Test media are then evaluated in a randomized method order.  The
raw cold and warm repetitions, operator/preconditioner calls, failures, setup
cost, and memory observations are retained instead of being collapsed into one
ambiguous "first solve" estimate.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from rte.collision_transport import CollisionTransportPreconditioner
from rte.residual_subspace import physical_residual_loss
from rte.run_large_angular_spectral_experiments import (
    ExperimentConfig, dataset_fingerprint, make_operator, setup_batch, source_from_data,
)
from rte.run_residual_subspace_experiment import measured_solver, synchronized, write_csv
from rte.train_collision_transport_ensemble import material_split
from rte.transport_baselines import ClassicalTransportStack


CLASSICAL_FAMILIES = ("hss", "tsa", "dsa", "dsa_transport", "reference")


def percentile(values, probability):
    """Linear-interpolated percentile without a NumPy dependency."""
    values = sorted(float(value) for value in values)
    if not values:
        return None
    position = (len(values) - 1) * float(probability)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    if not math.isfinite(values[high]):
        return values[high]
    return values[low] * (high - position) + values[high] * (position - low)


def resident_bytes():
    """Current Linux RSS, if available (not the process-wide historical peak)."""
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def memory_start(device):
    synchronized(device)
    rss = resident_bytes()
    if device.type != "cuda":
        return {"rss_before_bytes": rss}
    torch.cuda.reset_peak_memory_stats(device)
    return {"rss_before_bytes": rss,
            "cuda_allocated_before_bytes": torch.cuda.memory_allocated(device),
            "cuda_reserved_before_bytes": torch.cuda.memory_reserved(device)}


def memory_finish(device, before):
    synchronized(device)
    rss = resident_bytes()
    result = {**before, "rss_after_bytes": rss}
    if rss is not None and before["rss_before_bytes"] is not None:
        result["rss_delta_bytes"] = rss - before["rss_before_bytes"]
    if device.type == "cuda":
        result.update({
            "cuda_allocated_after_bytes": torch.cuda.memory_allocated(device),
            "cuda_reserved_after_bytes": torch.cuda.memory_reserved(device),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "cuda_peak_allocated_delta_bytes": max(
                0, torch.cuda.max_memory_allocated(device) - before["cuda_allocated_before_bytes"]),
        })
    return result


def checkpoint_state(payload):
    """Return validation-selected weights from either best.pt or last.pt."""
    if payload.get("best_state") is not None:
        return payload["best_state"], int(payload["best_epoch"])
    return payload["model"], int(payload.get("selected_epoch", payload.get("epoch", 0)))


def candidate_specs():
    specs = []
    for depth in (1, 2, 4):
        for scale in (.5, 1., 2., 4.):
            specs.append({"family": "hss", "depth": depth, "scale": scale})
    for family in CLASSICAL_FAMILIES[1:]:
        for depth in (1, 2, 4):
            for scale in (0., .03, .1):
                specs.append({"family": family, "depth": depth, "scale": scale})
    return specs


def method_label(spec):
    return "learned_CT" if spec["family"] == "learned_CT" else spec["family"] + "_tuned"


def make_method(spec, rte, learned):
    family, depth = spec["family"], int(spec.get("depth", 4))
    if family == "learned_CT":
        return learned
    scale = float(spec.get("scale", 0.))
    if family == "hss":
        return CollisionTransportPreconditioner(rte, depth=depth, shift_factor=scale).eval()
    return ClassicalTransportStack(rte, family, depth, scale).eval()


def action_metrics(rte, method, source):
    """One-step loss with every nested physical-operator call counted."""
    original = rte.A_op
    calls = 0

    def operator(value):
        nonlocal calls
        calls += 1
        return original(value)

    rte.A_op = operator
    try:
        with torch.no_grad():
            correction = method(source)
            loss = float(physical_residual_loss(rte.A_op, source, correction))
    finally:
        rte.A_op = original
    return loss, calls


def work_fields(method, row):
    preconditioner_calls = int(row.get("preconditioner_calls", 0))
    fields = {}
    if hasattr(method, "factor_stages_per_apply"):
        fields["factor_stage_calls"] = preconditioner_calls * int(method.factor_stages_per_apply)
    if isinstance(method, ClassicalTransportStack):
        fields["classical_stage_calls"] = preconditioner_calls * int(method.depth)
    if getattr(method, "base_calls_per_apply", 0):
        fields["base_calls"] = preconditioner_calls * int(method.base_calls_per_apply)
    return fields


def evaluate_method(spec, rte, source, solver_args, device, learned, repeats, phase, index, order):
    """Measure one method/medium pair and preserve all raw solve repetitions."""
    label = method_label(spec)
    common = {"phase": phase, "index": int(index), "method_order": int(order), "method": label,
              "family": spec["family"], "depth": spec.get("depth", ""), "scale": spec.get("scale", "")}
    before = memory_start(device)
    synchronized(device)
    start = time.perf_counter()
    try:
        method = make_method(spec, rte, learned)
        if hasattr(method, "prepare"):
            method.prepare()
        synchronized(device)
        setup_seconds = time.perf_counter() - start
    except Exception as error:
        synchronized(device)
        case = {**common, "setup_seconds": time.perf_counter() - start,
                "setup_failure": f"{type(error).__name__}: {error}",
                **memory_finish(device, before)}
        return case, []

    samples = []
    for repeat_kind, count in (("cold", 1), ("warm", int(repeats))):
        for repeat in range(count):
            sample = {**common, "repeat_kind": repeat_kind, "repeat": repeat}
            try:
                row = measured_solver(rte, method, source, solver_args)
                converged = math.isfinite(row["final"]) and row["final"] <= solver_args.tol
                sample.update(row)
                sample.update(work_fields(method, row))
                sample["converged"] = converged
                sample["time_to_tolerance_seconds"] = row["seconds"] if converged else ""
                if not converged:
                    sample["solve_failure"] = "nonfinite_residual" if not math.isfinite(row["final"]) else "iteration_budget"
            except Exception as error:
                sample.update({"converged": False, "solve_failure": f"{type(error).__name__}: {error}"})
            samples.append(sample)

    try:
        source_loss, source_operator_calls = action_metrics(rte, method, source)
        action = {"source_loss": source_loss, "source_action_preconditioner_calls": 1,
                  "source_action_operator_calls": source_operator_calls}
    except Exception as error:
        action = {"source_action_failure": f"{type(error).__name__}: {error}"}

    cold = samples[0]
    warm = samples[1:]
    successful_warm = [row for row in warm if row.get("converged")]
    finite_warm = [row for row in warm if isinstance(row.get("final"), (int, float)) and math.isfinite(row["final"])]
    case = {**common, "setup_seconds": setup_seconds,
            "cold_solve_seconds": cold.get("seconds", ""),
            "cold_end_to_end_seconds": setup_seconds + cold["seconds"] if "seconds" in cold else "",
            "cold_converged": cold.get("converged", False),
            "cold_final": cold.get("final", ""), "cold_iters": cold.get("iters", ""),
            "warm_repeats": len(warm), "warm_failures": sum(not row.get("converged", False) for row in warm),
            "warm_median_seconds": statistics.median(row["seconds"] for row in warm if "seconds" in row) if any("seconds" in row for row in warm) else "",
            "warm_p95_seconds": percentile([row["seconds"] for row in warm if "seconds" in row], .95) or "",
            "warm_median_time_to_tolerance_seconds": statistics.median(row["seconds"] for row in successful_warm) if successful_warm else "",
            "warm_p95_time_to_tolerance_seconds": percentile([row["seconds"] for row in successful_warm], .95) or "",
            "warm_median_final": statistics.median(row["final"] for row in finite_warm) if finite_warm else "",
            "warm_median_iters": statistics.median(row["iters"] for row in finite_warm) if finite_warm else "",
            **action, **memory_finish(device, before)}
    return case, samples


def candidate_score(rows, tol):
    failures = sum(row.get("warm_failures", 1) > 0 or not isinstance(row.get("warm_median_final"), (int, float))
                   or row["warm_median_final"] > tol for row in rows)
    finals = [row["warm_median_final"] for row in rows
              if isinstance(row.get("warm_median_final"), (int, float)) and math.isfinite(row["warm_median_final"])]
    quality = statistics.mean(math.log10(max(tol, value)) for value in finals) if failures and finals else (100. if failures else 0.)
    times = [row["warm_median_seconds"] for row in rows if isinstance(row.get("warm_median_seconds"), (int, float))]
    return failures, quality, statistics.mean(times) if times else float("inf")


def tune_classical(rte, validation, cfg, args, device, learned, out):
    specs = candidate_specs()
    cases, samples = [], []
    generator = torch.Generator().manual_seed(args.seed + 7100)
    count = min(args.validation_count, len(validation["sigma_t"]))
    for index in range(count):
        source = setup_batch(cfg, rte, validation, torch.tensor([index]), device)
        for order, position in enumerate(torch.randperm(len(specs), generator=generator).tolist()):
            case, raw = evaluate_method(specs[position], rte, source, args, device, learned,
                                        args.tuning_repeats, "validation", index, order)
            cases.append(case)
            samples.extend(raw)
        write_csv(out / "classical_tuning_cases.csv", cases)
        write_csv(out / "classical_tuning_samples.csv", samples)
        print(f"validation tuning medium {index + 1}/{count}", flush=True)

    summary, selected = [], {}
    for spec in specs:
        rows = [row for row in cases if row["family"] == spec["family"]
                and int(row["depth"]) == spec["depth"] and float(row["scale"]) == spec["scale"]]
        score = candidate_score(rows, args.tol)
        summary.append({**spec, "validation_failures": score[0], "validation_quality": score[1],
                        "validation_seconds": score[2], "selected": False})
    for family in CLASSICAL_FAMILIES:
        choice = min((row for row in summary if row["family"] == family),
                     key=lambda row: (row["validation_failures"], row["validation_quality"], row["validation_seconds"]))
        choice["selected"] = True
        selected[family] = {"family": family, "depth": int(choice["depth"]), "scale": float(choice["scale"])}
        print(f"selected {family}: depth={choice['depth']} scale={choice['scale']}", flush=True)
    write_csv(out / "classical_tuning.csv", summary)
    return selected


def load_tuning(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = {}
    for family in CLASSICAL_FAMILIES:
        candidates = [row for row in rows if row.get("family") == family]
        explicitly_selected = [row for row in candidates if row.get("selected", "").lower() in ("true", "1")]
        if explicitly_selected:
            choice = explicitly_selected[0]
        elif candidates:
            choice = min(candidates, key=lambda row: (int(float(row["validation_failures"])),
                         float(row["validation_quality"]), float(row["validation_seconds"])))
        else:
            raise ValueError(f"tuning CSV has no {family} candidates")
        selected[family] = {"family": family, "depth": int(float(choice["depth"])),
                            "scale": float(choice["scale"])}
    return selected


def aggregate_summary(cases, samples, tol):
    summary = []
    for method in dict.fromkeys(row["method"] for row in cases):
        method_cases = [row for row in cases if row["method"] == method]
        warm = [row for row in samples if row["method"] == method and row["repeat_kind"] == "warm"]
        successful = [row for row in warm if row.get("converged")]
        setup_failures = sum(bool(row.get("setup_failure")) for row in method_cases)
        censored_times = [row["seconds"] if row.get("converged") else float("inf") for row in warm]
        # A failed setup censors every intended warm attempt, not just one.
        planned_repeats = max((row.get("warm_repeats", 0) for row in method_cases), default=0) or 1
        censored_times.extend([float("inf")] * setup_failures * planned_repeats)
        all_p95 = percentile(censored_times, .95)
        numeric = lambda key, rows=warm: [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        summary.append({
            "method": method, "media": len(method_cases), "warm_solves": len(warm),
            "setup_failures": setup_failures,
            "cold_failures": sum(not row.get("cold_converged", False) for row in method_cases),
            "failed_media": sum(bool(row.get("setup_failure")) or row.get("warm_failures", 0) > 0 for row in method_cases),
            "failures": setup_failures + sum(not row.get("converged", False) for row in warm), "tolerance": tol,
            "mean_iterations": statistics.mean(numeric("iters")) if numeric("iters") else None,
            "mean_final_residual": statistics.mean(numeric("final")) if numeric("final") else None,
            "median_time_to_tolerance_seconds": percentile([row["seconds"] for row in successful], .5),
            "p95_time_to_tolerance_seconds": percentile([row["seconds"] for row in successful], .95),
            "p95_all_attempts_time_to_tolerance_seconds": all_p95 if all_p95 is not None and math.isfinite(all_p95) else None,
            "p95_all_attempts_unreached": all_p95 is not None and not math.isfinite(all_p95),
            "median_setup_seconds": percentile(numeric("setup_seconds", method_cases), .5),
            "median_cold_solve_seconds": percentile(numeric("cold_solve_seconds", method_cases), .5),
            "p95_cold_solve_seconds": percentile(numeric("cold_solve_seconds", method_cases), .95),
            "mean_operator_calls": statistics.mean(numeric("operator_calls")) if numeric("operator_calls") else None,
            "mean_preconditioner_calls": statistics.mean(numeric("preconditioner_calls")) if numeric("preconditioner_calls") else None,
            "peak_cuda_allocated_bytes": max(numeric("cuda_peak_allocated_bytes", method_cases), default=None),
        })
    return summary


def run(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    weights, selected_epoch = checkpoint_state(payload)
    saved = payload["args"]
    args.seed = int(saved["seed"])
    for key in ("size", "n_angles", "g", "regimes", "train_size", "val_size", "test_size", "batch_size"):
        setattr(args, key, saved[key])
    cfg = ExperimentConfig(dataset_size=args.train_size, eval_size=args.val_size, test_size=args.test_size,
                           size=args.size, n_angles=args.n_angles, anisotropy_g=args.g, regimes=args.regimes,
                           batch_size=args.batch_size, eta_scale=0.)
    split_args = SimpleNamespace(**{**saved, "dataset_dir": args.dataset_dir})
    validation = material_split(split_args, cfg, "validation")
    test = material_split(split_args, cfg, "test")
    validation_fingerprint, test_fingerprint = dataset_fingerprint(validation), dataset_fingerprint(test)
    expected_validation = payload.get("fingerprints", {}).get("validation")
    if expected_validation and validation_fingerprint != expected_validation:
        raise ValueError("validation data do not match checkpoint fingerprint")
    published_test_fingerprint = checkpoint_path.parent / "test_fingerprint.txt"
    if published_test_fingerprint.exists() and test_fingerprint != published_test_fingerprint.read_text().strip():
        raise ValueError("test data do not match published fingerprint")

    initial = torch.tensor([0])
    q = source_from_data(cfg, validation, initial, device)
    rte = make_operator(cfg, validation["sigma_t"][initial], validation["sigma_s"][initial], q, device)
    learned = CollisionTransportPreconditioner(rte, depth=4, learned=True, width=12).to(device).eval()
    learned.load_state_dict(weights, strict=True)

    settings = {**vars(args), "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "selected_epoch": selected_epoch, "validation_fingerprint": validation_fingerprint,
                "test_fingerprint": test_fingerprint, "torch": str(torch.__version__),
                "cuda": torch.version.cuda, "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "method_order": "randomized_within_each_medium", "schema": 2,
                "cache_guard": "setup_generation_and_retained_input_tensors_v3",
                "timing_scope": "existing process, resident weights, and already constructed physical operator; method preparation plus first solve for a medium",
                "excluded_timing": ["process/CUDA initialization", "checkpoint loading", "source transfer", "physical operator construction/setup"],
                "legacy_cold_label": "first solve after method preparation; not process-cold end-to-end latency",
                "percentile_scope": "legacy time-to-tolerance percentiles condition on success; all-attempt p95 reports unreached when censored",
                "memory_scope": "CUDA allocator peak for setup/solves/source-loss action includes live shared objects; CPU RSS is before/after, not peak",
                "tuning_sha256": hashlib.sha256(Path(args.tuning).read_bytes()).hexdigest() if args.tuning else None}
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    if args.tuning:
        selected = load_tuning(args.tuning)
    else:
        args.validation_count = min(args.validation_count, len(validation["sigma_t"]))
        selected = tune_classical(rte, validation, cfg, args, device, learned, out)
    specs = [{"family": "learned_CT", "depth": 4, "scale": ""}] + [selected[name] for name in CLASSICAL_FAMILIES]

    cases, samples = [], []
    generator = torch.Generator().manual_seed(args.seed + 6100)
    count = len(test["sigma_t"]) if args.test_count == 0 else min(args.test_count, len(test["sigma_t"]))
    for index in range(count):
        source = setup_batch(cfg, rte, test, torch.tensor([index]), device)
        for order, position in enumerate(torch.randperm(len(specs), generator=generator).tolist()):
            case, raw = evaluate_method(specs[position], rte, source, args, device, learned,
                                        args.repeats, "test", index, order)
            cases.append(case)
            samples.extend(raw)
        write_csv(out / "results.csv", cases)
        write_csv(out / "solve_samples.csv", samples)
        print(f"test medium {index + 1}/{count}", flush=True)
    summary = aggregate_summary(cases, samples, args.tol)
    write_csv(out / "summary.csv", summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="best.pt or last.pt; validation-selected weights are used")
    parser.add_argument("--dataset-dir", default="", help="Optional verified handoff snapshot directory")
    parser.add_argument("--tuning", default="", help="Optional prior classical_tuning.csv; otherwise retune on validation")
    parser.add_argument("--validation-count", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=0, help="Zero evaluates the complete saved/generated test split")
    parser.add_argument("--tuning-repeats", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--solver-steps", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0, help="Replaced by the checkpoint seed; retained in provenance")
    parser.add_argument("--output", default="rte/workstation_runs/ct_evaluation")
    args = parser.parse_args()
    if min(args.validation_count, args.tuning_repeats, args.repeats, args.solver_steps, args.threads) < 1 or args.test_count < 0:
        parser.error("counts, repeats, steps and threads must be positive (test-count may be zero)")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
