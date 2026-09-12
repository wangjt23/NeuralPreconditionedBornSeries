"""Reproduce the three supplemental fixed-medium checks from saved checkpoints.

Run with bytecode disabled to write only the three analysis CSVs:
    python -B -m rte.analyze_residual_bank
    python -B -m rte.analyze_residual_bank --check

Bank components include individual source/stage entries, pooled stages,
trajectory, probes and all entries. Validation is the original 'val' split.
Equal-budget results are prefixes of the same 12-iteration FGMRES evaluation
used by the experiment, not independent time-to-tolerance measurements.
Matched witnesses give numerical lower bounds, never uniform norm estimates.
Assertions check recombination, source pairing, recorded results, checkpoint
selection, and equal base-call budgets. No training or source-file edits occur.
"""

import argparse
import csv
from functools import partial
import hashlib
import io
import json
import math
from pathlib import Path

import torch

from rte.npbs_unet_ablation import (
    NPBSUNetPreconditioner, inner_base_apply, solve_fgmres,
)
from rte.probe_residual_amplification import residual_power_vector
from rte.residual_subspace import squared_norm
from rte.run_hard_case_experiments import make_case
from rte.run_residual_subspace_experiment import residual_bank, sources


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "rte/results/residual_subspace_mixed"
# These are the fixed bank-generation constants in the saved experiment.
SPLITS = (("train", 8, 1000, 4000), ("validation", 2, 2000, 5000),
          ("test", 3, 3000, 6000))
ROLLOUT_STAGES = 6
RANDOM_PROBES = 16


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    signature = f"{array.dtype}:{array.shape}:".encode()
    return hashlib.sha256(signature + array.tobytes()).hexdigest()


def close(actual, expected, context, atol=1e-7, rtol=1e-5):
    assert math.isfinite(float(actual)), (context, actual)
    assert math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=rtol), (
        context, actual, expected)


def unique_row(rows, **criteria):
    matches = [row for row in rows if all(row.get(k) == str(v) for k, v in criteria.items())]
    assert len(matches) == 1, (criteria, len(matches))
    return matches[0]


@torch.no_grad()
def losses(rte, method, bank):
    values = []
    for batch in bank.split(8):
        error = batch - rte.A_op(method(batch))
        values.append(squared_norm(error) / squared_norm(batch))
    return torch.cat(values).double()


def stats(values):
    ratio = values.sqrt()
    return {"sample_count": len(values), "physical_loss": float(values.mean()),
            "one_step_mean": float(ratio.mean()), "one_step_max": float(ratio.max())}


def analyze(directory):
    settings_path = directory / "settings.json"
    settings = json.loads(settings_path.read_text())
    results_path = directory / "results.csv"
    reference = read_csv(results_path)
    amplification_path = directory / "amplification.csv"
    amplification = read_csv(amplification_path)
    assert torch.is_grad_enabled(), "analysis requires gradients for witness search"
    assert settings["solver_steps"] == 12, "this analysis matches the recorded 12-step experiment"
    assert "legacy_raw" in settings["models"]
    source_hashes = {p: file_hash(p) for p in (settings_path, results_path, amplification_path)}
    bank_rows, witness_rows, budget_rows = [], [], []
    largest_bank_difference = 0.0
    for seed in settings["seeds"]:
        checkpoint_path = directory / f"{settings['case']}_{seed}_legacy_raw.pt"
        source_hashes[checkpoint_path] = file_hash(checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        args = state["args"]
        assert args == settings, "checkpoint arguments differ from settings.json"
        torch.set_num_threads(args["threads"])
        torch.manual_seed(seed)
        device = torch.device(args["device"])
        rte, _ = make_case(args["case"], args["size"], args["n_angles"],
                           torch.float32, .1, device=device)
        # Architecture constants match run_residual_subspace_experiment.train.
        model = NPBSUNetPreconditioner(rte, form="defect2", base_kind=args["base_kind"],
                                      width=16, levels=3, output_init="zero").to(device)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        assert model.base_calls_per_apply == 2
        methods = {"base2": partial(inner_base_apply, rte, depth=2, kind=args["base_kind"]),
                   "base4": partial(inner_base_apply, rte, depth=4, kind=args["base_kind"]),
                   "legacy_raw": model}
        metadata = {"case": args["case"], "size": args["size"], "n_angles": args["n_angles"],
                    "base_kind": args["base_kind"], "seed": seed,
                    "legacy_selected_step": state["selected_step"],
                    "checkpoint": checkpoint_path.name,
                    "checkpoint_sha256": source_hashes[checkpoint_path],
                    "results_sha256": source_hashes[results_path],
                    "settings_sha256": source_hashes[settings_path],
                    "torch_version": str(torch.__version__), "device": str(device)}
        test_sources = None
        for split, count, source_offset, bank_offset in SPLITS:
            source_list = sources(args["size"], args["n_angles"], count, seed + source_offset, device)
            bank = residual_bank(rte, source_list, args["base_kind"], seed + bank_offset)
            trajectory_count = count * ROLLOUT_STAGES
            assert len(bank) == trajectory_count + RANDOM_PROBES
            torch.testing.assert_close(squared_norm(bank), torch.ones(len(bank), device=device),
                                       atol=1e-6, rtol=1e-6)
            for index, source in enumerate(source_list):
                normalized = source / squared_norm(source).sqrt().view(-1, 1, 1, 1)
                torch.testing.assert_close(bank[index * ROLLOUT_STAGES:index * ROLLOUT_STAGES + 1],
                                           normalized, atol=1e-6, rtol=1e-6)
            if split == "test":
                test_sources = source_list
            for name, method in methods.items():
                value = losses(rte, method, bank)
                trajectory = value[:trajectory_count]
                probes = value[trajectory_count:]
                recombined = (trajectory.mean() * trajectory_count + probes.mean() * RANDOM_PROBES) / len(bank)
                close(value.mean(), recombined, "trajectory/probe recombination", atol=1e-12, rtol=1e-12)
                stage_means = torch.stack([trajectory[stage::ROLLOUT_STAGES].mean()
                                           for stage in range(ROLLOUT_STAGES)])
                close(trajectory.mean(), stage_means.mean(), "stage recombination", atol=1e-12, rtol=1e-12)
                source_means = trajectory.reshape(count, ROLLOUT_STAGES).mean(1)
                close(trajectory.mean(), source_means.mean(), "source recombination", atol=1e-12, rtol=1e-12)
                saved = unique_row(reference, seed=seed, method=name, solver="fgmres", test_source=0)
                close(value.mean(), saved[split + "_physical_loss"], f"saved {seed}/{split}/{name}")
                largest_bank_difference = max(largest_bank_difference,
                                               abs(float(value.mean()) - float(saved[split + "_physical_loss"])))
                if name == "legacy_raw":
                    assert int(saved["selected_step"]) == state["selected_step"]
                common = {**metadata, "split": split, "method": name,
                          "bank_sha256": tensor_hash(bank), "source_seed": seed + source_offset,
                          "bank_seed": seed + bank_offset}
                for index, source in enumerate(source_list):
                    for stage in range(ROLLOUT_STAGES):
                        offset = index * ROLLOUT_STAGES + stage
                        bank_rows.append({**common, "component": "source_stage", "source": index,
                                          "source_sha256": tensor_hash(source), "stage": stage,
                                          **stats(value[offset:offset + 1])})
                for stage in range(ROLLOUT_STAGES):
                    bank_rows.append({**common, "component": "stage", "source": "", "source_sha256": "",
                                      "stage": stage, **stats(trajectory[stage::ROLLOUT_STAGES])})
                for component, subset in (("trajectory", trajectory), ("probes", probes), ("all", value)):
                    bank_rows.append({**common, "component": component, "source": "", "source_sha256": "",
                                      "stage": "", **stats(subset)})

        saved_probe = unique_row(amplification, seed=seed, method="legacy_raw")
        assert int(saved_probe["selected_step"]) == state["selected_step"]
        power_steps = int(saved_probe["power_steps"])
        witness = residual_power_vector(rte.A_op, model, rte.source, seed + 7000, power_steps)
        for name, method in methods.items():
            with torch.no_grad():
                error = witness - rte.A_op(method(witness))
                one = float(error.norm() / witness.norm())
                error = witness
                for _ in range(12):
                    error = error - rte.A_op(method(error))
                twelve = float(error.norm() / witness.norm())
            if name == "legacy_raw":
                close(one, saved_probe["witnessed_one_step_ratio"], "saved witness one-step")
                close(twelve, saved_probe["witnessed_stationary_12_ratio"], "saved witness twelve-step")
            witness_rows.append({**metadata, "method": name, "witness_method": "legacy_raw",
                                 "witness_seed": seed + 7000, "power_steps": power_steps,
                                 "witness_sha256": tensor_hash(witness),
                                 "amplification_sha256": source_hashes[amplification_path],
                                 "one_step_ratio": one, "stationary_12_ratio": twelve,
                                 "interpretation": "finite_witness_not_uniform_bound"})

        assert test_sources is not None and len(test_sources) == 3
        for index, source in enumerate(test_sources):
            for name, method in methods.items():
                saved = unique_row(reference, seed=seed, method=name, solver="fgmres", test_source=index)
                result = solve_fgmres(rte, source, method, args["solver_steps"], args["tol"], real_stacked=True)
                iters, final, _, _, history, times, calls = result
                assert iters == int(saved["iters"]) == args["solver_steps"]
                close(final, saved["final"], "full-budget saved FGMRES", atol=1e-9, rtol=1e-4)
                with torch.no_grad():
                    source_one = float((source - rte.A_op(method(source))).norm() / source.norm())
                close(source_one, saved["test_source_one_step"], "paired source one-step")
                depth = 4 if name == "base4" else 2
                snapshot = 6 if name == "base4" else 12
                assert calls[snapshot] == snapshot
                assert depth * calls[snapshot] == 24
                budget_rows.append({**metadata, "test_source": index, "source_seed": seed + 3000,
                                    "source_sha256": tensor_hash(source), "method": name,
                                    "base_calls_per_apply": depth, "iters": snapshot,
                                    "preconditioner_calls": calls[snapshot], "base_calls": depth * calls[snapshot],
                                    "final": history[snapshot], "trace_elapsed_seconds": times[snapshot],
                                    "tol": args["tol"], "hit_tol": history[snapshot] <= args["tol"],
                                    "evaluation": "prefix_of_saved_budget_reproduction",
                                    "evaluation_maxiter": args["solver_steps"],
                                    "reproduced_full_final": final, "saved_full_final": saved["final"]})

    for seed in settings["seeds"]:
        matched = [r for r in witness_rows if r["seed"] == seed]
        assert len(matched) == 3 and len({r["witness_sha256"] for r in matched}) == 1
        for source in range(3):
            paired = [r for r in budget_rows if r["seed"] == seed and r["test_source"] == source]
            assert len(paired) == 3 and {r["method"] for r in paired} == set(methods)
            assert len({r["source_sha256"] for r in paired}) == 1
            assert {r["base_calls"] for r in paired} == {24}
    for path, digest in source_hashes.items():
        assert file_hash(path) == digest, f"source changed during analysis: {path}"
    print(f"Assertions passed; maximum saved-bank discrepancy {largest_bank_difference:.3e}.")
    return {"bank_components.csv": bank_rows, "matched_witness.csv": witness_rows,
            "equal_base_budget.csv": budget_rows}


def serialize(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", nargs="?", type=Path, default=DEFAULT)
    parser.add_argument("--check", action="store_true", help="verify existing outputs without writing; ignore new wall times")
    args = parser.parse_args()
    outputs = analyze(args.directory.resolve())
    for name, rows in outputs.items():
        path = args.directory / name
        if args.check:
            saved = read_csv(path)
            expected = list(csv.DictReader(io.StringIO(serialize(rows))))
            for group in (saved, expected):
                for row in group:
                    row.pop("trace_elapsed_seconds", None)
            assert saved == expected, f"non-timing output changed: {path}"
        else:
            path.write_text(serialize(rows), encoding="utf-8")
        print(f"{'Verified' if args.check else 'Wrote'} {path}: {len(rows)} rows")


if __name__ == "__main__":
    main()
