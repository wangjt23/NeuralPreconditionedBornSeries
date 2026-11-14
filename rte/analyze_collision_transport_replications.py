"""Consolidate paired collision/transport replication artifacts."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

from rte.run_residual_subspace_experiment import write_csv


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(value):
    if value in (None, ""):
        return None
    return float(value)


def bootstrap_mean_interval(values, seed, samples=10000):
    if not values:
        return None, None
    generator = np.random.default_rng(seed)
    estimates = generator.choice(values, size=(samples, len(values))).mean(axis=1)
    return tuple(float(x) for x in np.quantile(estimates, [.025, .975]))


def hierarchical_interval(rows, samples=10000):
    """Resample trained-model seeds, then paired media within each seed."""
    generator = np.random.default_rng(9299)
    seeds = sorted({row['seed'] for row in rows})
    clusters = [[row['delta_seconds'] for row in rows if row['seed'] == seed and row['delta_seconds'] is not None]
                for seed in seeds]
    if not clusters or not all(clusters):
        return None, None
    estimates = np.zeros(samples)
    for _ in seeds:
        means = np.stack([generator.choice(values, size=(samples, len(values))).mean(1) for values in clusters], 1)
        choices = generator.integers(len(seeds), size=samples)
        estimates += means[np.arange(samples), choices] / len(seeds)
    return tuple(float(x) for x in np.quantile(estimates, [.025, .975]))


def paired_summary(rows, seed):
    successes = [row for row in rows if row['delta_seconds'] is not None]
    deltas = [row["delta_seconds"] for row in successes]
    iterations = [row["delta_iterations"] for row in successes]
    ratios = [row["speedup_baseline_over_ct"] for row in successes]
    low, high = hierarchical_interval(rows) if seed == 'pooled' else bootstrap_mean_interval(deltas, 9200 + int(seed))
    return {
        "seed": seed, "baseline": rows[0]["baseline"], "media": len(rows),
        "paired_successes": len(successes),
        "ct_failed_media": sum(row['ct_failed'] for row in rows),
        "baseline_failed_media": sum(row['baseline_failed'] for row in rows),
        "interval_scope": 'resample model seeds then media; exploratory with few seeds' if seed == 'pooled' else 'media conditional on fixed trained model',
        "ct_faster_media": sum(value < 0 for value in deltas),
        "ties": sum(value == 0 for value in deltas),
        "mean_delta_seconds_ct_minus_baseline": statistics.mean(deltas) if deltas else None,
        "bootstrap95_mean_delta_low": low, "bootstrap95_mean_delta_high": high,
        "median_delta_seconds_ct_minus_baseline": statistics.median(deltas) if deltas else None,
        "median_speedup_baseline_over_ct": statistics.median(ratios) if ratios else None,
        "mean_delta_iterations_ct_minus_baseline": statistics.mean(iterations) if iterations else None,
    }


def run(args):
    if len(args.training_runs) != len(args.evaluation_runs):
        raise ValueError("training-runs and evaluation-runs must be paired")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    per_seed, paired, fingerprints = [], [], []
    for training_name, evaluation_name in zip(args.training_runs, args.evaluation_runs):
        training, evaluation = Path(training_name), Path(evaluation_name)
        train_settings = json.loads((training / "settings.json").read_text())
        eval_settings = json.loads((evaluation / "settings.json").read_text())
        seed = int(train_settings["seed"])
        losses = {row["split"]: float(row["physical_loss"]) for row in read_csv(training / "selected_losses.csv")}
        summaries = read_csv(evaluation / "summary.csv")
        for row in summaries:
            per_seed.append({
                "seed": seed, "selected_epoch": int(float(eval_settings["selected_epoch"])),
                "method": row["method"], "failures": int(float(row["failures"])),
                "mean_iterations": number(row["mean_iterations"]),
                "mean_final_residual": number(row["mean_final_residual"]),
                "median_time_to_tolerance_seconds": number(row["median_time_to_tolerance_seconds"]),
                "p95_time_to_tolerance_seconds": number(row["p95_time_to_tolerance_seconds"]),
                "median_setup_seconds": number(row["median_setup_seconds"]),
                "mean_operator_calls": number(row["mean_operator_calls"]),
                "mean_preconditioner_calls": number(row["mean_preconditioner_calls"]),
                "train_loss": losses["train"] if row["method"] == "learned_CT" else None,
                "validation_loss": losses["validation"] if row["method"] == "learned_CT" else None,
                "test_loss": losses["test"] if row["method"] == "learned_CT" else None,
            })
        cases = read_csv(evaluation / "results.csv")
        indexed = {(int(row["index"]), row["method"]): row for row in cases}
        for baseline in ("hss_tuned", "tsa_tuned", "reference_tuned", "dsa_tuned", "dsa_transport_tuned"):
            for index in sorted({key[0] for key in indexed}):
                ct, control = indexed[index, "learned_CT"], indexed[index, baseline]
                ct_failed = bool(ct.get('setup_failure')) or int(ct.get('warm_failures') or 0) > 0
                control_failed = bool(control.get('setup_failure')) or int(control.get('warm_failures') or 0) > 0
                ct_seconds = number(ct.get("warm_median_time_to_tolerance_seconds"))
                control_seconds = number(control.get("warm_median_time_to_tolerance_seconds"))
                both = not ct_failed and not control_failed and ct_seconds is not None and control_seconds is not None
                ct_iters, control_iters = number(ct.get('warm_median_iters')), number(control.get('warm_median_iters'))
                paired.append({
                    "seed": seed, "index": index, "baseline": baseline,
                    "ct_failed": ct_failed, "baseline_failed": control_failed,
                    "ct_seconds": ct_seconds, "baseline_seconds": control_seconds,
                    "delta_seconds": ct_seconds - control_seconds if both else None,
                    "speedup_baseline_over_ct": control_seconds / ct_seconds if both else None,
                    "ct_iterations": ct_iters, "baseline_iterations": control_iters,
                    "delta_iterations": ct_iters - control_iters if both else None,
                    "ct_source_loss": number(ct.get("source_loss")), "baseline_source_loss": number(control.get("source_loss")),
                })
        fingerprints.extend([
            {"seed": seed, "split": "train", "fingerprint": train_settings["fingerprints"]["train"]},
            {"seed": seed, "split": "validation", "fingerprint": eval_settings["validation_fingerprint"]},
            {"seed": seed, "split": "test", "fingerprint": eval_settings["test_fingerprint"]},
        ])

    paired_summaries = []
    for seed in sorted({row["seed"] for row in paired}):
        for baseline in sorted({row["baseline"] for row in paired}):
            rows = [row for row in paired if row["seed"] == seed and row["baseline"] == baseline]
            paired_summaries.append(paired_summary(rows, seed))
    for baseline in sorted({row["baseline"] for row in paired}):
        rows = [row for row in paired if row["baseline"] == baseline]
        paired_summaries.append(paired_summary(rows, "pooled"))

    if len({row["fingerprint"] for row in fingerprints}) != len(fingerprints):
        raise ValueError("a dataset fingerprint was reused across replications/splits")
    write_csv(out / "per_seed.csv", per_seed)
    write_csv(out / "paired_cases.csv", paired)
    write_csv(out / "paired_summary.csv", paired_summaries)
    write_csv(out / "fingerprints.csv", fingerprints)
    manifest = {"training_runs": args.training_runs, "evaluation_runs": args.evaluation_runs,
                "bootstrap_samples": 10000, "dataset_fingerprints_unique": True,
                "fingerprint_limit": "different whole-split hashes do not prove per-medium disjointness",
                "pooled_interval": "hierarchical by trained-model seed then paired medium; few seeds",
                "speed_comparison": "conditional on both methods succeeding; failed pairs explicitly retained"}
    (out / "settings.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return per_seed, paired_summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-runs", nargs="+", required=True)
    parser.add_argument("--evaluation-runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    per_seed, paired = run(args)
    print(json.dumps({"per_seed": per_seed, "paired_summary": paired}, indent=2))


if __name__ == "__main__":
    main()
