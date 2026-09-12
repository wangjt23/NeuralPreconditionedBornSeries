#!/usr/bin/env python3
"""Summarize seed-level prototype results without treating sources as media."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(directory):
    rows = [r for r in read(directory / "results.csv") if r["solver"] == "fgmres"]
    methods = list(dict.fromkeys(r["method"] for r in rows))
    seeds = sorted({int(r["seed"]) for r in rows})
    keys = ["train_physical_loss", "validation_physical_loss", "test_physical_loss",
            "test_source_one_step", "final", "seconds", "operator_calls", "base_calls"]
    records = []
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        row = {"method": method, "seeds": len(seeds), "test_sources_per_seed": len(subset) // len(seeds)}
        for key in keys:
            means = [np.mean([float(r[key]) for r in subset if int(r["seed"]) == seed]) for seed in seeds]
            row[key + "_mean"] = np.mean(means)
            row[key + "_seed_min"] = min(means)
            row[key + "_seed_max"] = max(means)
        records.append(row)
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    colors = {"legacy_raw": "#176c89", "legacy_replay": "#91652d",
              "angular_raw": "#b8424b", "angular_projected": "#387546"}
    for kind, color in colors.items():
        if kind not in methods:
            continue
        histories = [read(next(directory.glob(f"*_{seed}_{kind}_history.csv"))) for seed in seeds]
        steps = [int(r["step"]) for r in histories[0]]
        for prefix, style in (("train", "-"), ("validation", "--")):
            values = np.array([[float(r[prefix + "_physical_loss"]) for r in history] for history in histories])
            axes[0, 0].plot(steps, values.mean(0), style, color=color,
                            label=kind + (" train" if style == "-" else " validation"))
    axes[0, 0].set(xlabel="Optimizer updates", ylabel="Mean squared physical residual",
                   title="Fixed residual banks, before checkpoint selection", yscale="log")
    axes[0, 0].legend(fontsize=7, ncols=2)
    for ax, key, title in ((axes[0, 1], "test_physical_loss", "Test residual bank at selected checkpoint"),
                            (axes[1, 0], "final", "Test-source FGMRES after 12 applications"),
                            (axes[1, 1], "seconds", "Test-source solve wall time (CPU prototype)")):
        x = np.arange(len(methods))
        means = np.array([r[key + "_mean"] for r in records])
        lows = np.array([r[key + "_seed_min"] for r in records])
        highs = np.array([r[key + "_seed_max"] for r in records])
        ax.bar(x, means, color=["#757575" if m.startswith(("base", "MR_base", "MR_classical")) else
                               colors[m.removeprefix("MR_")] for m in methods])
        ax.errorbar(x, means, yerr=[means - lows, highs - means], fmt="none", ecolor="black", capsize=3)
        ax.set_xticks(x, methods, rotation=65, ha="right", fontsize=8)
        ax.set(title=title, yscale="log")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Mixed periodic RTE, 16 x 16 x 12; three seeds, one coefficient field\n"
                 "Bars: mean over seeds/sources; whiskers: range of seed means, not confidence intervals", fontsize=11)
    fig.savefig(directory / "comparison.pdf")
    fig.savefig(directory / "comparison.png", dpi=140)
    plt.close(fig)
    for row in records:
        print(f"{row['method']:24s} train={row['train_physical_loss_mean']:.5g} "
              f"val={row['validation_physical_loss_mean']:.5g} test={row['test_physical_loss_mean']:.5g} "
              f"source1={row['test_source_one_step_mean']:.5g} fgmres={row['final_mean']:.5g} "
              f"time={row['seconds_mean']:.5g} A={row['operator_calls_mean']:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("rte/results/residual_subspace_mixed"))
    summarize(parser.parse_args().directory)
