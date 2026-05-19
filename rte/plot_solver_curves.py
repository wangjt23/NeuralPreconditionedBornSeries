import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def group_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["case"],
            row["split"],
            row["solver"],
            row["base_kind"],
            row["size"],
        )
        grouped[key].append(row)
    return grouped


def method_series(rows):
    series = defaultdict(list)
    for row in rows:
        series[row["method"]].append(row)
    for method in series:
        series[method].sort(key=lambda r: int(r["iter"]))
    return series


def plot_group(key, rows, output_dir):
    case, split, solver, base_kind, size = key
    data = method_series(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_t, ax_c = axes

    for method, seq in sorted(data.items()):
        t = [float(r["elapsed"]) for r in seq]
        c = [float(r["base_calls"]) for r in seq]
        y = [max(float(r["relres"]), 1e-16) for r in seq]
        ax_t.plot(t, y, label=method, linewidth=1.8)
        ax_c.plot(c, y, label=method, linewidth=1.8)

    title = f"size={size} case={case} split={split} solver={solver} base={base_kind}"
    ax_t.set_title("Residual vs Wall Time")
    ax_t.set_xlabel("Wall time (s)")
    ax_t.set_ylabel("Relative residual")
    ax_t.set_yscale("log")
    ax_t.grid(True, alpha=0.3)

    ax_c.set_title("Residual vs Base Calls")
    ax_c.set_xlabel("Base preconditioner calls")
    ax_c.set_ylabel("Relative residual")
    ax_c.set_yscale("log")
    ax_c.grid(True, alpha=0.3)

    handles, labels = ax_t.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    fname = f"curve_size{size}_{case}_{split}_{solver}_{base_kind}.png"
    out_path = output_dir / fname
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot residual-vs-time and residual-vs-base-calls from NPBS trace CSV.")
    parser.add_argument("--trace-csv", required=True)
    parser.add_argument("--output-dir", default="rte/curve_plots")
    args = parser.parse_args()

    trace_path = Path(args.trace_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with trace_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    grouped = group_rows(rows)

    outputs = []
    for key, group in sorted(grouped.items()):
        outputs.append(plot_group(key, group, output_dir))

    print(f"wrote {len(outputs)} plot files to {output_dir}")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
