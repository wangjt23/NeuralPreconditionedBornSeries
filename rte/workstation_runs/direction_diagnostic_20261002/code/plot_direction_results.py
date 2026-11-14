"""Plot preserved development-only timing comparisons; no model execution."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def run(results, output):
    rows = json.loads((results / "runtime_comparisons.json").read_text())
    colors = ["#157F8A", "#BF6926", "#7756A7"]
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.8), sharey=True)
    for panel, comparisons in enumerate([
        [("frozen_classical", "tsa8_mixed", "Mixed after TSA8"),
         ("frozen_classical", "tsa16_mixed", "Mixed after TSA16"),
         ("frozen_classical", "tsa8_phase4", "Four-phase average after TSA8")],
        [("tsa8_reference_cost", "tsa8_mixed", "Mixed vs one-forward TSA8 control"),
         ("tsa16_reference_cost", "tsa16_mixed", "Mixed vs one-forward TSA16 control"),
         ("tsa8_reference_phase4_cost", "tsa8_phase4", "Average vs four-forward TSA8 control")]
    ]):
        ax = axes[panel]
        for offset, (baseline, candidate, label), color in zip([-.19, 0, .19], comparisons, colors):
            group = sorted([r for r in rows if r["classical"] == baseline and r["neural"] == candidate], key=lambda r: r["size"])
            assert len(group) == 3
            y = [r["saving_percent"] for r in group]
            errors = [[y[i] - r["saving_percent_ci95"][0] for i,r in enumerate(group)],
                      [r["saving_percent_ci95"][1] - y[i] for i,r in enumerate(group)]]
            ax.errorbar([i+offset for i in range(3)], y, yerr=errors, fmt="o", color=color, capsize=4, label=label, markersize=5)
        ax.axhline(0, color="#444444", linewidth=.8)
        if panel == 0:
            ax.axhline(5, color="#888888", linestyle="--", linewidth=.8)
        ax.set_xticks(range(3), ["N64", "N128", "N256"])
        ax.set_ylim(-25, 13)
        ax.grid(axis="y", alpha=.15)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        ax.set_title(["Actual speed: no candidate clears the gate", "Quality benefit at matched inference cost"][panel], fontsize=11)
    axes[0].set_ylabel("Mean solve-time saving (%)")
    fig.text(.08, .025, "Same 16 development media · medians of three repeats · paired 95% bootstrap intervals · positive values favor learning", fontsize=8)
    fig.tight_layout(rect=(0,.055,1,1))
    for ext in ("png", "svg"):
        fig.savefig(output / ("direction_runtime." + ext), dpi=180)
    plt.close(fig)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.results, a.output)
