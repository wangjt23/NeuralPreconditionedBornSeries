#!/usr/bin/env python3
"""Extract historical RTE evidence without importing or executing an RTE model.

Run from any directory: python3 scripts/summarize_rte_evidence.py
Add --pdf for the standalone report (requires reportlab), --check to compare
the TSV without writing, or --self-test for synthetic extraction tests.
--csv PATH (repeatable) replaces the default, explicitly cited CSV manifest.

TSV numbers retain source strings. Blank means unavailable, not zero/failure.
Tolerance snapshots use the historical reporting convention <=; the actual
solver stops on <. Iterations, preconditioner calls, and base calls are separate.
First-crossing time is an observation from a longer solve, not a rerun stopped
at that tolerance. Source hashes identify the artifacts, not the runtime code.
Batch endpoints are different samples; harness loss endpoints can belong to a
different training step than the selected model's source/solver evaluation.
No coefficient-heldout evidence is inferred from these source-heldout files.
"""

import argparse
import csv
from decimal import Decimal
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "npbs_unet_ablation_full12_legacy_teacherphysical_onecycle_"
FULL_RUNS = [
    ("L3600", "largebank_mixed_size64_steps3600"),
    ("L7200", "largebank_mixed_size64_steps7200"),
    ("XL7200", "xlargebank_mixed_size64_steps7200"),
    ("XL14400", "xlargebank_mixed_size64_steps14400"),
    ("C14400+3600", "xlargebank_mixed_size64_from14400_lr1e3_steps3600"),
    ("C18000+2400", "xlargebank_mixed_size64_from18000_lr5e4_steps2400"),
    ("C20400+1800", "xlargebank_mixed_size64_from20400_lr2p5e4_steps1800"),
    ("C22200+1200", "xlargebank_mixed_size64_from22200_lr1p25e4_steps1200"),
    ("C23100+900", "xlargebank_mixed_size64_from23100_lr6p25e5_steps900"),
    ("C24000+600", "xlargebank_mixed_size64_from24000_lr3p125e5_steps600"),
]
EXTRA_RUNS = [
    ("MomentPhysical1800", "npbs_unet_ablation_moment_structure_physical_size64_steps1800"),
    ("CoeffWeightedW64", "npbs_unet_ablation_coeff_weighted_tsa_m4_physical_w64_size64_steps1800"),
    ("RPPhysical5400", "npbs_unet_ablation_learned_rp_m4_physical_rollout_stability_size64_steps5400"),
    ("RPDiverse5400", "npbs_bottleneck_harness_learned_rp_raw_rollout_teacher05_onecycle_ckpt_diverse_mixed_size64_steps5400"),
    ("RPAnchorNoProbe", "npbs_bottleneck_harness_learned_rp_raw_rollout_teacher05_lowlr_anchor5e3_composite_noprobe_same_seed_mixed_size64_steps1200_eval100"),
]
TOLS = ("1e-04", "1e-05", "1e-06")
KEYS = ("size", "case", "split", "base_kind", "tsa_harmonics", "method")
BASE_FIELDS = [
    "run_id", "source_csv", "source_line", "source_sha256", "settings_file",
    "settings_sha256", "trace_csv", "trace_sha256", "size", "n_angles", "case",
    "split", "split_scope", "base_kind", "tsa_harmonics", "method",
    "base_calls_per_apply", "one_step_true", "one_step_base_metric",
    "train_batch_ratio_initial", "train_batch_ratio_final", "train_objective_initial",
    "train_objective_final", "training_metrics_scope", "selected_step", "steps",
    "checkpoint_metric", "residual_dataset_bank_size", "rollout_bank_size",
    "krylov_bank_size", "fgmres_mode", "configured_tol", "fgmres_budget",
    "stationary_budget", "settings_json", "source_record_json",
]
FIELDS = BASE_FIELDS + [
    f"{solver}_{field}" for solver in ("fgmres", "stationary")
    for field in ("iters", "final", "time", "trace_end_time", "trace_min", "trace_last10_min",
                  "trace_last10_max", "termination_assessment", "trace_check")
] + [
    f"{solver}_{tol}_{field}" for solver in ("fgmres", "stationary")
    for tol in TOLS for field in ("hit", "iter", "time_s", "preconditioner_calls",
                                  "base_calls", "relres", "trace_line")
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return [(reader.line_num, row) for row in reader]


def settings_for(path):
    report = path.with_name(path.stem.upper() + ".md")
    if not report.exists():
        return None, {}
    pairs = re.findall(r"^- `([^`]+)`: `(.*)`$", report.read_text(), re.MULTILINE)
    return report, dict(pairs)


def first_crossing(trace, tolerance):
    target = Decimal(tolerance)
    for line, row in trace:
        value = Decimal(row["relres"])
        if value.is_finite() and value <= target:
            return line, row
    return None


def same_number(a, b):
    x, y = Decimal(a), Decimal(b)
    return (x.is_nan() and y.is_nan()) or x == y


def trace_key(row):
    return tuple(row.get(key, "") for key in KEYS)


def trace_metrics(trace, row, solver, budget, tolerance):
    out = {}
    prefix = solver + "_"
    if not trace:
        return {prefix + "trace_check": "unavailable",
                prefix + "termination_assessment": "unavailable_without_trace"}
    for index, (_, point) in enumerate(trace):
        if int(point["iter"]) != index:
            raise ValueError(f"non-contiguous {solver} trace at iteration {index}")
        if index and float(point["elapsed"]) < float(trace[index - 1][1]["elapsed"]):
            raise ValueError(f"non-monotone {solver} elapsed time")
    last = trace[-1][1]
    # Stationary total time is sampled again after its final trace timestamp.
    comparisons = [("final", last["relres"]), ("iters", last["iter"])]
    if solver == "fgmres":
        comparisons.append(("time", last["elapsed"]))
    for key, value in comparisons:
        if row.get(prefix + key) and not same_number(row[prefix + key], value):
            raise ValueError(f"summary/trace mismatch: {solver} {key}")
    finite = [float(point["relres"]) for _, point in trace
              if math.isfinite(float(point["relres"]))]
    tail = [float(point["relres"]) for _, point in trace[-10:]
            if math.isfinite(float(point["relres"]))]
    out.update({prefix + "trace_end_time": last["elapsed"],
                prefix + "trace_min": repr(min(finite)) if finite else "",
                prefix + "trace_last10_min": repr(min(tail)) if tail else "",
                prefix + "trace_last10_max": repr(max(tail)) if tail else "",
                prefix + "trace_check": "matched"})
    stop = "unresolved_no_saved_stop_flag"
    if tolerance and float(last["relres"]) < float(tolerance):
        stop = "tolerance_satisfied"
    elif budget and int(last["iter"]) == int(budget):
        stop = "budget_reached_target_not_met"
    out[prefix + "termination_assessment"] = stop
    for tolerance in TOLS:
        stem = prefix + tolerance + "_"
        crossing = first_crossing(trace, tolerance)
        out[stem + "hit"] = str(crossing is not None)
        saved_hit = row.get(prefix + "hit_" + tolerance)
        if saved_hit and saved_hit != out[stem + "hit"]:
            raise ValueError(f"summary/trace hit mismatch: {solver} {tolerance}")
        if crossing:
            line, point = crossing
            out.update({stem + "iter": point["iter"], stem + "time_s": point["elapsed"],
                        stem + "preconditioner_calls": point["preconditioner_calls"],
                        stem + "base_calls": point["base_calls"],
                        stem + "relres": point["relres"], stem + "trace_line": str(line)})
            for saved, field in (("time", "elapsed"), ("base_calls", "base_calls")):
                value = row.get(prefix + saved + "_" + tolerance)
                if value and not same_number(value, point[field]):
                    raise ValueError(f"summary/trace snapshot mismatch: {solver} {tolerance}")
    return out


def extract(manifest):
    evidence, all_traces = [], {}
    for run_id, path in manifest:
        settings_path, settings = settings_for(path)
        trace_path = path.with_name(path.stem + "_traces.csv")
        traces = {}
        if trace_path.exists():
            for line, row in read_csv(trace_path):
                traces.setdefault((trace_key(row), row["solver"]), []).append((line, row))
        for line, row in read_csv(path):
            result = {key: row.get(key, "") for key in BASE_FIELDS}
            result.update({
                "run_id": run_id, "source_csv": display_path(path), "source_line": str(line),
                "source_sha256": sha256(path), "settings_file": display_path(settings_path) if settings_path else "",
                "settings_sha256": sha256(settings_path) if settings_path else "",
                "trace_csv": display_path(trace_path) if trace_path.exists() else "",
                "trace_sha256": sha256(trace_path) if trace_path.exists() else "",
                "method": row.get("method", row.get("model_kind", "")),
                "n_angles": row.get("n_angles", settings.get("n_angles", "")),
                "steps": row.get("steps", settings.get("steps", "")),
                "configured_tol": settings.get("tol", ""),
                "fgmres_budget": settings.get("fgmres_maxiter", ""),
                "stationary_budget": settings.get("maxiter", ""),
                "split_scope": "source_only_same_fixed_operator",
                "train_batch_ratio_initial": row.get("train_one_step_initial", row.get("raw_initial", "")),
                "train_batch_ratio_final": row.get("train_one_step_final", row.get("raw_final", "")),
                "train_objective_initial": row.get("train_loss_initial", row.get("loss_initial", "")),
                "train_objective_final": row.get("train_loss_final", row.get("loss_final", "")),
                "training_metrics_scope": ("training_endpoints_not_selected_checkpoint" if row.get("checkpoint_metric", "none") != "none"
                                           and row.get("checkpoint_metric") else "sampled_training_endpoints"),
                "settings_json": json.dumps(settings, sort_keys=True, separators=(",", ":")),
                "source_record_json": json.dumps(row, sort_keys=True, separators=(",", ":")),
            })
            for solver in ("fgmres", "stationary"):
                for field in ("iters", "final", "time"):
                    result[solver + "_" + field] = row.get(solver + "_" + field, "")
                trace = traces.get((trace_key(row), solver), [])
                if trace_path.exists() and not trace:
                    raise ValueError(f"missing trace group: {path}:{line} {solver}")
                result.update(trace_metrics(trace, row, solver, result[solver + "_budget"], result["configured_tol"]))
                all_traces[(run_id, trace_key(row), solver)] = trace
            evidence.append(result)
    return evidence, all_traces


def serialize(evidence):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(evidence)
    return buffer.getvalue()


def self_test():
    trace = [(2, {"iter": "0", "relres": "1", "elapsed": "0", "preconditioner_calls": "0", "base_calls": "0"}),
             (3, {"iter": "1", "relres": "1e-5", "elapsed": "0.12345678901234567", "preconditioner_calls": "1", "base_calls": "4"}),
             (4, {"iter": "2", "relres": "2e-5", "elapsed": "0.2", "preconditioner_calls": "2", "base_calls": "8"})]
    assert first_crossing(trace, "1e-05")[1]["iter"] == "1"
    assert first_crossing(trace, "1e-06") is None
    row = {"fgmres_final": "2e-5", "fgmres_iters": "2", "fgmres_time": "0.2"}
    out = trace_metrics(trace, row, "fgmres", "2", "1e-6")
    assert out["fgmres_1e-05_time_s"] == "0.12345678901234567"
    assert out["fgmres_1e-05_base_calls"] == "4"
    assert out["fgmres_termination_assessment"] == "budget_reached_target_not_met"
    assert "fgmres_1e-06_iter" not in out
    assert trace_metrics([], row, "fgmres", "2", "1e-6")["fgmres_trace_check"] == "unavailable"
    try:
        trace_metrics(trace, dict(row, fgmres_final="3e-5"), "fgmres", "2", "1e-6")
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched summary was accepted")
    print("Self-tests passed: equality, non-monotone residual, missing tolerance/trace, exact time, call counts, mismatch rejection.")


def make_pdf(path, evidence, traces):
    from reportlab.lib.colors import HexColor
    from reportlab.pdfgen import canvas

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=(864, 648), invariant=1)
    c.setTitle("Historical RTE evidence: learning gains and solver limits")
    colors = {"tsa2": "#64748B", "tsa4": "#17816A", "UNet-defect2-w1": "#C25B35"}
    label = {"tsa2": "TSA2", "tsa4": "TSA4", "UNet-defect2-w1": "U-Net"}

    def text(x, y, value, size=9, bold=False, color="#17232D"):
        c.setFillColor(HexColor(color))
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(x, y, str(value))

    def row(run, method="UNet-defect2-w1", split="heldout"):
        return next(r for r in evidence if r["run_id"] == run and r["method"] == method and r["split"] == split and r["case"] == "mixed")

    def series(run, method, solver):
        r = row(run, method)
        return traces[(run, trace_key(r), solver)]

    def legend(x, y, entries):
        for name, color in entries:
            c.setStrokeColor(HexColor(color)); c.setLineWidth(2)
            c.line(x, y + 3, x + 13, y + 3)
            text(x + 18, y, name, 8)
            x += 18 + c.stringWidth(name, "Helvetica", 8) + 17

    def axes(x, y, w, h, title, xticks, yticks, xmin, xmax, ymin, ymax, log=False):
        text(x, y + h + 25, title, 11, True)
        def xp(v):
            return x + (v - xmin) / (xmax - xmin) * w
        def yp(v):
            value = math.log10(v) if log else v
            return y + (value - ymin) / (ymax - ymin) * h
        c.setLineWidth(.4)
        for value, name in yticks:
            py = yp(value)
            c.setStrokeColor(HexColor("#DCE2E5")); c.line(x, py, x + w, py)
            text(x - 34, py - 3, name, 8)
        for value, name in xticks:
            text(xp(value) - c.stringWidth(name, "Helvetica", 8) / 2, y - 14, name, 8)
        c.setStrokeColor(HexColor("#8C989D")); c.line(x, y, x, y + h)
        c.line(x, y, x + w, y)
        return xp, yp

    def line(points, xp, yp, color, dots=False):
        c.setStrokeColor(HexColor(color)); c.setFillColor(HexColor(color)); c.setLineWidth(1.6)
        p = c.beginPath()
        for i, (x, y) in enumerate(points):
            if i == 0:
                p.moveTo(xp(x), yp(y))
            else:
                p.lineTo(xp(x), yp(y))
        c.drawPath(p)
        if dots:
            for x, y in points:
                c.circle(xp(x), yp(y), 2.4, stroke=0, fill=1)

    text(44, 607, "Historical RTE: learning gains, solver limits", 22, True)
    text(44, 585, "Fixed mixed medium | 64 x 64 | 12 angles | source-heldout only | saved GPU float32 runs", 10)
    runs = ["XL7200", "XL14400", "C24000+600"]
    xp, yp = axes(78, 375, 310, 135, "A  Batch residual vs full-source evaluation",
                  [(0, "XL7200"), (1, "XL14400"), (2, "Final cont.")],
                  [(v, f"{v:.2f}") for v in (.08, .12, .16, .20, .24)], 0, 2, .08, .24)
    entries = [("Batch end", "#7A5C91"), ("Eval train", "#256C99"), ("Eval heldout", "#C25B35")]
    for field, split, (_, color) in zip(["train_batch_ratio_final", "one_step_true", "one_step_true"], ["train", "train", "heldout"], entries):
        line([(i, float(row(run, split=split)[field])) for i, run in enumerate(runs)], xp, yp, color, True)
    legend(66, 342, entries)
    text(66, 327, "Unsquared ratios; batch and source distributions differ.", 8)

    text(469, 535, "B  Final continuation: first FGMRES hit at 1e-5", 11, True)
    text(469, 518, "Bar = elapsed seconds; labels = iterations / base calls", 8)
    for i, method in enumerate(colors):
        r = row("C24000+600", method)
        y = 476 - i * 49
        value = float(r["fgmres_1e-05_time_s"])
        text(469, y + 13, label[method], 9, True)
        c.setFillColor(HexColor(colors[method])); c.rect(520, y - 2, value / .23 * 230, 12, stroke=0, fill=1)
        text(520, y + 14, f"{value:.6f} s | {r['fgmres_1e-05_iter']} it / {r['fgmres_1e-05_base_calls']} base", 8)
    text(469, 327, "Fewer base calls does not mean fewer iterations or less time.", 8)

    for solver, x, title, xmax in [("fgmres", 78, "C  Final continuation: FGMRES trace", 40),
                                   ("stationary", 478, "D  Final continuation: stationary trace", 80)]:
        xp, yp = axes(x, 116, 310, 143, title,
                      [(i, str(i)) for i in range(0, xmax + 1, xmax // 4)],
                      [(10 ** i, f"1e{i}") for i in (-6, -4, -2, 0)], 0, xmax, -6.2, .1, True)
        for method in colors:
            points = [(int(t["iter"]), float(t["relres"])) for _, t in series("C24000+600", method, solver)]
            line(points, xp, yp, colors[method])
        text(x + 125, 87, "Iterations", 8)
        legend(x - 9, 70, [(label[m], colors[m]) for m in colors])
    text(44, 44, "Target 1e-6 was never reached in the full12 traces. The 40-iteration FGMRES budget was reached; no plateau stopping rule exists.", 8)
    text(44, 30, "Observed ~7e-6 to 8e-6 tails are consistent with precision stagnation, not proof of its cause. No universal learning failure follows.", 8)
    text(44, 16, "Source digits, trace lines and SHA-256: docs/research/historical_evidence.tsv | Generator: scripts/summarize_rte_evidence.py | 1 / 2", 7)
    c.showPage()

    text(44, 608, "Common-tolerance evidence and interpretation", 20, True)
    text(44, 587, "First trace crossing uses <= tolerance; runtime convergence uses < tolerance. Times are single-run observations, not speed estimates.", 9)
    columns = [(44, "Run / source"), (194, "Method"), (280, "1e-4: it / time (s)"), (457, "1e-5: it / time (s)"), (633, "1e-5 base calls"), (743, "1e-6 hit")]
    for x, title in columns:
        text(x, 559, title, 9, True)
    y = 541
    for run in runs:
        for split in ("train", "heldout"):
            for method in colors:
                r = row(run, method, split)
                values = [f"{run} / {split}", label[method]]
                for tol in ("1e-04", "1e-05"):
                    values.append(f"{r['fgmres_' + tol + '_iter']} / {float(r['fgmres_' + tol + '_time_s']):.9f}")
                values.extend([r["fgmres_1e-05_base_calls"], "No" if r["fgmres_1e-06_hit"] == "False" else "Yes"])
                for (x, _), value in zip(columns, values):
                    text(x, y, value, 8)
                y -= 14
        y -= 7
    text(44, 245, "Decisive statements", 12, True)
    statements = [
        "Learning produced substantial numerical gains: XL7200 -> XL14400 source ratios improve 0.121667/0.135623 -> 0.105624/0.111631.",
        "Heldout U-Net beats TSA4 at XL14400 (0.111631 vs 0.114459), then reaches 0.107681. The training source still trails TSA4 (0.098095).",
        "XL14400 sampled batch ratio falls 0.319635 -> 0.210184; its composite objective falls 0.293390 -> 0.095186. Neither is source error.",
        "Final heldout FGMRES at 1e-5: U-Net 15 iterations / 30 base calls; TSA4 11 / 44. U-Net recorded time is about twice TSA4 time.",
        "At 1e-6, all full12 methods used all 40 iterations without success. A precision/conditioning explanation needs a higher-precision rerun.",
        "No universal failure or established data bottleneck: one-step gains are real, but these files do not establish faster solves or new-media transfer.",
    ]
    for value in statements:
        y = 225 - statements.index(value) * 17
        text(44, y, value, 8.3)
    text(44, 109, "Provenance and boundaries", 11, True)
    text(44, 91, "Solver logic matches co-committed source bdccfab: restart=20, max_restarts=2 (up to three cycles), true residual checked each iteration.", 8)
    text(44, 76, "Runtime has no stagnation test; breakdown threshold is 1e-14. Trace files do not save a termination flag. Budget reached is observable.", 8)
    text(44, 61, "7200 vs 14400 changes the OneCycle horizon. C labels preserve filenames: 22200+1200 followed by from23100 is inconsistent.", 8)
    text(44, 46, "216 denotes rollout residual entries, not media. Coefficient environments were fixed; checkpoints/banks and FP64 reruns are unavailable here.", 8)
    text(44, 31, "The TSV also includes 24-angle CoeffWeighted/Mixed/LearnedRP comparisons. Do not pool those with the 12-angle full-TSA experiments.", 8)
    text(44, 16, "Default extraction validates every available paired trace against its summary. Missing trace metrics remain blank. Values above are rounded. | 2 / 2", 7)
    c.save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", action="append", type=Path, help="replace default manifest; repeatable")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/research/historical_evidence.tsv")
    parser.add_argument("--pdf", nargs="?", const=ROOT / "output/pdf/historical_rte_evidence.pdf", type=Path)
    parser.add_argument("--check", action="store_true", help="verify existing TSV without writing")
    parser.add_argument("--self-test", action="store_true", help="run synthetic extraction tests only")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    if args.check and args.pdf:
        parser.error("--check does not write a PDF")
    if args.pdf and args.csv:
        parser.error("the chart requires the default manifest")
    manifest = ([(p.stem, p.resolve()) for p in args.csv] if args.csv else
                [(name, ROOT / "rte" / (PREFIX + suffix + ".csv")) for name, suffix in FULL_RUNS] +
                [(name, ROOT / "rte" / (stem + ".csv")) for name, stem in EXTRA_RUNS])
    evidence, traces = extract(manifest)
    content = serialize(evidence)
    if args.check:
        if not args.output.exists() or args.output.read_text() != content:
            raise ValueError(f"TSV differs or is missing: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8", newline="")
    if args.pdf:
        make_pdf(args.pdf, evidence, traces)
    checked = sum(r.get(s + "_trace_check") == "matched" for r in evidence for s in ("fgmres", "stationary"))
    print(f"{'Verified' if args.check else 'Wrote'} {args.output}: {len(manifest)} CSVs, {len(evidence)} result rows, {checked} matched solver traces.")
    if args.pdf:
        print(f"Wrote {args.pdf}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(str(error))
