"""Frozen original versus CUDA-graph recipes on existing development media only.

All 48 operator checks and 12 ownership/guard bundles precede full solves. Timers enclose only
the original protocol.solve call and CUDA synchronization. Q-cost tables are
conditional setup-plus-warm estimates, not distinct-RHS or process-cold trials.
The launcher's owned-process timeout supplements the cooperative deadline here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import time

PROCESS_STARTED = time.monotonic()

import torch
from graph_classical_v2 import GraphClassicalV2
from rte.benchmark_tail_solver import solve
from rte.train_source_predictor import build_base, material_operator, source_family
from rte.transport_baselines import ClassicalTransportStack


GRIDS = (64, 128, 256)
IDENTITY_FIELDS = ("sigma_t", "sigma_s", "centers", "phases", "amplitudes", "angular_codes", "geometry_base64")
RECIPES = {"64": "reference_tail", "128": "reference_tail", "256": "tsa16_zero"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def relative(a, b):
    denominator = float(b.norm())
    return float((a - b).norm()) / denominator if math.isfinite(denominator) and denominator > 0 else float("inf")


def gpu_processes():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
                            text=True, capture_output=True, timeout=5)
    return dict(returncode=result.returncode, stdout=result.stdout.strip(), stderr=result.stderr.strip())


def check_originals(policy):
    mapping = policy["original_source_sha256"]
    if len(mapping) != 155:
        raise ValueError("policy must preserve the declared 155 original sources")
    for name, expected in mapping.items():
        if digest(name) != expected:
            raise ValueError("original source hash mismatch: " + name)
    return len(mapping)


def expected_parent(policy, size, path):
    mapping = policy["development_parent_sha256"]
    for key in (str(size), str(path), str(path.resolve())):
        if key in mapping:
            return mapping[key]
    raise ValueError("missing development-parent hash for N" + str(size))


def accuracy(high, value, status, margin, field_difference, same_iterations, same_calls):
    rhs = high.source
    source_norm, field_norm = float(rhs.norm()), float(value.norm())
    fields_finite = bool(torch.isfinite(rhs).all() and torch.isfinite(value).all())
    error_vector = rhs - high.A_op(value.to(torch.complex128))
    residual_norm = float(error_vector.norm())
    residual_finite = bool(torch.isfinite(error_vector).all()) and math.isfinite(residual_norm)
    relative_residual = residual_norm / source_norm if math.isfinite(source_norm) and source_norm > 0 else float("inf")
    valid_margin = math.isfinite(margin) and margin > 0
    delta = residual_norm / margin if valid_margin and residual_finite else float("inf")
    denominator = field_norm - delta
    bound = delta / denominator if math.isfinite(denominator) and denominator > 0 else float("inf")
    passed = (fields_finite and residual_finite and math.isfinite(source_norm) and source_norm > 0
              and valid_margin and math.isfinite(bound) and 0 <= bound <= 1e-3
              and math.isfinite(relative_residual) and relative_residual <= 1e-5
              and math.isfinite(field_difference) and field_difference <= 1e-4
              and status.get("failed") == 0 and same_iterations and same_calls)
    return dict(source_norm=source_norm, field_norm=field_norm, field_finite=fields_finite,
                residual_vector_finite=residual_finite, independent_residual_norm=residual_norm,
                independent_true_residual=relative_residual, evaluated_coercivity_margin=margin,
                coercivity_margin_finite_positive=valid_margin, evaluated_absolute_error_bound=delta,
                evaluated_relative_error_bound=bound, error_bound_finite=math.isfinite(bound),
                field_difference_vs_frozen=field_difference, paired_iterations_equal=same_iterations,
                paired_classical_calls_equal=same_calls, accuracy_passed=passed)


def rejection_check(graph, rhs, mutate, restore, rejected_state_check=None):
    mutate()
    try:
        try:
            graph(rhs)
        except RuntimeError:
            if rejected_state_check is not None and not rejected_state_check():
                raise AssertionError("stale-state guard rebuilt a missing cache before rejection")
            return True
        raise AssertionError("stale graph state was not rejected")
    finally:
        restore()


def _run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "settings.json", dict(complete=False, test_touched=False, training_run=False,
                                        runner_sha256=digest(__file__)))
    policy_path = Path(args.policy).resolve()
    policy = json.loads(policy_path.read_text())
    budget = min(float(args.max_wall_seconds), float(policy.get("soft_process_budget_seconds", 300.)))
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("positive finite process budget required")
    deadline = PROCESS_STARTED + budget
    parity, guards, rows, setups, aggregate = [], [], [], [], []
    settings = dict(complete=False, scope="16 existing development media at three nested grids; original recipes and exact-original graph replay",
                    independent_media=16, grids=list(GRIDS), recipes=RECIPES, parity_ids=[0, 3],
                    warmups=1, timed_repeats=2, max_full_solves=288, max_wall_seconds=budget,
                    test_touched=False, reserved_seed9626_touched=False, training_run=False,
                    policy_sha256=digest(policy_path), runner_sha256=digest(__file__),
                    helper_sha256=digest(Path(__file__).with_name("graph_classical_v2.py")),
                    deadline_enforcement="cooperative checks between phases/cases/solves; launcher owns hard process timeout",
                    accuracy_scope="unchanged original FP64 A/RHS; stored weighted-phase coercivity estimate without directed rounding; no continuous-error certificate",
                    precision_scope="complex64 graph replay, original complex128 native fallback; cached original reference inverse unchanged",
                    cost_scope="shared physical/required-base setup charged to both arms; graph warmup/capture extra; Q tables conditional setup-plus-warm estimates, not distinct-RHS experiments or process-cold latency")

    def checkpoint():
        settings["wall_seconds"] = time.monotonic() - PROCESS_STARTED
        save(output / "settings.json", settings)
        save(output / "parity_cases.json", parity)
        save(output / "guard_cases.json", guards)
        save(output / "solver_cases.json", rows)
        save(output / "setup_records.json", setups)
        save(output / "case_medians.json", aggregate)

    def bounded(phase):
        settings["last_phase"] = phase
        if time.monotonic() >= deadline:
            raise TimeoutError("owned process budget exhausted before " + phase)

    def load_development(size):
        bounded("development identity N" + str(size))
        path = Path(f"rte/workstation_runs/resume_audit_20261002/runtime_gate/validation_n{size}.pt")
        actual = digest(path)
        if actual != expected_parent(policy, size, path):
            raise ValueError("development-parent hash mismatch: " + str(path))
        data = torch.load(path, map_location="cpu", weights_only=True)
        cfg, regenerated = source_family(size, 16, 4626)
        for name in IDENTITY_FIELDS:
            if not torch.equal(data[name], regenerated[name]):
                raise ValueError("regenerated identity mismatch: " + name)
        if len(data["sigma_t"]) != 16:
            raise ValueError("full cohort must contain exactly 16 saved media")
        settings.setdefault("verified_development_parent_sha256", {})[str(size)] = actual
        return cfg, data

    checkpoint()
    try:
        bounded("policy and original-source verification")
        if policy.get("recipes", RECIPES) != RECIPES:
            raise ValueError("policy recipes differ from frozen original portfolio")
        for name, expected in policy.get("code_sha256", {}).items():
            if digest(policy_path.parent / name) != expected:
                raise ValueError("staged code hash mismatch: " + name)
        settings["original_sources_verified_before"] = check_originals(policy)
        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; graph screen cannot run")
        settings["gpu_processes_before"] = gpu_processes()
        torch.cuda.reset_peak_memory_stats()
        for n in GRIDS:
            cfg, data = load_development(n)
            for index in (0, 3):
                bounded(f"parity material N{n} ID{index}")
                low, high = material_operator(cfg, data, torch.tensor([index]), .1, "cuda")
                stripe = (-1.) ** torch.arange(n, device="cuda")
                grazing = torch.zeros_like(low.source)
                grazing[:, 0] = stripe[None, None, :]
                for depth in (1, 16):
                    bounded(f"parity capture N{n} ID{index} depth{depth}")
                    base = ClassicalTransportStack(low, "tsa", depth, 0.)
                    graph = GraphClassicalV2(base, low.source)
                    expected = dict(fftn=4 * depth - 1, ifftn=4 * depth - 1)
                    if graph.capture_fft_counts != expected:
                        raise AssertionError("captured original algorithm FFT count mismatch")
                    for dtype in (torch.complex64, torch.complex128):
                        for name, probe in (("existing_physical_source", low.source), ("deterministic_grazing_nyquist", grazing)):
                            bounded(f"parity apply N{n} ID{index} depth{depth} {dtype} {name}")
                            rhs = probe.to(dtype)
                            original = base(rhs).to(torch.complex128)
                            candidate = graph(rhs)
                            image_original, image_candidate = high.A_op(original), high.A_op(candidate)
                            finite = bool(torch.isfinite(original).all() and torch.isfinite(candidate).all()
                                          and torch.isfinite(image_original).all() and torch.isfinite(image_candidate).all())
                            field, image = relative(candidate, original), relative(image_candidate, image_original)
                            equal = bool(torch.equal(original, candidate))
                            limit = 1e-4 if dtype == torch.complex64 else 1e-9
                            parity.append(dict(size=n, index=index, depth=depth, input_dtype=str(dtype), probe=name,
                                               field_relative=field, image_relative=image, finite=finite, bitwise_equal=equal,
                                               limit=limit, native128_fallback=dtype == torch.complex128,
                                               capture_fft_counts=expected, passed=finite and equal and math.isfinite(field)
                                               and math.isfinite(image) and field <= limit and image <= limit))
                    bounded("ownership and stale-state guards")
                    held = graph(low.source)
                    held_copy = held.clone()
                    subsequent = graph(grazing)
                    pointers_distinct = (held.data_ptr() != subsequent.data_ptr()
                                         and held.data_ptr() != graph.static_output.data_ptr()
                                         and subsequent.data_ptr() != graph.static_output.data_ptr())
                    owned = (bool(torch.equal(held, held_copy)) and held.dtype == torch.complex128
                             and subsequent.dtype == torch.complex128 and pointers_distinct)
                    if not owned:
                        raise AssertionError("returned graph output was overwritten by later replay")
                    old_t, old_depth = low.sigma_t, base.depth
                    transposed = old_t.transpose(-1, -2)
                    same_pointer_shape = old_t.data_ptr() == transposed.data_ptr() and old_t.shape == transposed.shape
                    changed_stride = old_t.stride() != transposed.stride()
                    if not same_pointer_shape or not changed_stride:
                        raise AssertionError("transpose guard fixture does not exercise same-pointer/shape view replacement")
                    transpose = rejection_check(graph, low.source,
                        lambda: setattr(low, "sigma_t", transposed), lambda: setattr(low, "sigma_t", old_t))
                    depth_guard = rejection_check(graph, low.source,
                        lambda: setattr(base, "depth", old_depth + 1), lambda: setattr(base, "depth", old_depth))
                    inverse = base.reference._reference_transport_inverse
                    missing = rejection_check(graph, low.source,
                        lambda: setattr(base.reference, "_reference_transport_inverse", None),
                        lambda: setattr(base.reference, "_reference_transport_inverse", inverse),
                        lambda: base.reference._reference_transport_inverse is None)
                    restored, native_restored = graph(low.source), base(low.source).to(torch.complex128)
                    restored_finite = bool(torch.isfinite(restored).all() and torch.isfinite(native_restored).all())
                    restored_equal = bool(torch.equal(restored, native_restored))
                    if not restored_finite or not restored_equal:
                        raise AssertionError("restored graph state does not reproduce finite original action")
                    guards.append(dict(size=n, index=index, depth=depth, output_owned=owned,
                                       output_pointers_distinct_from_static_and_each_other=pointers_distinct,
                                       transpose_same_pointer_shape=same_pointer_shape, transpose_stride_changed=changed_stride,
                                       original_sigma_t_stride=old_t.stride(), transposed_sigma_t_stride=transposed.stride(),
                                       transpose_view_rejected=transpose, changed_depth_rejected=depth_guard,
                                       missing_cache_rejected=missing, restored_action_finite=restored_finite,
                                       restored_action_bitwise_equal=restored_equal,
                                       passed=owned and transpose and depth_guard and missing and restored_finite and restored_equal))
                    del graph, base, held, held_copy, subsequent, restored, native_restored
                checkpoint()
        settings.update(parity_checks=len(parity), all_parity_passed=len(parity) == 48 and all(r["passed"] for r in parity),
                        all_parity_bitwise_equal=all(r["bitwise_equal"] for r in parity),
                        all_ownership_and_guards_passed=len(guards) == 12 and all(r["passed"] for r in guards))
        checkpoint()
        if not settings["all_parity_passed"] or not settings["all_ownership_and_guards_passed"]:
            raise RuntimeError("entry parity/ownership/guard gate blocks full solves")

        for n in GRIDS:
            cfg, data = load_development(n)
            recipe = RECIPES[str(n)]
            base_name = "tsa8" if n < 256 else "tsa16"
            for index in range(16):
                bounded(f"full material N{n} ID{index}")
                torch.cuda.synchronize()
                began = time.perf_counter()
                low, high = material_operator(cfg, data, torch.tensor([index]), .1, "cuda")
                torch.cuda.synchronize()
                physical_setup = time.perf_counter() - began
                bounded("required original-base setup")
                began = time.perf_counter()
                base = build_base(low, base_name)
                torch.cuda.synchronize()
                base_setup = time.perf_counter() - began
                bounded("graph warmup and capture")
                graph = GraphClassicalV2(base, low.source)
                setups.append(dict(size=n, index=index, recipe=recipe, required_base=base_name,
                                   physical_setup_seconds=physical_setup, original_base_setup_seconds=base_setup,
                                   common_setup_seconds=physical_setup + base_setup,
                                   graph_extra_setup_seconds=graph.setup_seconds,
                                   graph_setup_includes_side_stream_warmup_and_capture=True,
                                   original_reference_inverse_bytes=base.reference.reference_transport_inverse.numel()
                                   * base.reference.reference_transport_inverse.element_size(),
                                   graph_shares_original_reference_cache=True, unused_bases_constructed=False))
                phase = high.phase_matrix.to(torch.complex128) * high.weights.to(torch.float64)[None, :]
                mu = float(torch.linalg.eigvalsh((phase + phase.mH) / 2).max())
                margin = float((high.sigma_t - high.sigma_s * mu).min())
                rng = random.Random(int(policy.get("timing_order_seed", 27003)) + n * 100 + index)
                warm_order = ["frozen_original", "frozen_graph"]
                rng.shuffle(warm_order)
                first_order = ["frozen_original", "frozen_graph"]
                rng.shuffle(first_order)
                for repeat, order in ((-1, warm_order), (0, first_order), (1, list(reversed(first_order)))):
                    values, statuses, durations = {}, {}, {}
                    for arm in order:
                        bounded(f"solve N{n} ID{index} repeat{repeat} {arm}")
                        bases = {base_name: graph if arm == "frozen_graph" else base}
                        torch.cuda.synchronize()
                        began = time.perf_counter()
                        value, status = solve(recipe, low, high, bases, {})
                        torch.cuda.synchronize()
                        durations[arm] = time.perf_counter() - began
                        values[arm], statuses[arm] = value, status
                    bounded("independent pair accuracy audit")
                    field_delta = relative(values["frozen_graph"], values["frozen_original"])
                    equal = bool(torch.equal(values["frozen_graph"], values["frozen_original"]))
                    same_iterations = statuses["frozen_original"].get("iterations") == statuses["frozen_graph"].get("iterations")
                    same_calls = statuses["frozen_original"].get("classical_calls") == statuses["frozen_graph"].get("classical_calls")
                    for position, arm in enumerate(order):
                        record = dict(size=n, index=index, recipe=recipe, method=arm, repeat=repeat,
                                      warmup=repeat == -1, order=position, pair_order=order, seconds=durations[arm],
                                      paired_full_solution_bitwise_equal=equal, angular_phase_hermitian_max_eigenvalue=mu,
                                      **statuses[arm])
                        record.update(accuracy(high, values[arm], statuses[arm], margin,
                                               field_delta if arm == "frozen_graph" else 0., same_iterations, same_calls))
                        record["timing_finite_positive"] = math.isfinite(durations[arm]) and durations[arm] > 0
                        record["accuracy_passed"] = record["accuracy_passed"] and equal and record["timing_finite_positive"]
                        rows.append(record)
                    checkpoint()
                medians = {arm: statistics.median(r["seconds"] for r in rows if r["size"] == n and r["index"] == index
                           and r["method"] == arm and not r["warmup"]) for arm in ("frozen_original", "frozen_graph")}
                original_time, graph_time = medians["frozen_original"], medians["frozen_graph"]
                saving = 1 - graph_time / original_time
                cost = {str(q): dict(original_estimated_seconds=physical_setup + base_setup + q * original_time,
                        graph_estimated_seconds=physical_setup + base_setup + graph.setup_seconds + q * graph_time,
                        estimated_saving_percent=100 * (1 - (physical_setup + base_setup + graph.setup_seconds + q * graph_time)
                        / (physical_setup + base_setup + q * original_time))) for q in (1, 4, 16, 64)}
                aggregate.append(dict(size=n, index=index, original_median_seconds=original_time,
                                      graph_median_seconds=graph_time, warm_saving_percent=100 * saving,
                                      individual_regression_exceeds_10_percent=graph_time > 1.1 * original_time,
                                      conditional_fixed_material_Q_cost=cost))
                checkpoint()
                del graph, base, low, high, values
                print(f"Completed frozen graph pair N{n} ID{index}", flush=True)
        bounded("final preservation and summary")
        settings["original_sources_verified_after"] = check_originals(policy)
        summaries = []
        for n in GRIDS:
            cases = [r for r in aggregate if r["size"] == n]
            original_mean = statistics.mean(r["original_median_seconds"] for r in cases)
            graph_mean = statistics.mean(r["graph_median_seconds"] for r in cases)
            summaries.append(dict(size=n, independent_media=16, original_mean_case_median_seconds=original_mean,
                                  graph_mean_case_median_seconds=graph_mean, saving_percent=100 * (1 - graph_mean / original_mean)))
        all_accuracy = len(rows) == 288 and all(r["accuracy_passed"] for r in rows)
        all_equal = len(rows) == 288 and all(r["paired_full_solution_bitwise_equal"] for r in rows)
        headroom = (all_accuracy and all_equal and not any(r["individual_regression_exceeds_10_percent"] for r in aggregate)
                    and all(r["saving_percent"] >= 5 for r in summaries if r["size"] in (128, 256)))
        settings.update(complete=True, full_solve_rows=len(rows), measured_solves=sum(not r["warmup"] for r in rows),
                        recorded_warmup_solves=sum(r["warmup"] for r in rows), all_solve_accuracy_passed=all_accuracy,
                        all_full_solution_bitwise_equal=all_equal, summaries=summaries,
                        exploratory_warm_headroom_gate_passed=headroom, winner_claim_allowed=headroom,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        memory_scope="whole-process peak includes entry captures/guards and full screen, not isolated one-method memory",
                        gpu_processes_after=gpu_processes())
        checkpoint()
    except Exception as error:
        settings.update(complete=False, error_type=type(error).__name__, error=str(error),
                        timed_out=isinstance(error, TimeoutError), winner_claim_allowed=False)
        checkpoint()
        raise


def run(args):
    existed = Path(args.output).exists()
    try:
        _run(args)
    except Exception as error:
        # Metadata/policy/imported-helper failures before _run's main try still
        # leave an explicit incomplete receipt; an existing output is untouched.
        output = Path(args.output)
        if not existed and output.is_dir():
            path = output / "settings.json"
            settings = json.loads(path.read_text()) if path.exists() else {}
            settings.update(complete=False, error_type=type(error).__name__, error=str(error),
                            timed_out=isinstance(error, TimeoutError), winner_claim_allowed=False,
                            wall_seconds=time.monotonic() - PROCESS_STARTED)
            save(path, settings)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-wall-seconds", type=float, default=300.)
    run(parser.parse_args())
