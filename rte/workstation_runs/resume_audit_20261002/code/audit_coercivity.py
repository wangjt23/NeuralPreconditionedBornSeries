"""Discrete rounded-phase coercivity audit; validation only, no teachers/training.

Run from the RTE repository root with PYTHONPATH=.; this file imports its current
rte package. Reported floating-point bounds implement an exact-arithmetic theorem,
not interval-certified arithmetic. Audit durations are separate from benchmarks.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from rte.benchmark_tail_solver import CLASSICAL, solve
from rte.mixed_precision_transport import high_precision_copy
from rte.models import PeriodicRTE2D
from rte.run_large_angular_spectral_experiments import dataset_fingerprint
from rte.train_source_predictor import build_base, material_operator, source_family


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def retain_nonfinite_fields(row):
    """Keep failed numerical evidence serializable rather than dropping a case."""
    bad = {key: str(value) for key, value in row.items()
           if isinstance(value, float) and not math.isfinite(value)}
    if bad:
        row = dict(row)
        for key in bad:
            row[key] = None
        row.update(failed=1, nonfinite_numeric_fields=bad)
    return row


@torch.no_grad()
def phase_margin(operator):
    """Use actual P columns weighted as in models.scattering_op, never ideal P."""
    phase = operator.phase_matrix.to(torch.float64)
    weights = operator.weights.to(torch.float64)
    matrix = phase * weights[None, :]
    hermitian = (matrix + matrix.mH) / 2
    mu = float(torch.linalg.eigvalsh(hermitian)[-1])
    margin = float((operator.sigma_t.double() - mu * operator.sigma_s.double()).min())
    matrix_norm = float(matrix.norm())
    return dict(
        phase_hermitian_max_eigenvalue=mu,
        coercivity_margin=margin,
        positive_margin=bool(math.isfinite(margin) and margin > 0),
        phase_relative_asymmetry=float((matrix - matrix.mH).norm()) / max(matrix_norm, 1e-300),
        phase_max_row_normalization_error=float((matrix.sum(1) - 1).abs().max()),
        phase_max_column_normalization_error=float((matrix.sum(0) - 1).abs().max()),
        uniform_angular_weights=bool(torch.equal(weights, weights[0].expand_as(weights))),
    )


@torch.no_grad()
def solution_bounds(operator, rhs, approximate, margin_record=None):
    """Recompute the original-equation residual independently after solving."""
    result = dict(phase_margin(operator) if margin_record is None else margin_record)
    rhs_norm = float(rhs.norm())
    field_norm = float(approximate.norm())
    residual_norm = float((rhs - operator.A_op(approximate)).norm())
    result.update(rhs_norm=rhs_norm, approximate_solution_norm=field_norm,
                  independently_checked_residual_norm=residual_norm,
                  independently_checked_relative_residual=residual_norm / rhs_norm if rhs_norm > 0 else None)
    delta = residual_norm / result["coercivity_margin"] if result["positive_margin"] else None
    valid = delta is not None and math.isfinite(delta) and math.isfinite(field_norm) and field_norm > delta
    result.update(absolute_solution_error_bound=delta,
                  relative_solution_error_bound=delta / (field_norm - delta) if valid else None,
                  relative_certificate_available=valid,
                  interpretation="algebraic bound evaluated in FP64; no interval rounding")
    return result


@torch.no_grad()
def manufactured_test():
    """A small CPU manufactured field checks absolute and computable bounds."""
    torch.set_num_threads(1)
    low = PeriodicRTE2D(shape=(4, 4), dx=.25, n_angles=12, anisotropy_g=.65,
                        eta_scale=0., fdtype=torch.float32)
    t = torch.full((1, 1, 4, 4), 3.)
    low.setup(t, torch.ones_like(t))
    high = high_precision_copy(low)
    generator = torch.Generator().manual_seed(10022026)
    true = torch.randn(1, 12, 4, 4, dtype=torch.complex128, generator=generator)
    approximate = true + .002 * torch.randn(true.shape, dtype=true.dtype, generator=generator)
    rhs = high.A_op(true)
    record = solution_bounds(high, rhs, approximate)
    error = float((true - approximate).norm())
    relative_error = error / float(true.norm())
    v = torch.randn(true.shape, dtype=true.dtype, generator=generator)
    w = torch.randn(true.shape, dtype=true.dtype, generator=generator)
    sv, sw = high.streaming_op(v), high.streaming_op(w)
    defect = ((v.conj() * sw).sum() + (sv.conj() * w).sum()).abs()
    scale = v.norm() * sw.norm() + sv.norm() * w.norm()
    skew_defect = float(defect / scale.clamp_min(1e-300))
    assert record["positive_margin"] and record["uniform_angular_weights"]
    assert error <= record["absolute_solution_error_bound"] * (1 + 1e-10) + 1e-12
    assert record["relative_certificate_available"]
    assert relative_error <= record["relative_solution_error_bound"] * (1 + 1e-10) + 1e-12
    assert skew_defect < 1e-12
    return dict(status="passed", scope="small CPU homogeneous manufactured field, no solver or teacher",
                actual_absolute_error=error, actual_relative_error=relative_error,
                streaming_relative_skew_adjoint_defect=skew_defect, **record)


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    selection_path = Path(args.classical_selection)
    selections = json.loads(selection_path.read_text())
    choices = {size: selections[str(size)]["classical"] for size in args.sizes}
    if any(method not in CLASSICAL for method in choices.values()):
        raise ValueError("only frozen classical selections are allowed")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    settings = dict(args=vars(args), complete=False, validation_seed=4626,
                    generated_validation_count=16, classical_choices=choices,
                    frozen_selection_sha256=digest(selection_path),
                    script_sha256=digest(__file__), torch=str(torch.__version__),
                    scope="existing validation family only; no teacher, neural training, tuning or test generation",
                    timing="single diagnostic solve and setup durations; NOT warm benchmark timings",
                    theorem="discrete exact-arithmetic coercivity; numerical FP64 evaluation without interval rounding",
                    streaming="pure-imaginary Fourier symbol with real stored directions/wavenumbers is skew-Hermitian",
                    source_sha256={name: digest(name) for name in (
                        "rte/models.py", "rte/mixed_precision_transport.py",
                        "rte/benchmark_tail_solver.py", "rte/train_source_predictor.py",
                        "rte/augmented_fgmres.py", "rte/source_initialization.py")})
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    rows, fingerprints = [], {}
    for size in args.sizes:
        # Count affects generator draw ordering. Generate the exact existing
        # sixteen-case cohort even when auditing only its first few cases.
        cfg, data = source_family(size, 16, 4626)
        fingerprints[str(size)] = dataset_fingerprint(data)
        method = choices[size]
        names = {"tsa8"}
        if method.endswith(("_balanced", "_zero")):
            names.add(method.rsplit("_", 1)[0])
        for index in range(args.count):
            row = dict(size=size, index=index, classical_method=method)
            try:
                synchronize(device)
                started = time.perf_counter()
                low, high = material_operator(cfg, data, torch.tensor([index]), .1, device)
                bases = {name: build_base(low, name) for name in sorted(names)}
                synchronize(device)
                row["audit_setup_seconds"] = time.perf_counter() - started
                margin = phase_margin(high)
                if not margin["uniform_angular_weights"]:
                    raise ValueError("certificate derivation assumes uniform angular weights")
                synchronize(device)
                started = time.perf_counter()
                approximate, status = solve(method, low, high, bases, {})
                synchronize(device)
                row.update(audit_single_solve_seconds=time.perf_counter() - started, **status)
                started = time.perf_counter()
                row.update(solution_bounds(high, high.source, approximate, margin))
                synchronize(device)
                row["audit_bound_seconds"] = time.perf_counter() - started
            except Exception as error:
                row.update(failed=1, audit_failure=f"{type(error).__name__}: {error}")
            row = retain_nonfinite_fields(row)
            rows.append(row)
            (out / "cases.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)
    settings.update(complete=True, validation_fingerprints=fingerprints,
                    cases=len(rows), solve_or_audit_failures=sum(bool(r.get("failed")) for r in rows))
    (out / "settings.json").write_text(json.dumps(settings, indent=2, allow_nan=False) + "\n")
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manufactured-test", action="store_true")
    parser.add_argument("--classical-selection", default="rte/workstation_runs/tail_solver_benchmark_20260926/frozen_selection.json")
    parser.add_argument("--sizes", type=int, nargs="+", choices=(64, 128, 256), default=[64, 128, 256])
    parser.add_argument("--count", type=int, default=16, help="First 1–16 cases of the exact sixteen-case validation cohort")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if not 1 <= args.count <= 16 or len(set(args.sizes)) != len(args.sizes):
        parser.error("count must be 1–16 and sizes must be distinct")
    if not args.manufactured_test and not args.output:
        parser.error("--output is required for a full audit")
    result = manufactured_test() if args.manufactured_test else run(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
