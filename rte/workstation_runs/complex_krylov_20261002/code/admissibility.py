"""Frozen-probe complex-linearity consistency checks; scalar receipts only.

This is not a global linearity certificate. The caller owns the original
action, fixed probes, dtype/cast path, numerical cap and execution budget.
No operator, material, cache, random field or reference inverse is created.
The gate uses noncancelling *reference-output* sums, as frozen before runs.
Symmetric-output normalizers are additionally recorded, never used to relax
that gate. All images remain temporary tensors within check_action.
"""

import math

import torch


DEFAULT_ALPHA = .73 + .29j
DEFAULT_BETA = -.41 + .37j
PHASE_ANGLE = .37
PHASE = complex(math.cos(PHASE_ANGLE), math.sin(PHASE_ANGLE))
REAL_SCALE = -1.7


def _finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _norm(value):
    # Difference/norm accumulation is diagnostic128, not a replacement for
    # the actual action's native/cast precision or operation order.
    return float(torch.linalg.vector_norm(value.to(torch.complex128)))


def _metadata(value):
    norm = _norm(value)
    real_norm = float(torch.linalg.vector_norm(value.real.to(torch.float64)))
    imag_norm = (float(torch.linalg.vector_norm(value.imag.to(torch.float64)))
                 if value.is_complex() else 0.)
    finite = bool(torch.isfinite(value).all())
    valid_norm = math.isfinite(norm) and norm > 0
    return dict(dtype=str(value.dtype), device=str(value.device),
                shape=list(value.shape), stride=list(value.stride()),
                finite=finite, norm=_finite_number(norm),
                real_norm=_finite_number(real_norm),
                imaginary_norm=_finite_number(imag_norm),
                imaginary_norm_fraction=imag_norm / norm if valid_norm else None,
                imaginary_energy_fraction=(imag_norm / norm)**2 if valid_norm else None)


def _identity(lhs, predicted, reference_terms, tolerance):
    """reference_terms contains (scalar magnitude, recorded output norm)."""
    numerator = _norm(lhs.to(torch.complex128) - predicted.to(torch.complex128))
    lhs_norm, predicted_norm = _norm(lhs), _norm(predicted)
    denominator = sum(weight * norm for weight, norm in reference_terms)
    symmetric_denominator = lhs_norm + denominator
    finite = (bool(torch.isfinite(lhs).all()) and bool(torch.isfinite(predicted).all())
              and all(math.isfinite(v) for v in
                      (numerator, lhs_norm, predicted_norm, denominator,
                       symmetric_denominator)))
    eligible = finite and denominator > 0
    ratio = numerator / denominator if eligible else None
    symmetric_ratio = (numerator / symmetric_denominator
                       if eligible and symmetric_denominator > 0 else None)
    # Finite raw norms can still yield an overflowing quotient. Keep strict
    # JSON receipts and fail such an uninformative numerical comparison.
    ratio_overflow = ratio is not None and not math.isfinite(ratio)
    if ratio_overflow:
        finite, eligible, ratio = False, False, None
    if symmetric_ratio is not None and not math.isfinite(symmetric_ratio):
        finite, eligible, symmetric_ratio = False, False, None
    return dict(numerator=_finite_number(numerator),
                denominator=_finite_number(denominator), ratio=ratio,
                symmetric_denominator=_finite_number(symmetric_denominator),
                symmetric_ratio=symmetric_ratio, lhs_norm=_finite_number(lhs_norm),
                predicted_norm=_finite_number(predicted_norm), finite=finite,
                eligible=eligible, ratio_overflow=ratio_overflow,
                passed=bool(eligible and ratio is not None and ratio <= tolerance))


@torch.no_grad()
def check_action(A, x, y, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, tolerance=None,
                 expected_output_dtype=None):
    """Check the actual supplied action on one frozen pair, returning JSON data.

    Ten action calls: zero, x, y, and i/phase/real scalings of BOTH probes,
    plus alpha*x+beta*y. Input expressions and output predictions retain
    their actual tensor dtype; only difference/norm diagnostics promote128.
    Snapshots prevent reused graph-output storage from corrupting receipts.
    Input mutation, invalid outputs, action errors or null denominators fail.
    Output dtype defaults to the input dtype; a deliberate conversion must
    declare expected_output_dtype. Snapshots do not establish caller ownership
    of the raw returned storage: that remains a separate cross-call check.

    The observed input->output dtypes are recorded. Internal casts cannot be
    inferred from a callable; the driver must separately freeze its declared
    wrapper/cast path. No Tensor, field/image values or storage addresses are
    returned. An absent/invalid tolerance is a configuration error.
    """
    if not callable(A):
        raise TypeError("A must be callable")
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("x and y must be tensors")
    if (not x.is_complex() or not y.is_complex() or x.shape != y.shape
            or x.dtype != y.dtype or x.device != y.device):
        raise ValueError("use matching complex input shape, dtype and device")
    expected_output_dtype = x.dtype if expected_output_dtype is None else expected_output_dtype
    if expected_output_dtype not in (torch.complex64, torch.complex128):
        raise ValueError("declare a supported complex output dtype")
    if tolerance is None or isinstance(tolerance, bool):
        raise ValueError("provide an explicit positive finite tolerance")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("provide an explicit positive finite tolerance")
    alpha, beta = complex(alpha), complex(beta)
    if not all(math.isfinite(v) for v in (alpha.real, alpha.imag, beta.real, beta.imag)):
        raise ValueError("combination scalars must be finite")

    # Owned probe copies preserve caller fields, including when A is invalid.
    x, y = x.detach().clone(), y.detach().clone()
    result = dict(tolerance=tolerance, finite=False, exact_zero=False,
                  passed=False, max_normalized_defect=None,
                  i_homogeneity=None, phase_homogeneity=None,
                  real_scaling=None, complex_combination=None,
                  normalization="reference output sum; no denominator clamp",
                  symmetric_normalization="lhs norm plus reference output sum; diagnostic only",
                  metric_arithmetic="native input/output expressions; complex128 differences/norms",
                  complex_linearity_certificate=False, action_calls=0,
                  coefficients=dict(alpha=dict(real=alpha.real, imaginary=alpha.imag),
                                    beta=dict(real=beta.real, imaginary=beta.imag),
                                    phase_angle=PHASE_ANGLE,
                                    phase=dict(real=PHASE.real, imaginary=PHASE.imag),
                                    real_scale=REAL_SCALE),
                  input=dict(x=_metadata(x), y=_metadata(y)), outputs={},
                  observed_dtype_paths={}, internal_casts_introspected=False,
                  expected_output_dtype=str(expected_output_dtype),
                  output_dtype_consistent=False,
                  input_mutated=False, output_contract_passed=True,
                  errors=[], checks={})
    images = {}

    def call(name, value):
        # Clone per call so an invalid mutating action cannot alter later probes.
        argument = value.detach().clone()
        original = argument.clone()
        before_version = argument._version
        result["action_calls"] += 1
        try:
            image = A(argument)
            mutated = (argument._version != before_version
                       or not torch.equal(argument, original))
            result["input_mutated"] |= mutated
            if not isinstance(image, torch.Tensor):
                raise TypeError("action output must be a tensor")
            contract = (image.shape == value.shape and image.device == value.device
                        and image.is_complex() and image.dtype == expected_output_dtype)
            result["output_contract_passed"] &= bool(contract)
            if not contract:
                result["errors"].append(dict(call=name, type="OutputContract",
                    message="matching shape/device and declared complex output dtype required"))
            result["observed_dtype_paths"][name] = dict(input_dtype=str(argument.dtype),
                                                       output_dtype=str(image.dtype))
            # A static-buffer action may overwrite its next return value.
            snapshot = image.detach().clone()
            result["outputs"][name] = _metadata(snapshot)
            if image.shape == value.shape and image.device == value.device:
                images[name] = snapshot
        except Exception as error:
            result["output_contract_passed"] = False
            result["errors"].append(dict(call=name, type=type(error).__name__,
                                         message=str(error)[:300]))
        finally:
            # Also detects mutation before an action raises.
            result["input_mutated"] |= (argument._version != before_version
                                        or not torch.equal(argument, original))

    call("zero", torch.zeros_like(x))
    call("x", x)
    call("y", y)
    for suffix, value in (("x", x), ("y", y)):
        call("i_" + suffix, 1j * value)
        call("phase_" + suffix, PHASE * value)
        call("real_" + suffix, REAL_SCALE * value)
    call("complex_combination", alpha * x + beta * y)

    result["exact_zero"] = bool("zero" in images
                                 and torch.count_nonzero(images["zero"]) == 0)
    result["finite"] = (all(item["finite"] for item in result["input"].values())
                         and len(result["outputs"]) == 10
                         and all(item["finite"] for item in result["outputs"].values()))
    result["output_dtype_consistent"] = bool(len(result["outputs"]) == 10
        and {item["dtype"] for item in result["outputs"].values()} == {str(expected_output_dtype)})
    if "x" in images and "y" in images:
        norms = {name: _norm(images[name]) for name in ("x", "y")}
        for category, scalar in (("i", 1j), ("phase", PHASE), ("real", REAL_SCALE)):
            for suffix in ("x", "y"):
                name = category + "_" + suffix
                if name in images:
                    predicted = scalar * images[suffix]
                    result["checks"][name] = _identity(images[name], predicted,
                        [(abs(scalar), norms[suffix])], tolerance)
        if "complex_combination" in images:
            predicted = alpha * images["x"] + beta * images["y"]
            result["checks"]["complex_combination"] = _identity(
                images["complex_combination"], predicted,
                [(abs(alpha), norms["x"]), (abs(beta), norms["y"])], tolerance)

    categories = {"i_homogeneity": ("i_x", "i_y"),
                  "phase_homogeneity": ("phase_x", "phase_y"),
                  "real_scaling": ("real_x", "real_y"),
                  "complex_combination": ("complex_combination",)}
    for category, names in categories.items():
        values = [result["checks"].get(name, {}).get("ratio") for name in names]
        if all(value is not None for value in values):
            result[category] = max(values)
    values = [item["ratio"] for item in result["checks"].values()
              if item["ratio"] is not None]
    result["max_normalized_defect"] = max(values) if values else None
    result["passed"] = bool(result["finite"] and result["exact_zero"]
        and result["output_contract_passed"] and result["output_dtype_consistent"]
        and not result["input_mutated"]
        and not result["errors"] and len(result["checks"]) == 7
        and all(item["passed"] for item in result["checks"].values()))
    return result
