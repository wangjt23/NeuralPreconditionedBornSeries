"""Scalar-only frozen-certifier receipt analysis; no field/backend execution.

Usage: python analyze_certifier.py receipts/certifier_execution --output report.json
Optional --illustrate-equal-certificate prints a labelled algebraic scenario,
never an integrated solver speed measurement. Existing output is not replaced.
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics


OLD_COMPONENTS = {
    128: dict(prior_q1_mean_ms=223.3476, direct_q1_mean_ms=176.9263,
              preserved_reported_speedup_percent=20.7844),
    256: dict(prior_q1_mean_ms=162.1031, direct_q1_mean_ms=130.2582,
              preserved_reported_speedup_percent=19.6448),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_float(value):
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("Finite scalar numeric value required")
    return Fraction.from_float(float(value))


def stats(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return dict(count=0, minimum=None, maximum=None, mean=None, median=None)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Nonfinite recorded statistic")
    return dict(count=len(values), minimum=min(values), maximum=max(values),
                mean=statistics.mean(values), median=statistics.median(values))


def ratios_and_predicates(certificate):
    """Reconstruct exact cross-products from recorded directed scalar bounds."""
    c = certificate
    tau = Fraction(c['policy_tolerance_rational'])
    alpha = Fraction(c['reserve_fraction_rational'])
    solution_tau = Fraction(c['policy_solution_tolerance_rational'])
    if tau != Fraction(1, 100000) or alpha != Fraction(1, 100):
        raise ValueError("Unexpected frozen tolerance/reserve")
    q = exact_float(c['source_norm_lower'])
    center = exact_float(c['residual_center_norm_upper'])
    radius = exact_float(c['residual_radius_norm_upper'])
    residual = exact_float(c['residual_norm_upper'])
    x = exact_float(c['candidate_norm_lower'])
    margin = Fraction(c['coercivity_lower_rational'])
    if margin <= 0:
        raise ValueError("Nonpositive recorded exact margin")
    if margin != Fraction(c['coercivity_lower_numerator'], c['coercivity_lower_denominator']):
        raise ValueError("Exact margin representations differ")
    denominator = margin*x-residual
    predicates = dict(
        center=q > 0 and center <= (1-alpha)*tau*q,
        radius=q > 0 and radius <= alpha*tau*q,
        direct_residual=q > 0 and residual <= tau*q,
        solution=denominator > 0 and residual <= solution_tau*denominator)
    aliases = dict(center='reserve_center_accepted', radius='reserve_radius_accepted',
                   direct_residual='residual_accepted', solution='solution_error_accepted')
    mismatches = [name for name, field in aliases.items()
                  if bool(c[field]) != predicates[name]]
    expected = all(predicates.values())
    if bool(c['accepted']) != expected:
        mismatches.append('combined_acceptance')
    if bool(c['fixed_one_percent_reserve_accepted']) != all(
            predicates[k] for k in ('center', 'radius', 'direct_residual')):
        mismatches.append('combined_reserve')
    ratio = lambda a, b: float(a/b) if b > 0 else None
    return dict(
        predicates=predicates, acceptance_mismatches=mismatches,
        residual_over_tau_source=ratio(residual, tau*q),
        center_over_allocated_budget=ratio(center, (1-alpha)*tau*q),
        radius_over_allocated_budget=ratio(radius, alpha*tau*q),
        exact_relative_solution_upper=ratio(residual, denominator),
        source_norm_width=float(exact_float(c['source_norm_upper'])-q),
        center_norm_width=float(exact_float(c['residual_center_norm_upper'])-
                                exact_float(c['residual_center_norm_lower'])),
    )


def analyze(receipt_root, illustrate_equal_certificate=False, authored_code_dir=None):
    root = Path(receipt_root)
    paths = dict(certificates=root/'results/certificates.json',
                 settings=root/'results/settings.json', policy=root/'policy.json')
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Awaiting frozen receipt files: "+', '.join(missing))
    rows = json.loads(paths['certificates'].read_text())
    settings = json.loads(paths['settings'].read_text())
    policy = json.loads(paths['policy'].read_text())
    if settings['full_solves'] != 0 or settings['global_gpu_contract_established']:
        raise ValueError("Unexpected solver/global-contract scope")
    execution_path = root/'execution_receipt.json'
    execution = json.loads(execution_path.read_text()) if execution_path.is_file() else None
    authored_code_dir = (Path(authored_code_dir) if authored_code_dir is not None
                         else Path(__file__).resolve().parent)
    code_sources = {}
    for name in policy['code_sha256']:
        parts = Path(name).parts
        if len(parts) != 2 or parts[0] != 'code':
            raise ValueError("Unexpected authored code path")
        cached = root/name
        local = authored_code_dir/parts[1]
        code_sources[name] = cached if cached.is_file() else local
    code_checks = {name: path.is_file() and sha(path) == policy['code_sha256'][name]
                   for name, path in code_sources.items()}
    if not all(code_checks.values()):
        raise ValueError("Frozen code hash mismatch")
    recorded_policy = settings.get('policy_sha256')
    if recorded_policy != sha(paths['policy']):
        raise ValueError("Settings/policy hash mismatch")
    counters = settings['counters']
    actual = Counter(row['kind'] for row in rows)
    counter_names = dict(saved_teacher='saved_teacher_certificates',
                         manufactured_M_source='manufactured_fixture_certificates',
                         original_source_zero='negative_control_certificates')
    if set(actual)-set(counter_names):
        raise ValueError("Unexpected certificate kind")
    for kind, field in counter_names.items():
        if actual[kind] != counters[field]:
            raise ValueError("Certificate population/counter mismatch: "+kind)
    if settings['complete']:
        if len(rows) != policy['max_certificates']:
            raise ValueError("Completed certificate count mismatch")
        if counters['explicit_gpu_H_actions'] != policy['max_explicit_H_actions']:
            raise ValueError("Completed GPU diagnostic count mismatch")
        if counters['preconditioner_actions'] != policy['max_M_actions']:
            raise ValueError("Completed factor count mismatch")
    groups = defaultdict(list)
    events = []
    max_seen = 0
    for ordinal, row in enumerate(rows):
        n = int(row['size'])
        cache_class = ('first_process_certificate' if ordinal == 0 else
                       'extends_prior_twiddle_size' if n > max_seen else
                       'later_existing_twiddle_size')
        detail = ratios_and_predicates(row['certificate'])
        if detail['acceptance_mismatches']:
            raise ValueError("Exact acceptance mismatch at row%d: %s" %
                             (ordinal, detail['acceptance_mismatches']))
        event = dict(ordinal=ordinal, kind=row['kind'], size=n, index=row['index'],
                     inferred_cache_class=cache_class, prior_maximum_size=max_seen,
                     complete_verification_seconds=row['complete_verification_seconds'],
                     certificate_seconds=row['certificate']['elapsed_seconds'],
                     explicit_gpu_H_seconds=row['explicit_gpu_H_seconds'],
                     accepted=row['certificate']['accepted'], **detail)
        events.append(event)
        groups[(row['kind'], n)].append((row, event))
        max_seen = max(max_seen, n)
    summaries = []
    for (kind, n), items in sorted(groups.items()):
        costs = sorted(set().union(*(row['certificate']['costs'] for row, _ in items)))
        predicate_counts = {name: dict(passed=sum(event['predicates'][name] for _, event in items),
                                      total=len(items))
                            for name in ('center', 'radius', 'direct_residual', 'solution')}
        summary = dict(kind=kind, size=n, indices=[row['index'] for row, _ in items],
                       count=len(items), accepted=sum(row['certificate']['accepted'] for row, _ in items),
                       exact_predicate_counts=predicate_counts,
                       complete_verification_seconds=stats(row['complete_verification_seconds'] for row, _ in items),
                       certificate_seconds=stats(row['certificate']['elapsed_seconds'] for row, _ in items),
                       residual_relative_upper=stats(row['certificate']['relative_residual_upper'] for row, _ in items),
                       relative_solution_error_upper=stats(row['certificate']['relative_solution_error_upper'] for row, _ in items),
                       vector_radius_norm_upper=stats(row['certificate']['residual_radius_norm_upper'] for row, _ in items),
                       radius_over_fixed_reserve=stats(event['radius_over_allocated_budget'] for _, event in items),
                       center_over_fixed_budget=stats(event['center_over_allocated_budget'] for _, event in items),
                       direct_residual_over_tolerance=stats(event['residual_over_tau_source'] for _, event in items),
                       cost_parts_seconds={name: stats(row['certificate']['costs'].get(name) for row, _ in items)
                                           for name in costs},
                       first_or_extended_cache_verification_seconds=stats(
                           row['complete_verification_seconds'] for row, event in items
                           if event['inferred_cache_class'] != 'later_existing_twiddle_size'),
                       later_existing_cache_verification_seconds=stats(
                           row['complete_verification_seconds'] for row, event in items
                           if event['inferred_cache_class'] == 'later_existing_twiddle_size'))
        if illustrate_equal_certificate and n in OLD_COMPONENTS:
            old = OLD_COMPONENTS[n]
            measured_c_ms = 1000.*summary['complete_verification_seconds']['mean']
            difference = old['prior_q1_mean_ms']-old['direct_q1_mean_ms']
            summary['illustrative_equal_certificate_scenario'] = dict(
                assumed_same_C_ms=measured_c_ms,
                formula='(Tprior-Tdirect)/(Tprior+C)',
                hypothetical_fraction=difference/(old['prior_q1_mean_ms']+measured_c_ms),
                measured_integrated_speedup=False,
                caveat='C is a teacher/fixture certificate mean. Equal real/complex control verification cost and integrated solver cost were not measured; this is algebra only.')
        summaries.append(summary)
    return dict(
        receipt_root=str(root), receipt_sha256={name: sha(path) for name, path in paths.items()},
        execution_receipt_sha256=sha(execution_path) if execution else None,
        frozen_code_hashes_verified=True,
        code_hash_verification_sources={name: str(path) for name, path in code_sources.items()},
        complete=settings['complete'],
        execution=execution, counters=counters, rows=len(rows), groups=summaries,
        verification_events=events,
        cache_scope='Cold/extension/later labels inferred from this fresh driver call order, not cache telemetry or repeated cold controls.',
        verification_cost_scope='Capture originalD64/all CPU transfers/two hash passes/full interval certificate; separate GPU diagnostics and JSON persistence excluded; external whole-process receipt is authoritative.',
        raw_settings=settings,
        preserved_old_solver_components={str(n): dict(v, scope='Preserved prior solver components, separate from this certifier execution; rounded means and reported prior percentages retained without recomputation.')
                                         for n, v in OLD_COMPONENTS.items()},
        integrated_solver_speedup_measured=False, equal_control_certificate_cost_measured=False,
        global_gpu_contract_established=False, full_solves=0,
        interpretation='Input-specific nominal certificates under recorded native CPU/compiler contracts; no universal GPU/FFT proof, new integrated solver speedup, learned-method win or field export.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt_root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--authored-code-dir', type=Path,
                        help='Frozen authored sources when scalar-only receipts omit code; defaults to this script directory')
    parser.add_argument('--illustrate-equal-certificate', action='store_true')
    args = parser.parse_args()
    result = analyze(args.receipt_root, args.illustrate_equal_certificate,
                     args.authored_code_dir)
    encoded = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(encoded)
        print(json.dumps(dict(output=str(args.output), sha256=sha(args.output),
                              complete=result['complete'], rows=result['rows'])))
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()
