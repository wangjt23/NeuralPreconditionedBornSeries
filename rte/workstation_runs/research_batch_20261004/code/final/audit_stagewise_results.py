"""Stdlib-only audit of final A/B scalar receipts; never executes RTE/FFT.

Final receipts must be complete. Exact gates and radius propagation are
reconstructed from recorded binary64 endpoints and rational margins. Original
omitted arrays/actions, persisted NPZ files and remote preservation are not
independently reexecuted by this audit.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import statistics


TAU, ALPHA, SOLUTION, PAIR = F(1, 100000), F(1, 100), F(1, 1000), F(1, 10000)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def scalar(value):
    check(not isinstance(value, bool) and math.isfinite(float(value)), 'Nonfinite scalar')
    return F.from_float(float(value))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def up(value):
    result = float(value)
    check(math.isfinite(result), 'Scalar overflow')
    if scalar(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def down(value):
    result = float(value)
    check(math.isfinite(result), 'Scalar overflow')
    if scalar(result) > value:
        result = math.nextafter(result, -math.inf)
    return result


def strip_times(value):
    if isinstance(value, dict):
        return {k: strip_times(v) for k, v in value.items()
                if not k.endswith('_seconds') and k != 'costs'}
    if isinstance(value, list):
        return [strip_times(v) for v in value]
    return value


def guard(value, library):
    check(value['guard_flags'] == 127 and value['library_sha256'] == library,
          'Guard/library mismatch')
    check(not value['floating_control_modes_modified'], 'Mode change')


def pair_gates(row):
    check(F(row['paired_field_tolerance_rational']) == PAIR, 'Paired tolerance')
    dl, du = scalar(row['paired_field_difference_norm_lower']), scalar(row['paired_field_difference_norm_upper'])
    bl, bu = scalar(row['paired_field_baseline_norm_lower']), scalar(row['paired_field_baseline_norm_upper'])
    check(0 <= dl <= du and 0 <= bl <= bu, 'Pair norm ordering')
    accepted = bl > 0 and du <= PAIR*bl
    rejected = bu > 0 and dl > PAIR*bu
    check(row['paired_field_accepted'] == accepted and
          row['paired_field_rigorous_rejection'] == rejected, 'Exact paired gate')
    return dict(accepted=accepted, rigorous_rejection=rejected)


def stage_audit(c, size):
    rows = c['stage_ledger']; previous = F(0); collision = None
    kinds = []; stages = 0
    for row in rows:
        kind = row['kind']; kinds.append(kind)
        local = scalar(row['local_radius_upper'])
        check(local >= 0, 'Negative local radius')
        for name, value in row.items():
            if name.endswith('_seconds'):
                check(scalar(value) >= 0, 'Negative stage timing')
        if kind == 'collision_point_enclosure':
            collision = local
            continue
        check(scalar(row['incoming_radius_upper']) == previous, 'Lost incoming radius')
        if kind in ('forward_fft_stage', 'inverse_fft_stage'):
            factor = scalar(row['stage_operator_norm_upper'])
            check(factor > 0 and factor*factor >= 2, 'sqrt2 upper bound')
            check(scalar(math.nextafter(float(factor), 0.))**2 < 2, 'Minimal sqrt2 upper')
            target = factor*previous+local; stages += 1
            check(row['length'] == size and row['axis'] in (2, 3), 'Spatial stage layout')
        elif kind == 'inverse_normalization':
            check(F(row['normalization_rational']) == F(1, size), 'Inverse normalization')
            target = previous/F(size)+local
        elif kind == 'stored_D64_multiplication':
            factor = scalar(row['operator_norm_upper'])
            check(factor >= 0, 'Negative D norm')
            target = factor*previous+local
        elif kind == 'stream_collision_addition':
            check(collision is not None and scalar(row['collision_radius_upper']) == collision,
                  'Collision radius mismatch')
            target = previous+collision+local
        elif kind == 'source_action_subtraction':
            target = previous+local
        else:
            raise ValueError('Unknown radius stage '+kind)
        check(row['radius_upper'] == up(target), 'Outward radius recurrence')
        previous = scalar(row['radius_upper'])
    bits = size.bit_length()-1
    expected = (['forward_fft_stage']*(2*bits)+['stored_D64_multiplication']+
                ['inverse_fft_stage']*bits+['inverse_normalization']+
                ['inverse_fft_stage']*bits+['inverse_normalization']+
                ['collision_point_enclosure', 'stream_collision_addition', 'source_action_subtraction'])
    check(kinds == expected and stages == 4*bits, 'Exact stage sequence/count')
    check(previous == scalar(c['residual_radius_norm_upper']), 'Final radius mismatch')
    return dict(fft_stages=stages, ledger_rows=len(rows), all_recurrences_exactly_reproduced=True)


def certificate_audit(row, library):
    c = row['certificate']; size = row['size']
    check(F(c['policy_tolerance_rational']) == TAU and F(c['reserve_fraction_rational']) == ALPHA and
          F(c['policy_solution_tolerance_rational']) == SOLUTION, 'Fixed accuracy policy')
    margin = F(c['coercivity_lower_rational'])
    check(margin == F(c['coercivity_lower_numerator'], c['coercivity_lower_denominator']) and margin > 0,
          'Exact margin representations')
    check(margin <= F(c['derived_coercivity_lower_rational']), 'Supplied margin exceeds derived')
    q, x = scalar(c['source_norm_lower']), scalar(c['candidate_norm_lower'])
    cl, cu, e = (scalar(c[k]) for k in ('residual_center_norm_lower', 'residual_center_norm_upper', 'residual_radius_norm_upper'))
    rl, ru = scalar(c['residual_norm_lower']), scalar(c['residual_norm_upper'])
    check(0 <= cl <= cu and q > 0 and x > 0 and e >= 0, 'Norm validity')
    check(float(rl) == down(max(F(0), cl-e)) and float(ru) == up(cu+e), 'Residual norm reconstruction')
    denominator = margin*x-ru
    gates = dict(center=cu <= (1-ALPHA)*TAU*q, radius=e <= ALPHA*TAU*q,
                 residual=ru <= TAU*q, solution=denominator > 0 and ru <= SOLUTION*denominator)
    fields = dict(center='reserve_center_accepted', radius='reserve_radius_accepted',
                  residual='residual_accepted', solution='solution_error_accepted')
    for gate, name in fields.items():
        check(c[name] == gates[gate], 'Exact '+gate+' gate')
    pair = pair_gates(c)
    check(c['fixed_one_percent_reserve_accepted'] == all(gates[k] for k in ('center', 'radius', 'residual')),
          'Combined reserve')
    check(c['accuracy_accepted_without_paired_field'] == all(gates.values()), 'Combined accuracy')
    check(c['paired_field_gate_required'] and c['paired_field_gate_evaluated'], 'Missing actual baseline')
    check(c['paired_field_gate_established'] == pair['accepted'], 'Pair established flag')
    check(c['accepted'] == (all(gates.values()) and pair['accepted']), 'Combined admission')
    check(c['certified_under_cpu_contract'] and not c['global_gpu_contract_established'], 'Contract scope')
    for name in ('native_backend_guard_before', 'native_backend_guard_after',
                 'paired_field_guard_before', 'paired_field_guard_after'):
        guard(c[name], library)
    stage = stage_audit(c, size)
    return dict(size=size, index=row['index'], role=row['role'], snapshot_sha256=row['snapshot_sha256'],
                gates=gates, paired_gates=pair, accepted=c['accepted'], stage_audit=stage,
                relative_residual_upper=c['relative_residual_upper'],
                relative_radius_upper=c['relative_radius_norm_upper'],
                radius_reserve_fraction=float(e/(ALPHA*TAU*q)),
                relative_solution_error_upper=c['relative_solution_error_upper'],
                paired_field_relative_upper=c['paired_field_relative_upper'],
                certificate_call_seconds=row['certificate_call_seconds'])


def audit(receipt_root, authored_root):
    root, authored = Path(receipt_root), Path(authored_root)
    paths = dict(A_certificates=root/'jobA/results/certificates.json', A_pairs=root/'jobA/results/pairs.json',
                 A_settings=root/'jobA/results/settings.json', B_certificates=root/'jobB/certificates.json',
                 B_settings=root/'jobB/settings.json', snapshots=root/'jobA/snapshot_ledger.json',
                 preflight=root/'preflight_receipt.json', A_execution=root/'A_stagewise.execution.json',
                 B_execution=root/'B_optimized.execution.json', freeze_execution=root/'freeze.execution.json')
    data = {key: json.loads(path.read_text()) for key, path in paths.items()}
    a, b, sa, sb = (data[k] for k in ('A_certificates', 'B_certificates', 'A_settings', 'B_settings'))
    check(sa['complete'] and sb['complete'], 'Await final completed A/B receipts')
    check(len(a) == 8 and len(b) == 4 and len(data['A_pairs']) == 12, 'Population count')
    for settings in (sa, sb):
        check(settings['full_solves'] == 0 and settings['training_steps'] == 0, 'Execution scope')
        for name in ('source_arrays_exported', 'paired_solver_comparison', 'global_gpu_contract_established'):
            if name in settings:
                check(not settings[name], 'Execution scope flag '+name)
    for name in ('A_execution', 'B_execution', 'freeze_execution'):
        execution = data[name]
        check(execution['returncode'] == 0 and not execution['timeout'] and
              execution['external_process_seconds'] <= execution['hard_seconds'], 'Owned process budget '+name)
    preflight = data['preflight']; library = preflight['native_sha256']
    check(preflight['tests_returncode'] == 0 and '47 passed' in preflight['tests_stdout'] and
          '34 subtests passed' in preflight['tests_stdout'], 'FWI tiny preflight')
    policy = json.loads((authored/'policy.json').read_text())
    for name, expected in policy['code_sha256'].items():
        check(sha(authored/name) == expected, 'Local frozen code seal '+name)
    snapshots = data['snapshots']
    check(snapshots['complete'] and snapshots['all_snapshots_persisted_before_checker'] and
          snapshots['checker_calls'] == 0 and len(snapshots['fixtures']) == 6, 'Prechecker persistence')
    check(snapshots['counts'] == dict(full_solves=0, training_steps=0, new_seed_generation=0, M=2, H=2), 'Frozen action counts')
    expected_a = {(n, i, role) for n in (64, 128, 256) for i in (0, 3)
                  for role in (('baseline', 'positive') if n == 256 else ('baseline',))}
    expected_b = {(256, i, role) for i in (0, 3) for role in ('baseline', 'positive')}
    key = lambda row: (row['size'], row['index'], row['role'])
    check({key(row) for row in a} == expected_a and {key(row) for row in b} == expected_b, 'Exact matched cases')
    check({key(row) for row in data['A_pairs']} ==
          {(n, i, role) for n in (64, 128, 256) for i in (0, 3) for role in ('positive', 'negative')},
          'Exact paired-control cases')
    fixture_map = {(row['size'], row['index']): row for row in snapshots['fixtures']}
    for row in a+b:
        check(row['snapshot_sha256'] == fixture_map[(row['size'], row['index'])]['sha256'], 'Snapshot linkage')
    aa, bb = [certificate_audit(row, library) for row in a], [certificate_audit(row, library) for row in b]
    amap = {key(row): row for row in a}; matches = []
    ignored = {'native_backend_guard_before', 'native_backend_guard_after',
               'paired_field_guard_before', 'paired_field_guard_after'}
    for row in b:
        other = amap[key(row)]
        check(strip_times(other['certificate']['stage_ledger']) == strip_times(row['certificate']['stage_ledger']),
              'Matched non-timing stage ledger differs')
        ca = {k: v for k, v in strip_times(other['certificate']).items() if k not in ignored}
        cb = {k: v for k, v in strip_times(row['certificate']).items() if k not in ignored}
        check(ca == cb, 'Matched enclosures/gates differ')
        ta, tb = other['certificate_call_seconds'], row['certificate_call_seconds']
        matches.append(dict(size=256, index=row['index'], role=row['role'], A_seconds=ta, B_seconds=tb,
                            observed_certificate_call_reduction_fraction=(ta-tb)/ta,
                            scalar_enclosures_and_non_timing_ledger_identical=True))
    pairs = []
    for row in data['A_pairs']:
        gates = pair_gates(row)
        check(gates['accepted'] if row['role'] == 'positive' else gates['rigorous_rejection'], 'Direct pair control outcome')
        pairs.append(dict(size=row['size'], index=row['index'], role=row['role'], **gates))
    return dict(audit_success=True, scope='Exact scalar reconstruction only; no omitted arrays/actions/NPZ or FWI seal reexecution',
                receipt_sha256={name: sha(path) for name, path in paths.items()},
                source_hashes_verified=len(policy['code_sha256']), native_library_sha256=library,
                preflight_tests=47, preflight_subtests=34, A=aa, B=bb, direct_pair_controls=pairs,
                matched_cases=matches, all_matched_enclosures_and_stage_ledgers_identical=True,
                A_settings=sa, B_settings=sb, A_execution=data['A_execution'], B_execution=data['B_execution'],
                B_unreported_scope_flags=[name for name in ('source_arrays_exported', 'paired_solver_comparison', 'global_gpu_contract_established') if name not in sb],
                freeze_execution=data['freeze_execution'], snapshot_costs=snapshots['costs'],
                snapshot_seconds=snapshots['elapsed_seconds'],
                observed_mean_matched_call_reduction_fraction=statistics.mean(r['observed_certificate_call_reduction_fraction'] for r in matches),
                cost_caveats=['Concurrent bounded batch load; no dedicated paired randomized repeat benchmark.',
                             'A has eight certificates and prior N64/N128 twiddle warming; B has four N256 certificates and its own cold process.',
                             'Matched call costs are observations, not an integrated solver or learned-method speedup.',
                             'Snapshot construction, startup/readback/hash/serialization/native preflight and preservation are additionally billable; nested timers must not be added twice.'],
                full_solver_comparison_admitted=False, learned_advantage_established=False,
                independently_reexecuted_original_fields=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipts', type=Path, default=Path('receipts/batch_20261004_execution'))
    parser.add_argument('--authored', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = audit(args.receipts, args.authored)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(audit_success=True, sha256=sha(args.output), matched_cases=len(result['matched_cases']))))


if __name__ == '__main__':
    main()
