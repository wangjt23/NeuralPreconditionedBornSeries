"""Reproduce the C/D audit from scalar receipts only; no NumPy/Torch/RTE imports."""
import argparse
import hashlib
import json
import math
import pathlib
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    value = json.loads(path.read_text())
    def finite(x):
        if isinstance(x, dict):
            for v in x.values():
                finite(v)
        elif isinstance(x, list):
            for v in x:
                finite(v)
        elif isinstance(x, float):
            assert math.isfinite(x), path
    finite(value)
    return value


def average(values):
    return statistics.mean(values)


def audit(root, authored):
    policy_path = authored / 'policy.json'
    policy = read(policy_path)
    preflight = read(root / 'preflight_receipt.json')
    assert policy['max_full_solves'] == 0
    assert policy['development_ids'] == [0, 3]
    assert not policy['training'] and not policy['new_seed_generation']
    assert preflight['head'] == policy['head']
    assert preflight['tests_returncode'] == 0
    assert '47 passed, 34 subtests passed' in preflight['tests_stdout']
    assert preflight['staged_sha256']['policy.json'] == sha(policy_path)
    for name, digest in policy['code_sha256'].items():
        assert sha(authored / name) == digest, name
        assert preflight['staged_sha256'][name] == digest, name

    output = dict(
        schema='RTE_C_D_scalar_audit_20261004_v1',
        scope='Frozen action-only jobs C/D; independent scalar audit, no FWI reexecution or field export',
        policy_sha256=sha(policy_path),
        authored_code_hashes_all_match=True,
        preflight=dict(tests=47, subtests=34, returncode=0,
                       native_compile_seconds=preflight['compile_seconds'],
                       tests_external_seconds=preflight['tests_seconds'],
                       pre_batch_preservation_seconds=preflight['preservation_seconds'],
                       prior_files_root_verified=preflight['prior_files_verified']),
        receipt_sha256={}, jobs={},
        boundaries=dict(full_solver_admission=False, learned_advantage_established=False,
                        original_source_solution_accuracy_established=False,
                        actual_AZ_projector_used=False, holdout_or_training_used=False,
                        persisted_binary_fields_independently_rehashed=False,
                        historical_teacher_acquisition_cost='unknown, not zero',
                        new_research_authorized=False),
    )
    expected_cases = {(n, i) for n in (128, 256) for i in (0, 3)}
    contexts = ('source', 'one_strong_action_defect')
    all_manifests = {}
    for job, mode, maps, expected_counts, expected_snapshots in (
        ('jobC', 'local', ('strong', 'reference', 'material_local'),
         dict(H=52, M=28, S=16, full_solves=0, training_steps=0, new_seed_generation=0), 60),
        ('jobD', 'coarse', ('teacher_column', 'reference_column'),
         dict(H=36, M=4, S=0, full_solves=0, training_steps=0, new_seed_generation=0), 36),
    ):
        rows = read(root / job / 'results.json')
        settings = read(root / job / 'settings.json')
        setup = read(root / job / 'setup.json')
        manifest = read(root / job / 'candidates/manifest.json')
        execution_name = 'C_local.execution.json' if job == 'jobC' else 'D_coarse.execution.json'
        execution = read(root / execution_name)
        assert settings['complete'] and settings['mode'] == mode
        assert settings['counts'] == expected_counts
        assert settings['full_solves'] == settings['training_steps'] == 0
        assert execution['returncode'] == 0 and not execution['timeout']
        assert execution['hard_seconds'] == policy['gpu_job_hard_seconds']
        assert execution['external_process_seconds'] < execution['hard_seconds']
        assert settings['elapsed_seconds'] < policy['gpu_job_soft_seconds']
        assert settings['gpu_peak_reserved_bytes'] <= policy['max_gpu_reserved_bytes']
        assert settings['warm_apply_is_component_only']
        assert settings['no_solution_residual_accuracy_or_pair_claim']
        keys = {(r['size'], r['index'], r['context'], r['map']) for r in rows}
        expected_keys = {(n, i, c, m) for n, i in expected_cases for c in contexts for m in maps}
        assert len(rows) == len(keys) and keys == expected_keys
        assert len(setup) == 4 and {(s['size'], s['index']) for s in setup} == expected_cases
        for r in rows:
            assert len(r['apply_seconds']) == 3 and min(r['apply_seconds']) >= 0
            assert sorted(r['apply_seconds'])[1] == r['median_apply_seconds']
            assert r['ordinary_fp64_diagnostic'] and not r['certification_established']
            assert r['no_fullsolver_result']
            assert r['constructed_image_linearity_gap'] <= 1e-10
            assert abs(r['constructed_field_actual_H_line_ratio'] -
                       r['measured_image_best_real_line_ratio']) <= 1e-10
            assert r['constructed_field_actual_H_line_ratio'] >= 0
        assert len(manifest['snapshots']) == expected_snapshots
        assert sum(s['bytes'] for s in manifest['snapshots']) == manifest['total_bytes']
        assert manifest['total_bytes'] <= policy['max_gpu_job_snapshot_bytes']
        names = {}
        for s in manifest['snapshots']:
            name = pathlib.PurePosixPath(s['path']).name
            assert name not in names
            names[name] = s
            assert s['dtype'] == 'torch.complex128' and s['saved_before_physical_check']
            assert s['shape'][:2] == [1, 12] and s['shape'][2:] in ([128, 128], [256, 256])
            assert len(s['sha256']) == len(s['field_sha256']) == 64
            int(s['sha256'], 16); int(s['field_sha256'], 16)
        all_manifests[job] = names
        aggregates = []
        for n in (128, 256):
            for context in contexts:
                result = dict(size=n, context=context, maps={})
                for name in maps:
                    selected = [r for r in rows if r['size'] == n and r['context'] == context and r['map'] == name]
                    result['maps'][name] = dict(
                        mean_actual_constructed_residual_ratio=average(r['constructed_field_actual_H_line_ratio'] for r in selected),
                        mean_actual_constructed_squared_residual_ratio=average(r['constructed_field_actual_H_line_ratio']**2 for r in selected),
                        mean_component_median_apply_seconds=average(r['median_apply_seconds'] for r in selected),
                    )
                if job == 'jobC':
                    local, strong = result['maps']['material_local'], result['maps']['strong']
                    result['residual_excess_mean_square'] = local['mean_actual_constructed_squared_residual_ratio'] - strong['mean_actual_constructed_squared_residual_ratio']
                    result['component_apply_cost_ratio_to_strong'] = local['mean_component_median_apply_seconds'] / strong['mean_component_median_apply_seconds']
                    result['residual_gate_pass'] = result['residual_excess_mean_square'] <= .02
                    result['component_cost_gate_pass'] = result['component_apply_cost_ratio_to_strong'] <= .25
                    result['stratum_gate_pass'] = result['residual_gate_pass'] and result['component_cost_gate_pass']
                else:
                    teacher, reference = result['maps']['teacher_column'], result['maps']['reference_column']
                    result['teacher_squared_residual_gain_over_reference'] = reference['mean_actual_constructed_squared_residual_ratio'] - teacher['mean_actual_constructed_squared_residual_ratio']
                    result['stratum_gate_pass'] = result['teacher_squared_residual_gain_over_reference'] >= .02
                aggregates.append(result)
        timing_keys = [k for k in rows[0] if k.endswith('_seconds') and k != 'apply_seconds']
        component_sums = {k: sum(r[k] for r in rows) for k in timing_keys if k != 'median_apply_seconds'}
        component_sums['all_three_measured_apply_repeats_seconds'] = sum(sum(r['apply_seconds']) for r in rows)
        setup_sums = {k: sum(s[k] for s in setup) for k in setup[0] if k.endswith('_seconds')}
        diagnostics = [s['candidate_diagnostics'] for s in setup]
        if job == 'jobC':
            for d in diagnostics:
                assert d['condition_2'] <= 100 and d['reconstruction_relative'] <= 1e-12
                assert d['eigen_residual_relative'] <= 1e-12 and d['reference_matrix_closure_max'] <= 1e-10
                assert d['min_denominator_magnitude'] > 0 and not d['exact_or_interval_certificate']
                assert not d['parameter_sweep'] and d['full_original_phase_used']
            commutator = read(root / job / 'commutator.json')
            assert len(commutator) == 8
            assert {(r['size'], r['index'], r['context']) for r in commutator} == {(n, i, c) for n, i in expected_cases for c in contexts}
            assert all(r['identity_discrepancy_relative'] <= 1e-10 and r['ordinary_fp64_diagnostic_not_proof'] for r in commutator)
            commutator_summary = dict(
                reference_closure_max=max(r['reference_closure_relative'] for r in commutator),
                local_collision_closure_max=max(r['local_collision_closure_relative'] for r in commutator),
                streaming_material_penalty_min=min(r['streaming_material_penalty_relative'] for r in commutator),
                streaming_material_penalty_max=max(r['streaming_material_penalty_relative'] for r in commutator),
                identity_discrepancy_max=max(r['identity_discrepancy_relative'] for r in commutator),
                max_norm_difference_penalty_vs_actual_residual=max(abs(r['actual_residual_relative']-r['streaming_material_penalty_relative']) for r in commutator),
                diagnostics_seconds=sum(r['diagnostic_seconds'] for r in commutator),
            )
        else:
            for d in diagnostics:
                assert d['old_teacher_acquisition_seconds'] is None
                assert d['acquisition_cost_unknown_not_zero'] and d['no_new_teacher_solve']
                assert d['reused_same_development_source_teacher']
                assert all(d['crossgrid_source_geometry_parameter_identity'].values())
                assert d['full_angles'] == 12 and d['coarse_size'] == 64
                assert d['interpolation_relative_error'] <= 1e-12
            commutator_summary = None
            for n, i in expected_cases:
                for m in maps:
                    source = names[f'n{n}_id{i}_source_{m}.pt']['field_sha256']
                    frontier = names[f'n{n}_id{i}_one_strong_action_defect_{m}.pt']['field_sha256']
                    assert source == frontier, 'Fixed source column must be unchanged across residuals'
        summary = dict(
            rows=len(rows), counts=expected_counts, snapshot_count=expected_snapshots,
            snapshot_bytes=manifest['total_bytes'], scalar_payloads_finite=True,
            external_process_seconds=execution['external_process_seconds'],
            driver_seconds=settings['elapsed_seconds'], gpu_peak_reserved_bytes=settings['gpu_peak_reserved_bytes'],
            gpu_peak_allocated_bytes=settings['gpu_peak_allocated_bytes'],
            constructed_field_original_H_checks=2*len(rows),
            constructed_image_linearity_gap_max=max(r['constructed_image_linearity_gap'] for r in rows),
            imagefit_actual_constructed_ratio_difference_max=max(abs(r['constructed_field_actual_H_line_ratio']-r['measured_image_best_real_line_ratio']) for r in rows),
            aggregates=aggregates, screen_gate_pass=all(r['stratum_gate_pass'] for r in aggregates),
            setup_ledger_sums_seconds=setup_sums, measured_component_sums_seconds=component_sums,
            setup_cases=setup, commutator_summary=commutator_summary,
            only_component_cost_not_full_solver=True,
        )
        # There are len(rows) candidate-H checks and len(rows) constructed-H checks.
        summary['candidate_H_checks'] = len(rows)
        summary['constructed_field_original_H_checks'] = len(rows)
        output['jobs'][job] = summary
        paths = sorted((root / job).rglob('*.json')) + [root / execution_name]
        for path in paths:
            output['receipt_sha256'][path.relative_to(root).as_posix()] = sha(path)
    for n, i in expected_cases:
        name = f'n{n}_id{i}_strong_source.pt'
        assert all_manifests['jobC'][name]['field_sha256'] == all_manifests['jobD'][name]['field_sha256']
    output['identical_recorded_strong_source_fields_across_C_D'] = True
    output['receipt_sha256']['preflight_receipt.json'] = sha(root / 'preflight_receipt.json')
    output['decision'] = 'Both fixed action-only gates fail; retain negatives and stop new research this round'
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipts', type=pathlib.Path, default=pathlib.Path('receipts/batch_20261004_execution'))
    parser.add_argument('--authored', type=pathlib.Path, default=pathlib.Path('deliverables/batch_20261004'))
    parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('deliverables/batch_20261004/RESULT_DIRECTION_AUDIT.json'))
    args = parser.parse_args()
    value = audit(args.receipts, args.authored)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(dict(output=args.output.name, jobs={k: dict(rows=v['rows'], gate=v['screen_gate_pass']) for k, v in value['jobs'].items()})))


if __name__ == '__main__':
    main()
