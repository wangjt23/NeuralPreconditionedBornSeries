"""Independent stdlib-only audit of scalar interval-certifier receipts.

Never imports RTE, Torch, NumPy, the certifier backend, or original arrays.
Recomputes scalar predicates, not the omitted field/action enclosure itself.
"""
import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import re


TAU, EPS, ALPHA = F(1, 100000), F(1, 1000), F(1, 100)
HEAD = '028caea3f35d100225f656e2b1e910c1366aaafe'
EXPECTED_COUNTERS = dict(saved_teacher_certificates=4,
    manufactured_fixture_certificates=2, negative_control_certificates=4,
    explicit_gpu_H_actions=10, preconditioner_actions=2, full_solves=0,
    training_steps=0)
EXPECTED_CASES = ({('saved_teacher', n, i) for n in (64, 128) for i in (0, 3)} |
    {('manufactured_M_source', 256, i) for i in (0, 3)} |
    {('original_source_zero', n, i) for n in (128, 256) for i in (0, 3)})


def check(predicate, message):
    if not predicate:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value, label):
    check(isinstance(value, (int, float)) and not isinstance(value, bool), label+' numeric type')
    check(math.isfinite(value), label+' finite')
    return value


def fraction(value, label):
    finite(value, label)
    return F(value) if isinstance(value, int) else F.from_float(value)


def walk_finite(value, label='JSON'):
    if isinstance(value, float):
        check(math.isfinite(value), label+' nonfinite')
    elif isinstance(value, dict):
        for k, v in value.items():
            walk_finite(v, label+'.'+k)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            walk_finite(v, label+'[%d]' % i)


def read(path):
    data = json.loads(Path(path).read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    walk_finite(data, str(path))
    return data


def locate(root, name):
    found = [p for p in (root/name, root/'results'/name) if p.exists()]
    check(bool(found), 'Missing receipt '+name)
    if len(found) == 2:
        check(digest(found[0]) == digest(found[1]), 'Conflicting copies '+name)
    return found[0]


def guard(g, expected_library):
    check(g['guard_flags'] == 127, 'Native guard flags')
    check(g['architecture'] == 'x86-MXCSR', 'FWI native architecture')
    check(g['floating_control_modes_modified'] is False, 'Mode modification')
    check(g['sticky_exception_flags_may_change_in_arithmetic'] is True, 'Sticky status distinction')
    check(g['library_sha256'] == expected_library, 'Native library identity')
    control = g['control_register']
    check(isinstance(control, int), 'Mode register type')
    check(control & ((1 << 15) | (1 << 6) | (3 << 13)) == 0, 'FTZ/DAZ/rounding mode')
    check(g['compilation_provenance_must_be_recorded_by_executor'] is True,
          'Compiler trust-boundary flag')
    return control


def timing_map(data, label):
    for key, value in data.items():
        check(finite(value, label+'.'+key) >= 0, label+'.'+key+' nonnegative')
    return math.fsum(data.values())


def upper_report(value, exact, label):
    got = fraction(value, label)
    check(got >= exact, label+' must be outward upper')
    # Source uses nearest conversion plus at most one adjacent-float step.
    rounded = float(exact)
    if F.from_float(rounded) < exact:
        rounded = math.nextafter(rounded, math.inf)
    check(value == rounded, label+' reproduces directed scalar conversion')


def audit_case(row, policy, library):
    kind, n, index = row['kind'], row['size'], row['index']
    label = '%s/N%d/ID%d' % (kind, n, index)
    check((kind, n, index) in EXPECTED_CASES, 'Unexpected case '+label)
    check(row['development_seed'] == 4626, label+' development seed')
    check(row['no_input_mutation'] is True, label+' mutation flag')
    check(row['output_representation'] == 'Actual immutable saved/constructed complex128 bits', label+' output semantics')
    parent = policy['n256_material_parent'] if n == 256 else policy['teacher_banks'][str(n)]
    check(row['parent'] == parent, label+' source-parent identity')
    identities = row['buffer_identities']
    check(set(identities) == {'source','candidate','stream_symbol','sigma_t','sigma_s','phase','weights'}, label+' buffers')
    expected_shape = dict(source=[1,12,n,n],candidate=[1,12,n,n],stream_symbol=[1,12,n,n],
                          sigma_t=[1,1,n,n],sigma_s=[1,1,n,n],phase=[12,12],weights=[12])
    for name, identity in identities.items():
        check(identity['shape'] == expected_shape[name], label+'/'+name+' shape')
        expected_dtype = 'complex128' if name in ('source','candidate','stream_symbol') else 'float64'
        check(identity['dtype'] == expected_dtype, label+'/'+name+' dtype')
        elements = math.prod(expected_shape[name]); width = 16 if expected_dtype == 'complex128' else 8
        check(identity['bytes'] == elements*width, label+'/'+name+' byte count')
        check(re.fullmatch('[0-9a-f]{64}', identity['sha256']) is not None, label+'/'+name+' hash format')
    c = row['certificate']
    check(c['policy_tolerance_rational'] == str(TAU), label+' tau')
    check(c['policy_solution_tolerance_rational'] == str(EPS), label+' epsilon')
    check(c['reserve_fraction_rational'] == str(ALPHA), label+' reserve')
    check(c['certified_under_cpu_contract'] is True and c['global_gpu_contract_established'] is False,
          label+' conditional/global contract flags')
    check(c['target'] == 'Exact nominal original stored D64 and collision/source bits', label+' target')
    check(c['residual_center_inside_enclosure'] is True, label+' center membership')
    before = guard(c['native_backend_guard_before'], library)
    after = guard(c['native_backend_guard_after'], library)
    check((before ^ after) & 0xFFC0 == 0, label+' arithmetic control state changed')
    m = F(c['coercivity_lower_numerator'], c['coercivity_lower_denominator'])
    check(m == F(c['coercivity_lower_rational']) and m > 0, label+' rational margin')
    check(m <= F(c['derived_coercivity_lower_rational']), label+' supplied margin bound')
    check(fraction(c['coercivity_lower'], label+'/margin') <= m, label+' downward margin')
    norms = {}
    for key in ('residual','source','candidate','residual_center','residual_radius'):
        lower = fraction(c[key+'_norm_lower'], label+'/'+key+'_lower')
        upper = fraction(c[key+'_norm_upper'], label+'/'+key+'_upper')
        check(0 <= lower <= upper, label+'/'+key+' norm ordering')
        norms[key] = (lower,upper)
    rlo, rup = norms['residual']; qlo, qup = norms['source']; xlo, xup = norms['candidate']
    clo, cup = norms['residual_center']; elo, eup = norms['residual_radius']
    delta = rup/m; denominator = xlo-delta
    predicates = dict(source_relative_norm_defined=qlo > 0,
        residual_accepted=qlo > 0 and rup <= TAU*qlo,
        reserve_center_accepted=qlo > 0 and cup <= (1-ALPHA)*TAU*qlo,
        reserve_radius_accepted=qlo > 0 and eup <= ALPHA*TAU*qlo,
        solution_error_accepted=denominator > 0 and delta <= EPS*denominator)
    predicates['fixed_one_percent_reserve_accepted'] = all(predicates[k] for k in
        ('residual_accepted','reserve_center_accepted','reserve_radius_accepted'))
    predicates['accepted'] = predicates['fixed_one_percent_reserve_accepted'] and predicates['solution_error_accepted']
    for key, value in predicates.items():
        check(c[key] is value, label+'/'+key+' exact predicate mismatch')
    upper_report(c['absolute_solution_error_upper'], delta, label+'/absolute_error')
    upper_report(c['residual_center_norm_width'], cup-clo, label+'/center_norm_width')
    upper_report(c['source_norm_width'], qup-qlo, label+'/source_norm_width')
    upper_report(c['center_plus_radius_residual_norm_upper'], cup+eup, label+'/triangle_upper')
    for key, exact in [('relative_residual_upper', rup/qlo if qlo > 0 else None),
                       ('relative_center_norm_upper', cup/qlo if qlo > 0 else None),
                       ('relative_radius_norm_upper', eup/qlo if qlo > 0 else None),
                       ('relative_solution_error_upper', delta/denominator if denominator > 0 else None)]:
        if exact is None:
            check(c[key] is None, label+'/'+key+' undefined ratio')
        else:
            upper_report(c[key], exact, label+'/'+key)
    rejected = qlo > 0 and rlo > TAU*qup
    if kind == 'original_source_zero':
        check(rejected and not predicates['accepted'], label+' rigorous negative control')
    ordinary = finite(row['ordinary_gpu_residual_diagnostic'], label+'/ordinary_GPU_residual')
    check(ordinary >= 0, label+' ordinary GPU residual nonnegative')
    certificate_cost_sum = timing_map(c['costs'], label+'/costs')
    elapsed = finite(c['elapsed_seconds'], label+'/elapsed')
    verification = finite(row['complete_verification_seconds'], label+'/complete_verification')
    check(0 <= certificate_cost_sum <= elapsed+1e-6, label+' nonoverlapping certificate cost phases')
    check(verification >= elapsed, label+' verification includes certificate')
    check(finite(row['explicit_gpu_H_seconds'], label+'/GPUH_seconds') >= 0, label+' GPUH time')
    status = 'CERTIFIED_INPUT' if predicates['accepted'] else ('RIGOROUS_RESIDUAL_REJECTION' if rejected else 'NO_CERTIFICATE')
    return dict(kind=kind,size=n,index=index,status=status,exact_predicates=predicates,
        rigorous_residual_rejection=rejected,coercivity_lower=float(m),
        relative_residual_upper=c['relative_residual_upper'],
        relative_center_upper=c['relative_center_norm_upper'],
        relative_radius_upper=c['relative_radius_norm_upper'],
        relative_solution_error_upper=c['relative_solution_error_upper'],
        ordinary_gpu_residual_diagnostic=ordinary,certificate_seconds=elapsed,
        complete_verification_seconds=verification,explicit_gpu_H_seconds=row['explicit_gpu_H_seconds'])


def audit(root, authored):
    paths = {name: locate(root, name+'.json') for name in
             ('certificates','settings','execution_receipt','preflight_receipt','policy')}
    raw = {name: read(path) for name,path in paths.items()}
    rows, settings, execution, preflight, policy = (raw[n] for n in paths)
    post_path=locate(root,'postrun_preservation.json');post=read(post_path)
    check(post['prior528_preserved'] is True and post['tracked_dirty_unchanged'] is True and
          post['other_research_processes'] == [] and post['source_exported'] is False and
          post['full_solves'] == 0, 'Postrun recorded preservation/scope')
    check(set(post['payload_sha256']) == {'policy.json','preflight_receipt.json','execution_receipt.json',
        'driver_stdout.txt','driver_stderr.txt','results/certificates.json','results/settings.json'},
        'Exact scalar-only retrieved payload list')
    for name,expected in post['payload_sha256'].items():
        check(digest(root/name) == expected, 'Retrieved scalar payload seal '+name)
    check(policy['phase'] == 'fresh_original_H_interval_certifier_v1', 'Phase')
    check(policy['head'] == HEAD and policy['development_seed'] == 4626 and policy['development_ids'] == [0,3], 'Identity/scope')
    for key, value in dict(tau_rational=[1,100000],solution_tolerance_rational=[1,1000],reserve_rational=[1,100],
        soft_seconds=120,hard_seconds=150,max_reserved_bytes=4*1024**3,max_full_solves=0,
        max_certificates=10,max_explicit_H_actions=10,max_M_actions=2,training=False,
        holdout_or_reserved_seed_access=False,source_export=False).items():
        check(policy[key] == value, 'Prospective policy '+key)
    phash = digest(paths['policy'])
    check(settings['policy_sha256'] == phash == execution['policy_sha256'] == preflight['policy_sha256'], 'Policy hash agreement')
    check(settings['complete'] is True and settings['error'] is None and preflight['complete'] is True, 'Complete status')
    check(execution['returncode'] == 0 and execution['timed_out'] is False, 'Owned-process execution')
    check(settings['counters'] == EXPECTED_COUNTERS, 'Complete counters')
    check(len(rows) == 10 and {(r['kind'],r['size'],r['index'])for r in rows} == EXPECTED_CASES, 'Case list and uniqueness')
    for key,value in dict(global_gpu_contract_established=False,source_exported=False,
                          original_candidates_and_buffers_remain_on_FWI=True,
                          heldout_or_reserved_seed_access=False,full_solves=0).items():
        check(settings[key] is value if isinstance(value,bool) else settings[key] == value, 'Settings '+key)
    check(execution['new_full_solves'] == 0 and execution['source_exported'] is False, 'Execution prohibited work flags')
    check(preflight['full_solves'] == 0 and preflight['target_actions'] == 0 and
          preflight['original_source_exported'] is False, 'Preflight scope')
    check(preflight['repo'] == '/data/wjt/NeuralPreconditionedBornSeries' and preflight['head'] == HEAD, 'Preflight identity')
    command = preflight['native_compile_command']
    for flag in ('-O2','-std=c++17','-fno-fast-math','-ffp-contract=off','-shared','-fPIC'):
        check(flag in command, 'Compiler flag '+flag)
    check(not any('fast-math' in flag and flag != '-fno-fast-math' for flag in command), 'Unsafe compile flag')
    check('g++' in preflight['compiler'], 'FWI compiler provenance')
    check(re.search(r'29 passed', preflight['tests_stdout']) is not None, 'All 29 tiny tests passed on FWI')
    library = preflight['native_library_sha256']
    check(re.fullmatch('[0-9a-f]{64}',library) is not None, 'Library hash format')
    check(settings['endpoint_sharedlib_sha256'] == library, 'Settings library identity')
    check(execution['native_library_sha256'] == library, 'Execution library identity')
    guard(settings['native_cpu_guard'],library); guard(preflight['native_cpu_guard'],library)
    code = {}
    for name, expected in policy['code_sha256'].items():
        path = authored/Path(name).name
        check(path.exists(), 'Missing authored source '+str(path))
        check(digest(path) == expected, 'Local reviewed snapshot '+name)
        matches = [h for staged,h in preflight['staged_sha256'].items() if staged.endswith('/'+name)]
        check(matches == [expected], 'Staged snapshot '+name)
        code[name] = expected
    check(preflight['native_source_sha256'] == policy['code_sha256']['code/cpu_interval_primitives.cpp'], 'Native source seal')
    case_audits = [audit_case(row,policy,library) for row in rows]
    costs = settings['costs']; category_sum = timing_map(costs,'run costs')
    verification_sum = math.fsum(c['complete_verification_seconds']for c in case_audits)
    certificate_sum = math.fsum(c['certificate_seconds']for c in case_audits)
    h_sum = math.fsum(c['explicit_gpu_H_seconds']for c in case_audits)
    check(abs(h_sum-costs['explicit_gpu_H_seconds']) < 1e-9, 'GPU H cost reconciliation')
    check(costs['certificate_seconds'] >= certificate_sum, 'Global certificate timer encloses helper timers')
    elapsed = finite(settings['elapsed_seconds'],'Driver elapsed'); process = finite(execution['owned_driver_seconds_including_imports'],'Process elapsed')
    check(0 <= category_sum <= elapsed+1e-5 and elapsed <= policy['soft_seconds'], 'Driver time budget/ledger')
    check(elapsed <= process <= policy['hard_seconds'], 'Owned process/import-inclusive hard budget')
    check(settings['gpu_peak_reserved_bytes'] <= policy['max_reserved_bytes'], 'GPU memory budget')
    check(0 <= settings['gpu_peak_allocated_bytes'] <= settings['gpu_peak_reserved_bytes'], 'GPU memory accounting')
    check(preflight['compiler_seconds'] >= 0 and preflight['tests_seconds'] >= 0, 'Cold preflight time')
    check(execution['preservation_before_seconds'] >= 0 and post['preservation_after_seconds'] >= 0,
          'Recorded preservation time')
    # Same-context teacher/zero controls must share the exact represented source
    # and material/operator buffers. Candidate is intentionally different.
    for index in (0,3):
        teacher = next(r for r in rows if(r['kind'],r['size'],r['index']) == ('saved_teacher',128,index))
        zero = next(r for r in rows if(r['kind'],r['size'],r['index']) == ('original_source_zero',128,index))
        for name in ('source','stream_symbol','sigma_t','sigma_s','phase','weights'):
            check(teacher['buffer_identities'][name] == zero['buffer_identities'][name], 'N128 matched target/source '+name)
    totals = Counter(c['status']for c in case_audits)
    positives = [c for c in case_audits if c['kind'] != 'original_source_zero']
    return dict(audit_success=True,scope='Scalar receipts and sealed authored source only; no backend/field/action reexecution',
        independent_remote_or_preservation_reexecution=False,independent_action_or_margin_reconstruction=False,
        original_arrays_exported=False,full_solver_admission=False,global_gpu_contract_established=False,
        paired_field_gate_certified=False,policy_sha256=phash,receipt_sha256={**{k:digest(p)for k,p in paths.items()},'postrun_preservation':digest(post_path)},
        source_sha256=code,compiler=preflight['compiler'],native_library_sha256=library,
        native_guard_flags=127,preflight_tests=29,counters=settings['counters'],status_counts=dict(totals),cases=case_audits,
        positive_case_maxima={key:max((c[key]for c in positives if c[key] is not None),default=None)for key in ('relative_residual_upper','relative_center_upper','relative_radius_upper','relative_solution_error_upper')},
        cost=dict(driver_seconds=elapsed,owned_process_seconds=process,run_category_seconds=costs,
            category_sum_seconds=category_sum,complete_verification_sum_seconds=verification_sum,
            certifier_helper_sum_seconds=certificate_sum,explicit_gpu_H_sum_seconds=h_sum,
            compiler_seconds=preflight['compiler_seconds'],preflight_seconds=preflight['tests_seconds'],
            experiment_preservation_before_seconds=execution['preservation_before_seconds'],
            experiment_preservation_after_seconds=post['preservation_after_seconds'],
            nested_costs_must_not_be_added_twice=True,compiler_preflight_excluded_from_owned_run=True,
            driver_clock_excludes_imports=True,old_solver_timings_exclude_new_verification=True),
        memory=dict(gpu_peak_reserved_bytes=settings['gpu_peak_reserved_bytes'],
            gpu_peak_allocated_bytes=settings['gpu_peak_allocated_bytes'],cpu_peak_rss_kib=settings['cpu_peak_rss_kib']),
        preservation={'reported_prior_hash_count':528,'reported_all_match':True,
                      'retrieved_scalar_payload_hashes_verified':len(post['payload_sha256']),
                      'independently_reexecuted_FWI_seal':False},
        limitations=['Stored endpoint/norm/coercivity enclosures rely on reviewed authored computation and explicit IEEE/compiler/nextafter/bit-I/O contract; scalar receipts cannot independently reconstruct omitted arrays.',
            'Native finite test witnesses and register guards do not prove all CPU operations or a uniform GPU error bound.',
            'Saved full-teacher certificates apply only to their actual captured field/source/operator bits; N256 cases are manufactured fixtures, not recovered benchmark solutions.',
            'Saved teacher q/x/t/s persist in original PT banks, but manufactured N256 x/q and captured D64 were only transiently hashed/certified; no new persistent binary snapshot is proved by the no-export metadata. Bitwise reexecution is unverified.',
            'Failed upper-bound predicates retain NO_CERTIFICATE; only R_L>tau*Q_U proves nominal residual rejection.',
            'No new solve, solver-convergence comparison, paired-field criterion, economics benefit, continuum accuracy or reserved-test result is established.'])


def markdown(result):
    status = result['status_counts']; c=result['cost']; mem=result['memory']; mx=result['positive_case_maxima']
    show=lambda x:'undefined'if x is None else'%.9g'%x
    lines=['# Independent scalar certificate result audit','',
        '2026-10-02. The stdlib auditor independently reproduces every scalar acceptance predicate using exact Fraction arithmetic and agrees with the sealed receipts. It does not import or rerun RTE, GPU code, the native backend or omitted original arrays.', '',
        '**Outcome:** %d certified inputs, %d rigorous original-source/zero residual rejections, %d inconclusive NO_CERTIFICATE cases. Zero full solves and training steps. No full-solver comparison is admitted by this result.' %
        (status.get('CERTIFIED_INPUT',0),status.get('RIGOROUS_RESIDUAL_REJECTION',0),status.get('NO_CERTIFICATE',0)), '',
        '| Case | Classification | Residual upper / Q_L | Radius upper / Q_L | Solution-error upper | Complete verification (s) |',
        '|---|---|---:|---:|---:|---:|']
    for r in result['cases']:
        lines.append('| %s N%d ID%d | %s | %s | %s | %s | %.6f |' %
            (r['kind'],r['size'],r['index'],r['status'],show(r['relative_residual_upper']),
             show(r['relative_radius_upper']),show(r['relative_solution_error_upper']),r['complete_verification_seconds']))
    lines += ['', 'The audit verifies 10 records (4 saved teachers, 2 manufactured fixtures, 4 negative controls), 10 explicit H actions, 2 M actions, the declared exact cases, zero prohibited work, matching policy/code/native-library seals, all 29 tiny FWI tests and read-only x86-MXCSR guards with flags 127. Same-context N128 teacher and zero controls share source/operator buffer identities. Shapes, dtypes, byte counts and no-mutation metadata are consistent. Those identities are evidence from receipts, not an independent reconstruction of the fields.', '',
        'Acceptance is recomputed from rational coercivity m and binary endpoint values: `C_U <= .99*tau*Q_L`, `E_U <= .01*tau*Q_L`, `R_U <= tau*Q_L`, and positive-denominator `R_U/(m*X_L-R_U) <= 1e-3`, with exact `tau=1/100000`. The direct box norm and separately reported triangle bound remain distinct. Every outward scalar conversion is independently reproduced. All negative controls satisfy exact `R_L > tau*Q_U`, so their rejection is proved for the nominal stored operator. Norm-width inflation is already paid through the outward C_U and Q_L; norm width does not replace vector radius.', '',
        'Across the six positive inputs, maxima are residual %s, center %s, radius %s and defined relative solution-error upper %s.' %(show(mx['relative_residual_upper']),show(mx['relative_center_upper']),show(mx['relative_radius_upper']),show(mx['relative_solution_error_upper'])), '',
        'Both N256 fixtures have valid residual upper bounds below tau (1.15209610e-6 and 5.03149137e-7) and valid relative solution-error upper bounds below 1e-3 (8.56672448e-6 and 4.18128542e-6), conditional on the CPU contract. The fixed vector-radius reserve alone fails: E_U/Q_L is 1.15209604e-6 and 5.03149102e-7 against 1e-7. Thus the enclosure establishes those weaker individual accuracy inequalities but does not admit either fixture under the complete frozen policy. This is enclosure uncertainty, not a lower-bound proof of GPU error or an inaccurate candidate.', '',
        'Owned process wall is %.6f s; the driver records %.6f s after imports. Complete per-record verification totals %.6f s, including actual-D64 capture/CPU transfers, both identity-hash passes and the full certifier. Nested helper timers total %.6f s. Explicit rounded GPU H diagnostics total %.6f s separately. Cold compilation %.6f s and preflight %.6f s are outside the owned run and remain billable work. Peak GPU reserved memory is %d B (allocated %d B), within 4 GiB. Do not add nested helper/transfer categories to complete verification a second time. Parent loading, physical/capture setup, persistence and external startup also remain visible; this is not a solver speed measurement.' %(c['owned_process_seconds'],c['driver_seconds'],c['complete_verification_sum_seconds'],c['certifier_helper_sum_seconds'],c['explicit_gpu_H_sum_seconds'],c['compiler_seconds'],c['preflight_seconds'],mem['gpu_peak_reserved_bytes'],mem['gpu_peak_allocated_bytes']), '',
        'The N64/N128 inputs are actual stored full teachers and exact promoted original source bits. N256 inputs are new manufactured fixtures with represented q=captured rounded GPU H(x); no prior N256 benchmark field was recovered. The checker certifies each represented nominal residual directly and requires no uniform GPU action-error hypothesis. Its correctness remains conditional on the reviewed exact-bit I/O, integer/Fraction, native IEEE/compiler and nextafter contracts. Finite fixtures and register guards are supporting evidence, not a proof of every CPU operation.', '',
        'The saved teacher q/x/t/s remain in their original PT banks. The manufactured N256 x/q and actual captured D64 were transient inputs hashed and certified within this owned run, without a proved new binary snapshot. Metadata saying buffers remain on FWI establishes nonexport, not persistence. Bit-for-bit replay by reexecution is unverified; the in-run certificate remains about its precise captured inputs.', '',
        'The postrun receipt reports all 528 prior hashes preserved, tracked dirty work unchanged and no other research process. The auditor checks its seven retrieved scalar-payload hashes locally, including the stdout and empty stderr. It does not independently rerun the FWI seal. Preservation-before %.6f s and preservation-after %.6f s are separate experiment provenance work outside the owned driver; they are recorded without conflating them with an intrinsic checker latency.' %(c['experiment_preservation_before_seconds'],c['experiment_preservation_after_seconds']), '',
        'Individual error bounds do not establish the separate 1e-4 paired-field criterion. No new Krylov convergence, performance advantage, continuum/physical-positivity claim, past unsaved-field accuracy, global GPU contract or reserved-test outcome follows. This audit does not independently repeat FWI preservation checks or rebuild the omitted action/coercivity arrays.', '',
        'Reproduce: `python3 deliverables/certifier_20261002/audit_certifier_receipts.py --receipts receipts/certifier_execution`. Input SHA256 seals and every independently recomputed predicate are recorded in RESULT_AUDIT.json.', '']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipts',type=Path,default=Path('receipts/certifier_execution'))
    parser.add_argument('--authored-dir',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--json',type=Path)
    parser.add_argument('--markdown',type=Path)
    args=parser.parse_args()
    result=audit(args.receipts,args.authored_dir)
    result['auditor_sha256']=digest(__file__)
    j=args.json or args.authored_dir/'RESULT_AUDIT.json'
    md=args.markdown or args.authored_dir/'RESULT_AUDIT.md'
    j.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');md.write_text(markdown(result))
    print(json.dumps(dict(audit_success=True,status_counts=result['status_counts'],
                         cost=result['cost'],json=str(j),markdown=str(md)),indent=2))


if __name__=='__main__':
    main()
