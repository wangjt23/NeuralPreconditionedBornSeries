"""Audit byte-level provenance of the completed Krylov-tail research campaign."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'rte/workstation_runs/tail_solver_report_20260926'
HANDOFF = 'docs/research/RTE_WORKSTATION_HANDOFF_2026_09_12.md'
SNAPSHOT = 'rte/workstation_runs/tail_solver_report_20260926/previous_handoff_snapshot.md'


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    audits = []
    manifests = ['rte/handoff/2026-09-12/SHA256SUMS.tsv',
        'rte/workstation_runs/recursive_response_report_20260919/artifact_hashes.csv',
        'rte/workstation_runs/source_initialization_report_20260926/artifact_hashes.csv',
        'rte/workstation_runs/source_characteristic_report_20260926/artifact_hashes.csv']
    for name in manifests:
        with (ROOT/name).open() as handle:
            rows = list(csv.DictReader(handle, delimiter='\t' if name.endswith('.tsv') else ','))
        mismatches, mapped = [], []
        for row in rows:
            path = row['path']
            if path == HANDOFF and 'source_characteristic_report' in name:
                path = SNAPSHOT
                mapped.append(dict(original=row['path'], preserved_snapshot=path, sha256=row['sha256']))
            target = ROOT/path
            if not target.is_file() or target.stat().st_size != int(row['bytes']) or sha(target) != row['sha256']:
                mismatches.append(row['path'])
        audits.append(dict(manifest=name, count=len(rows), mismatches=mismatches, explicit_snapshot_mapping=mapped))
    provenance = []
    for directory in ('tail_injection_gate_20260926', 'tail_direction_pilot_20260926',
        'tail_optical_moment_20260926', 'tail_solver_benchmark_20260926',
        'tail_lookahead_smoke_20260926', 'tail_lookahead_20260926', 'tail_lookahead_gate_20260926',
        'tail_multiresolution_bank_20260926'):
        path = ROOT/'rte/workstation_runs'/directory/'settings.json'
        if not path.exists():
            continue
        settings = json.loads(path.read_text())
        for name, expected in settings.get('source_sha256', {}).items():
            actual_name = name
            if directory == 'tail_direction_pilot_20260926' and name == 'rte/train_tail_direction.py':
                actual_name = 'rte/workstation_runs/tail_direction_pilot_20260926/train_tail_direction_snapshot.py'
            actual = sha(ROOT/actual_name)
            provenance.append(dict(run=directory, original=name, resolved=actual_name,
                expected=expected, actual=actual, matches=actual == expected))
    audit = dict(old_manifests=audits, executed_sources=provenance,
        note='Historical manifests are unmodified. The evolving handoff is verified at its byte-identical pre-continuation snapshot; the original tail training script also has an exact execution snapshot.')
    (OUT/'provenance_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    if any(a['mismatches'] for a in audits) or any(not p['matches'] for p in provenance):
        raise RuntimeError('historical or executed-source hash mismatch; inspect provenance audit')
    files = set()
    names = ['augmented_fgmres', 'tail_response_network', 'audit_tail_injection',
        'train_tail_direction', 'tail_moment_space', 'probe_tail_moments',
        'optical_tail_network', 'benchmark_tail_solver', 'audit_tail_generalization',
        'probe_tail_frontier', 'tail_lookahead', 'train_tail_lookahead',
        'benchmark_tail_lookahead', 'prepare_tail_multiresolution', 'report_tail_solver']
    for name in names:
        files.add(ROOT/f'rte/{name}.py')
    for pattern in ('test_augmented_fgmres.py', 'test_tail*.py', 'test_optical_tail_network.py'):
        files.update((ROOT/'rte/tests').glob(pattern))
    files.add(ROOT/HANDOFF)
    files.add(ROOT/'docs/research/RTE_KRYLOV_TAIL_RESULTS_2026_09_26.md')
    for directory in (ROOT/'rte/workstation_runs').glob('tail_*_20260926'):
        files.update(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts
            and p.name != 'artifact_hashes.csv')
    rows = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha(p))
        for p in sorted(files) if p.is_file()]
    with (OUT/'artifact_hashes.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'bytes', 'sha256'])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(old_manifests=[(a['manifest'], a['count']) for a in audits],
        executed_sources=len(provenance), new_artifacts=len(rows)), indent=2))


if __name__ == '__main__':
    run()
