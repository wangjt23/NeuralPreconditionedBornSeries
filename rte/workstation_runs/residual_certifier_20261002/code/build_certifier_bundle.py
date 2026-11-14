"""Deterministic bundle of authored certifier code and scalar receipts only.

No original repository source, bank, candidate, tensor or native binary is
copied. Outputs are created exclusively. This performs no Library/network/
host action and does not run or compile a certifier.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


CODE = (
    'interval_fft.py', 'certified_twiddles.py', 'cpu_interval_primitives.cpp',
    'certifier_driver.py', 'test_interval_fft.py', 'test_interval_fft_audit.py',
    'test_certified_twiddles.py', 'analyze_certifier.py', 'build_certifier_bundle.py')
RECEIPTS = (
    'policy.json', 'preflight_receipt.json', 'execution_receipt.json',
    'postrun_preservation.json', 'driver_stdout.txt', 'driver_stderr.txt',
    'results/certificates.json', 'results/settings.json')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def payloads(authored_root, receipt_root):
    authored_root, receipt_root = Path(authored_root), Path(receipt_root)
    policy_bytes = (receipt_root/'policy.json').read_bytes()
    policy = json.loads(policy_bytes)
    output = {}
    for name in CODE:
        path = authored_root/name
        data = path.read_bytes()
        if 'code/'+name in policy['code_sha256']:
            if digest(data) != policy['code_sha256']['code/'+name]:
                raise ValueError('Executed authored-code seal mismatch: '+name)
        output['code/'+name] = data
    for path in sorted(authored_root.glob('*.md')):
        output['reports/'+path.name] = path.read_bytes()
    analysis = authored_root/'SCALAR_ANALYSIS.json'
    if analysis.is_file():
        output['reports/'+analysis.name] = analysis.read_bytes()
    for name in RECEIPTS:
        output['receipts/'+name] = (receipt_root/name).read_bytes()
    post = json.loads((receipt_root/'postrun_preservation.json').read_text())
    for name, expected in post['payload_sha256'].items():
        if name not in RECEIPTS or digest((receipt_root/name).read_bytes()) != expected:
            raise ValueError('Post-run scalar payload seal mismatch: '+name)
    if not post['prior528_preserved'] or not post['tracked_dirty_unchanged']:
        raise ValueError('Original preservation failed')
    output['README.md'] = (
        '# Authored input-specific interval certifier and scalar evidence\n\n'
        'This bundle contains newly authored code/proofs and scalar/hash receipts. '
        'It contains no original repository source, field/bank tensors or compiled native binary.\n\n'
        'Verify every listed SHA256 in MANIFEST.json. Existing NumPy and a system '
        'C++ compiler are required for local fixtures; no package installation is performed.\n\n'
        'Run CPU fixtures with python code/test_interval_fft.py, '
        'python code/test_certified_twiddles.py and python code/test_interval_fft_audit.py. '
        'The native arithmetic/CPU/compiler contract is documented in reports/IMPLEMENTATION_CONTRACT.md.\n\n'
        'Reproduce scalar analysis with python code/analyze_certifier.py receipts '
        '--authored-code-dir code --output reproduced_analysis.json. '
        'This creates a new scalar file and never runs transport actions.\n\n'
        'Reproducing project-field certificates requires the exact separately held '
        'original repository/banks/buffers sealed in the policy and an authorized FWI executor. '
        'The driver is not a standalone substitute for that workspace.\n\n'
        'Prior GPU solver components remain separate from measured CPU certification. '
        'No integrated solver speedup, equal-control certificate cost or learned-method win '
        'was measured by this screen.\n'
    ).encode()
    manifest = dict(
        scope='Authored new certifier sources/proofs and scalar/hash evidence only',
        executed_policy_sha256=digest(policy_bytes),
        files={name: dict(sha256=digest(data), bytes=len(data))
               for name, data in sorted(output.items())},
        manifest_self_hash_excluded=True,
        deterministic_archive='Sorted paths; mode0644, uid/gid0, empty owners, mtime0; gzip mtime0/empty filename',
        original_source_exported=False, original_field_arrays_included=False,
        compiled_native_binary_included=False, full_solver_execution=False,
        library_or_network_action_performed=False)
    output['MANIFEST.json'] = (json.dumps(manifest, indent=2, sort_keys=True)+'\n').encode()
    return output


def build(authored_root, receipt_root, output_path):
    files = payloads(authored_root, receipt_root)
    output_path = Path(output_path)
    with output_path.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0,
                           compresslevel=9) as zipped:
            with tarfile.open(fileobj=zipped, mode='w', format=tarfile.PAX_FORMAT) as archive:
                for name, data in sorted(files.items()):
                    info = tarfile.TarInfo(name)
                    info.size = len(data); info.mode = 0o644
                    info.uid = info.gid = 0; info.uname = info.gname = ''
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(data))
    return dict(path=str(output_path), sha256=digest(output_path.read_bytes()),
                bytes=output_path.stat().st_size, files=len(files),
                manifest_sha256=digest(files['MANIFEST.json']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authored-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--receipt-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.authored_root, args.receipt_root, args.output)))


if __name__ == '__main__':
    main()
