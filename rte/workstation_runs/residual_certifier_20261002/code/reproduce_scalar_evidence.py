"""Verify bundle SHA256s and reproduce three scalar artifacts byte for byte.

Uses only Python's standard library. Does not import the certifier, NumPy,
Torch, RTE, or a native backend; does not execute a field/action/certificate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reproduce(bundle_root):
    root = Path(bundle_root).resolve()
    manifest = json.loads((root/'MANIFEST.json').read_text())
    checked = 0
    for name, expected in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe manifest path: '+name)
        path = root/relative
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing/nonregular payload: '+name)
        if path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
            raise ValueError('Payload hash mismatch: '+name)
        checked += 1
    authored = root/'deliverables/certifier_20261002'
    receipts = root/'receipts/certifier_execution'
    with tempfile.TemporaryDirectory(prefix='rte-scalar-reproduction-') as temporary:
        output = Path(temporary)
        subprocess.run([sys.executable, str(authored/'audit_certifier_receipts.py'),
                        '--receipts', str(receipts), '--authored-dir', str(authored),
                        '--json', str(output/'RESULT_AUDIT.json'),
                        '--markdown', str(output/'RESULT_AUDIT.md')],
                       cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run([sys.executable, str(authored/'analyze_certifier.py'), str(receipts),
                        '--authored-code-dir', str(authored),
                        '--output', str(output/'SCALAR_ANALYSIS.json')],
                       cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        artifacts = {}
        for name in ('RESULT_AUDIT.json', 'RESULT_AUDIT.md', 'SCALAR_ANALYSIS.json'):
            stored, regenerated = authored/name, output/name
            if stored.read_bytes() != regenerated.read_bytes():
                raise ValueError('Scalar reduction is not byte exact: '+name)
            artifacts[name] = dict(sha256=sha(regenerated), bytes=regenerated.stat().st_size,
                                   byte_exact=True)
    return dict(verified_payloads=checked, artifacts=artifacts,
                standard_library_only=True, field_action_or_certifier_execution=False,
                native_backend_execution=False, network_or_library_action=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-root', type=Path,
                        default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(reproduce(args.bundle_root), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
