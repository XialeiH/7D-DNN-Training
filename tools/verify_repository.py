#!/usr/bin/env python3
"""Read-only integrity and syntax checks; never import submission scripts."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'provenance/source_manifest.json').read_text())
    for record in manifest['files']:
        path = ROOT / record['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], path
    python_files = list(ROOT.rglob('*.py'))
    shell_files = list(ROOT.rglob('*.sh'))
    for path in python_files:
        ast.parse(path.read_text(), filename=str(path))
    for path in shell_files:
        subprocess.run(['bash', '-n', str(path)], check=True)
    for path in ROOT.rglob('*.json'):
        json.loads(path.read_text())
    norm = ROOT / '7D/all615/data_split_whitelist17/feature_norm.json'
    assert hashlib.sha256(norm.read_bytes()).hexdigest() == 'b34059b1260e5d7003ac8d642425c689b83c96e745247702b5ca2ae82bc91b0f'
    expected = {'LocalResponse_6D_MLP_prefAngle.m': '41c7fdc837617cd66ee81b0a7f7dc2bc3af45ed8c38e67bc3e85abed8b6bb161',
                'predict_pref6D_S.m': '0eaf440aebc1ad3194c01efb3a807ab8eb03d8dff223a64fdd1565935da6049a',
                'predict_pref6D_C.m': '1efcf8260cae702885f775e8875e098a9658d09aad7f74631ff701ec28709ae2',
                'predict_pref6D_I.m': '6be47fb16e2b913f6cb7957b6b9daad9b6862f9b85fdd05822c4b54c9ac7c971'}
    for name, digest in expected.items():
        assert hashlib.sha256((ROOT / 'Previous 5D training/runtime' / name).read_bytes()).hexdigest() == digest
    print(json.dumps({'source_hashes': len(manifest['files']), 'python_syntax': len(python_files),
                      'shell_syntax': len(shell_files), 'normalization_and_manuscript_runtime': 'passed'}))


if __name__ == '__main__':
    main()
