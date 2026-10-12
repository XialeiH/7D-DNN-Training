#!/usr/bin/env python3
"""Run retained CPU scientific tests in temporary copies, never live run roots."""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix='dnn_contracts_') as tmp:
        run = Path(tmp)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(run / 'mpl'))
        shutil.copytree(ROOT / '7D/all615/code', run / 'code')
        shutil.copy2(ROOT / '7D/all615/source_trainer.py', run / 'source_trainer.py')
        shutil.copytree(ROOT / '7D/all615/data_split_whitelist17', run / 'data_split_whitelist17')
        norm = str(run / 'data_split_whitelist17/feature_norm.json')
        tests = [('test_source_contract.py', []), ('test_original_loss.py', []),
                 ('test_whitelist.py', [norm])]
        for name, args in tests:
            subprocess.run([sys.executable, str(run / 'code' / name), *args], env=env, check=True)
        smooth = run / 'smooth'
        shutil.copytree(ROOT / '7D/experiments/smoothness/code', smooth)
        subprocess.run([sys.executable, str(smooth / 'test_smoothness.py'), norm], env=env, check=True)
        print('All four retained scientific contract suites passed; no cluster jobs submitted.')


if __name__ == '__main__':
    main()
