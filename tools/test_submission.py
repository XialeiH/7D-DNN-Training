#!/usr/bin/env python3
"""Exercise submission receipts and refusal paths with a fake sbatch, never Slurm."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.split = self.base / 'split with spaces'
        (self.split / 'shards').mkdir(parents=True)
        (self.split / 'dataset_validation.json').write_text('{"verified": true}')
        shutil.copyfile(ROOT / '7D/all615/data_split_whitelist17/feature_norm.json', self.split / 'feature_norm.json')
        for target in 'SCI':
            for part in ['train', 'val', 'test']:
                (self.split / 'shards' / f'{target}_{part}.h5').touch()
        bin_dir = self.base / 'bin'
        bin_dir.mkdir()
        self.calls = self.base / 'calls'
        fake = bin_dir / 'sbatch'
        fake.write_text(f'''#!{sys.executable}
import os,sys
from pathlib import Path
p=Path({str(self.calls)!r})
n=int(p.read_text())+1 if p.exists() else 1
p.write_text(str(n))
if os.getenv('FAIL_SECOND') and n==2:
    print('simulated scheduler rejection',file=sys.stderr);sys.exit(1)
print(900000+n)
''')
        fake.chmod(0o755)
        self.env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'])
        self.runroot = self.base / 'new run'

    def call(self, *extra, recipe='h1024x4_relu_original'):
        return subprocess.run([sys.executable, str(ROOT / 'jobs/submit_training.py'),
                               '--recipe', str(ROOT / 'configs' / (recipe + '.json')),
                               '--split', str(self.split), '--run-root', str(self.runroot), *extra],
                              text=True, capture_output=True, env=self.env)

    def test_dry_run_is_read_only(self):
        result = self.call()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.runroot.exists())
        self.assertFalse(self.calls.exists())

    def test_confirmed_receipts_and_duplicate_refusal(self):
        result = self.call('--submit')
        self.assertEqual(result.returncode, 0, result.stderr)
        for target in 'SCI':
            rec = json.loads((self.runroot / 'receipts' / (target + '.json')).read_text())
            self.assertEqual(rec['status'], 'confirmed')
            self.assertTrue(rec['job_id'].isdigit())
        self.assertNotEqual(self.call('--submit').returncode, 0)
        self.assertEqual(self.calls.read_text(), '3')

    def test_partial_failure_preserves_receipts(self):
        self.env['FAIL_SECOND'] = '1'
        self.assertNotEqual(self.call('--submit').returncode, 0)
        self.assertEqual(json.loads((self.runroot / 'receipts/S.json').read_text())['status'], 'confirmed')
        self.assertIn('unconfirmed', json.loads((self.runroot / 'receipts/C.json').read_text())['status'])
        self.assertFalse((self.runroot / 'receipts/I.json').exists())
        self.assertNotEqual(self.call('--submit').returncode, 0)
        self.assertEqual(self.calls.read_text(), '2')

    def test_invalid_norm_refused_before_submission(self):
        (self.split / 'feature_norm.json').write_text('{}')
        self.assertNotEqual(self.call('--submit').returncode, 0)
        self.assertFalse(self.runroot.exists())
        self.assertFalse(self.calls.exists())

    def test_lineage_recipes_refused(self):
        for name in ['h1024x4_relu_Cseed43', 'h512x5_smooth_low']:
            self.assertNotEqual(self.call('--submit', recipe=name).returncode, 0)
        self.assertFalse(self.calls.exists())


if __name__ == '__main__':
    unittest.main()
