#!/usr/bin/env python3
"""Stage and submit the recorded all615 scratch recipe; dry-run by default."""
import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NORM_SHA = 'b34059b1260e5d7003ac8d642425c689b83c96e745247702b5ca2ae82bc91b0f'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--account', default='torch_pr_482_general')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    if recipe['name'] != 'h1024x4_relu_original' or set(recipe['target_arguments']) != set('SCI'):
        parser.error('Use the dedicated lineage gates for C-only retry or smoothness recipes.')
    split, root = args.split.resolve(), args.run_root.resolve()
    if root.exists():
        parser.error('Run directory exists. Reconcile its receipts instead of resubmitting.')
    code = REPO / recipe['code_directory']
    records = json.loads((REPO / 'provenance/source_manifest.json').read_text())['files']
    for rec in records:
        path = REPO / rec['path']
        if path.parent == code:
            if hashlib.sha256(path.read_bytes()).hexdigest() != rec['sha256']:
                parser.error(f'Source hash changed: {path}')
    scripts = {}
    commands = {}
    for target in 'SCI':
        argv = [str(args.python), str(root / 'code' / recipe['entrypoint'])]
        argv += [v.replace('{SPLIT}', str(split)).replace('{RUN}', str(root))
                 for v in recipe['target_arguments'][target]]
        scripts[target] = '\n'.join([
            '#!/usr/bin/env bash', 'set -euo pipefail',
            'module purge', 'module load anaconda3/2025.06',
            'source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh',
            'conda activate py310', 'export PYTHONNOUSERSITE=1',
            'export TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}_all615_${SLURM_JOB_ID}}"',
            'export MPLCONFIGDIR="$TMPDIR/mpl"', 'mkdir -p "$MPLCONFIGDIR"',
            shlex.join(argv), ''])
        commands[target] = ['sbatch', '--parsable', '--account=' + args.account,
                            '--gres=gpu:1', '--cpus-per-task=8', '--mem=64G',
                            '--time=24:00:00', '--job-name=all615_' + target,
                            '--output=' + str(root / 'logs' / (target + '.%j.out')),
                            '--error=' + str(root / 'logs' / (target + '.%j.err')),
                            str(root / 'code' / (target + '.sh'))]
        subprocess.run(['bash', '-n'], input=scripts[target], text=True, check=True)
        print(shlex.join(commands[target]))
        print(scripts[target])
    if not args.submit:
        print('Dry run only. No files created and no jobs submitted. Split checks run with --submit.')
        return
    if not json.loads((split / 'dataset_validation.json').read_text()).get('verified'):
        parser.error('Dataset is not verified.')
    if hashlib.sha256((split / 'feature_norm.json').read_bytes()).hexdigest() != NORM_SHA:
        parser.error('Normalization hash differs from the recorded all615 split.')
    for target in 'SCI':
        for part in ('train', 'val', 'test'):
            if not (split / 'shards' / f'{target}_{part}.h5').is_file():
                parser.error(f'Missing shard: {target}_{part}.h5')
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(code, root / 'code', ignore=shutil.ignore_patterns('__pycache__'))
    (root / 'logs').mkdir()
    (root / 'receipts').mkdir()
    (root / 'recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')
    for target in 'SCI':
        (root / 'train' / target).mkdir(parents=True)
        (root / 'code' / (target + '.sh')).write_text(scripts[target])
        receipt = root / 'receipts' / (target + '.json')
        state = {'status': 'intent; reconcile if interrupted', 'command': commands[target]}
        with receipt.open('x') as handle:
            json.dump(state, handle, indent=2)
        result = subprocess.run(commands[target], text=True, capture_output=True)
        state.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
        job_id = result.stdout.strip().split(';')[0]
        state['status'] = 'confirmed' if result.returncode == 0 and job_id.isdigit() else 'unconfirmed; reconcile before retry'
        if state['status'] == 'confirmed':
            state['job_id'] = job_id
        receipt.write_text(json.dumps(state, indent=2) + '\n')
        if state['status'] != 'confirmed':
            raise RuntimeError(f'Submission not confirmed; inspect {receipt}')
        print(target, job_id)


if __name__ == '__main__':
    main()
