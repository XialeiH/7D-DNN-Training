"""Prepare isolated packages, preflight their gates, then submit each gate once."""
from pathlib import Path
import copy, datetime, hashlib, json, shlex, shutil, subprocess, sys
import torch
from checkpoint_gate import validate

BASE = Path(__file__).resolve().parent
PY = '/scratch/xh2906/conda/envs/py310/bin/python'
SOURCE = '/scratch/xh2906/librarySCI_runs/h96_tfsf7d_h1024x4_scratch_fixedbaseline_goodFRgt20_whitelist17_all615DS10_20261004_original_loss_relu/posttraining_7d_tf32_globalall7'
PHYSICAL_SOURCE = '/home/xh2906/librarySCI_active_posttraining/' + Path(SOURCE).parent.name + '/posttraining_7d_tf32_globalall7'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
trials = json.loads((BASE/'trials.json').read_text())
torch.set_num_threads(2)
reports = {}

for label, trial in trials.items():
    root = Path(trial['root']); post = root/'posttraining_7d_tf32_globalall7'
    physical = Path('/home/xh2906/librarySCI_active_posttraining')/root.name/post.name
    manifest = json.loads((root/'submitted_jobs.json').read_text())
    assert manifest['target_jobs'] == trial['jobs'] and manifest['strength'] == label.removeprefix('smooth_')
    assert not (root/'posttraining_submission.json').exists() and not post.exists() and not post.is_symlink() and not physical.exists(), 'Existing package: reconcile before retry'
    physical.mkdir(parents=True); post.symlink_to(physical, target_is_directory=True)
    (post/'logs').mkdir(); (post/'template_code').mkdir()
    provenance = {}
    for source in (BASE/'template_code').iterdir():
        if source.name.startswith('._'): continue
        assert source.suffix in {'.py','.m'}, source
        destination = post/'template_code'/source.name
        text = source.read_text().replace(PHYSICAL_SOURCE,str(post)).replace(SOURCE,str(post))
        destination.write_text(text)
        if destination.suffix == '.py': compile(text,str(destination),'exec')
        provenance[source.name] = dict(source_sha256=sha(source),copied_sha256=sha(destination),paths_rebound_only=True)
    (post/'template_code/template_source_sha.json').write_text(json.dumps(provenance,indent=2))
    for name in ['launch_smooth_post.py','checkpoint_gate.py','run_archive_exports.py','local_reference_inventory.json']:
        shutil.copy2(BASE/name,post/name)
    (post/'storage_provenance.json').write_text(json.dumps(dict(logical=str(post),physical=str(physical),reason='Dedicated Home avoids limited scratch inode headroom; no unrelated files moved'),indent=2))
    norm_path = Path(manifest['split'])/'feature_norm.json'
    assert sha(norm_path) == manifest['normalization_sha256']
    norm = json.loads(norm_path.read_text()); calibration = json.loads((root/'smoothness_calibration.json').read_text())
    initial = json.loads((root/'initial_checkpoint_provenance.json').read_text()); tests = []
    for target in 'SCI':
        source_path = root/'initial_checkpoints'/f'{target}.pt'; assert sha(source_path) == initial[target]['sha256']
        source = torch.load(source_path,map_location='cpu',weights_only=False)
        fixture = copy.deepcopy(source); train = fixture['train_config']
        train.update(lr_stage_values=manifest['schedule']['lr_stage_values'],curvature_mode='axes4_periodic_angle',init_checkpoint=str(source_path))
        coefficient = calibration['targets'][target]['lambda_by_strength'][manifest['strength']]
        train.update(curvature_lambda=coefficient,ckpt_global_curvature_beta=coefficient)
        fit = float(source['extra']['best_val_total_loss'])
        fixture['extra'].update(best_epoch=0,best_val_total_loss=fit,best_global_curvature_penalty=1.,best_ckpt_selection_score=fit+coefficient)
        validate(fixture,manifest,target,norm,source,calibration)
        for kind in ['wrong_coefficient','wrong_activation','wrong_norm','wrong_score','wrong_epoch_zero_weights']:
            bad = copy.deepcopy(fixture)
            if kind == 'wrong_coefficient': bad['train_config']['curvature_lambda'] *= 2
            if kind == 'wrong_activation': bad['model_config']['activation_seq'] = ['tanh']*5
            if kind == 'wrong_norm': bad['feature_norm'] = {}
            if kind == 'wrong_score': bad['extra']['best_ckpt_selection_score'] += 1
            if kind == 'wrong_epoch_zero_weights': bad['model']['net.0.weight'][0,0] += 1
            try: validate(bad,manifest,target,norm,source,calibration)
            except AssertionError: tests.append(target+':'+kind+' rejected')
            else: raise AssertionError('Gate accepted invalid fixture: '+kind)
        live = root/'train'/target/'best.pt'
        if live.exists():
            validate(torch.load(live,map_location='cpu',weights_only=False),manifest,target,norm,source,calibration)
            tests.append(target+':current checkpoint validated read-only; not frozen')
    script = post/'checkpoint_gate.sh'
    script.write_text('#!/bin/bash\nset -euo pipefail\nmodule load anaconda3/2025.06\nsource /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh\nconda activate py310\nexport TMPDIR=/tmp/xh2906_smoothpost_${SLURM_JOB_ID}\nmkdir -p "$TMPDIR"\n'+shlex.join([PY,str(post/'checkpoint_gate.py'),str(root)])+'\n')
    subprocess.run(['bash','-n',str(script)],check=True)
    state = dict(run_root=str(root),post_root=str(post),physical_post_root=str(physical),training_job_ids=manifest['target_jobs'],
                 local_folder=trial['local_folder'],status='prepared; awaiting deferred gate submission',atlas_cases=139,dense_conditions=97,
                 representative_cases=len(json.loads((BASE/'local_reference_inventory.json').read_text())),completion_email_recipient='xialei.huang@nyu.edu',
                 checkpoint_policy='All three targets terminal; valid selected best.pt; fine-tune epoch0 is trained source fallback',frequency_axes='linear')
    (root/'posttraining_submission.json').write_text(json.dumps(state,indent=2))
    reports[label] = dict(preflight_passed=True,tests=tests,epoch_zero_test='In-memory trained-source fine-tuning fixture only; not saved as a production checkpoint',state=state)

(BASE/'attachment_preflight.json').write_text(json.dumps(reports,indent=2))
for label, trial in trials.items():
    root = Path(trial['root']); post = root/'posttraining_7d_tf32_globalall7'
    args = ['sbatch','--parsable','--account=torch_pr_482_general','--partition=cpu_short,cs','--cpus-per-task=2','--mem=16G','--time=01:00:00',
            '--dependency=afterany:'+':'.join(trial['jobs'].values()),'--job-name=post_'+label,'--output='+str(post/'logs/checkpoint_gate.%j.out'),
            '--error='+str(post/'logs/checkpoint_gate.%j.err'),str(post/'checkpoint_gate.sh')]
    intent = post/'deferred_post_submission_intent.json'
    with intent.open('x') as f: json.dump(dict(state='submitting',command=args),f)
    job = subprocess.check_output(args,text=True).strip().split(';')[0]; assert job.isdigit()
    receipt = dict(state='confirmed',job_id=job,command=args,training_jobs=trial['jobs'],submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (post/'deferred_post_submission.json').write_text(json.dumps(receipt,indent=2)); intent.write_text(json.dumps(receipt,indent=2))
    state = json.loads((root/'posttraining_submission.json').read_text()); state.update(deferred_checkpoint_gate_job=job,status='deferred until all three training jobs end and checkpoints validate')
    (root/'posttraining_submission.json').write_text(json.dumps(state,indent=2))
    reports[label].update(job_id=job,command=args)
    (BASE/'attachment_submission.json').write_text(json.dumps(reports,indent=2))
    print(label,job,flush=True)
