"""Prepare four pinned post-training packages, then attach afterany checkpoint gates."""
from pathlib import Path
import datetime, hashlib, json, shlex, shutil, subprocess, urllib.request

PY='/scratch/xh2906/conda/envs/py310/bin/python'
TEMPLATE=Path('/scratch/xh2906/librarySCI_runs/h96_tfsf7d_h512x4_scratch_fixedbaseline_goodFRgt20_whitelist17_all615DS10_20261002_original_loss/posttraining_7d_tf32_globalall7/code')
BASE=Path('/home/xh2906/librarySCI_activation_lr_trials_20261004')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
trials=json.loads((BASE/'post_trials.json').read_text());prepared=[]
for label,trial in trials.items():
    R=Path(trial['run_root']);P=R/'posttraining_7d_tf32_globalall7'
    H=Path('/home/xh2906/librarySCI_active_posttraining')/R.name/P.name
    assert not (R/'posttraining_submission.json').exists() and not P.exists() and not P.is_symlink() and not H.exists(), 'Existing post-training must be reconciled'
    manifest=json.loads((R/'submitted_jobs.json').read_text());assert manifest['target_jobs']==trial['target_jobs']
    H.mkdir(parents=True);P.symlink_to(H,target_is_directory=True)
    (P/'logs').mkdir();(P/'template_code').mkdir()
    files={}
    for source in sorted(TEMPLATE.iterdir()):
        if not source.is_file():continue
        dest=P/'template_code'/source.name;shutil.copy2(source,dest)
        if source.suffix in ['.py','.m','.sh']:
            dest.write_text(source.read_text().replace(str(TEMPLATE.parent.resolve()),str(P)).replace(str(TEMPLATE.parent),str(P)))
        files[source.name]={'source':str(source),'source_sha256':sha(source),'copied_sha256':sha(dest),'only_run_root_paths_rebound':True}
        if dest.suffix=='.py':compile(dest.read_text(),str(dest),'exec')
    (P/'template_code/template_source_sha.json').write_text(json.dumps(files,indent=2))
    for name in ['launch_trial_post.py','checkpoint_gate.py','run_archive_exports.py','trial_reference_inventory.json']:
        urlname='trial_checkpoint_gate.py' if name=='checkpoint_gate.py' else name
        payload=urllib.request.urlopen('http://127.0.0.1:8765/'+urlname,timeout=30).read()
        dest=P/('local_reference_inventory.json' if name=='trial_reference_inventory.json' else name);dest.write_bytes(payload)
        if name.endswith('.py'):compile(payload,str(dest),'exec')
    storage={'logical_post_root':str(P),'physical_post_root':str(H),'reason':'Dedicated compute-visible Home output; four post-training packages exceed remaining scratch inode headroom. Existing data and jobs unchanged.','created_utc':now()}
    (P/'storage_provenance.json').write_text(json.dumps(storage,indent=2))
    script=P/'checkpoint_gate.sh'
    script.write_text('#!/bin/bash\nset -euo pipefail\nmodule purge\nmodule load anaconda3/2025.06\nsource /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh\nconda activate py310\nexport PYTHONDONTWRITEBYTECODE=1\nexport TMPDIR=/tmp/xh2906_trial_post_${SLURM_JOB_ID}\nexport MPLCONFIGDIR=$TMPDIR/mpl\nmkdir -p "$TMPDIR" "$MPLCONFIGDIR"\n'+shlex.join([PY,str(P/'checkpoint_gate.py'),str(R)])+'\n')
    subprocess.run(['bash','-n',str(script)],check=True)
    dependency='afterany:'+':'.join(manifest['target_jobs'][t] for t in 'SCI')
    command=['sbatch','--parsable','--account=torch_pr_482_general','--partition=cpu_short,cs','--cpus-per-task=2','--mem=16G','--time=01:00:00','--dependency='+dependency,'--job-name=post_'+label,'--output='+str(P/'logs/checkpointgate.%j.out'),'--error='+str(P/'logs/checkpointgate.%j.err'),str(script)]
    state={'run_root':str(R),'post_root':str(P),'physical_post_root':str(H),'required_training_jobs':manifest['target_jobs'],'dependency':dependency,'condition':'All S/C/I training jobs terminal AND all three nonempty, readable, finite best.pt checkpoints match this run architecture, activations, normalization and losses. Freeze only after these checks.','status':'prepared; not yet attached','expected_atlas_count':139,'expected_dense_conditions':97,'representative_reference_count':len(json.loads((P/'local_reference_inventory.json').read_text())),'local_folder':trial['local_folder'],'completion_email_to':'xialei.huang@nyu.edu','launch_script_sha256':sha(P/'launch_trial_post.py'),'checkpoint_gate_sha256':sha(P/'checkpoint_gate.py'),'created_utc':now()}
    (R/'posttraining_submission.json').write_text(json.dumps(state,indent=2));prepared.append((label,R,P,command,state))

for label,R,P,command,state in prepared:
    receipt=P/'deferred_post_submission_intent.json'
    with receipt.open('x') as f:json.dump({'state':'submitting','command':command,'created_utc':now()},f)
    result=subprocess.run(command,text=True,capture_output=True)
    if result.returncode:
        receipt.write_text(json.dumps({'state':'rejected','command':command,'stdout':result.stdout,'stderr':result.stderr},indent=2));raise RuntimeError(result.stderr)
    job=result.stdout.strip().split(';')[0];assert job.isdigit(),result.stdout
    receipt.write_text(json.dumps({'state':'confirmed','job_id':job,'command':command,'confirmed_utc':now()},indent=2))
    state.update(checkpoint_gate_job=job,status='attached; waiting for training termination and valid checkpoints')
    (P/'deferred_post_submission.json').write_text(json.dumps(state,indent=2));(R/'posttraining_submission.json').write_text(json.dumps(state,indent=2))
    print('POST_ATTACHED',label,job,state['dependency'],flush=True)
print('ALL_FOUR_POSTTRAINING_GATES_CONFIRMED',flush=True)
