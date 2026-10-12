from pathlib import Path
import copy,datetime,hashlib,json,shlex,shutil,subprocess,sys
import torch
from checkpoint_gate import validate
from learning_health import assess
BASE=Path(__file__).resolve().parent
PARENT=Path('/scratch/xh2906/librarySCI_runs/h96_tfsf7d_h1024x4_scratch_fixedbaseline_goodFRgt20_whitelist17_all615DS10_20261004_original_loss_relu')
SEED=int(sys.argv[1]);assert 43<=SEED<=47
R=Path('/home/xh2906/librarySCI_C_seed_retry')/(PARENT.name+f'_Cseed{SEED}_20261005');P=R/'posttraining_7d_tf32_globalall7'
PY='/scratch/xh2906/conda/envs/py310/bin/python';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((R/'submitted_jobs.json').read_text());assert set(m['target_jobs'])=={'C'}
assert not P.exists() and not (R/'posttraining_submission.json').exists(), 'Post package already exists: do not duplicate'
P.mkdir();(P/'logs').mkdir();(P/'template_code').mkdir()
source=str(PARENT/'posttraining_7d_tf32_globalall7')
physical=str(Path('/home/xh2906/librarySCI_active_posttraining')/PARENT.name/'posttraining_7d_tf32_globalall7')
provenance={}
for f in (BASE/'template_code').iterdir():
 assert f.suffix in {'.py','.m'} and not f.name.startswith('._')
 out=P/'template_code'/f.name;text=f.read_text().replace(physical,str(P)).replace(source,str(P));out.write_text(text)
 if f.suffix=='.py':compile(text,str(out),'exec')
 provenance[f.name]=dict(source_sha256=sha(f),copied_sha256=sha(out),paths_rebound_only=True)
(P/'template_code/template_source_sha.json').write_text(json.dumps(provenance,indent=2))
for name in ['checkpoint_gate.py','learning_health.py','launch_retry_post.py','run_archive_exports.py','local_reference_inventory.json']:shutil.copy2(BASE/name,P/name)
torch.set_num_threads(2);norm=json.loads((Path(m['split'])/'feature_norm.json').read_text());tests=[]
for t in 'SI':
 c=torch.load(R/'train'/t/'best.pt',map_location='cpu',weights_only=False);validate(c,m,t,norm);assert sha(R/'train'/t/'best.pt')==m['fixed_checkpoints'][t]['sha256'];tests.append(t+' preserved checkpoint valid')
old=torch.load(PARENT/'posttraining_7d_tf32_globalall7/checkpoints/C.pt',map_location='cpu',weights_only=False)
fixture=copy.deepcopy(old);fixture['train_config']['seed']=SEED
try:validate(fixture,m,'C',norm)
except AssertionError:tests.append('epoch0 initial C rejected')
else:raise AssertionError('Gate accepts untrained C')
fixture['extra']['best_epoch']=1;validate(fixture,m,'C',norm);tests.append('in-memory positive epoch schema accepted; no fixture saved as checkpoint')
bad=copy.deepcopy(fixture);bad['train_config']['seed']=42
try:validate(bad,m,'C',norm)
except AssertionError:tests.append('wrong C seed rejected')
else:raise AssertionError('Gate accepts wrong seed')
bad=copy.deepcopy(fixture);bad['model']['net.0.weight'][0,0]=float('nan')
try:validate(bad,m,'C',norm)
except AssertionError:tests.append('nonfinite model rejected')
else:raise AssertionError('Gate accepts NaN')
health=assess(PARENT);assert health['persistent_zero_gradients'] and not health['normal_learning'];tests.append('known failed original C detected as collapse')
(P/'gate_preflight.json').write_text(json.dumps(dict(passed=True,tests=tests),indent=2))
script=P/'checkpoint_gate.sh';script.write_text('#!/bin/bash\nset -euo pipefail\nmodule load anaconda3/2025.06\nsource /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh\nconda activate py310\nexport PYTHONDONTWRITEBYTECODE=1\nexport TMPDIR=/tmp/xh2906_Cretry_post_${SLURM_JOB_ID}\nmkdir -p "$TMPDIR"\n'+shlex.join([PY,str(P/'checkpoint_gate.py'),str(R)])+'\n')
subprocess.run(['bash','-n',str(script)],check=True)
state=dict(run_root=str(R),post_root=str(P),training_job_ids=m['target_jobs'],reused_target_jobs=m['reused_target_jobs'],local_folder='/Users/xialeihuang/Desktop/Neuroscience_Project/7D NN Training/'+PARENT.name,status='prepared; awaiting afterany gate',seed=SEED,checkpoint_policy='C terminal, selected epoch>0, validation improved, finite recent metrics and nonzero recent fit gradient; S84/I65 byte-identical',atlas_cases=139,dense_conditions=97,representative_cases=88,frequency_axes='linear',completion_email_recipient='xialei.huang@nyu.edu',local_update_policy='Back up old verified package outside OUT, stage and fully validate replacement before replacing user original folder; preserve prior email receipt as historical')
(R/'posttraining_submission.json').write_text(json.dumps(state,indent=2))
args=['sbatch','--parsable','--account=torch_pr_482_general','--partition=cpu_short,cs','--cpus-per-task=2','--mem=16G','--time=01:00:00','--dependency=afterany:'+m['target_jobs']['C'],'--job-name=Cseed'+str(SEED)+'_postgate','--output='+str(P/'logs/gate.%j.out'),'--error='+str(P/'logs/gate.%j.err'),str(script)]
intent=P/'deferred_post_submission_intent.json'
with intent.open('x') as f:json.dump(dict(state='submitting',command=args),f)
j=subprocess.check_output(args,text=True).strip().split(';')[0];assert j.isdigit()
receipt=dict(state='confirmed',job_id=j,command=args,training_jobs=m['target_jobs'],submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(P/'deferred_post_submission.json').write_text(json.dumps(receipt,indent=2));intent.write_text(json.dumps(receipt,indent=2))
state.update(deferred_checkpoint_gate_job=j,status='waiting for terminal C and normal-learning validation');(R/'posttraining_submission.json').write_text(json.dumps(state,indent=2))
print(json.dumps(state,indent=2))

