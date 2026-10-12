from pathlib import Path
import datetime, hashlib, json, os, shlex, shutil, subprocess, sys
PARENT=Path('/scratch/xh2906/librarySCI_runs/h96_tfsf7d_h1024x4_scratch_fixedbaseline_goodFRgt20_whitelist17_all615DS10_20261004_original_loss_relu')
SEED=int(sys.argv[2]) if len(sys.argv)>2 else 43
assert SEED in range(43,48), 'Only five seeds 43 through 47 authorized'
ROOT=Path('/home/xh2906/librarySCI_C_seed_retry')/(PARENT.name+f'_Cseed{SEED}_20261005')
PY='/scratch/xh2906/conda/envs/py310/bin/python'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):p.write_text(json.dumps(v,indent=2))
def prepare():
 if SEED>43:
  previous=ROOT.with_name(PARENT.name+f'_Cseed{SEED-1}_20261005')
  decision=json.loads((previous/'retry_decision.json').read_text())
  assert decision['reason']=='learning_failure' and decision['training_terminal_verified'] and decision['next_seed']==SEED, 'Previous attempt must be documented as a terminal learning failure'
 assert not ROOT.exists(), 'Existing retry: reconcile before preparing'
 for d in ['code','logs','receipts','train/C','train/S','train/I','tmp','mpl']:(ROOT/d).mkdir(parents=True,exist_ok=True)
 parent=json.loads((PARENT/'submitted_jobs.json').read_text())
 data=Path(parent['split']);assert json.loads((data/'dataset_validation.json').read_text())['verified']
 assert sha(data/'feature_norm.json')==parent['normalization_sha256']
 (ROOT/'data_split_whitelist17').symlink_to(data,target_is_directory=True)
 for name,digest in parent['code_sha256'].items():
  assert sha(PARENT/'code'/name)==digest
  shutil.copy2(PARENT/'code'/name,ROOT/'code'/name)
 shutil.copy2(PARENT/'source_trainer.py',ROOT/'source_trainer.py')
 assert sha(ROOT/'source_trainer.py')==parent['reference_trainer_sha256']
 fixed={}
 for t in 'SI':
  # Authority is the completed evaluated-runtime provenance, cross-checked below.
  src=PARENT/'posttraining_7d_tf32_globalall7/checkpoints'/f'{t}.pt'
  provenance=json.loads((PARENT/'posttraining_7d_tf32_globalall7/checkpoint_provenance.json').read_text())
  assert sha(src)==provenance[t]['sha256']
  shutil.copy2(src,ROOT/'train'/t/'best.pt')
  fixed[t]={'source':str(src),'sha256':sha(src),'selected_epoch':provenance[t]['extra']['best_epoch'],'reused_unchanged':True}
 assert fixed['S']['selected_epoch']==84 and fixed['I']['selected_epoch']==65
 original=shlex.split(parent['commands']['C']);args=original.copy()
 args[1]=str(ROOT/'code/train_lowtfsf_grad01.py')
 args[args.index('--out_dir')+1]=str(ROOT/'train/C')
 assert '--seed' not in args and not any(k in args for k in ['--resume','--init_checkpoint','--init_from_5d_dir'])
 args+=['--seed',str(SEED)]
 assert sum(a!=b for a,b in zip(args,original))==2
 command=shlex.join(args)
 wrapper=(PARENT/'code/all615_h1024x4_relu_C.sh').read_text()
 assert parent['commands']['C'] in wrapper
 wrapper=wrapper.replace(parent['commands']['C'],command).replace(str(PARENT/'loss_validation.json'),str(ROOT/'loss_validation.json')).replace('xh2906_h1024x4_relu_all615_',f'xh2906_Cseed{SEED}_')
 script=ROOT/f'code/C_seed{SEED}.sh';script.write_text(wrapper);subprocess.run(['bash','-n',str(script)],check=True)
 old=json.loads((PARENT/'receipts/all615_h1024x4_relu_C.json').read_text())['command']
 cmd=[v.replace(str(PARENT),str(ROOT)).replace('--job-name=all615_h1024x4_relu_C',f'--job-name=h1024relu_Cseed{SEED}') for v in old];cmd[-1]=str(script)
 resource=lambda c:[v for v in c if v.startswith(('--account=','--cpus-per-task=','--mem=','--gres=','--time=','--partition='))]
 assert resource(cmd)==resource(old)
 state={k:parent[k] for k in ['split','max_epochs','supervised_loss','allowed_tf_sf_pairs','native_grid_points','code_sha256','normalization_sha256','reference_trainer_sha256','architecture','activation_seq','lr_stage_values']}
 state.update(run_root=str(ROOT),reference_run=str(PARENT),created_utc=now(),seed=SEED,initialization='C random initialization; S/I unchanged frozen original checkpoints',commands={'C':command},submission_commands={'C':cmd},target_jobs={},reused_target_jobs={t:parent['target_jobs'][t] for t in 'SI'},fixed_checkpoints=fixed,status='preparing')
 write(ROOT/'submitted_jobs.json',state)
 write(ROOT/'plan.json',dict(state,scope='Only C is retrained with the recorded retry seed. After normal learning and terminal C, evaluate unchanged S84/I65 plus new C. Update original local folder after complete verification.'))
 write(ROOT/'experiment_command_diff.json',{'only_scientific_change':{'seed':{'before':42,'after':SEED}},'path_changes':[{'before':original[i],'after':args[i]} for i in range(len(original)) if original[i]!=args[i]],'appended':['--seed',str(SEED)],'resources_unchanged':resource(cmd),'no_initial_checkpoint':True})
 write(ROOT/'storage_provenance.json',{'root':str(ROOT),'filesystem':'Home','reason':'Scratch quota 5000201/5000000 files; all new writes in dedicated Home. Dataset read-only in scratch. Job temporary storage node-local.'})
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TMPDIR=str(ROOT/'tmp'),MPLCONFIGDIR=str(ROOT/'mpl'))
 for name,tail in [('test_original_loss.py',[]),('test_whitelist.py',[str(data/'feature_norm.json')])]:
  subprocess.run([PY,str(ROOT/'code'/name),*tail],env=env,check=True)
 subprocess.run([PY,str(Path(__file__).with_name('seed_preflight.py')),str(ROOT)],env=env,check=True)
 state.update(status='preflight passed; not submitted');write(ROOT/'submitted_jobs.json',state)
 print(json.dumps({'prepared':str(ROOT),'preflight':json.loads((ROOT/'seed_validation.json').read_text())}),flush=True)
def submit():
 state=json.loads((ROOT/'submitted_jobs.json').read_text())
 assert state['status']=='preflight passed; not submitted' and not state['target_jobs']
 assert json.loads((ROOT/'seed_validation.json').read_text())['passed']
 assert json.loads((ROOT/'loss_validation.json').read_text())['passed']
 for name,digest in state['code_sha256'].items():assert sha(ROOT/'code'/name)==digest
 receipt=ROOT/f'receipts/C_seed{SEED}.json';cmd=state['submission_commands']['C']
 with receipt.open('x') as f:json.dump({'state':'submitting','command':cmd,'created_utc':now()},f)
 r=subprocess.run(cmd,capture_output=True,text=True)
 if r.returncode:write(receipt,dict(state='rejected',command=cmd,stderr=r.stderr,stdout=r.stdout));raise RuntimeError(r.stderr)
 job=r.stdout.strip().split(';')[0];assert job.isdigit()
 write(receipt,dict(state='confirmed',job_id=job,command=cmd,stdout=r.stdout,confirmed_utc=now()))
 state['target_jobs']={'C':job};state['status']=f'C seed{SEED} retry submitted; original S/I retained';write(ROOT/'submitted_jobs.json',state)
 print(json.dumps({'job_id':job,'root':str(ROOT),'receipt':str(receipt)}),flush=True)
if __name__=='__main__':{'prepare':prepare,'submit':submit}[sys.argv[1]]()

