from pathlib import Path
import datetime, hashlib, json, os, shlex, shutil, subprocess, sys

R = Path('/scratch/xh2906/librarySCI_runs/h96_tfsf7d_h512x4_scratch_fixedbaseline_goodFRgt20_whitelist17_all615DS10_20261002_original_loss')
PY = '/scratch/xh2906/conda/envs/py310/bin/python'
parent = json.loads((R/'submitted_jobs.json').read_text())
data = Path(parent['split'])
validation = json.loads((data/'dataset_validation.json').read_text())
assert validation['verified'] and validation['no_subsampling'] and validation['native_points_per_block'] == 615
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(data/'feature_norm.json') == parent['normalization_sha256']
env = os.environ.copy()
env['PYTHONDONTWRITEBYTECODE'] = '1'
env['OMP_NUM_THREADS'] = '2'
env['MKL_NUM_THREADS'] = '2'
prepared = []

def write(path, obj):
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)

smoke = '''import sys,json,torch
from pathlib import Path
root=Path(sys.argv[1]);width=int(sys.argv[2]);depth=int(sys.argv[3]);sys.path.insert(0,str(root/'code'))
from common import ModelConfig,build_model,constant_baseline_rule
torch.set_num_threads(2);norm=json.loads((root/'data_split_whitelist17/feature_norm.json').read_text());counts={}
for t in 'SCI':
 cfg=ModelConfig(target=t,feature_mode='tf_sf7d',input_dim=7,hidden_dims=(width,)*depth,activation_seq=('tanh',)*depth)
 cfg.constant_baseline_rule=constant_baseline_rule(norm,t);m=build_model(cfg)
 assert [(l.in_features,l.out_features) for l in m.net if isinstance(l,torch.nn.Linear)]==[(7,width)]+[(width,width)]*(depth-1)+[(width,1)]
 assert sum(isinstance(l,torch.nn.Tanh) for l in m.net)==depth
 raw=torch.tensor([[1,50,20,20000,10000,10,2.5],[1,0,20,20000,10000,10,2.5],[1,50,20,20000,10000,0,2.5],[1,50,20,20000,10000,10,0],[1,50,20,20000,10000,51,2.5],[1,50,20,20000,10000,10,16]],dtype=torch.float32);x=raw.clone()
 for j,k in enumerate(['OriCos2','Contrast','L6','L4E','L4I','lgnTF','lgnSF']):
  if j:x[:,j]=(raw[:,j]-norm[k]['mean'])/norm[k]['std']
 y=m(x);assert torch.isfinite(y).all();torch.testing.assert_close(y[1:],torch.full((5,),{'S':2.5,'C':4.5,'I':12.5}[t]),rtol=0,atol=0)
 y[0].backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
 counts[t]=sum(p.numel() for p in m.parameters());assert counts[t]==(7*width+width)+(depth-1)*(width*width+width)+width+1
(root/'architecture_validation.json').write_text(json.dumps({'passed':True,'architecture':[7]+[width]*depth+[1],'parameters_per_target':counts,'baseline_checks':15,'finite_forward_backward_all_targets':True,'hidden_activations':'tanh'},indent=2))
print('ARCHITECTURE PASSED',width,depth,counts,flush=True)
'''

# Both variants must pass all checks before any job is submitted.
for width, depth, label, stem in [(1024,4,'h1024x4','all615_h1024x4'), (512,5,'h512x5','all615_h512x5')]:
    N = R.with_name(R.name.replace('h512x4',label).replace('20261002','20261003'))
    assert not N.exists(), f'Existing run must be reconciled before retry: {N}'
    for d in ['code','logs','receipts','train/S','train/C','train/I']:
        (N/d).mkdir(parents=True, exist_ok=True)
    (N/'data_split_whitelist17').symlink_to(data, target_is_directory=True)
    for name, digest in parent['code_sha256'].items():
        assert sha(R/'code'/name) == digest, name
        shutil.copy2(R/'code'/name, N/'code'/name)
        assert sha(N/'code'/name) == digest, name
    shutil.copy2(R/'source_trainer.py', N/'source_trainer.py')
    assert sha(N/'source_trainer.py') == parent['reference_trainer_sha256']
    state = {k: parent[k] for k in ['split','max_epochs','supervised_loss','allowed_tf_sf_pairs','native_grid_points','code_sha256','normalization_sha256','reference_trainer_sha256']}
    state.update(run_root=str(N), created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), reference_run=str(R), initialization='random; no checkpoint', architecture=[7]+[width]*depth+[1], activation_seq=['tanh']*depth, commands={}, target_jobs={}, status='preparing; not submitted')
    write(N/'submitted_jobs.json', state)
    subprocess.run([PY,str(N/'code/test_original_loss.py')], env=env, check=True)
    subprocess.run([PY,str(N/'code/test_whitelist.py'),str(data/'feature_norm.json')], env=env, check=True)
    subprocess.run([PY,'-c',smoke,str(N),str(width),str(depth)], env=env, check=True)
    assert json.loads((N/'loss_validation.json').read_text())['passed']
    submissions = {}
    differences = {}
    for t in 'SCI':
        original = shlex.split(parent['commands'][t])
        args = original.copy()
        replacements = {1:str(N/'code/train_lowtfsf_grad01.py'), original.index('--hidden_dims')+1:','.join([str(width)]*depth), original.index('--activation_seq')+1:','.join(['tanh']*depth), original.index('--out_dir')+1:str(N/'train'/t)}
        for index, value in replacements.items(): args[index] = value
        changed = {i for i,(a,b) in enumerate(zip(original,args)) if a != b}
        assert len(args) == len(original) and changed <= set(replacements)
        assert not any(k in args for k in ['--resume','--init_checkpoint','--init_from_5d_dir'])
        command = shlex.join(args)
        state['commands'][t] = command
        differences[t] = [{'index':i,'before':original[i],'after':args[i]} for i in sorted(changed)]
        source = (R/f'code/all615_h512_{t}.sh').read_text()
        assert parent['commands'][t] in source
        wrapper = source.replace(parent['commands'][t], command).replace(str(R/'loss_validation.json'),str(N/'loss_validation.json')).replace('xh2906_h512_all615_',f'xh2906_{label}_all615_')
        f = N/f'code/{stem}_{t}.sh'
        f.write_text(wrapper)
        subprocess.run(['bash','-n',str(f)],check=True)
        oldcmd = json.loads((R/f'receipts/all615_h512_{t}.json').read_text())['command']
        assert not any(x.startswith('--dependency=') for x in oldcmd)
        cmd = [x.replace(str(R),str(N)).replace('all615_h512_',stem+'_') for x in oldcmd]
        cmd[-1] = str(f)
        # All scheduler resources remain exactly as submitted for the parent.
        fixed = [x for x in oldcmd if x.startswith(('--account=','--cpus-per-task=','--mem=','--time=','--gres=','--partition='))]
        assert [x for x in cmd if x.startswith(('--account=','--cpus-per-task=','--mem=','--time=','--gres=','--partition='))] == fixed
        submissions[t] = cmd
    state['validation'] = {'original_loss_passed':True,'whitelist_checks_passed':True,'architecture_and_baselines_passed':True,'only_architecture_and_isolated_paths_changed':True,'scheduler_resources_unchanged':True}
    state['status'] = 'all preflight checks passed; not submitted'
    write(N/'submitted_jobs.json', state)
    write(N/'architecture_command_diff.json', differences)
    plan = json.loads((R/'plan.json').read_text())
    plan.update(run_root=str(N),reference_run=str(R),shared_split=str(data),architecture=state['architecture'],activation_seq=state['activation_seq'])
    write(N/'plan.json',plan)
    write(N/'dataset_validation_at_submission.json',validation)
    prepared.append((N,state,submissions,stem))

for N, state, submissions, stem in prepared:
    for t in 'SCI':
        receipt = N/f'receipts/{stem}_{t}.json'
        cmd = submissions[t]
        with receipt.open('x') as f: json.dump({'state':'submitting','command':cmd},f)
        result = subprocess.run(cmd,text=True,capture_output=True)
        if result.returncode:
            write(receipt,{'state':'rejected','command':cmd,'stdout':result.stdout,'stderr':result.stderr})
            raise RuntimeError(result.stderr)
        job = result.stdout.strip().split(';')[0]
        assert job.isdigit(), result.stdout
        write(receipt,{'state':'confirmed','job_id':job,'command':cmd,'confirmed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        state['target_jobs'][t] = job
        write(N/'submitted_jobs.json',state)
        print('SUBMITTED',N.name,t,job,flush=True)
    state['status'] = 'three random-initialization scratch trainings submitted'
    write(N/'submitted_jobs.json',state)

print('CONFIRMED',json.dumps([{'run_root':str(N),'target_jobs':state['target_jobs'],'architecture':state['architecture']} for N,state,_,_ in prepared]),flush=True)
