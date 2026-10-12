"""Prepare, then submit four isolated activation/LR trials from unchanged parents."""
from pathlib import Path
import datetime, hashlib, json, os, shlex, shutil, subprocess, sys

BASE = Path('/scratch/xh2906/librarySCI_runs')
OUTPUTS = Path('/home/xh2906/librarySCI_active_training')
PY = '/scratch/xh2906/conda/envs/py310/bin/python'
TRIALS = [(label, variant) for variant in ['relu', 'tanh_lr001'] for label in ['h1024x4', 'h512x5']]
SETUP = Path('/home/xh2906/librarySCI_activation_lr_trials_20261004')
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TMPDIR=str(SETUP/'tmp'), MPLCONFIGDIR=str(SETUP/'mpl'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()

def paths(label, variant):
    parent = BASE/f'h96_tfsf7d_{label}_scratch_fixedbaseline_goodFRgt20_whitelist17_all615DS10_20261003_original_loss'
    new = BASE/f'h96_tfsf7d_{label}_scratch_fixedbaseline_goodFRgt20_whitelist17_all615DS10_20261004_original_loss_{variant}'
    return parent, new, f'all615_{label}_{variant}'

def write(path, obj):
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)

SMOKE = '''import sys,json,torch,numpy as np
from pathlib import Path
r=Path(sys.argv[1]);s=json.loads((r/'submitted_jobs.json').read_text());sys.path.insert(0,str(r/'code'))
from common import ModelConfig,TrainConfig,build_model,constant_baseline_rule
import train_sixd_single_longrun as base
import train_lowtfsf_grad01 as trainer
torch.set_num_threads(2);norm=json.loads((r/'data_split_whitelist17/feature_norm.json').read_text());width=s['architecture'][1];depth=len(s['activation_seq']);act=s['activation_seq'][0];results={}
def normalize(raw):
 z=raw.clone()
 for j,k in enumerate(['OriCos2','Contrast','L6','L4E','L4I','lgnTF','lgnSF']):
  if j:z[:,j]=(raw[:,j]-norm[k]['mean'])/norm[k]['std']
 return z
for t in 'SCI':
 torch.manual_seed(42)
 cfg=ModelConfig(target=t,feature_mode='tf_sf7d',input_dim=7,hidden_dims=(width,)*depth,activation_seq=tuple(s['activation_seq']))
 cfg.constant_baseline_rule=constant_baseline_rule(norm,t);m=build_model(cfg)
 assert [(l.in_features,l.out_features) for l in m.net if isinstance(l,torch.nn.Linear)]==[(7,width)]+[(width,width)]*(depth-1)+[(width,1)]
 expected_act=torch.nn.ReLU if act=='relu' else torch.nn.Tanh
 assert sum(isinstance(l,expected_act) for l in m.net)==depth and m.softplus_out
 raw=torch.tensor([[1,0,20,20000,10000,10,2.5],[1,50,20,20000,10000,0,2.5],[1,50,20,20000,10000,10,0],[1,50,20,20000,10000,51,2.5],[1,50,20,20000,10000,10,16]],dtype=torch.float32)
 torch.testing.assert_close(m(normalize(raw)),torch.full((5,),{'S':2.5,'C':4.5,'I':12.5}[t]),rtol=0,atol=0)
 raw=torch.tensor([[1,50,20,20000,10000,tf,sf] for tf,sf in s['allowed_tf_sf_pairs']],dtype=torch.float32);x=normalize(raw)
 lc=TrainConfig(train_mat='',split_dir=s['split'],out_dir='',target=t,device='cpu',feature_mode='tf_sf7d',w_log=.5,w_rel=.5,variance_weight_mode='soft_inv_sem',tf_unimodal_lambda=5,sf_unimodal_lambda=5)
 pred,terms=trainer.compute_loss_terms(m,x,torch.full((len(raw),),{'S':5.,'C':10.,'I':20.}[t]),{'FR_ResponseVar':np.zeros(len(raw)),'FR_ResponseN':np.ones(len(raw))},lc,torch.device('cpu'),None,norm)
 loss,_,_=base.build_total_loss_from_terms(terms,lc);assert torch.isfinite(pred).all() and torch.isfinite(loss);loss.backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
 assert m.net[0].weight.grad.abs().sum()>0
 count=sum(p.numel() for p in m.parameters());assert count==(7*width+width)+(depth-1)*(width*width+width)+width+1
 results[t]={'parameters':count,'regularized_loss':float(loss),'finite_nonzero_backward':True}
(r/'architecture_validation.json').write_text(json.dumps({'passed':True,'architecture':s['architecture'],'hidden_activations':act,'baseline_checks':15,'softplus_output':True,'actual_model_regularized_forward_backward':results,'lr_stage_values':s['lr_stage_values']},indent=2))
print('MODEL/LOSS/BASELINE PASSED',r.name,results,flush=True)
'''

def prepare():
    for d in ['tmp','mpl']: (SETUP/d).mkdir(parents=True, exist_ok=True)
    for label, variant in TRIALS:
        R, N, stem = paths(label, variant)
        parent = json.loads((R/'submitted_jobs.json').read_text())
        data = Path(parent['split']); validation = json.loads((data/'dataset_validation.json').read_text())
        assert validation['verified'] and validation['no_subsampling'] and validation['native_points_per_block']==615
        assert sha(data/'feature_norm.json')==parent['normalization_sha256']
        physical = OUTPUTS/N.name/'train'
        assert not N.exists() and not N.is_symlink() and not physical.parent.exists(), str(N)
        for d in ['code','logs','receipts']: (N/d).mkdir(parents=True)
        physical.mkdir(parents=True)
        (N/'train').symlink_to(physical, target_is_directory=True)
        for t in 'SCI': (N/'train'/t).mkdir()
        (N/'data_split_whitelist17').symlink_to(data, target_is_directory=True)
        for name, digest in parent['code_sha256'].items():
            assert sha(R/'code'/name)==digest, name
            shutil.copy2(R/'code'/name,N/'code'/name);assert sha(N/'code'/name)==digest
        shutil.copy2(R/'source_trainer.py',N/'source_trainer.py');assert sha(N/'source_trainer.py')==parent['reference_trainer_sha256']
        state={k:parent[k] for k in ['split','max_epochs','supervised_loss','allowed_tf_sf_pairs','native_grid_points','code_sha256','normalization_sha256','reference_trainer_sha256','architecture']}
        activation='relu' if variant=='relu' else 'tanh'
        lrs='1e-2,5e-3,2e-3,1e-3,5e-4,2e-4' if variant=='relu' else '1e-3,5e-4,2e-4'
        state.update(run_root=str(N),reference_run=str(R),created_utc=now(),initialization='random; no checkpoint',trial=variant,activation_seq=[activation]*len(parent['activation_seq']),lr_stage_values=[float(v) for v in lrs.split(',')],commands={},target_jobs={},submission_commands={},status='preparing; not submitted')
        write(N/'submitted_jobs.json',state)
        write(N/'storage_provenance.json',{'logical_training_root':str(N/'train'),'physical_training_root':str(physical),'reason':'Scratch has 1610 file slots; 12 targets x 40 stages x 4 diagnostic files alone could require 1920. Dedicated Home outputs avoid scratch inode exhaustion. No existing files moved or deleted.','scientific_settings_unchanged':True})
        for t in 'SCI':
            probe=N/'train'/t/'write_probe.tmp';probe.write_text('new run output writable');assert probe.read_text()=='new run output writable';probe.unlink()
        for test,args in [('test_original_loss.py',[]),('test_whitelist.py',[str(data/'feature_norm.json')])]:
            subprocess.run([PY,str(N/'code'/test),*args],env=ENV,check=True)
        subprocess.run([PY,'-c',SMOKE,str(N)],env=ENV,check=True)
        differences={}
        for t in 'SCI':
            original=shlex.split(parent['commands'][t]);args=original.copy()
            setting='--activation_seq' if variant=='relu' else '--lr_stage_values'
            replacements={1:str(N/'code/train_lowtfsf_grad01.py'),original.index('--out_dir')+1:str(N/'train'/t),original.index(setting)+1:','.join(state['activation_seq']) if variant=='relu' else lrs}
            for i,value in replacements.items():args[i]=value
            changed={i for i,(a,b) in enumerate(zip(original,args)) if a!=b};assert changed==set(replacements) and len(args)==len(original)
            assert not any(k in args for k in ['--resume','--init_checkpoint','--init_from_5d_dir'])
            command=shlex.join(args);state['commands'][t]=command
            differences[t]=[{'index':i,'before':original[i],'after':args[i]} for i in sorted(changed)]
            source=(R/f'code/all615_{label}_{t}.sh').read_text();assert parent['commands'][t] in source
            wrapper=source.replace(parent['commands'][t],command).replace(str(R/'loss_validation.json'),str(N/'loss_validation.json')).replace(f'xh2906_{label}_all615_',f'xh2906_{label}_{variant}_all615_')
            f=N/f'code/{stem}_{t}.sh';f.write_text(wrapper);subprocess.run(['bash','-n',str(f)],check=True)
            oldcmd=json.loads((R/f'receipts/all615_{label}_{t}.json').read_text())['command']
            assert not any(x.startswith('--dependency=') for x in oldcmd)
            cmd=[x.replace(str(R),str(N)).replace(f'all615_{label}_',stem+'_') for x in oldcmd];cmd[-1]=str(f)
            fixed=lambda c:[x for x in c if x.startswith(('--account=','--cpus-per-task=','--mem=','--time=','--gres=','--partition='))]
            assert fixed(cmd)==fixed(oldcmd);state['submission_commands'][t]=cmd
        state['validation']={'original_loss_passed':True,'whitelist_passed':True,'model_loss_and_baselines_passed':True,'architecture_unchanged':True,'only_requested_setting_and_isolated_paths_changed':True,'scheduler_resources_unchanged':True,'source_code_sha_unchanged':True}
        state['status']='all preflight checks passed; not submitted';write(N/'submitted_jobs.json',state);write(N/'experiment_command_diff.json',differences)
        plan=json.loads((R/'plan.json').read_text());plan.update(run_root=str(N),reference_run=str(R),shared_split=str(data),architecture=state['architecture'],activation_seq=state['activation_seq'],lr_stage_values=state['lr_stage_values'],scope='Three scratch trainings; activation-only or starting-LR trial. No posttraining.')
        write(N/'plan.json',plan);write(N/'dataset_validation_at_submission.json',validation)
        print('PREPARED',N.name,flush=True)

def submit():
    prepared=[]
    for label,variant in TRIALS:
        R,N,stem=paths(label,variant);state=json.loads((N/'submitted_jobs.json').read_text())
        assert state['status']=='all preflight checks passed; not submitted' and not state['target_jobs']
        assert json.loads((N/'architecture_validation.json').read_text())['passed'] and json.loads((N/'loss_validation.json').read_text())['passed']
        for name,digest in state['code_sha256'].items():assert sha(N/'code'/name)==digest
        for t in 'SCI':assert not (N/f'receipts/{stem}_{t}.json').exists()
        prepared.append((N,stem,state))
    for N,stem,state in prepared:
        for t in 'SCI':
            receipt=N/f'receipts/{stem}_{t}.json';cmd=state['submission_commands'][t]
            with receipt.open('x') as f:json.dump({'state':'submitting','command':cmd,'created_utc':now()},f)
            result=subprocess.run(cmd,text=True,capture_output=True)
            if result.returncode:
                write(receipt,{'state':'rejected','command':cmd,'stdout':result.stdout,'stderr':result.stderr});raise RuntimeError(result.stderr)
            job=result.stdout.strip().split(';')[0];assert job.isdigit(),result.stdout
            write(receipt,{'state':'confirmed','job_id':job,'command':cmd,'stdout':result.stdout,'confirmed_utc':now()})
            state['target_jobs'][t]=job;write(N/'submitted_jobs.json',state);print('SUBMITTED',N.name,t,job,flush=True)
        state['status']='three random-initialization scratch trainings submitted';write(N/'submitted_jobs.json',state)
    print('ALL TWELVE SUBMISSIONS CONFIRMED',flush=True)

if __name__=='__main__':
    {'prepare':prepare,'submit':submit}[sys.argv[1]]()
