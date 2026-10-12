from pathlib import Path
import datetime, hashlib, importlib.util, json, shlex, subprocess, sys, urllib.request

SETUP=Path('/home/xh2906/librarySCI_activation_lr_trials_20261004')
spec=importlib.util.spec_from_file_location('trials',SETUP/'submit_trials.py')
trials=importlib.util.module_from_spec(spec);spec.loader.exec_module(trials)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'mode':sys.argv[1],'runs':[]}
jobs=[]
for label,variant in trials.TRIALS:
    R,N,stem=trials.paths(label,variant)
    old=json.loads((R/'submitted_jobs.json').read_text());state=json.loads((N/'submitted_jobs.json').read_text())
    assert state['architecture']==old['architecture'] and state['split']==old['split']
    assert state['supervised_loss']==old['supervised_loss'] and state['normalization_sha256']==old['normalization_sha256']
    for name,digest in state['code_sha256'].items():assert sha(N/'code'/name)==sha(R/'code'/name)==digest
    assert sha(Path(state['split'])/'feature_norm.json')==state['normalization_sha256']
    files=['architecture_validation.json','loss_validation.json','loss_ratio_validation.json','experiment_command_diff.json','storage_provenance.json','plan.json']
    audits={name:json.loads((N/name).read_text()) for name in files}
    assert all(audits[f]['passed'] for f in files[:3])
    assert (N/'train').resolve()==Path(audits['storage_provenance.json']['physical_training_root'])
    receipts={}
    for t in 'SCI':
        a=shlex.split(old['commands'][t]);b=shlex.split(state['commands'][t])
        allowed={1,a.index('--out_dir')+1,a.index('--activation_seq' if variant=='relu' else '--lr_stage_values')+1}
        assert len(a)==len(b) and {i for i,(x,y) in enumerate(zip(a,b)) if x!=y}==allowed
        old_scheduler=json.loads((R/f'receipts/all615_{label}_{t}.json').read_text())['command']
        cmd=state['submission_commands'][t]
        assert not any(x.startswith('--dependency=') for x in cmd)
        resources=lambda c:[x for x in c if x.startswith(('--account=','--cpus-per-task=','--mem=','--time=','--gres=','--partition='))]
        assert resources(old_scheduler)==resources(cmd)
        if sys.argv[1]=='submitted':
            receipt=json.loads((N/f'receipts/{stem}_{t}.json').read_text())
            assert receipt['state']=='confirmed' and receipt['job_id']==state['target_jobs'][t] and receipt['command']==cmd
            jobs.append(receipt['job_id']);receipts[t]=receipt
        else:
            assert state['status']=='all preflight checks passed; not submitted' and not state['target_jobs']
    report['runs'].append({'label':label,'trial':variant,'manifest':state,'audits':audits,'receipts':receipts})
if sys.argv[1]=='submitted':
    assert len(jobs)==len(set(jobs))==12
    queries={'squeue':['squeue','-h','-j',','.join(jobs),'-o','%i|%T|%M|%R|%P|%j'],
             'sacct':['sacct','-X','-n','-P','-j',','.join(jobs),'--format=JobIDRaw,State,Elapsed,ExitCode,Reason,Partition']}
    report['scheduler']={}
    for name,cmd in queries.items():
        p=subprocess.run(cmd,text=True,capture_output=True);assert p.returncode==0,p.stderr
        report['scheduler'][name]={'command':cmd,'stdout':p.stdout,'stderr':p.stderr}
    report['scontrol']={}
    for job in jobs:
        p=subprocess.run(['scontrol','show','job',job,'-o'],text=True,capture_output=True);assert p.returncode==0,p.stderr
        fields=dict(item.split('=',1) for item in p.stdout.split() if '=' in item)
        assert fields['Account']=='torch_pr_482_general' and fields['NumCPUs']=='8' and fields['TimeLimit']=='1-00:00:00'
        assert 'mem=64G' in fields['ReqTRES'] and 'gres/gpu=1' in fields['ReqTRES'] and fields.get('Dependency')=='(null)'
        report['scontrol'][job]=p.stdout
report['verified']=True
dest=SETUP/f'{sys.argv[1]}_verification.json';dest.write_text(json.dumps(report,indent=2))
req=urllib.request.Request('http://127.0.0.1:8765/remote_audit.json',data=dest.read_bytes(),method='POST')
with urllib.request.urlopen(req,timeout=60) as response:assert response.status==200
print('VERIFIED',sys.argv[1],[(r['label'],r['trial'],r['manifest']['target_jobs']) for r in report['runs']],flush=True)
if 'scheduler' in report:print(report['scheduler']['squeue']['stdout'],flush=True)
