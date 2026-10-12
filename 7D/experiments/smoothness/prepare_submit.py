"""Isolated, receipt-guarded preparation/submission of three SCI fine-tunes."""
from pathlib import Path
import datetime, hashlib, json, os, shlex, shutil, subprocess, sys
import torch

PARENT = Path('/scratch/xh2906/librarySCI_runs/h96_tfsf7d_h512x5_scratch_fixedbaseline_goodFRgt20_whitelist17_all615DS10_20261004_original_loss_relu')
SETUP = Path('/home/xh2906/librarySCI_smooth_finetune_20261004')
ROOT = Path('/scratch/xh2906/librarySCI_runs')
HOME_OUTPUTS = Path('/home/xh2906/librarySCI_active_training')
STRENGTHS = {'low': .05, 'medium': .20, 'high': .80}
PY = '/scratch/xh2906/conda/envs/py310/bin/python'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TMPDIR=str(SETUP/'tmp'), MPLCONFIGDIR=str(SETUP/'mpl'))


def write(p, value):
    p.write_text(json.dumps(value, indent=2))


def runroot(level):
    return ROOT/f'h96_tfsf7d_h512x5_finetune_fixedbaseline_goodFRgt20_whitelist17_all615DS10_20261004_original_loss_relu_smooth4_{level}'


def prepare():
    assert not (SETUP/'preparation_complete.json').exists()
    torch.set_num_threads(2)
    parent = json.loads((PARENT/'submitted_jobs.json').read_text())
    data = Path(parent['split']); norm = json.loads((data/'feature_norm.json').read_text())
    valid = json.loads((data/'dataset_validation.json').read_text())
    assert valid['verified'] and valid['no_subsampling'] and valid['native_points_per_block']==615
    assert sha(data/'feature_norm.json')==parent['normalization_sha256']
    for name in ['tmp','mpl','initial_checkpoints']: (SETUP/name).mkdir(parents=True, exist_ok=True)
    if not (SETUP/'data_split_whitelist17').exists(): (SETUP/'data_split_whitelist17').symlink_to(data)
    for name,digest in parent['code_sha256'].items():
        assert sha(PARENT/'code'/name)==digest
        shutil.copy2(PARENT/'code'/name,SETUP/'code'/name)
    shutil.copy2(PARENT/'source_trainer.py',SETUP/'source_trainer.py')
    assert sha(SETUP/'source_trainer.py')==parent['reference_trainer_sha256']
    for test,args in [('test_original_loss.py',[]),('test_whitelist.py',[str(data/'feature_norm.json')]),('test_smoothness.py',[str(data/'feature_norm.json')])]:
        result=subprocess.run([PY,str(SETUP/'code'/test),*args],env=ENV,text=True,capture_output=True)
        (SETUP/(test+'.stdout')).write_text(result.stdout);(SETUP/(test+'.stderr')).write_text(result.stderr)
        assert result.returncode==0,(test,result.stdout,result.stderr)
        print('PASSED',test,result.stdout[-800:],flush=True)
    sys.path.insert(0,str(SETUP/'code'))
    from common import ModelConfig,TrainConfig,build_model
    from smoothness import SETTINGS,AXES,curvature_terms,balanced_curvature
    import train_smooth as trainer
    calibration={'settings':SETTINGS,'targets':{},'created_utc':now(),'source_run':str(PARENT)}
    provenance={}
    for target in 'SCI':
        src=PARENT/'train'/target/'best.pt';dst=SETUP/'initial_checkpoints'/(target+'.pt')
        stat=src.stat();digest=sha(src)
        if dst.exists(): assert sha(dst)==digest
        else: shutil.copy2(src,dst)
        assert sha(src)==sha(dst)==digest and src.stat().st_mtime_ns==stat.st_mtime_ns
        q=torch.load(dst,map_location='cpu',weights_only=False)
        assert q['feature_norm']==norm and q['model_config']['input_dim']==7
        assert tuple(q['model_config']['hidden_dims'])==(512,)*5 and tuple(q['model_config']['activation_seq'])==('relu',)*5
        assert q['model_config']['softplus_out'] and q['extra']['best_epoch']>0
        model=build_model(ModelConfig(**q['model_config']));model.load_state_dict(q['model'],strict=True)
        assert all(torch.isfinite(x).all() for x in model.parameters())
        with torch.no_grad(): terms=curvature_terms(model,torch.zeros(1,7),norm,target,validation=True)
        raw={k:float(v) for k,v in terms.items()};scales={k:max(v,SETTINGS['scale_floor']) for k,v in raw.items()}
        assert all(0<=v<float('inf') for v in raw.values())
        fitting=float(q['extra']['best_ckpt_selection_score']);r0=sum(raw[k]/scales[k] for k in AXES)/4
        assert fitting>0 and r0>0
        weights={level:fraction*fitting/r0 for level,fraction in STRENGTHS.items()}
        calibration['targets'][target]={'axis_curvature':raw,'axis_scales':scales,'initial_balanced_curvature':r0,'source_validation_fitting_loss':fitting,'lambda_by_strength':weights}
        provenance[target]={'source':str(src),'frozen':str(dst),'sha256':digest,'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'selected_epoch':q['extra']['best_epoch'],'source_extra':q['extra'],'source_model_config':q['model_config'],'normalization_sha256':parent['normalization_sha256']}
        dst.chmod(0o444)
    write(SETUP/'smoothness_calibration.json',calibration);write(SETUP/'initial_checkpoint_provenance.json',provenance)
    trainer.install(calibration)
    probes={}
    for target in 'SCI':
        probes[target]={}
        for level in STRENGTHS:
            q=torch.load(SETUP/'initial_checkpoints'/(target+'.pt'),map_location='cpu',weights_only=False)
            model=build_model(ModelConfig(**q['model_config']));model.load_state_dict(q['model'],strict=True)
            torch.manual_seed(42)
            terms=curvature_terms(model,torch.zeros(64,7),norm,target)
            loss=calibration['targets'][target]['lambda_by_strength'][level]*balanced_curvature(terms,calibration['targets'][target])
            loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            grad=sum(float(p.grad.square().sum()) for p in model.parameters())**.5;assert grad>0
            torch.optim.AdamW(model.parameters(),lr=2e-4).step();assert all(torch.isfinite(p).all() for p in model.parameters())
            probes[target][level]={'weighted_smoothness':float(loss.detach()),'finite_nonzero_gradient_norm':grad,'finite_AdamW_step':True}
    write(SETUP/'preflight_verification.json',{'passed':True,'actual_frozen_checkpoint_probes':probes,'source_epochs':{t:p['selected_epoch'] for t,p in provenance.items()},'original_loss_and_whitelist_passed':True,'analytic_boundary_periodicity_and_integration_passed':True})
    runs={}
    for level,fraction in STRENGTHS.items():
        r=runroot(level);physical=HOME_OUTPUTS/r.name/'train';assert not r.exists() and not physical.parent.exists()
        for d in ['code','logs','receipts']: (r/d).mkdir(parents=True)
        physical.mkdir(parents=True);(r/'train').symlink_to(physical);(r/'data_split_whitelist17').symlink_to(data);(r/'initial_checkpoints').symlink_to(SETUP/'initial_checkpoints')
        for f in (SETUP/'code').glob('*.py'): shutil.copy2(f,r/'code'/f.name)
        shutil.copy2(SETUP/'source_trainer.py',r/'source_trainer.py')
        for name in ['smoothness_calibration.json','initial_checkpoint_provenance.json','preflight_verification.json','loss_validation.json','loss_ratio_validation.json']:
            shutil.copy2(SETUP/name,r/name)
        state={'run_root':str(r),'parent_run':str(PARENT),'split':str(data),'normalization_sha256':parent['normalization_sha256'],'architecture':parent['architecture'],'activation_seq':parent['activation_seq'],'initialization':'Strict frozen source S71/C47/I66 weights; independent fresh AdamW optimizer','strength':level,'initial_smoothness_fraction_of_source_fitting_loss':fraction,'schedule':{'lr_stage_values':[2e-4,5e-5,1e-5],'stage_epochs':200,'max_stages':40,'min_improve':.001,'min_best_rel_gain':.0005,'stop_on_lowest_lr_plateau':True},'smoothness':SETTINGS,'commands':{},'submission_commands':{},'target_jobs':{},'status':'preparing','source_code_sha256':parent['code_sha256'],'code_sha256':{f.name:sha(f) for f in (r/'code').glob('*.py')}}
        differences={}
        for target in 'SCI':
            original=shlex.split(parent['commands'][target]);args=original.copy();args[1]=str(r/'code/train_smooth.py')
            replacements={'--out_dir':str(r/'train'/target),'--lr_stage_values':'2e-4,5e-5,1e-5','--curvature_lambda':str(calibration['targets'][target]['lambda_by_strength'][level])}
            for flag,value in replacements.items(): args[args.index(flag)+1]=value
            additions=['--init_checkpoint',str(r/'initial_checkpoints'/(target+'.pt')),'--global_curvature_balance_path',str(r/'smoothness_calibration.json'),'--global_curvature_balance_mode','baseline_relative','--curvature_max_points','256','--ckpt_global_curvature_points','1024','--ckpt_global_curvature_beta',str(calibration['targets'][target]['lambda_by_strength'][level])]
            args+=additions;command=shlex.join(args);state['commands'][target]=command
            differences[target]={'changed':[{'index':i,'before':a,'after':b} for i,(a,b) in enumerate(zip(original,args)) if a!=b],'added':additions}
            assert len(differences[target]['changed'])==4
            source=(PARENT/f'code/all615_h512x5_relu_{target}.sh').read_text();assert parent['commands'][target] in source
            wrapper=source.replace(parent['commands'][target],command).replace(str(PARENT/'loss_validation.json'),str(r/'loss_validation.json')).replace('h512x5_relu_all615','h512x5_smooth_'+level)
            check="import json,hashlib;from pathlib import Path;r=Path("+repr(str(r))+");assert json.load(open(r/'preflight_verification.json'))['passed'];p=r/'initial_checkpoints'/"+repr(target+'.pt')+";assert hashlib.sha256(p.read_bytes()).hexdigest()==json.load(open(r/'initial_checkpoint_provenance.json'))["+repr(target)+"]['sha256']"
            wrapper=wrapper.replace(command,PY+' -c '+shlex.quote(check)+'\n'+command)
            f=r/f'code/smooth_{level}_{target}.sh';f.write_text(wrapper);subprocess.run(['bash','-n',str(f)],check=True)
            submission=parent['submission_commands'][target]
            submission=[x.replace(str(PARENT),str(r)).replace('all615_h512x5_relu_'+target,'all615_h512x5_smooth_'+level+'_'+target) for x in submission];submission[-1]=str(f)
            fixed=lambda cmd:[x for x in cmd if x.startswith(('--account=','--cpus-per-task=','--mem=','--time=','--gres=','--partition='))]
            assert fixed(submission)==fixed(parent['submission_commands'][target]);state['submission_commands'][target]=submission
        assert all(sha(r/'code'/name)==digest for name,digest in parent['code_sha256'].items())
        state['status']='preflight passed; not submitted';write(r/'submitted_jobs.json',state);write(r/'plan.json',state);write(r/'experiment_command_diff.json',differences)
        write(r/'storage_provenance.json',{'logical_training_root':str(r/'train'),'physical_training_root':str(physical),'reason':'Dedicated Home outputs preserve scarce scratch inode headroom; no existing outputs moved/deleted','training_tmp':'compute-node-local /tmp'})
        runs[level]=state;print('PREPARED',level,str(r),flush=True)
    write(SETUP/'preparation_complete.json',{'passed':True,'runs':runs,'created_utc':now()})


def submit():
    prepared=json.loads((SETUP/'preparation_complete.json').read_text());assert prepared['passed']
    for level in STRENGTHS:
        r=runroot(level);s=json.loads((r/'submitted_jobs.json').read_text());assert s['status']=='preflight passed; not submitted' and not s['target_jobs']
        assert json.loads((r/'preflight_verification.json').read_text())['passed']
        for name,digest in s['code_sha256'].items(): assert sha(r/'code'/name)==digest
        for target in 'SCI': assert not (r/f'receipts/smooth_{level}_{target}.json').exists()
    for level in STRENGTHS:
        r=runroot(level);s=json.loads((r/'submitted_jobs.json').read_text())
        for target in 'SCI':
            receipt=r/f'receipts/smooth_{level}_{target}.json';command=s['submission_commands'][target]
            with receipt.open('x') as f:json.dump({'state':'submitting','command':command,'created_utc':now()},f)
            result=subprocess.run(command,text=True,capture_output=True)
            if result.returncode:
                write(receipt,{'state':'rejected','command':command,'stdout':result.stdout,'stderr':result.stderr});raise RuntimeError(result.stderr)
            job=result.stdout.strip().split(';')[0];assert job.isdigit(),result.stdout
            write(receipt,{'state':'confirmed','job_id':job,'command':command,'stdout':result.stdout,'confirmed_utc':now()});s['target_jobs'][target]=job;write(r/'submitted_jobs.json',s)
            print('SUBMITTED',level,target,job,flush=True)
        s['status']='three fine-tuning jobs submitted';write(r/'submitted_jobs.json',s)
    print('ALL NINE SUBMISSIONS CONFIRMED',flush=True)


if __name__=='__main__': {'prepare':prepare,'submit':submit}[sys.argv[1]]()
