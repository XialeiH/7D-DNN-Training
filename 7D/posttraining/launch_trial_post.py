from pathlib import Path
import json,shutil,hashlib,subprocess,sys,shlex,fcntl
import torch,numpy as np
from scipy.io import savemat,loadmat
R=Path(sys.argv[1]);width=int(sys.argv[2]);depth=int(sys.argv[3]);torch.set_num_threads(2)
P=R/'posttraining_7d_tf32_globalall7';CODE=P/'code';A=P/'atlas_139';PY='/scratch/xh2906/conda/envs/py310/bin/python';EXTRA=Path('/scratch/xh2906/librarySCI_runs/h96_tfsf7d_extra50_full_libequ70_20260905');REF=Path('/scratch/xh2906/smooth7_representative192_20260919')
P.mkdir(exist_ok=True); lock=(P/'launch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (P/'launch.json').exists(),'Existing submission: reconcile before retry'
for d in ['code','logs','checkpoints']:(P/d).mkdir(exist_ok=True)
TEMPLATE=Path('/scratch/xh2906/librarySCI_runs/h96_tfsf7d_h512x4_scratch_fixedbaseline_goodFRgt20_whitelist17_all615DS10_20261002_original_loss/posttraining_7d_tf32_globalall7/code')
assert (P/'template_code/template_source_sha.json').is_file(), 'Pinned template required'
TEMPLATE=P/'template_code'
shutil.copytree(TEMPLATE,CODE,dirs_exist_ok=True)
for f in CODE.iterdir():
 if f.is_file() and f.suffix in ['.py','.m','.sh']:
  f.write_text(f.read_text().replace(str(TEMPLATE.parent.resolve()),str(P)).replace(str(TEMPLATE.parent),str(P)))
# New-run wrappers only; all scientific routines are copied without changing numerical logic.
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
meta={};parity={};rng=np.random.default_rng(921)
x=np.column_stack([rng.uniform(-1,1,64),rng.uniform(.1,95,64),rng.uniform(1,40,64),rng.uniform(500,47000,64),rng.uniform(218,35000,64),rng.choice([.5,1,2,4,10,16,24,32],64),rng.choice([.5,1,2,2.5,3,5,10],64)])
x[:8,5]=[.5,1,2,4,10,16,24,32];parity['X']=x
for t in 'SCI':
 src=R/'train'/t/'best.pt'; dst=P/'checkpoints'/(t+'.pt');shutil.copy2(src,dst)
 c=torch.load(dst,map_location='cpu',weights_only=False);assert c['model_config']['input_dim']==7
 assert c['feature_norm']['feature_names']['order']==['OriCos2','Contrast','L6','L4E','L4I','lgnTF','lgnSF']
 assert list(c['model_config']['hidden_dims']) == [width]*depth
 expected=json.loads((R/'submitted_jobs.json').read_text())
 assert list(c['model_config']['activation_seq']) == expected['activation_seq']
 assert set(c['model_config']['activation_seq']) <= {'tanh','relu'}
 meta[t]={'source':str(src),'snapshot':str(dst),'sha256':sha(dst),'bytes':src.stat().st_size,'mtime_ns':src.stat().st_mtime_ns,'extra':c['extra'],'model_config':c['model_config'],'feature_norm':c['feature_norm'],'train_config':c['train_config']}
 norm=c['feature_norm'];features=['OriCos2','Contrast','L6','L4E','L4I','lgnTF','lgnSF']
 mu=np.array([0 if k=='OriCos2' else norm[k]['mean'] for k in features],dtype=np.float32).astype(float);sd=np.array([1 if k=='OriCos2' else norm[k]['std'] for k in features],dtype=np.float32).astype(float)
 h=torch.tensor((x-mu)/sd,dtype=torch.float64)
 indices=sorted(int(k.split('.')[1]) for k in c['model'] if k.startswith('net.') and k.endswith('.weight'))
 assert len(indices)==depth+1
 for layer,i in enumerate(indices):
  h=h@c['model'][f'net.{i}.weight'].double().T+c['model'][f'net.{i}.bias'].double()
  if i!=indices[-1]:
   activation=c['model_config']['activation_seq'][layer]
   h=torch.relu(h) if activation=='relu' else torch.tanh(h)
 parity[t]=torch.nn.functional.softplus(h).numpy()
 sys.path.insert(0,str(R/'code'))
 from common import ModelConfig,build_model
 model=build_model(ModelConfig(**c['model_config'])).double();model.load_state_dict(c['model']);model.eval()
 with torch.no_grad():actual=model(torch.tensor((x-mu)/sd,dtype=torch.float64)).numpy()
 assert np.allclose(actual.reshape(-1),parity[t].reshape(-1),rtol=0,atol=1e-9), 'Independent PyTorch forward mismatch'
 meta[t]['parity_reference']='64 physical inputs, matching checkpoint evaluated in PyTorch float64; includes TF32'
 assert sha(src)==sha(dst) and src.stat().st_mtime_ns==meta[t]['mtime_ns'], 'Checkpoint changed while freezing'
(P/'checkpoint_provenance.json').write_text(json.dumps(meta,indent=2,default=str));savemat(P/'parity_inputs.mat',parity)
# Validate reference files without changing the shared reference manifest.
s=(EXTRA/'code/validate_extra50_references.py').read_text().replace("(ROOT / 'references_ready.json')",repr(str(P/'reference_validation.json')))
s=s.replace("'"+str(P/'reference_validation.json')+"'.write_text", "Path("+repr(str(P/'reference_validation.json'))+").write_text")
s="from pathlib import Path\nimport sys\nsys.path.insert(0,"+repr(str(EXTRA/'code'))+")\n"+s
(CODE/'validate_references.py').write_text(s);compile(s,str(CODE/'validate_references.py'),'exec')
# Both plotting paths remain linear under the user's explicit display instruction.
plot=CODE/'plot_dense.py';s=plot.read_text();s=s.replace("set_xscale('log')","set_xscale('linear')").replace('set_xscale("log")','set_xscale("linear")');s=s.replace("set_yscale('log')","set_yscale('linear')").replace('set_yscale("log")','set_yscale("linear")');plot.write_text(s)
f=CODE/'finalize_global_smooth_posttraining.py';s=f.read_text();s=s.replace('Path("/scratch/xh2906/h96_tfsf7d_log_variant_20260727/submit_runtime_tuning_curves.py")','Path('+repr(str(CODE/'submit_runtime_dense_tuning.py'))+')');f.write_text(s)
f=CODE/'prepare_archive.py';s=f.read_text().replace('this frozen fine-tuned network','this frozen scratch network');f.write_text(s)
# New-run export runner only; MATLAB numerical routines are unchanged.
archive=CODE/'submit_archive.py';original=archive.read_text();lines=original.splitlines()
indices=[i for i,line in enumerate(lines) if line.startswith("script=code/'build_archive.sh'")];assert len(indices)==1
lines[indices[0]]="script=code/'build_archive.sh';script.write_text('#!/bin/bash\\nset -euo pipefail\\nmodule purge\\nmodule load anaconda3/2025.06 matlab/2025b\\nsource /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh\\nconda activate py310\\nexport TMPDIR=/tmp/xh2906_archvariant_${SLURM_JOB_ID}\\nexport MATLAB_PREFDIR=$TMPDIR/prefs\\nmkdir -p \\\"$TMPDIR\\\" \\\"$MATLAB_PREFDIR\\\"\\n'+shlex.join([PY,str(code/'prepare_archive.py')])+'\\n'+shlex.join([PY,str(code/'run_archive_exports.py')])+'\\n')"
original='\n'.join(lines)+'\n';compile(original,str(archive),'exec');archive.write_text(original)
shutil.copy2(P/'run_archive_exports.py',CODE/'run_archive_exports.py')

# Only currently available local representative conditions, with numerical references.
inv=json.loads((P/'local_reference_inventory.json').read_text());cases=json.loads((REF/'cases.json').read_text());ids=[];records=[]
for q in inv:
 hits=[i for i,c in enumerate(cases) if all(float(c[a])==q[b] for a,b in [('angle','orientation'),('contrast','contrast'),('tf','tf'),('sf','sf')])];assert len(hits)==1
 i=hits[0];folder=Path('/scratch/xh2906/representative_full_library_192_20260913/equilibrium_ready13')/f'case{i:03d}'
 files=list(folder.glob('libonly*.mat'));assert len(files)==1,(i,files)
 z=loadmat(folder/'maps.mat',simplify_cells=True);assert np.isfinite(z['repMaps']).all();assert np.array_equal(z['repQ'],[q[k] for k in ['orientation','contrast','tf','sf']])
 ids.append(i);records.append(dict(q,case=i,reference=str(files[0]),reference_sha256=sha(files[0]),maps=str(folder/'maps.mat'),maps_sha256=sha(folder/'maps.mat')))
(P/'representative_reference_manifest.json').write_text(json.dumps(records,indent=2))
rep=P/'representative192'
for d in ['code','runtime','data','pdf','logs']:(rep/d).mkdir(parents=True,exist_ok=True)
shutil.copytree(REF/'code',rep/'code',dirs_exist_ok=True);shutil.copy2(REF/'cases.json',rep/'cases.json');(rep/'network').symlink_to(REF/'network',target_is_directory=True)
# Launcher prepares/validates runtime, then starts independent CPU evaluations.
coord='''from pathlib import Path
import subprocess,json,shutil,sys
P=Path(POST);C=P/'code';A=P/'atlas_139';PY=sys.executable
cmd=[PY,str(C/'prepare_submit_logtfsf_atlas102.py'),'--profile','all615post','--atlas-root',str(A),'--require-transform','identity','--require-tf-max','32','--save-operating-points','--extra50-root',EXTRA,'--mode','prepare']
for t in 'SCI':cmd+=['--model-'+t.lower(),str(P/'checkpoints'/(t+'.pt'))]
subprocess.run([PY,str(C/'validate_references.py')],check=True)
subprocess.run(cmd,check=True)
subprocess.run(['matlab','-singleCompThread','-batch',"addpath('"+str(C)+"'); verify_runtime('"+str(P)+"')"],check=True)
subprocess.run([PY,str(C/'build_submit_logtfsf_atlas102.py'),'--profile','all615post','--root',str(A),'--runtime',str(A/'runtime_h96iter'),'--save-operating-points','--extra50-root',EXTRA,'--mode','submit'],check=True)
subprocess.run([PY,str(C/'submit_dense.py'),'--post-root',str(P),'--runtime',str(A/'runtime_h96iter')],check=True)
rep=P/'representative192';shutil.copytree(A/'runtime_h96iter',rep/'runtime',dirs_exist_ok=True)
job=subprocess.check_output(['sbatch','--parsable','--account=torch_pr_482_general','--partition=cpu_short,cs','--cpus-per-task=1','--mem=12G','--time=01:00:00','--array='+','.join(map(str,IDS))+'%16','--job-name=aug20_rep','--output='+str(rep/'logs/%A_%a.out'),'--error='+str(rep/'logs/%A_%a.err'),str(rep/'job.sh')],text=True).strip()
(rep/'submission.json').write_text(json.dumps({'job_id':job,'case_ids':IDS,'count':len(IDS)},indent=2))
(P/'launch_complete.json').write_text(json.dumps({'atlas':str(A),'representative_job':job,'representative_count':len(IDS)},indent=2))
'''.replace('POST',repr(str(P))).replace('EXTRA',repr(str(EXTRA))).replace('IDS',repr(sorted(ids)))
(CODE/'coordinate_post.py').write_text(coord);compile(coord,str(CODE/'coordinate_post.py'),'exec')
(rep/'job.sh').write_text('#!/bin/bash\nset -euo pipefail\nmodule load matlab/2025b\nexport TMPDIR=/tmp/xh2906_aug20ft_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}\nexport MATLAB_PREFDIR=$TMPDIR/prefs\nmkdir -p \"$TMPDIR\" \"$MATLAB_PREFDIR\"\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\nmatlab -singleCompThread -batch "addpath(\''+str(rep/'code')+'\'); run_one(\''+str(rep)+'\',str2double(getenv(\'SLURM_ARRAY_TASK_ID\')))"\n')
ENV='set -euo pipefail\nmodule purge\nmodule load anaconda3/2025.06 matlab/2025b\nsource /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh\nconda activate py310\nexport TMPDIR=/tmp/xh2906_aug20ft_${SLURM_JOB_ID}\nexport MATLAB_PREFDIR=$TMPDIR/prefs\nmkdir -p \"$TMPDIR\" \"$MATLAB_PREFDIR\"\n'
def submit(label,cmd,dep=None):
 script=CODE/(label+'.sh');script.write_text('#!/bin/bash\n'+ENV+shlex.join(cmd)+'\n');subprocess.run(['bash','-n',str(script)],check=True)
 args=['sbatch','--parsable','--account=torch_pr_482_general','--partition=cpu_short,cs','--cpus-per-task=2','--mem=16G','--time=01:00:00','--job-name='+label,'--output='+str(P/'logs/%x.%j.out'),'--error='+str(P/'logs/%x.%j.err')]
 if dep:args+=['--dependency=afterok:'+dep]
 args+=[str(script)]
 intent=P/(label+'_intent.json')
 with intent.open('x') as f:json.dump({'state':'submitting','command':args},f)
 j=subprocess.check_output(args,text=True).strip().split(';')[0];intent.write_text(json.dumps({'state':'confirmed','job_id':j,'command':args},indent=2));return j
j=submit('all615_post_launch',[PY,str(CODE/'coordinate_post.py')]);state=json.loads((R/'posttraining_submission.json').read_text());state.update({'posttraining_launcher_job':j,'atlas_root':str(A),'case_count':139,'checkpoint_epochs':{t:meta[t]['extra']['best_epoch'] for t in 'SCI'},'representative_cases':len(ids),'dense_conditions':97,'status':'submitted after terminal training and valid checkpoints','post_root':str(P)})
(P/'launch.json').write_text(json.dumps(state,indent=2));(R/'posttraining_submission.json').write_text(json.dumps(state,indent=2))
f=submit('all615_post_final',[PY,str(CODE/'finalize_global_smooth_posttraining.py'),'--run-root',str(R),'--profile','all615post','--atlas-name','atlas_139'],j);state['posttraining_finalizer_job']=f
(P/'launch.json').write_text(json.dumps(state,indent=2));(R/'posttraining_submission.json').write_text(json.dumps(state,indent=2));print(json.dumps(state,indent=2))
