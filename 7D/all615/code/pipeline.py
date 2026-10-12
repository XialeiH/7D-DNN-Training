"""Bounded full-grid simulation waves followed by verified-data scratch training."""
from pathlib import Path
import json,shlex,subprocess,sys,hashlib,shutil,os,time,fcntl,re
import h5py
R=Path(__file__).resolve().parents[1];plan=json.loads((R/'plan.json').read_text());PY='/scratch/xh2906/conda/envs/py310/bin/python';ENV='module purge\nmodule load anaconda3/2025.06\nsource /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh\nconda activate py310\nexport PYTHONDONTWRITEBYTECODE=1\nexport TMPDIR="/tmp/xh2906_all615_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"\nexport MPLCONFIGDIR="$TMPDIR/mpl"\nexport MATLAB_PREFDIR="$TMPDIR/prefs"\nexport H96_MATLAB_TMP_ROOT="$TMPDIR/pool"\nexport H96_PARPOOL_WORKERS=8\nmkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$MATLAB_PREFDIR"\n'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(state):
 p=R/'submitted_jobs.json.tmp';p.write_text(json.dumps(state,indent=2));p.replace(R/'submitted_jobs.json')
def script(name,command,matlab=False):
 p=R/'code'/(name+'.sh');p.write_text('#!/bin/bash\nset -euo pipefail\n'+ENV+('module load matlab/2025b\n' if matlab else '')+command+'\n');subprocess.run(['bash','-n',str(p)],check=True);return p
def submit(name,p,args):
 rec=R/'receipts'/(name+'.json');cmd=['sbatch','--parsable','--account=torch_pr_482_general','--job-name='+name,'--chdir='+str(R),*args,str(p)]
 with rec.open('x') as f:json.dump({'status':'intent; reconcile if uncertain','command':cmd},f)
 z=subprocess.run(cmd,text=True,capture_output=True)
 if z.returncode:rec.write_text(json.dumps({'status':'rejected','command':cmd,'stdout':z.stdout,'stderr':z.stderr}));raise RuntimeError(z.stderr)
 job=z.stdout.strip().split(';')[0];assert job.isdigit();rec.write_text(json.dumps({'status':'confirmed','job_id':job,'command':cmd},indent=2));print('SUBMITTED',name,job,flush=True);return job

def cpu48_queue():
 p=subprocess.run(['squeue','--array','-u','xh2906','-h','-o','%i|%q'],text=True,capture_output=True,check=True)
 return [line.split('|') for line in p.stdout.splitlines()]

def available_cpu48_submissions(queue=None):
 if queue is None:queue=cpu48_queue()
 active=sum(row[1]=='cpu48' for row in queue)
 return max(0,3000-active-2)  # Reserve successor-controller capacity while this controller is active.

def active_source_units(state,queue):
 waves={str(w['job_id']):json.loads(Path(w['unit_manifest']).read_text())['unit_ids'] for w in state['waves']}
 active={}
 for jid,qos in queue:
  m=re.fullmatch(r'(\d+)_(\d+)',jid)
  if m and m[1] in waves:
   unit=waves[m[1]][int(m[2])];assert unit not in active,('Duplicate active source unit',unit)
   active[unit]=m[1]
 return active

def init():
 assert not (R/'submitted_jobs.json').exists(),'Already initialized; reconcile receipts before retry'
 ref=json.loads(Path(plan['reference_run'],'submitted_jobs.json').read_text())
 for name,digest in ref['code_sha256'].items():assert sha(R/'code'/name)==digest,(name,'trainer changed')
 shutil.copy2(Path(plan['reference_run'])/'source_trainer.py',R/'source_trainer.py')
 D=R/'data_split_whitelist17';D.mkdir();shutil.copy2(Path(plan['reference_split'])/'feature_norm.json',D/'feature_norm.json')
 subprocess.run([PY,str(R/'code/test_original_loss.py')],check=True);subprocess.run([PY,str(R/'code/test_whitelist.py'),str(D/'feature_norm.json')],check=True)
 cases=json.loads(Path('/scratch/xh2906/training_source_equilibria_20260913/cases.json').read_text());reuse={};bg={};pairs=set(map(tuple,plan['allowed_tf_sf_pairs']))
 for c in cases:
  if (c['tf'],c['sf']) not in pairs or c['source_type']!='retained128_unsmoothed_linear':continue
  for lg in range(4):
   gamma=(c['angle']+45*lg)%180;gamma=min(gamma,180-gamma);key=(gamma,c['contrast'],c['tf'],c['sf']);reuse.setdefault(key,c['block_paths'][lg*40:(lg+1)*40])
  bg.setdefault((c['tf'],c['sf']),c['block_paths'][160:200])
 units=[]
 for tf,sf in plan['allowed_tf_sf_pairs']:
  for c in plan['contrasts']:
   for gamma in plan['foreground_gammas']:
    units.append({'id':len(units),'angle':gamma,'gamma':gamma,'contrast':c,'tf':tf,'sf':sf,'background':False,'representative_LGNctgr':next(lg+1 for a in plan['orientations'] for lg in range(4) if min((a+lg*45)%180,180-(a+lg*45)%180)==gamma),'reuse_blocks':reuse.get((gamma,c,tf,sf),['']*40)})
  units.append({'id':len(units),'angle':0,'gamma':0,'contrast':14,'tf':tf,'sf':sf,'background':True,'reuse_blocks':bg.get((tf,sf),['']*40)})
 assert len(units)==plan['source_units'];(R/'source_units.json').write_text(json.dumps(units))
 state={'run_root':str(R),'reference_run':plan['reference_run'],'split':str(D),'initialization':'random; no checkpoint','max_epochs':8000,'allowed_tf_sf_pairs':plan['allowed_tf_sf_pairs'],'native_grid_points':615,'source_units':len(units),'supervised_loss':ref['supervised_loss'],'code_sha256':ref['code_sha256'],'status':'submitting','target_jobs':{},'commands':{},'waves':[],'attempts':{},'source_complete':0}
 save(state)
 dataset=script('all615_dataset',f"{PY} {R}/code/build_all615.py")
 state['dataset_job']=submit('all615_dataset',dataset,['--hold','--cpus-per-task=8','--mem=64G','--time=12:00:00','--partition=cs','--output='+str(R/'logs/dataset.%j.out'),'--error='+str(R/'logs/dataset.%j.err')]);save(state)
 for t in 'SCI':
  args=shlex.split(ref['commands'][t]);args[1]=str(R/'code/train_lowtfsf_grad01.py')
  for key,value in [('--split_dir',str(D)),('--out_dir',str(R/'train'/t))]:args[args.index(key)+1]=value
  assert not any(k in args for k in ['--init_checkpoint','--resume','--init_from_5d_dir']);state['commands'][t]=shlex.join(args)
  gate=f"{PY} -c "+shlex.quote("import json;assert json.load(open("+repr(str(D/'dataset_validation.json'))+"))['verified'];assert json.load(open("+repr(str(R/'loss_validation.json'))+"))['passed']")
  p=script('all615_'+t,gate+'\n'+state['commands'][t]);state['target_jobs'][t]=submit('all615_'+t,p,['--dependency=afterok:'+state['dataset_job'],'--cpus-per-task=8','--mem=64G','--time=24:00:00','--gres=gpu:1','--output='+str(R/'logs/%x.%j.out'),'--error='+str(R/'logs/%x.%j.err')]);save(state)
 p=script('all615_grid',f"{PY} {R}/code/preflight.py",True)
 state['grid_job']=submit('all615_grid',p,['--cpus-per-task=1','--mem=16G','--time=01:00:00','--partition=cpu_short,cs','--output='+str(R/'logs/grid.%j.out'),'--error='+str(R/'logs/grid.%j.err')]);save(state)
 p=script('all615_pilot',f"{PY} {R}/code/run_source_unit.py 0 --pilot",True)
 state['pilot_job']=submit('all615_pilot',p,['--dependency=afterok:'+state['grid_job'],'--cpus-per-task=8','--mem=24G','--time=48:00:00','--partition=cs','--output='+str(R/'logs/pilot.%j.out'),'--error='+str(R/'logs/pilot.%j.err')]);save(state)
 p=script('all615_advance',f"{PY} {R}/code/pipeline.py advance")
 state['controller_job']=submit('all615_controller_0',p,['--dependency=afterany:'+state['pilot_job'],'--cpus-per-task=1','--mem=8G','--time=02:00:00','--partition=cpu_short,cs','--output='+str(R/'logs/controller.%j.out'),'--error='+str(R/'logs/controller.%j.err')]);state['status']='submitted; grid and pilot gate simulation waves, held dataset gates scratch';save(state)

def advance():
 state=json.loads((R/'submitted_jobs.json').read_text());units=json.loads((R/'source_units.json').read_text())
 queue=cpu48_queue();active=active_source_units(state,queue)
 assert json.loads((R/'grid_validation.json').read_text())['verified']
 if not state['waves']:
  with h5py.File(R/'source_bundles/unit_0000.h5') as f:assert f.attrs['status']=='pilot_verified' and 'L6_01' in f and 'L6_40' in f
 complete=[]
 for u in units:
  p=R/'source_bundles'/f"unit_{u['id']:04d}.h5"
  if p.exists():
   try:
    with h5py.File(p) as f:
     if f.attrs.get('complete',False):
      assert f.attrs['status']=='complete' and all(f'L6_{n:02d}' in f for n in range(1,41));complete.append(u['id'])
   except OSError:
    if u['id'] not in active:raise
 state['source_complete']=len(complete);missing=[u['id'] for u in units if u['id'] not in set(complete)]
 if not missing:
  subprocess.run(['scontrol','release',state['dataset_job']],check=True);state['status']='all source units verified; dataset released, three trainings depend afterok';save(state);print(state['status'],flush=True);return
 uncovered=[i for i in missing if i not in active]
 if any(state['attempts'].get(str(i),0)>=3 for i in uncovered):state['status']='blocked: source unit still unfinished after three resumable attempts';save(state);raise RuntimeError(state['status'])
 available=available_cpu48_submissions(queue);chosen=uncovered[:available]
 wave=len(state['waves'])+1
 if chosen:
  wp=R/'submissions'/f'wave_{wave:03d}.json';wp.write_text(json.dumps({'unit_ids':chosen}))
  p=script(f'all615_wave_{wave:03d}',f"{PY} {R}/code/run_source_unit.py --wave {wp} \"$SLURM_ARRAY_TASK_ID\"",True)
  job=submit(f'all615_wave_{wave:03d}',p,['--array=0-'+str(len(chosen)-1)+'%'+str(plan['array_throttle']),'--cpus-per-task=8','--mem=24G','--time=48:00:00','--partition=cs','--output=/dev/null','--error=/dev/null'])
  state['waves'].append({'job_id':job,'unit_manifest':str(wp),'units':len(chosen)})
  for i in chosen:state['attempts'][str(i)]=state['attempts'].get(str(i),0)+1
  save(state)
 p=script('all615_advance',f"{PY} {R}/code/pipeline.py advance")
 state['controller_job']=submit(f'all615_controller_{wave}_{time.time_ns()}',p,['--begin=now+30minutes','--cpus-per-task=1','--mem=8G','--time=02:00:00','--partition=cpu_short,cs','--output='+str(R/'logs/controller.%j.out'),'--error='+str(R/'logs/controller.%j.err')]);state['status']='saved-data controller checks every 30 minutes; active units reconciled; dataset held; training dependent';save(state)

with (R/'pipeline.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if sys.argv[1]=='init':init()
 elif sys.argv[1]=='advance':advance()
 else:raise ValueError(sys.argv[1])
