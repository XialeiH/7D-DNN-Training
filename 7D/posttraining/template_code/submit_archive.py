from pathlib import Path
import subprocess,json,shlex
P=Path(__file__).resolve().parent.parent;receipt=P/'archive_submission.json'
if receipt.exists():print(receipt.read_text());raise SystemExit(0)
PY='/scratch/xh2906/conda/envs/py310/bin/python';code=P/'code'
dense=json.loads((P/'dense_tuning/submission.json').read_text());rep=json.loads((P/'representative192/submission.json').read_text())
script=code/'build_archive.sh';script.write_text('#!/bin/bash\nset -euo pipefail\nmodule purge\nmodule load anaconda3/2025.06 matlab/2025b\nsource /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh\nconda activate py310\nexport TMPDIR=/tmp/xh2906_archvariant_${SLURM_JOB_ID}\nexport MATLAB_PREFDIR=$TMPDIR/prefs\nmkdir -p \"$TMPDIR\" \"$MATLAB_PREFDIR\"\n'+shlex.join([PY,str(code/'prepare_archive.py')])+'\n'+shlex.join([PY,str(code/'run_archive_exports.py')])+'\n')
args=['sbatch','--parsable','--account=torch_pr_482_general','--partition=cpu_short,cs','--cpus-per-task=2','--mem=16G','--time=01:00:00','--job-name=aug20_archive','--output='+str(P/'logs/archive.%j.out'),'--error='+str(P/'logs/archive.%j.err'),str(script)]
ids=[dense['plot_job'],rep['job_id']]
accounting=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(ids),'-o','JobID,State'],text=True)
active={'PENDING','RUNNING','CONFIGURING','COMPLETING','REQUEUED'};deps=[]
for job in ids:
 states=[line.split('|')[1].split()[0] for line in accounting.splitlines() if line.split('|')[0]==job or line.split('|')[0].startswith(job+'_')]
 if states and all(x=='COMPLETED' for x in states):continue
 if states and all(x=='COMPLETED' or x in active for x in states):deps.append(job)
 else:raise RuntimeError('Archive prerequisite unavailable/failed: '+job+' '+str(states))
if deps:args.insert(-1,'--dependency=afterok:'+':'.join(deps))
else:
 assert (P/'dense_tuning/dense_tuning_complete.txt').is_file()
 assert all((P/'representative192/data'/('case_%03d.mat'%i)).is_file() for i in rep['case_ids'])

intent=P/'archive_submission_intent.json'
with intent.open('x') as f:json.dump({'state':'submitting','command':args},f)
j=subprocess.check_output(args,text=True).strip().split(';')[0];state={'job_id':j,'output':str(P/'local_archive'),'command':args};receipt.write_text(json.dumps(state,indent=2));intent.write_text(json.dumps(dict(state,state='confirmed'),indent=2));print(receipt.read_text())
