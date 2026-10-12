import argparse,csv,json,shlex,shutil,subprocess,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
PY='/scratch/xh2906/conda/envs/py310/bin/python'
def sb(cmd,name,root,dep=None,array=None):
 args=['sbatch','--parsable','--account=torch_pr_482_general','--partition=cpu_short,cs','--cpus-per-task=8','--mem=32G','--time=01:00:00',f'--job-name={name}',f'--output={root}/logs/%x.%A_%a.out',f'--error={root}/logs/%x.%A_%a.err']
 if dep:args.append('--dependency=afterok:'+dep)
 if array:args.append('--array='+array)
 setup='set -euo pipefail\nmodule purge\nmodule load matlab/2025b\nexport TMPDIR="/tmp/xh2906_aug20_dense/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"\nexport MATLAB_PREFDIR="$TMPDIR/prefs"\nmkdir -p "$MATLAB_PREFDIR"\n'
 return subprocess.check_output(args+['--wrap',setup+cmd],text=True).strip().split(';')[0]
def main():
 a=argparse.ArgumentParser();a.add_argument('--post-root',type=Path,required=True);a.add_argument('--runtime',type=Path,required=True);a.add_argument('--dependency');a.add_argument('--smoke',action='store_true');v=a.parse_args()
 root=v.post_root/('dense_tuning_smoke' if v.smoke else 'dense_tuning');receipt=root/'submission.json'
 if receipt.exists():print(receipt.read_text());return
 for x in ['code','logs','equilibrium']:(root/x).mkdir(parents=True,exist_ok=True)
 for x in ['dense_nn_driver.m','run_dense_task.m','plot_dense.py']:shutil.copy2(P/x,root/'code'/x)
 tf=sorted(set([.5,.75,1,1.5,2,3,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32]))
 sf=sorted(set([.5,.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.5,4,4.5,5,6,7,8,9,10]))
 cs=sorted(set(list(range(5,101,5))+[14,24,33,42,52,61,71,80,89,95]))
 rows=[dict(kind='tf',angle=0,contrast=100,tf=t,sf=2.5) for t in tf]+[dict(kind='sf',angle=0,contrast=100,tf=10,sf=s) for s in sf]+[dict(kind='contrast',angle=a,contrast=c,tf=10,sf=2.5) for a in [0,22.5] for c in cs]
 # Shared TF10/SF2.5 retained under both curve labels; each has an independently auditable task.
 if v.smoke:rows=[dict(kind='contrast',angle=0,contrast=95,tf=10,sf=2.5)]
 for k,r in enumerate(rows):r['task']=k
 with (root/'cases.tsv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['task','kind','angle','contrast','tf','sf'],delimiter='\t');w.writeheader();w.writerows(rows)
 hashes={str(f.relative_to(v.runtime)):hashlib.sha256(f.read_bytes()).hexdigest() for f in v.runtime.rglob('*.m')}
 (root/'runtime_provenance.json').write_text(json.dumps({'runtime':str(v.runtime),'sha256':hashes},indent=2))
 cmd=f"matlab -batch \"addpath('{root}/code'); run_dense_task('{root}','{v.runtime}')\""
 job=sb(cmd,'dense7d',root,v.dependency,f'0-{len(rows)-1}%8')
 plot=None if v.smoke else sb(f'{PY} {shlex.quote(str(root / "code/plot_dense.py"))} {shlex.quote(str(root))}','denseplot',root,job)
 receipt.write_text(json.dumps({'array_job':job,'plot_job':plot,'cases':len(rows),'tf':tf,'sf':sf,'contrasts':cs,'runtime':str(v.runtime),'hcnorm_valid':False},indent=2)+'\n');print(receipt.read_text())
if __name__=='__main__':main()
