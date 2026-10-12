import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run-root');p.add_argument('--post-root');p.add_argument('--atlas-root');p.add_argument('--runtime');p.add_argument('--profile');a=p.parse_args()
assert Path('/scratch/xh2906/librarySCI_runs/h96_tfsf7d_tf32_tailpair_selected_1000ep_20260911_104654/posttraining_savedbest_20260912/posttraining_7d_tf32_globalall7/dense_tuning_smoke/parity_verified.json').is_file(), 'Dense parity validation required'
post=Path(a.post_root);state=post/'tuning_curve_submission.json'
if not state.exists():subprocess.run([sys.executable,'/scratch/xh2906/librarySCI_runs/h96_tfsf7d_h1024x4_scratch_fixedbaseline_goodFRgt20_whitelist17_all615DS10_20261004_original_loss_relu/posttraining_7d_tf32_globalall7/code/submit_runtime_tuning_curves_quota.py']+sys.argv[1:],check=True)
subprocess.run([sys.executable,str(Path(__file__).with_name('submit_dense.py')),'--post-root',a.post_root,'--runtime',a.runtime,*[]],check=True)
s=json.loads(state.read_text());d=json.loads((post/'dense_tuning/submission.json').read_text())
if (post/'dense_tuning/dense_tuning_complete.txt').is_file():
 s['dense_tuning']=d;state.write_text(json.dumps(s,indent=2)+'\n');sys.exit(0)
if 'dense_tuning' not in s:
 old=s['verification_job']
 j=subprocess.check_output(['sbatch','--parsable','--account=torch_pr_482_general','--partition=cpu_short,cs','--cpus-per-task=1','--mem=1G','--time=00:05:00','--dependency=afterok:'+d['plot_job'],'--job-name=densecurvecheck','--output='+str(post/'densecurvecheck.%j.out'),'--error='+str(post/'densecurvecheck.%j.err'),'--wrap','test -s '+str(post/'dense_tuning/dense_tuning_complete.txt')],text=True).strip().split(';')[0]
 s['original_verification_job']=old;s['verification_job']=j;s['dense_tuning']=d
 state.write_text(json.dumps(s,indent=2)+'\n')
