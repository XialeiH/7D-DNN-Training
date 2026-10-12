from pathlib import Path
import io,json,shlex,subprocess,tarfile
B=Path(__file__).resolve().parent;plan=json.loads((B/'plan.json').read_text());R=plan['run_root']
ssh=['ssh','-S','/Users/xialeihuang/.ssh/torch-codex-cm','-o','BatchMode=yes','-o','ControlMaster=no','-o','ProxyCommand=false','xh2906@login.torch.hpc.nyu.edu']
assert not (B/'submission_attempt.json').exists(),'Prior attempt exists: reconcile remote receipts before retry'
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as t:
 for f in [B/'plan.json',B/'stop_training_receipt.json',B/'simulator_changes.diff',*sorted((B/'code').glob('*.py')),*sorted((B/'code').glob('*.m'))]:t.add(f,arcname=str(f.relative_to(B)))
receiver='from pathlib import Path;import sys,tarfile;R=Path('+repr(R)+');R.mkdir(exist_ok=False);tarfile.open(fileobj=sys.stdin.buffer,mode="r|gz").extractall(R);[(R/n).mkdir() for n in ["receipts","logs","submissions","source_bundles","train"]];print("UPLOADED",flush=True)'
env='module load anaconda3/2025.06\nsource /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh\nconda activate py310\nexport PYTHONDONTWRITEBYTECODE=1\n'
cmd=env+'/scratch/xh2906/conda/envs/py310/bin/python -c '+shlex.quote(receiver)
z=subprocess.run(ssh+['bash -lc '+shlex.quote(cmd)],input=buf.getvalue(),capture_output=True,timeout=300)
(B/'upload_output.txt').write_bytes(z.stdout+z.stderr);assert z.returncode==0,z.stderr
(B/'submission_attempt.json').write_text(json.dumps({'state':'attempting; reconcile before retry','run_root':R}))
cmd=env+f'/scratch/xh2906/conda/envs/py310/bin/python {R}/code/test_source_contract.py\n'+f'/scratch/xh2906/conda/envs/py310/bin/python {R}/code/pipeline.py init'
z=subprocess.run(ssh+['bash -lc '+shlex.quote(cmd)],capture_output=True,text=True,timeout=600)
(B/'submission_output.txt').write_text(z.stdout+z.stderr);print(z.stdout,z.stderr,flush=True)
assert z.returncode==0,'Inspect remote receipts: do not retry blindly'
(B/'submission_attempt.json').write_text(json.dumps({'state':'confirmed','run_root':R}))
z=subprocess.run(ssh+['cat '+shlex.quote(R+'/submitted_jobs.json')],capture_output=True,text=True,timeout=60);assert z.returncode==0
(B/'confirmed_submission.json').write_text(z.stdout);print('CONFIRMED',json.dumps(json.loads(z.stdout)['target_jobs']),flush=True)
