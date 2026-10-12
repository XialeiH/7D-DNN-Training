from pathlib import Path
import datetime,hashlib,json,math,subprocess,sys
import torch
from learning_health import assess
PY='/scratch/xh2906/conda/envs/py310/bin/python'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def validate(c,manifest,t,norm):
 assert c['model_config']['input_dim']==7 and list(c['model_config']['hidden_dims'])==[1024]*4
 assert list(c['model_config']['activation_seq'])==['relu']*4
 assert c['feature_norm']==norm
 cfg=c['train_config'];assert cfg['seed']==(manifest['seed'] if t=='C' else 42)
 assert list(cfg['lr_stage_values'])==manifest['lr_stage_values']
 for k,v in {'w_log':.5,'w_rel':.5,'raw_mse_lambda':0,'tf_unimodal_lambda':5,'sf_unimodal_lambda':5,'curvature_lambda':0,'tfsf_smooth_lambda':0}.items():assert cfg[k]==v,k
 assert 0<int(c['extra']['best_epoch'])<=8000, 'No trained selected checkpoint'
 assert math.isfinite(float(c['extra']['best_val_total_loss']))
 assert all(torch.isfinite(v).all() for v in c['model'].values() if torch.is_tensor(v))
def main():
 root=Path(sys.argv[1]);p=root/'posttraining_7d_tf32_globalall7';manifest=json.loads((root/'submitted_jobs.json').read_text());torch.set_num_threads(2)
 try:
  assert not (p/'launch.json').exists(), 'Existing launcher; reconcile before retry'
  ids=list(manifest['target_jobs'].values())
  query=subprocess.run(['sacct','-X','-n','-P','-j',','.join(ids),'--format=JobIDRaw,State,Elapsed,ExitCode'],capture_output=True,text=True,check=True)
  rows=[line.split('|') for line in query.stdout.splitlines() if line.split('|')[0] in ids]
  terminal={'COMPLETED','CANCELLED','FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
  assert len(rows)==len(ids) and all(row[1].split()[0].split('+')[0] in terminal for row in rows), 'Terminal accounting not confirmed'
  health=assess(root);assert health['normal_learning'], 'C did not pass normal-learning checks; monitor must investigate or try next authorized seed'
  norm_path=Path(manifest['split'])/'feature_norm.json';assert sha(norm_path)==manifest['normalization_sha256'];norm=json.loads(norm_path.read_text())
  checkpoints={}
  for t in 'SCI':
   src=root/'train'/t/'best.pt';c=torch.load(src,map_location='cpu',weights_only=False);validate(c,manifest,t,norm)
   if t in 'SI':assert sha(src)==manifest['fixed_checkpoints'][t]['sha256']
   if t=='C':
    assert c['extra']['best_val_total_loss']<health['initial_validation_loss']
    assert math.isclose(float(c['extra']['best_val_total_loss']),health['lowest_logged_validation_loss'],rel_tol=1e-6,abs_tol=1e-8), 'Checkpoint and log selection mismatch'
   checkpoints[t]=dict(path=str(src),sha256=sha(src),selected_epoch=c['extra']['best_epoch'],selected_validation_loss=c['extra']['best_val_total_loss'],seed=c['train_config']['seed'])
  passed=dict(passed=True,all_three_checkpoints_valid=True,checkpoints=checkpoints,learning_health=health,training_accounting=query.stdout,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
  (p/'checkpoint_gate_passed.json').write_text(json.dumps(passed,indent=2))
  subprocess.run([PY,str(p/'launch_retry_post.py'),str(root),'1024','4'],check=True)
 except Exception as e:
  (p/'checkpoint_gate_failure.json').write_text(json.dumps(dict(error=repr(e),run_root=str(root),checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),automatic_relaunch=False),indent=2));raise
if __name__=='__main__':main()

