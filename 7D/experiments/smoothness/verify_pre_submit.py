"""Validate the nine final CLIs and frozen-source selection scores before sbatch."""
from pathlib import Path
import datetime, hashlib, json, shlex, subprocess, sys, tempfile
import torch

setup = Path('/home/xh2906/librarySCI_smooth_finetune_20261004')
sys.path.insert(0, str(setup/'code'))
import train_smooth as trainer
from common import ModelConfig, build_model

torch.set_num_threads(2)
prepared = json.loads((setup/'preparation_complete.json').read_text())
cal = json.loads((setup/'smoothness_calibration.json').read_text())
proof = json.loads((setup/'initial_checkpoint_provenance.json').read_text())
assert prepared['passed'] and json.loads((setup/'preflight_verification.json').read_text())['passed']
trainer.install(cal)
checks = {}
for level, state in prepared['runs'].items():
    root = Path(state['run_root'])
    current = json.loads((root/'submitted_jobs.json').read_text())
    assert current == state and not current['target_jobs']
    for name, digest in current['code_sha256'].items():
        assert hashlib.sha256((root/'code'/name).read_bytes()).hexdigest() == digest
    checks[level] = {}
    for target in 'SCI':
        argv = shlex.split(current['commands'][target])
        sys.argv = argv[1:]
        cfg = trainer.base.parse_args()
        weight = cal['targets'][target]['lambda_by_strength'][level]
        assert cfg.curvature_mode == 'axes4_periodic_angle'
        assert cfg.curvature_lambda == cfg.ckpt_global_curvature_beta == weight
        assert list(cfg.lr_stage_values) == [2e-4, 5e-5, 1e-5]
        assert tuple(cfg.hidden_dims) == (512,)*5 and tuple(cfg.activation_seq) == ('relu',)*5
        assert cfg.stage_epochs == 200 and cfg.max_stages == 40
        assert cfg.min_improve == .001 and cfg.min_best_rel_gain == .0005
        assert cfg.w_log == cfg.w_rel == .5 and cfg.tf_unimodal_lambda == cfg.sf_unimodal_lambda == 5
        assert cfg.curvature_grad_ratio == 0 and not cfg.subloss_norm_enable
        assert cfg.batch_size == 8192 and cfg.eval_batch_size == 65536
        assert cfg.max_train_batches == 40 and cfg.max_eval_batches == 10
        assert cfg.variance_weight_mode == 'soft_inv_sem' and cfg.use_amp
        assert cfg.weight_anchor_lambda == cfg.distill_anchor_lambda == cfg.mono_pair_lambda == cfg.mono_grad_lambda == cfg.tfsf_smooth_lambda == 0
        assert hashlib.sha256(Path(cfg.init_checkpoint).read_bytes()).hexdigest() == proof[target]['sha256']
        assert str(Path(cfg.split_dir).resolve()) == str(Path(current['split']).resolve())
        assert not list((root/'receipts').glob('smooth_*.json'))
        subprocess.run(['bash', '-n', current['submission_commands'][target][-1]], check=True)
        checks[level][target] = {'CLI_parsed': True, 'lambda': weight, 'source_epoch': proof[target]['selected_epoch'], 'config': vars(cfg)}

# Recheck the actual wrapper's deterministic validation result for each source model.
for target in 'SCI':
    q = torch.load(setup/'initial_checkpoints'/(target+'.pt'), map_location='cpu', weights_only=False)
    model = build_model(ModelConfig(**q['model_config']))
    model.load_state_dict(q['model'], strict=True)
    model.eval()
    sys.argv = shlex.split(prepared['runs']['low']['commands'][target])[1:]
    cfg = trainer.base.parse_args()
    with tempfile.TemporaryDirectory(dir=setup/'tmp') as out:
        cfg.out_dir = out
        r = trainer.validation_curvature(model, cfg, q['feature_norm'], torch.device('cpu'))
    assert abs(r - cal['targets'][target]['initial_balanced_curvature']) < 1e-6
    for level in checks:
        checks[level][target]['source_fixed_bank_curvature'] = r
        checks[level][target]['initial_combined_selection_score'] = cal['targets'][target]['source_validation_fitting_loss'] + checks[level][target]['lambda']*r

parent = Path(prepared['runs']['low']['parent_run'])
jobs = json.loads((parent/'submitted_jobs.json').read_text())['target_jobs']
accounting = subprocess.run(['sacct', '-n', '-X', '-P', '-j', ','.join(str(j) for j in jobs.values()), '--format=JobIDRaw,State,Elapsed,ExitCode'], text=True, capture_output=True, check=True)
assert all(x.split('|')[1].split()[0] in {'CANCELLED','COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'} for x in accounting.stdout.splitlines() if x.strip())
audit = {'passed': True, 'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'runs': checks, 'source_accounting': accounting.stdout}
(setup/'command_preflight_verification.json').write_text(json.dumps(audit, indent=2))
print(json.dumps({'passed': True, 'nine_CLIs_verified': True, 'source_accounting': accounting.stdout, 'lambdas': {l: {t:v['lambda'] for t,v in targets.items()} for l,targets in checks.items()}}))
