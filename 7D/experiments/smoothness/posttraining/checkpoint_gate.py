"""After all three fine-tunes end, validate their selected snapshots and launch once."""
from pathlib import Path
import datetime, hashlib, json, math, subprocess, sys
import torch

PY = '/scratch/xh2906/conda/envs/py310/bin/python'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()


def validate(c, manifest, target, norm, source, calibration):
    model, train, extra = c['model_config'], c['train_config'], c['extra']
    assert model['target'] == target and model['input_dim'] == 7
    assert list(model['hidden_dims']) == manifest['architecture'][1:-1]
    assert list(model['activation_seq']) == manifest['activation_seq']
    assert model['softplus_out'] and model['constant_baseline_rule'] == source['model_config']['constant_baseline_rule']
    assert c['feature_norm'] == norm == source['feature_norm']
    assert norm['feature_names']['order'] == ['OriCos2','Contrast','L6','L4E','L4I','lgnTF','lgnSF']
    assert [float(v) for v in train['lr_stage_values']] == manifest['schedule']['lr_stage_values']
    assert train['w_log'] == .5 and train['w_rel'] == .5 and train['raw_mse_lambda'] == 0
    assert train['tf_unimodal_lambda'] == 5 and train['sf_unimodal_lambda'] == 5
    assert train['curvature_mode'] == 'axes4_periodic_angle'
    coefficient = calibration['targets'][target]['lambda_by_strength'][manifest['strength']]
    assert train['curvature_lambda'] == coefficient == train['ckpt_global_curvature_beta']
    assert Path(train['init_checkpoint']).resolve() == (Path(manifest['run_root'])/'initial_checkpoints'/f'{target}.pt').resolve()
    assert all(torch.isfinite(v).all().item() for v in c['model'].values() if torch.is_tensor(v))
    epoch = int(extra['best_epoch'])
    assert 0 <= epoch <= 8000 and int(source['extra']['best_epoch']) == {'S':71,'C':47,'I':66}[target]
    fit, curvature, score = [float(extra[k]) for k in ['best_val_total_loss','best_global_curvature_penalty','best_ckpt_selection_score']]
    assert all(math.isfinite(v) and v >= 0 for v in [fit, curvature, score])
    assert math.isclose(score, fit + coefficient*curvature, rel_tol=1e-7, abs_tol=1e-9)
    if epoch == 0:
        assert c['model'].keys() == source['model'].keys()
        assert all(torch.equal(c['model'][k], source['model'][k]) for k in c['model'])
    return dict(selected_epoch=epoch, fitting_loss=fit, curvature=curvature, coefficient=coefficient,
                selection_score=score, source_epoch=int(source['extra']['best_epoch']),
                epoch_zero_meaning='trained source checkpoint; no fine-tuning improvement' if epoch == 0 else None)


def main():
    torch.set_num_threads(2)
    root = Path(sys.argv[1]); post = root/'posttraining_7d_tf32_globalall7'
    manifest = json.loads((root/'submitted_jobs.json').read_text())
    try:
        assert not (post/'launch.json').exists(), 'Existing launch receipt; reconcile before retry'
        ids = list(manifest['target_jobs'].values())
        result = subprocess.run(['sacct','-X','-n','-P','-j',','.join(ids),'-o','JobIDRaw,State,Elapsed,ExitCode'], capture_output=True, text=True, check=True)
        states = {f[0]:dict(state=f[1],elapsed=f[2],exit_code=f[3]) for f in (line.split('|') for line in result.stdout.splitlines()) if f[0] in ids}
        # The afterany dependency supplies termination evidence if accounting is delayed.
        assert not any(v['state'].split()[0].rstrip('+') in {'PENDING','RUNNING','CONFIGURING','COMPLETING','REQUEUED'} for v in states.values())
        norm_path = Path(manifest['split'])/'feature_norm.json'
        assert sha(norm_path) == manifest['normalization_sha256']
        norm = json.loads(norm_path.read_text())
        calibration = json.loads((root/'smoothness_calibration.json').read_text())
        initial = json.loads((root/'initial_checkpoint_provenance.json').read_text())
        checkpoints = {}
        for target in 'SCI':
            source_path = root/'initial_checkpoints'/f'{target}.pt'
            assert sha(source_path) == initial[target]['sha256']
            source = torch.load(source_path, map_location='cpu', weights_only=False)
            path = root/'train'/target/'best.pt'
            assert path.is_file() and path.stat().st_size > 0, f'Missing {target} best.pt'
            before = path.stat(); digest = sha(path)
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            record = validate(checkpoint, manifest, target, norm, source, calibration)
            assert path.stat().st_mtime_ns == before.st_mtime_ns and sha(path) == digest
            checkpoints[target] = dict(record, path=str(path),sha256=digest,bytes=before.st_size,mtime_ns=before.st_mtime_ns)
        receipt = dict(passed=True,checked_utc=now(),training_job_ids=manifest['target_jobs'],training_accounting=states,
                       accounting_stdout=result.stdout,all_three_checkpoints_valid=True,checkpoints=checkpoints)
        (post/'checkpoint_gate_passed.json').write_text(json.dumps(receipt,indent=2))
        subprocess.run([PY,str(post/'launch_smooth_post.py'),str(root),'512','5'],check=True)
    except Exception as exc:
        failure = dict(failed_utc=now(),error_type=type(exc).__name__,error=str(exc),run_root=str(root),
                       training_jobs=manifest['target_jobs'],launch_receipt_exists=(post/'launch.json').exists(),automatic_retry_authorized=False)
        (post/'checkpoint_gate_failure.json').write_text(json.dumps(failure,indent=2))
        state = json.loads((root/'posttraining_submission.json').read_text())
        state.update(status='checkpoint gate or setup failed; inspect before recovery',failure_receipt=str(post/'checkpoint_gate_failure.json'))
        (root/'posttraining_submission.json').write_text(json.dumps(state,indent=2))
        raise


if __name__ == '__main__':
    main()
