"""Fine-tune unchanged parent fitting terms with balanced four-axis curvature."""
import json
from pathlib import Path
import torch
import train_lowtfsf_grad01 as low
from smoothness import SETTINGS, curvature_terms, balanced_curvature

base = low.base
CALIBRATION = None
original_parse = base.parse_args


def parse_args():
    cfg = original_parse()
    cfg.curvature_mode = 'axes4_periodic_angle'
    return cfg


def training_curvature(model, xb, cfg, feature_norm):
    terms = curvature_terms(model, xb, feature_norm, cfg.target)
    penalty = balanced_curvature(terms, CALIBRATION['targets'][cfg.target])
    call = int(getattr(cfg, 'smoothness_probe_calls', 0)) + 1
    cfg.smoothness_probe_calls = call
    if (call - 1) % 40 == 0:
        record = {'phase': 'train', 'probe_call': call, 'lambda': cfg.curvature_lambda,
                  'balanced_curvature': float(penalty.detach()),
                  'axis_curvature': {k: float(v.detach()) for k, v in terms.items()}}
        with (Path(cfg.out_dir) / 'smoothness_diagnostics.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
    return penalty


def validation_curvature(model, cfg, feature_norm, device):
    with torch.no_grad():
        terms = curvature_terms(model, torch.zeros((1, 7), device=device), feature_norm, cfg.target, validation=True)
        penalty = balanced_curvature(terms, CALIBRATION['targets'][cfg.target])
    record = {'phase': 'fixed_validation_probe_bank', 'lambda': cfg.curvature_lambda,
              'balanced_curvature': float(penalty),
              'axis_curvature': {k: float(v) for k, v in terms.items()}}
    with (Path(cfg.out_dir) / 'smoothness_diagnostics.jsonl').open('a') as f:
        f.write(json.dumps(record) + '\n')
    return float(penalty)


def install(calibration):
    global CALIBRATION
    CALIBRATION = calibration
    base.parse_args = parse_args
    base.compute_loss_terms = low.compute_loss_terms
    base.central_curvature_loss = training_curvature
    base.deterministic_global_curvature_penalty = validation_curvature


if __name__ == '__main__':
    import sys
    argv = sys.argv[1:]
    path = Path(argv[argv.index('--global_curvature_balance_path') + 1])
    install(json.loads(path.read_text()))
    out = Path(argv[argv.index('--out_dir') + 1]); out.mkdir(parents=True, exist_ok=True)
    (out / 'smoothness_config.json').write_text(json.dumps(SETTINGS, indent=2))
    print('[smoothness] Balanced finite-difference curvature on periodic orientation,contrast,TF,SF; original fitting/unimodality/low-frequency gradient rule retained. Checkpoint score=fitting validation+lambda*fixed-bank curvature; plateau rule uses unchanged fitting validation.', flush=True)
    base.main()
