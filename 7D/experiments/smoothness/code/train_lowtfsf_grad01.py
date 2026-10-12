"""Scratch trainer with exact per-output gradient scaling at low TF AND SF."""
import functools
import torch
import train_sixd_single_longrun as base


def install_low_frequency_gradient_hook(model, feature_norm):
    def hook(module, inputs, output):
        if not output.requires_grad:
            return
        x = inputs[0]
        masks = []
        for column, name in [(5, 'lgnTF'), (6, 'lgnSF')]:
            grid = x.new_tensor([base.normalized_feature_value(feature_norm, name, v) for v in (0.5, 1.0)])
            masks.append(torch.isclose(x[:, column, None], grid[None, :], rtol=0, atol=1e-6).any(dim=1))
        low = masks[0] & masks[1]
        scale = torch.where(low, x.new_tensor(0.1), x.new_tensor(1.0))
        scale = scale.reshape((-1,) + (1,) * (output.ndim - 1))
        output.register_hook(lambda gradient: gradient * scale)
    return model.register_forward_hook(hook)


original_compute_loss_terms = base.compute_loss_terms


@functools.wraps(original_compute_loss_terms)
def compute_loss_terms(model, xb, yb, meta, cfg, device, grad_teacher, feature_norm, **kwargs):
    handle = install_low_frequency_gradient_hook(model, feature_norm)
    try:
        return original_compute_loss_terms(model, xb, yb, meta, cfg, device, grad_teacher, feature_norm, **kwargs)
    finally:
        handle.remove()


if __name__ == '__main__':
    base.compute_loss_terms = compute_loss_terms
    print('[gradient_policy] TF in {0.5,1} AND SF in {0.5,1}: output-gradient multiplier 0.1; otherwise 1. No batch renormalization. Includes synthetic penalty points. Forward/validation unchanged.', flush=True)
    base.main()
