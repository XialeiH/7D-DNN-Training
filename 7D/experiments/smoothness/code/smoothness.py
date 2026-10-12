"""Four-axis, unlabeled finite-difference curvature probes."""
import math
import torch

FEATURES = ('OriCos2', 'Contrast', 'L6', 'L4E', 'L4I', 'lgnTF', 'lgnSF')
AXES = {'orientation': 0, 'contrast': 1, 'tf': 5, 'sf': 6}
SETTINGS = {
    'train_centers': 256, 'validation_centers': 1024,
    'validation_seed': 1045125, 'half_steps': [0.01, 0.025, 0.05],
    'minimum_half_step': 0.001, 'scale_floor': 1e-8,
    'orientation_bounds_degrees': [0.0, 180.0],
    'contrast_bounds': [0.1, 100.0], 'tf_bounds': [0.5, 32.0],
    'sf_bounds': [0.5, 10.0], 'batch_anchor_fraction': 0.5,
}


def coordinates(norm, like):
    bounds = [(0., 1.), SETTINGS['contrast_bounds'],
              (norm['L6']['domain_min'], norm['L6']['domain_max']),
              (norm['L4E']['domain_min'], norm['L4E']['domain_max']),
              (norm['L4I']['domain_min'], norm['L4I']['domain_max']),
              SETTINGS['tf_bounds'], SETTINGS['sf_bounds']]
    lo = like.new_tensor([x[0] for x in bounds])
    width = like.new_tensor([x[1] - x[0] for x in bounds])
    mu = like.new_tensor([0. if k == 'OriCos2' else norm[k]['mean'] for k in FEATURES])
    sd = like.new_tensor([1. if k == 'OriCos2' else norm[k]['std'] for k in FEATURES])
    return lo, width, mu, sd


def to_model(u, lo, width, mu, sd):
    physical = lo + u * width
    physical = physical.clone()
    # u0 is theta/180, whereas the unchanged NN input is cos(2*theta).
    physical[:, 0] = torch.cos(2 * math.pi * u[:, 0])
    return (physical - mu) / sd


def second_difference(minus, center, plus, half_step):
    return (minus - 2 * center + plus) / half_step.square()


def curvature_terms(model, xb, norm, target, *, validation=False):
    with torch.autocast(device_type=xb.device.type, enabled=False):
        dtype = torch.float64 if xb.dtype == torch.float64 else torch.float32
        xb = xb.to(dtype=dtype)
        lo, width, mu, sd = coordinates(norm, xb)
        n = SETTINGS['validation_centers'] if validation else SETTINGS['train_centers']
        generator = torch.Generator(device=xb.device).manual_seed(SETTINGS['validation_seed']) if validation else None
        centers = torch.rand((n, 7), device=xb.device, dtype=dtype, generator=generator)
        if not validation:
            physical = xb.detach() * sd + mu
            batch_u = (physical - lo) / width
            batch_u[:, 0] = torch.acos(physical[:, 0].clamp(-1., 1.)) / (2 * math.pi)
            valid = ((batch_u >= 0) & (batch_u <= 1)).all(dim=1)
            batch_u = batch_u[valid]
            if len(batch_u):
                ix = torch.randint(len(batch_u), (n // 2,), device=xb.device)
                centers[n // 2:] = batch_u[ix]
        half = xb.new_tensor(SETTINGS['half_steps'])[torch.arange(n, device=xb.device) % 3]
        output_scale = math.log1p({'S': 30., 'C': 100., 'I': 140.}[target])
        terms = {}
        for name, col in AXES.items():
            h = half if col == 0 else torch.minimum(half, torch.minimum(centers[:, col], 1 - centers[:, col]))
            keep = h >= SETTINGS['minimum_half_step']
            u, h = centers[keep], h[keep]
            minus, plus = u.clone(), u.clone()
            minus[:, col] -= h
            plus[:, col] += h
            if col == 0:
                minus[:, col] = minus[:, col].remainder(1.)
                plus[:, col] = plus[:, col].remainder(1.)
            x = to_model(torch.cat([minus, u, plus]), lo, width, mu, sd)
            rate = model(x).reshape(3, -1)
            g = torch.log1p(rate.clamp_min(0)) / output_scale
            terms[name] = second_difference(g[0], g[1], g[2], h).square().mean()
        return terms


def balanced_curvature(terms, target_calibration):
    return torch.stack([terms[k] / target_calibration['axis_scales'][k] for k in AXES]).mean()
