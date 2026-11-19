"""Numerical checks of Sections 5.1–5.3 and regeneration of Table 1."""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from objectives import diagnostics, jpi


def upper_cvar(scores, base, delta):
    order = np.argsort(scores)[::-1]
    weights = base[order]
    before = np.cumsum(weights) - weights
    mass = np.minimum(weights, np.maximum(delta - before, 0))
    return float(mass @ scores[order] / delta)


def bias_experiment(rng):
    law = np.array([.2, .5, .3])
    ratios = np.array([.5, 1.2, 1.])
    scores = np.array([-1., .2, 1.4])
    beta = .8
    x, y = ratios * np.exp(beta * scores), ratios
    mx, my = law @ x, law @ y
    target = np.log(mx / my) / beta
    coefficient = ((law @ (y-my)**2)/my**2 - (law @ (x-mx)**2)/mx**2)/(2*beta)
    results = []
    for n in (16, 64, 256, 1024):
        counts = rng.multinomial(n, law, size=200000)
        estimates = np.log((counts @ x)/(counts @ y))/beta
        results.append(dict(n=n, target=target, measured_bias=float(estimates.mean()-target),
                            predicted_bias=float(coefficient/n),
                            mc_se=float(estimates.std()/np.sqrt(len(estimates)))))
    return results


def run(root='results'):
    root = Path(root)
    (root / 'tables').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(41)
    x = torch.tensor(rng.normal(size=(64, 5)))
    theta = torch.tensor([.05, -.02, .03, .01, .05], requires_grad=True, dtype=x.dtype)
    constant = torch.full((64,), 3., dtype=x.dtype)
    logits = x @ theta
    rows = []
    for normalized, ref_obj, ref_norm in [(False, 3.0115, .1571), (True, 3., 5.8e-17)]:
        val = jpi(logits, constant, normalized=normalized)
        gradient = torch.autograd.grad(val, theta, retain_graph=True)[0]
        rows.append(dict(method='SN-JPI' if normalized else 'Raw Basic JPI',
                         objective=val.item(), gradient_norm=gradient.norm().item(),
                         paper_objective=ref_obj, paper_gradient_norm=ref_norm))
    assert abs(rows[1]['objective'] - 3) < 1e-13 and rows[1]['gradient_norm'] < 1e-13
    assert rows[0]['gradient_norm'] > .01
    scores = torch.tensor(rng.normal(size=64))
    expected = x.T @ (scores-scores.mean()) / len(scores)
    zero = torch.zeros_like(theta).detach().requires_grad_()
    errors = {}
    for beta in (.1, .01, .001, .0001, .00001):
        val = jpi(x @ zero, scores, beta)
        grad = torch.autograd.grad(val, zero)[0]
        errors[beta] = (grad-expected).norm().item()
    assert all(a > b for a, b in zip(errors.values(), list(errors.values())[1:]))
    rows[0].update(error_small_beta='', paper_error_small_beta='')
    rows[1].update(error_small_beta=errors[.00001], paper_error_small_beta=9.7e-6)
    tilted, base = (logits + .8*scores).softmax(0), logits.softmax(0)
    gradient = torch.autograd.grad(jpi(logits, scores), theta)[0]
    torch.testing.assert_close(gradient, x.T @ (tilted-base)/.8)
    assert torch.isfinite(jpi(logits.detach() * 100000, scores))
    for _ in range(100):
        a = torch.tensor(rng.normal(size=16))
        ell = torch.tensor(rng.normal(size=16))
        v = ell.softmax(0).numpy()
        mean = float(v @ a.numpy())
        std = np.sqrt(v @ (a.numpy()-mean)**2)
        objective = jpi(ell, a).item()
        d = diagnostics(ell, a, .8)
        assert d['delta_inf'] <= d['delta2'] + 1e-12
        assert 0 <= objective-mean <= std*np.sqrt(1/d['delta2']-1) + 1e-12
        for delta in (.1, .5, 1., d['delta_inf']):
            tail = upper_cvar(a.numpy(), v, delta)
            assert 0 <= tail-mean + 1e-12 <= std*np.sqrt(1/delta-1) + 2e-12
            assert tail <= objective + np.log(1/delta)/.8 + 1e-12
            if delta <= d['delta_inf']:
                assert objective <= tail + 1e-12
        slope = (jpi(ell, a, .0001).item()-mean)/.0001
        assert abs(slope - .5*std**2) < .001
    bias = bias_experiment(rng)
    with (root / 'tables/table1.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / 'theory.json').write_text(json.dumps(dict(seed=41, n=64, beta=.8,
        dtype='float64', theta=theta.tolist(), small_beta_gradient_errors=errors,
        bias=bias, envelopes_checked=100), indent=2))
    print('Table 1 and theory checks complete')


if __name__ == '__main__':
    run()
