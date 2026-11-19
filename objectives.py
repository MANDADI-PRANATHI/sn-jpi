import math
import torch


def jpi(log_ratio, scores, beta=.8, normalized=True):
    denominator = (torch.logsumexp(log_ratio, 0) if normalized
                   else math.log(log_ratio.numel()))
    return (torch.logsumexp(log_ratio + beta * scores, 0) - denominator) / beta


def diagnostics(log_ratio, scores, beta):
    tilted = log_ratio + beta * scores
    log_base = log_ratio.log_softmax(0)
    log_weights = tilted.log_softmax(0)
    return {
        'ess_return': (-torch.logsumexp(2 * (beta * scores).log_softmax(0), 0)).exp().item(),
        'ess_ratio': (-torch.logsumexp(2 * log_base, 0)).exp().item(),
        'ess_combined': (-torch.logsumexp(2 * log_weights, 0)).exp().item(),
        'delta2': (-torch.logsumexp(2 * log_weights - log_base, 0)).exp().item(),
        'delta_inf': (log_base - log_weights).min().exp().item(),
        'log_ratios': log_ratio.tolist(),
        'ratio_weights': log_base.exp().tolist(),
    }
