import numpy as np
import torch

def sample_np(rng, x, log_weights, num_samples):
    """Categorical sampling using numpy."""
    weights = np.exp(log_weights - log_weights.max())
    weights = weights / weights.sum()
    return np.interp(
        np.linspace(0, 1, num_samples),
        np.cumsum(weights),
        x)

def sample(rng, x, log_weights, num_samples):
    """Categorical sampling using either numpy or torch."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(log_weights, torch.Tensor):
        log_weights = log_weights.detach().cpu().numpy()
    return sample_np(rng, x, log_weights, num_samples)