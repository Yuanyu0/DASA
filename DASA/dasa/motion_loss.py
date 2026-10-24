"""Detached temporal weights for distribution matching distillation."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_motion_weights(latents, scale):
    """Return broadcastable weights; identical to the original multi-frame formula."""
    channel_dim = 2 if latents.ndim == 5 else -1
    if latents.shape[1] < 2:
        shape = list(latents.shape)
        shape[channel_dim] = 1
        return latents.new_ones(shape)
    sim = F.cosine_similarity(latents[:, 1:], latents[:, :-1], dim=channel_dim)
    sim = torch.cat([sim[:, 0:1, ...], sim], dim=1)
    weights = 1.0 + scale * torch.relu(1.0 - sim)
    return weights.unsqueeze(channel_dim)
