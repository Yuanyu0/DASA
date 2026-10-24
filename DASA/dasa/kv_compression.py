"""KV retention policy. Cache storage, positions and writes stay in each framework."""

import torch
import torch.nn.functional as F


def layer_similarity_threshold(shallow_threshold, deep_threshold, layer_index, num_layers):
    return shallow_threshold - (shallow_threshold - deep_threshold) / num_layers * (layer_index + 1)


def compute_retention_mask(features, *, tail_frames, threshold, last_update_step,
                           current_frame_num, max_token_age, training=False,
                           random_keep_probability=0.05):
    """Retain changed or aged tokens and the full tail; update ages in place."""
    previous = features[:, :-tail_frames, ...]
    following = features[:, 1:(-tail_frames + 1) or None, ...]
    mask = F.cosine_similarity(previous, following, dim=-1) < threshold
    if training:
        mask = mask | (torch.rand_like(mask.float()) < random_keep_probability)
    for frame_index in range(mask.shape[1]):
        aged = current_frame_num - last_update_step >= max_token_age
        frame_mask = mask[:, frame_index] | aged
        mask[:, frame_index] = frame_mask
        last_update_step[frame_mask] = current_frame_num
        current_frame_num += 1
    tail = torch.ones((features.shape[0], tail_frames, features.shape[2]),
                      device=features.device, dtype=torch.bool)
    return torch.cat([mask, tail], dim=1), last_update_step, current_frame_num
