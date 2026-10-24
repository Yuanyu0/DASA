"""Local latent matching and dynamic token selection shared by both frameworks."""

import torch
import torch.nn.functional as F


def local_nxn_match(reference, target, n=3, padding_mode="replicate", eps=1e-6):
    """Return locally aligned references [B, T, C, H, W] and cosine similarities [B, T, H, W]."""
    if reference.shape != target.shape:
        raise ValueError("Reference and target must have identical shapes")
    if n < 1 or n % 2 == 0:
        raise ValueError("The matching window must be a positive odd integer")
    B, T, C, H, W = target.shape
    tgt_flat = target.flatten(0, 1)
    ref_flat = reference.flatten(0, 1)
    tgt_vec = tgt_flat.view(B * T, C, H * W)
    ref_pad = F.pad(ref_flat, (n // 2,) * 4, mode=padding_mode)
    ref_neigh = F.unfold(ref_pad, kernel_size=n, stride=1)
    ref_neigh = ref_neigh.view(B * T, C, n * n, H * W)
    tgt_norm = F.normalize(tgt_vec, dim=1, eps=eps)
    ref_norm = F.normalize(ref_neigh, dim=1, eps=eps)
    sim_scores = (tgt_norm.unsqueeze(2) * ref_norm).sum(dim=1)
    best_idx = sim_scores.argmax(dim=1)
    gather_idx = best_idx.view(B * T, 1, 1, H * W).expand(B * T, C, 1, H * W)
    out = ref_neigh.gather(dim=2, index=gather_idx).squeeze(2).view(B, T, C, H, W)
    return out, F.cosine_similarity(out, target, dim=2)


def pool_token_similarities(cosine_sims):
    """Conservatively map 2x2 latent patches to Wan's spatial tokens."""
    return -F.max_pool2d(-cosine_sims, kernel_size=2, stride=2)


def dilate_token_mask(mask, dilation_size=1):
    if dilation_size < 1 or dilation_size % 2 == 0:
        raise ValueError("The dilation size must be a positive odd integer")
    if dilation_size == 1:
        return mask
    return F.max_pool2d(mask.float(), kernel_size=dilation_size, stride=1,
                        padding=dilation_size // 2).bool()


def select_dynamic_tokens(cosine_sims, threshold, dilation_size=1):
    """Return [B, T, H/2, W/2]; batching/indexing remains framework-owned."""
    return dilate_token_mask(pool_token_similarities(cosine_sims) < threshold, dilation_size)
