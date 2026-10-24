"""Recompute attention without changing a streaming model's live KV state."""

import torch


_ATTENTION_STATE = (
    "last_chunk_clean_k",
    "last_chunk_clean_v",
    "last_reference_frame",
    "last_update_step",
    "current_frame_num",
)


def _snapshot(value):
    return value.detach().clone() if isinstance(value, torch.Tensor) else value


def create_cache_checkpoint_forward(module):
    """Recompute with a private KV snapshot; leave live streaming state untouched."""
    first_call = True
    saved_cache = None
    saved_attention = None
    attention = module.self_attn if hasattr(module, "self_attn") else module

    def forward(*inputs, **kwargs):
        nonlocal first_call, saved_cache, saved_attention
        cache = kwargs.get("kv_cache")
        if first_call:
            first_call = False
            with torch.no_grad():
                if cache is not None:
                    length = int(cache["local_end_index"].item())
                    saved_cache = {
                        key: _snapshot(
                            value[:, :length] if key in ("k", "v") else value
                        )
                        for key, value in cache.items()
                    }
                saved_attention = {
                    name: _snapshot(getattr(attention, name))
                    for name in _ATTENTION_STATE
                    if hasattr(attention, name)
                }
            return module(*inputs, **kwargs)

        if cache is not None:
            with torch.no_grad():
                private_cache = {
                    key: _snapshot(value)
                    for key, value in saved_cache.items()
                    if key not in ("k", "v")
                }
                for key in ("k", "v"):
                    private_cache[key] = torch.zeros_like(cache[key])
                    private_cache[key][:, : saved_cache[key].shape[1]].copy_(
                        saved_cache[key]
                    )
            kwargs["kv_cache"] = private_cache

        live_attention = {name: getattr(attention, name) for name in saved_attention}
        try:
            for name, value in saved_attention.items():
                setattr(attention, name, _snapshot(value))
            return module(*inputs, **kwargs)
        finally:
            # Restore live attention state even when checkpoint recomputation exits early.
            for name, value in live_attention.items():
                setattr(attention, name, value)

    return forward
