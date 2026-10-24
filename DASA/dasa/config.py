"""DASA settings, retaining the original framework and train/inference defaults."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DASAConfig:
    similarity_threshold: float = 0.92
    search_size: int = 3
    dilation_size: int = 3
    shallow_threshold: float = 0.8
    deep_threshold: float = 0.2
    max_token_age: int = 4
    motion_loss_scale: float = 3.0

    @classmethod
    def from_config(cls, config: Any = None, *, framework: str, training: bool = False):
        if framework not in ("self_forcing", "longlive"):
            raise ValueError(f"Unknown framework: {framework}")
        if config is None:
            raw = {}
        elif hasattr(config, "get"):
            raw = config.get("dasa", {}) or {}
        else:
            raw = getattr(config, "dasa", {}) or {}
        selection = raw.get("token_selection", {}) or {}
        compression = raw.get("kv_compression", {}) or {}
        motion = raw.get("motion_loss", {}) or {}
        is_longlive = framework == "longlive"
        result = cls(
            similarity_threshold=float(selection.get("similarity_threshold", 0.8 if is_longlive else (0.9 if training else 0.92))),
            search_size=int(selection.get("search_size", 3)),
            dilation_size=int(selection.get("dilation_size", 1 if training else 3)),
            shallow_threshold=float(compression.get("shallow_threshold", 0.8)),
            deep_threshold=float(compression.get("deep_threshold", 0.2)),
            max_token_age=int(compression.get("max_token_age", 6 if is_longlive else 4)),
            motion_loss_scale=float(motion.get("scale", 2.0 if is_longlive else 3.0)),
        )
        result.validate()
        return result

    def validate(self):
        for name in ("similarity_threshold", "shallow_threshold", "deep_threshold"):
            if not -1.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between -1 and 1")
        for name in ("search_size", "dilation_size"):
            value = getattr(self, name)
            if value < 1 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer")
        if self.max_token_age < 1 or self.motion_loss_scale < 0:
            raise ValueError("max_token_age must be positive and motion loss scale nonnegative")

    def apply_to_generator(self, generator):
        """Configure policies without changing the framework's cache storage or RoPE."""
        generator.dasa_config = self
        for block in generator.model.blocks:
            attn = block.self_attn
            attn.max_similarity_ratio = self.shallow_threshold
            attn.min_similarity_ratio = self.deep_threshold
            attn.max_token_age = self.max_token_age
