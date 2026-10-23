"""
Memory Policy - the single VRAM tier table used across the codebase.

MemoryPolicy is model-agnostic: it only reasons about a VRAM budget in GB and
returns strategy decisions. Model-specific application of those decisions
(e.g. calling `pipe.enable_sequential_cpu_offload()`) lives next to the model
code (see checkpoint_loader/sdxl/memory_strategy.py::apply_to_pipeline).
"""

import logging
from typing import Literal

logger = logging.getLogger(__name__)

OffloadStrategy = Literal["sequential", "model", "none"]
AttentionSlicing = Literal["max", "auto", "none"]


class MemoryPolicy:
    """
    Determines memory optimization decisions based on available VRAM.

    VRAM tiers:
    - < 8GB:  aggressive  (sequential offload, max attention slicing, VAE slicing+tiling)
    - 8-12GB: balanced    (model offload, auto attention slicing, VAE slicing+tiling)
    - 12-16GB: light      (no offload, VAE slicing only)
    - >= 16GB: minimal    (GPU-resident, no VAE slicing/tiling)
    """

    def __init__(self, vram_gb: float):
        self.vram_gb = vram_gb

    def get_offload_strategy(self) -> OffloadStrategy:
        if self.vram_gb < 8:
            return "sequential"
        elif self.vram_gb < 12:
            return "model"
        return "none"

    def get_attention_slicing(self) -> AttentionSlicing:
        if self.vram_gb < 8:
            return "max"
        elif self.vram_gb < 12:
            return "auto"
        return "none"

    def should_enable_vae_slicing(self) -> bool:
        return self.vram_gb < 16

    def should_enable_vae_tiling(self) -> bool:
        return self.vram_gb < 12

    def should_enable_xformers(self) -> bool:
        return True

    def should_enable_tf32(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"MemoryPolicy(vram_gb={self.vram_gb:.2f})"
