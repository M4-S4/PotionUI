from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FLUX3 = "flux3"


@dataclass(frozen=True)
class Flux3Config:
    hidden: int
    heads: int
    head_dim: int
    depth: int
    depth_single: int
    in_channels: int
    context_in_dim: int
    mlp_hidden: int
    axes_dim: tuple[int, ...]
    theta: int
    flow_map: bool
    has_image_cond: bool
    fused_qkv: bool

    @classmethod
    def from_detect_config(cls, config: dict[str, Any]) -> "Flux3Config":
        hidden = int(config["hidden_size"])
        heads = int(config["num_heads"])
        head_dim = int(config.get("head_dim", hidden // heads))
        axes_dim = tuple(int(a) for a in config.get("axes_dim", [head_dim // 4] * 4))
        if sum(axes_dim) != head_dim:
            raise ValueError(f"flux3 axes_dim {axes_dim} must sum to head_dim {head_dim}")
        return cls(
            hidden=hidden,
            heads=heads,
            head_dim=head_dim,
            depth=int(config["depth"]),
            depth_single=int(config["depth_single_blocks"]),
            in_channels=int(config["in_channels"]),
            context_in_dim=int(config["context_in_dim"]),
            mlp_hidden=int(config["mlp_hidden"]),
            axes_dim=axes_dim,
            theta=int(config.get("theta", 10000)),
            flow_map=bool(config.get("flow_map", False)),
            has_image_cond=bool(config.get("has_image_cond", True)),
            fused_qkv=bool(config.get("fused_qkv", False)),
        )
