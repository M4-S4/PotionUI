from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MOSS_IMAGE = "moss_image"
AXES = 4
TEXT_BLOCKS = 2


@dataclass(frozen=True)
class MossImageConfig:
    dim: int
    heads: int
    head_dim: int
    ffn_dim: int
    depth: int
    text_dim: int
    in_channels: int
    freq_dim: int
    rope_base: float

    @classmethod
    def from_detect_config(cls, config: dict[str, Any]) -> "MossImageConfig":
        dim = int(config["hidden_size"])
        heads = int(config["num_heads"])
        head_dim = int(config.get("head_dim", dim // heads))
        freq_dim = int(config["freq_dim"])
        if heads * head_dim != dim or head_dim % (2 * AXES) or freq_dim % 2:
            raise ValueError(
                f"moss_image needs dim == heads * head_dim, head_dim divisible by {2 * AXES} "
                f"and an even freq_dim; got dim={dim} heads={heads} head_dim={head_dim} freq_dim={freq_dim}"
            )
        return cls(
            dim=dim,
            heads=heads,
            head_dim=head_dim,
            ffn_dim=int(config["ffn_dim"]),
            depth=int(config["depth"]),
            text_dim=int(config["context_in_dim"]),
            in_channels=int(config["in_channels"]),
            freq_dim=freq_dim,
            rope_base=float(config.get("rope_base", 10000.0)),
        )
