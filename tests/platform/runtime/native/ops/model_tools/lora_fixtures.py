from __future__ import annotations

import json

import torch
from safetensors.torch import save_file

from src.platform.runtime.native.arch.moss_image.config import MossImageConfig
from src.platform.runtime.native.arch.moss_image.model import MossImage
from src.platform.runtime.native.kernels import pick_operations

DIM = 16
CFG = MossImageConfig(
    dim=DIM, heads=2, head_dim=8, ffn_dim=24, depth=2, text_dim=12,
    in_channels=4, freq_dim=8, rope_base=10000.0,
)


def build_moss(dtype: torch.dtype = torch.float32, seed: int = 0) -> MossImage:
    gen = torch.Generator().manual_seed(seed)
    module = MossImage(CFG, operations=pick_operations(torch.float32, torch.float32))
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn(param.shape, generator=gen) * 0.05)
    module.eval()
    return module.to(dtype)


def split_qkv(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in sd.items():
        if key.endswith(".proj.qkv.weight"):
            base = key[: -len("qkv.weight")]
            for name, part in zip("qkv", value.chunk(3, dim=0)):
                out[f"{base}{name}.weight"] = part.contiguous()
        else:
            out[key] = value
    return out


def save_model(path, sd: dict[str, torch.Tensor], metadata: dict[str, str] | None = None) -> str:
    save_file({k: v.contiguous() for k, v in sd.items()}, str(path), metadata=metadata)
    return str(path)


def lora_pair(stem: str, out: int, inf: int, rank: int, seed: int, alpha: float | None = None, up="lora_up", down="lora_down"):
    gen = torch.Generator().manual_seed(seed)
    sd = {
        f"{stem}.{up}.weight": torch.randn(out, rank, generator=gen) * 0.1,
        f"{stem}.{down}.weight": torch.randn(rank, inf, generator=gen) * 0.1,
    }
    if alpha is not None:
        sd[f"{stem}.alpha"] = torch.tensor(float(alpha))
    return sd


def lokr(stem: str, shape1: tuple[int, int], shape2: tuple[int, int], seed: int, alpha: float | None = None, factor_w2: int | None = None):
    gen = torch.Generator().manual_seed(seed)
    sd = {f"{stem}.lokr_w1": torch.randn(*shape1, generator=gen) * 0.3}
    if factor_w2 is None:
        sd[f"{stem}.lokr_w2"] = torch.randn(*shape2, generator=gen) * 0.3
    else:
        sd[f"{stem}.lokr_w2_a"] = torch.randn(shape2[0], factor_w2, generator=gen) * 0.3
        sd[f"{stem}.lokr_w2_b"] = torch.randn(factor_w2, shape2[1], generator=gen) * 0.3
    if alpha is not None:
        sd[f"{stem}.alpha"] = torch.tensor(float(alpha))
    return sd


def save_lora(path, sd: dict[str, torch.Tensor]) -> str:
    save_file({k: v.contiguous() for k, v in sd.items()}, str(path))
    return str(path)


def drain(plan, tensors):
    out = dict(tensors)
    assert [s.key for s in plan] == list(out)
    return out


def descriptor_bytes() -> torch.Tensor:
    blob = json.dumps({"format": "float8_e4m3fn", "full_precision_matrix_mult": True}).encode()
    return torch.frombuffer(bytearray(blob), dtype=torch.uint8)


import pytest


@pytest.fixture
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
