from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from .errors import BakeError, MergeError, ModelToolsError, UnsupportedFamilyError


@dataclass(frozen=True)
class LoraInput:
    path: str
    strength: float = 1.0


FAMILY_ALIASES: dict[str, tuple[str, ...]] = {
    "flux": ("flux",),
    "flux1": ("flux",),
    "flux2": ("flux3",),
    "flux3": ("flux3",),
    "klein": ("flux3",),
    "qwen": ("qwen_image", "qwen_image21"),
    "qwen_image": ("qwen_image",),
    "qwen_image21": ("qwen_image21",),
    "wan": ("wan",),
    "z_image": ("z_image",),
    "zimage": ("z_image",),
    "krea2": ("krea2",),
    "krea_2": ("krea2",),
    "moss": ("moss_image",),
    "moss_image": ("moss_image",),
    "ltx": ("ltx",),
}

SUPPORTED_SPEC_FAMILIES = frozenset(f for group in FAMILY_ALIASES.values() for f in group)


def resolve_family(name: str) -> tuple[str, ...]:
    norm = re.sub(r"[\s\-]+", "_", name.strip().lower())
    group = FAMILY_ALIASES.get(norm)
    if group is None:
        raise UnsupportedFamilyError(
            f"family {name!r} is not supported by LoRA tools (supported: {', '.join(sorted(FAMILY_ALIASES))})"
        )
    return group


_TE_STEM = re.compile(
    r"^(lora_te\d*[_.]|text_encoders?\.|text_model\.|te\d*\.|conditioner\.|clip_[lg]\.|t5xxl\.|lora_prior_te)",
    re.IGNORECASE,
)


def is_text_encoder_stem(stem: str) -> bool:
    head = stem
    for prefix in ("base_model.model.", "transformer.", "diffusion_model."):
        if head.startswith(prefix):
            head = head[len(prefix):]
    return bool(_TE_STEM.match(head))


def load_lora_state(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix.lower() not in (".safetensors", ".sft"):
        raise ModelToolsError(f"{path.name}: LoRA files must be safetensors")
    if not path.is_file():
        raise ModelToolsError(f"{path.name}: file not found")
    return load_file(str(path), device="cpu")


def refusal_for_unsupported_kind(name: str, shapes: dict[str, tuple[int, ...]]) -> str | None:
    if any(".hada_" in k for k in shapes):
        return f"{name}: LoHa adapters (hada_*) are not supported"
    if any(k.endswith(".dora_scale") for k in shapes):
        return f"{name}: DoRA adapters (dora_scale) are not supported because their magnitude vector cannot be merged into a plain delta"
    for key, shape in shapes.items():
        if key.endswith((".lora_up.weight", ".lora_down.weight", ".lora_B.weight", ".lora_A.weight")) and len(shape) != 2:
            return f"{name}: convolution LoRAs are not supported ({key} has {len(shape)} dimensions)"
        if key.endswith(".lokr_t2"):
            return f"{name}: convolution LoKr (Tucker core lokr_t2) is not supported"
    return None
