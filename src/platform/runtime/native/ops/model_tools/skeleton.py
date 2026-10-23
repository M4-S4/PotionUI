from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.platform.runtime.model_headers.signatures import detect_denoiser_prefix
from vendor.gpl.comfyui.ops import pick_operations

from ...detect.registry import match_model_spec
from ...detect.unet_detect import detect_unet_config
from .errors import UnsupportedFamilyError
from .lora_io import BakeError, SUPPORTED_SPEC_FAMILIES, resolve_family
from .spec import DTYPE_TO_TORCH, TensorSpec


@dataclass(frozen=True)
class SourcePart:
    file_key: str
    row_start: int
    rows: int


@dataclass
class Skeleton:
    module: nn.Module
    family: str
    variant: str
    prefix: str
    sources: dict[str, TensorSpec]

    def param_shape(self, param: str) -> tuple[int, ...]:
        return tuple(self.module.get_parameter(param).shape)

    def locate(self, param: str) -> list[SourcePart]:
        shape = self.param_shape(param)
        direct = self.sources.get(self.prefix + param)
        if direct is not None:
            if tuple(direct.shape) != shape:
                raise BakeError(
                    f"shape mismatch on {self.prefix + param}: the model file has {tuple(direct.shape)} "
                    f"but the {self.family} architecture expects {shape}"
                )
            return [SourcePart(self.prefix + param, 0, shape[0])]
        for pattern, parts in _FUSED_SOURCES.get(self.family, ()):
            match = pattern.match(param)
            if match is None:
                continue
            keys = [self.prefix + match.group(1) + part for part in parts]
            if not all(k in self.sources for k in keys):
                continue
            located: list[SourcePart] = []
            start = 0
            for key in keys:
                spec = self.sources[key]
                if len(spec.shape) != 2 or spec.shape[1] != shape[1]:
                    raise BakeError(f"shape mismatch on {key}: {tuple(spec.shape)} does not fit {param} {shape}")
                located.append(SourcePart(key, start, spec.shape[0]))
                start += spec.shape[0]
            if start != shape[0]:
                raise BakeError(
                    f"shape mismatch: {', '.join(keys)} total {start} rows but {param} has {shape[0]}"
                )
            return located
        raise BakeError(f"missing key: {self.prefix + param} is not in the model file")


_FUSED_SOURCES: dict[str, tuple[tuple[re.Pattern[str], tuple[str, ...]], ...]] = {
    "moss_image": (
        (re.compile(r"^(.*\.proj)\.qkv\.weight$"), (".q.weight", ".k.weight", ".v.weight")),
    ),
    "qwen_image21": (
        (re.compile(r"^(.*\.img_mlp)\.gate_up\.weight$"), (".gate_layer.weight", ".proj.weight")),
    ),
}


def build_skeleton(
    plan: Sequence[TensorSpec], metadata: dict[str, str] | None, family: str | None
) -> Skeleton:
    allowed = resolve_family(family) if family else tuple(sorted(SUPPORTED_SPEC_FAMILIES))
    keys = [s.key for s in plan]
    prefix = detect_denoiser_prefix(keys) or ""
    sources = {s.key: s for s in plan}
    meta_sd = {
        s.key[len(prefix):]: torch.empty(s.shape, dtype=DTYPE_TO_TORCH[s.dtype], device="meta")
        for s in plan
        if s.key.startswith(prefix)
    }
    try:
        config = detect_unet_config(meta_sd, metadata)
    except Exception as exc:
        raise UnsupportedFamilyError(f"could not read the model architecture from its tensors: {exc}") from exc
    if config is None:
        raise UnsupportedFamilyError("could not identify the model family from its tensor names")
    try:
        spec = match_model_spec(config)
    except Exception as exc:
        raise UnsupportedFamilyError(f"no native architecture matches this model: {exc}") from exc
    if spec.family not in SUPPORTED_SPEC_FAMILIES:
        raise UnsupportedFamilyError(f"family {spec.family!r} is not supported by LoRA tools")
    if spec.family not in allowed:
        raise BakeError(f"the model file is {spec.family!r} but family {family!r} was requested")
    with torch.device("meta"):
        module = spec.resolve_model_class().from_config(config, pick_operations(torch.bfloat16, torch.bfloat16))
    return Skeleton(module=module, family=spec.family, variant=spec.variant, prefix=prefix, sources=sources)
