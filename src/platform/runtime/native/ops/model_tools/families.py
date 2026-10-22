from __future__ import annotations

import fnmatch
import re
from collections.abc import Sequence

import torch

from ...detect.registry import match_model_spec
from ...detect.te_detect import detect_te_config
from ...detect.unet_detect import detect_unet_config
from .errors import UnsupportedFamilyError
from .spec import TensorSpec

STRIP_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.", "model.")

COMMON_KEEP = (
    r"norm",
    r"embed",
    r"modulation",
    r"scale_shift_table",
    r"adaln",
    r"(^|\.)mod(\.|$)",
    r"_mod(\.|$)",
    r"(^|\.)mod_",
    r"^(time|vector|guidance)_in\.",
    r"lm_head",
    r"relative_attention_bias",
    r"^shared\.",
    r"position",
)

DIT_IO_KEEP = (r"^img_in\.", r"^txt_in\.", r"^x_embedder", r"^final_layer", r"^proj_out")

UNET_FAMILY_KEEP: dict[str, tuple[str, ...]] = {
    "flux": DIT_IO_KEEP,
    "flux3": DIT_IO_KEEP,
    "krea2": DIT_IO_KEEP,
    "moss_image": DIT_IO_KEEP + (r"^final_linear",),
    "anima": DIT_IO_KEEP + (r"^llm_adapter\.",),
    "qwen_image": DIT_IO_KEEP + (r"^time_text_embed", r"^norm_out"),
    "qwen_image21": DIT_IO_KEEP + (r"^time_text_embed", r"^norm_out"),
    "z_image": DIT_IO_KEEP + (r"^cap_embedder", r"^t_embedder", r"^all_"),
    "wan": (r"^head\.", r"^text_embedding", r"^img_emb", r"^time_projection", r"^patch_embedding"),
    "ltx": (
        r"^(audio_)?patchify_proj",
        r"^(audio_)?adaln_single",
        r"^(audio_)?caption_projection",
        r"^(audio_)?proj_out",
        r"^av_ca_",
    ),
}

TE_FAMILY_KEEP: dict[str, tuple[str, ...]] = {
    "t5xxl": (),
    "umt5": (),
    "qwen3": (),
    "qwen35": (r"^visual\.",),
    "clip_l": (r"^text_projection",),
}

FAMILY_KEEP = {**UNET_FAMILY_KEEP, **TE_FAMILY_KEEP}

REFUSED_NOTES = {
    "sdxl": "SDXL and other UNet checkpoints are not supported by the quantizer",
}


def strip_prefix(key: str) -> str:
    for prefix in STRIP_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _is_unet(keys: Sequence[str]) -> bool:
    return any(
        ".input_blocks." in k or ".down_blocks." in k or k.startswith(("input_blocks.", "down_blocks.", "conditioner."))
        for k in keys
    )


def detect_family(plan: Sequence[TensorSpec], metadata: dict[str, str] | None = None) -> str:
    keys = [s.key for s in plan]
    if _is_unet(keys):
        raise UnsupportedFamilyError(REFUSED_NOTES["sdxl"])
    sd = {strip_prefix(s.key): torch.empty(s.shape, dtype=torch.uint8, device="meta") for s in plan}
    config = None
    try:
        config = detect_unet_config(sd, metadata)
    except Exception:
        config = None
    if config is not None:
        try:
            family = match_model_spec(config).family
        except Exception:
            family = None
        if family in UNET_FAMILY_KEEP:
            return family
        if family is not None:
            raise UnsupportedFamilyError(f"family {family!r} is not supported by the quantizer")
    try:
        te = detect_te_config(sd)
    except Exception:
        te = None
    if te is not None and te.get("te_type") in TE_FAMILY_KEEP:
        return te["te_type"]
    raise UnsupportedFamilyError("could not identify the model family from its tensor names")


def supported_families() -> tuple[str, ...]:
    return tuple(FAMILY_KEEP)


def keep_patterns(family: str) -> tuple[re.Pattern[str], ...]:
    if family not in FAMILY_KEEP:
        note = REFUSED_NOTES.get(family) or f"family {family!r} is not supported by the quantizer"
        raise UnsupportedFamilyError(note)
    return tuple(re.compile(p, re.IGNORECASE) for p in COMMON_KEEP + FAMILY_KEEP[family])


def is_kept(key: str, patterns: Sequence[re.Pattern[str]], keep_extra: Sequence[str] = ()) -> bool:
    stripped = strip_prefix(key)
    if any(p.search(stripped) for p in patterns):
        return True
    return any(fnmatch.fnmatchcase(key, g) or fnmatch.fnmatchcase(stripped, g) for g in keep_extra)
