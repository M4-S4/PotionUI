# Derived from: comfy/ops.py (ComfyUI, GPL-3.0); see vendor/NOTICE.md
from __future__ import annotations

import logging

import torch

from vendor.gpl.comfyui.ops import _FP8_DTYPES

from .casting import disable_weight_init, manual_cast
from .nvfp4_linear import Nvfp4Linear

logger = logging.getLogger(__name__)


QUANT_FP8_SCALED = "fp8_scaled"


class fp8_ops(manual_cast):
    Linear = Nvfp4Linear


def detect_quant_format(
    metadata: dict[str, str],
    sd: dict[str, torch.Tensor],
) -> str | None:
    if "_quantization_metadata" in metadata:
        if any(t.dtype in _FP8_DTYPES for t in sd.values()):
            return QUANT_FP8_SCALED
    if "scaled_fp8" in sd:
        return QUANT_FP8_SCALED
    for k in sd:
        if (
            k.endswith(".weight_scale")
            or k.endswith(".scale_weight")
            or k.endswith(".weight_scale_2")
            or k.endswith(".comfy_quant")
        ):
            return QUANT_FP8_SCALED
    return None


def pick_operations(
    storage_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    quant_format: str | None = None,
) -> type[disable_weight_init]:
    if quant_format == QUANT_FP8_SCALED or storage_dtype in _FP8_DTYPES:
        logger.debug("pick_operations -> fp8_ops (quant=%s storage=%s)", quant_format, storage_dtype)
        return fp8_ops
    if storage_dtype != compute_dtype:
        logger.debug("pick_operations -> manual_cast (storage=%s compute=%s)", storage_dtype, compute_dtype)
        return manual_cast
    logger.debug("pick_operations -> disable_weight_init (dtype=%s)", storage_dtype)
    return disable_weight_init
