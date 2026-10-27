from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch

from vendor.gpl.comfyui import ops

from .fp8_linear import Fp8ScaledLinear, _scaled_mm_fast_path_reject_reason
from .gemm import gemm_support
from .nvfp4_linear import _nvfp4_fast_path_reject_reason

_FP8_FLAG = "native_fp8_matmul"
_NVFP4_FLAG = "native_nvfp4_matmul"
_ACTIVATION_DTYPE = torch.bfloat16


class FastPathKind(StrEnum):
    FP8_SCALED_MM = "fp8_scaled_mm"
    NVFP4_GEMM = "nvfp4_gemm"
    DEQUANT = "dequant"


class QuantFormat(StrEnum):
    FP8 = "fp8"
    INT8 = "int8"
    INT8_CONVROT = "int8_convrot"
    NVFP4 = "nvfp4"


@dataclass(frozen=True)
class FastPath:
    kind: FastPathKind
    quant_format: QuantFormat
    requires_resident_weight: bool
    lora_fused: bool
    reason: str | None


def fast_path(layer: object) -> FastPath | None:
    layer_format = quant_format(layer)
    if layer_format is None:
        return None
    if layer_format is QuantFormat.NVFP4:
        reason = _nvfp4_reason(layer)
        gemm_kind = FastPathKind.NVFP4_GEMM
    else:
        reason = _scaled_mm_reason(layer)
        gemm_kind = FastPathKind.FP8_SCALED_MM
    output_side, _ = ops.partition_output_branch_deltas(layer.lora_deltas, layer.out_features)
    kind = gemm_kind if reason is None else FastPathKind.DEQUANT
    return FastPath(
        kind=kind,
        quant_format=layer_format,
        requires_resident_weight=kind is FastPathKind.NVFP4_GEMM,
        lora_fused=bool(output_side) and ops._lora_fused_enabled(),
        reason=reason,
    )


def quant_format(layer: object) -> QuantFormat | None:
    if not isinstance(layer, Fp8ScaledLinear):
        return None
    if getattr(layer, "_is_nvfp4", False):
        return QuantFormat.NVFP4
    weight = layer.weight
    if layer.weight_scale is None or weight is None:
        return None
    if weight.dtype in ops._FP8_DTYPES:
        return QuantFormat.FP8
    if weight.dtype == torch.int8:
        return QuantFormat.INT8 if layer.convrot_hadamard is None else QuantFormat.INT8_CONVROT
    return None


def _scaled_mm_reason(layer: Fp8ScaledLinear) -> str | None:
    reason = _scaled_mm_fast_path_reject_reason(
        weight_dtype=layer.weight.dtype,
        has_weight_scale=True,
        lora_deltas=layer.lora_deltas,
        input_dtype=_ACTIVATION_DTYPE,
        input_is_cuda=True,
        weight_is_cuda=True,
        in_features=layer.in_features,
        out_features=layer.out_features,
        allow_weight_staging=True,
    )
    if reason is not None:
        return reason
    if layer.weight_scale.numel() != 1:
        return "weight_scale_not_scalar"
    if layer.input_scale is not None and layer.input_scale.numel() != 1:
        return "input_scale_not_scalar"
    if not ops.RUNTIME_FLAGS.get(_FP8_FLAG, False):
        return f"{_FP8_FLAG} is off"
    support = gemm_support()
    return None if support.fp8_scaled_mm else support.fp8_reason


def _nvfp4_reason(layer: Fp8ScaledLinear) -> str | None:
    reason = _nvfp4_fast_path_reject_reason(
        lora_deltas=layer.lora_deltas,
        input_dtype=_ACTIVATION_DTYPE,
        input_is_cuda=True,
        weight_is_cuda=True,
        in_features=layer.in_features,
        out_features=layer.out_features,
    )
    if reason is not None:
        return reason
    if not ops.RUNTIME_FLAGS.get(_NVFP4_FLAG, False):
        return f"{_NVFP4_FLAG} is off"
    support = gemm_support()
    return None if support.nvfp4 else support.nvfp4_reason
