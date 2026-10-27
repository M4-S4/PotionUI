# Derived from: comfy/ops.py (ComfyUI, GPL-3.0); see vendor/NOTICE.md
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from vendor.gpl.comfyui import ops
from vendor.gpl.comfyui.ops import (
    _NVFP4_BLOCK,
    _add_lora_output_branch,
    _deltas_output_branch_ok,
    _dequant_values,
    _lora_dequant_path,
    _lora_fused_enabled,
    _lora_output_branch,
    _note_lora_path,
    _quantize_nvfp4_dynamic,
    _scale_values,
    _with_lokr,
    cast_to,
    linear_with_lora_deltas,
    partition_output_branch_deltas,
    precompute_nvfp4_scale,
)

from .fp8_linear import Fp8ScaledLinear
from .gemm import gemm_support

logger = logging.getLogger(__name__)

NVFP4_MATMUL_FLAG = "native_nvfp4_matmul"


def _nvfp4_matmul_enabled() -> bool:
    if not ops.RUNTIME_FLAGS.get(NVFP4_MATMUL_FLAG, False):
        return False
    return gemm_support().nvfp4


def _nvfp4_fast_path_reject_reason(
    *,
    lora_deltas: "list | None",
    input_dtype: torch.dtype,
    input_is_cuda: bool,
    weight_is_cuda: bool,
    in_features: int,
    out_features: int,
) -> str | None:
    if lora_deltas and not _deltas_output_branch_ok(lora_deltas, out_features):
        return "lora_deltas"
    if input_dtype not in (torch.float16, torch.bfloat16):
        return f"input_dtype={input_dtype}"
    if not input_is_cuda:
        return "input_not_cuda"
    if not weight_is_cuda:
        return "weight_not_cuda"
    if in_features % 32:
        return "in_features_not_multiple_of_32"
    if out_features % 16:
        return "out_features_not_multiple_of_16"
    return None


def _nvfp4_scaled_mm_fast_path_ok(
    *,
    lora_deltas: "list | None",
    input_dtype: torch.dtype,
    input_is_cuda: bool,
    weight_is_cuda: bool,
    in_features: int,
    out_features: int,
) -> bool:
    return _nvfp4_fast_path_reject_reason(
        lora_deltas=lora_deltas, input_dtype=input_dtype,
        input_is_cuda=input_is_cuda, weight_is_cuda=weight_is_cuda,
        in_features=in_features, out_features=out_features,
    ) is None


_nvfp4_fast_path_reject_reasons_logged: set[str] = set()


def _log_nvfp4_fast_path_rejection(reason: str) -> None:
    if reason in _nvfp4_fast_path_reject_reasons_logged:
        return
    _nvfp4_fast_path_reject_reasons_logged.add(reason)
    logger.warning(
        "nvfp4 matmul: fast path unavailable (%s); using LUT dequant instead "
        "(logged once per distinct reason)", reason,
    )


def reset_nvfp4_fast_path_rejection_log() -> None:
    _nvfp4_fast_path_reject_reasons_logged.clear()


class Nvfp4Linear(Fp8ScaledLinear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._is_nvfp4 = False
        self.register_buffer("nvfp4_packed", None, persistent=False)
        self.register_buffer("nvfp4_scale", None, persistent=False)
        self.register_buffer("nvfp4_weight_block_scale", None, persistent=False)
        self.register_buffer("nvfp4_weight_global_scale", None, persistent=False)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ) -> None:
        ws2 = state_dict.pop(prefix + "weight_scale_2", None)
        if ws2 is None:
            super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs,
            )
            return

        packed = state_dict.pop(prefix + "weight", None)
        block = state_dict.pop(prefix + "weight_scale", None)
        state_dict.pop(prefix + "input_scale", None)
        state_dict.pop(prefix + "comfy_quant", None)
        if packed is None or block is None:
            raise ValueError(f"nvfp4 layer '{prefix.rstrip('.')}' missing weight/weight_scale")

        self._is_nvfp4 = True
        self.nvfp4_packed = packed if packed.dtype == torch.uint8 else packed.view(torch.uint8)
        global_scale = ws2.reshape(())
        self.nvfp4_scale = precompute_nvfp4_scale(block, global_scale, self.out_features, self.in_features)
        self.nvfp4_weight_block_scale = block if block.dtype == torch.float8_e4m3fn else block.to(torch.float8_e4m3fn)
        self.nvfp4_weight_global_scale = global_scale.to(torch.float32)
        self.weight = None

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )
        for name in ("weight", "weight_scale", "weight_scale_2", "input_scale", "comfy_quant", "pre_quant_scale"):
            key = prefix + name
            if key in missing_keys:
                missing_keys.remove(key)

    def forward_comfy_cast_weights(self, input: torch.Tensor) -> torch.Tensor:
        if not self._is_nvfp4:
            return super().forward_comfy_cast_weights(input)
        if self.pre_quant_scale is not None:
            input = input * cast_to(
                self.pre_quant_scale, input.dtype, input.device, non_blocking=self.stream_non_blocking,
            )
        dequant_reason = "nvfp4 matmul off"
        if _nvfp4_matmul_enabled():
            reject_reason = _nvfp4_fast_path_reject_reason(
                lora_deltas=self.lora_deltas,
                input_dtype=input.dtype,
                input_is_cuda=input.is_cuda,
                weight_is_cuda=self.nvfp4_packed.is_cuda,
                in_features=self.in_features,
                out_features=self.out_features,
            )
            if reject_reason is None:
                fast = self._forward_nvfp4_scaled_mm(input)
                if fast is not None:
                    return fast
                dequant_reason = "kernel rejected"
            else:
                _log_nvfp4_fast_path_rejection(reject_reason)
                dequant_reason = reject_reason
        if self.lora_deltas:
            output_side, weight_side = partition_output_branch_deltas(self.lora_deltas, self.out_features)
            _note_lora_path(self, _lora_dequant_path(weight_side, dequant_reason, output_side))
        values = _dequant_values(self.nvfp4_packed, self.out_features, self.in_features)
        num_blocks = self.in_features // _NVFP4_BLOCK
        weight = _scale_values(values, self.nvfp4_scale, self.out_features, self.in_features, num_blocks)
        weight = weight.to(device=input.device, dtype=input.dtype)
        bias = None if self.bias is None else self.bias.to(input.dtype)
        return linear_with_lora_deltas(input, weight, bias, self.lora_deltas, self.out_features, owner=self)

    def _forward_nvfp4_scaled_mm(self, input: torch.Tensor) -> torch.Tensor | None:
        device = input.device
        w_block = self.nvfp4_weight_block_scale
        w_global = self.nvfp4_weight_global_scale
        if w_block is None or w_global is None:
            return None
        if w_block.device != device or self.nvfp4_packed.device != device:
            return None

        orig_shape = input.shape
        x2d = input.reshape(-1, orig_shape[-1])
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()

        try:
            x_packed, x_block, x_global = _quantize_nvfp4_dynamic(x2d)
        except Exception:
            logger.debug("nvfp4 matmul: activation quant failed; using dequant path", exc_info=True)
            return None

        weight_packed = self.nvfp4_packed if self.nvfp4_packed.is_contiguous() else self.nvfp4_packed.contiguous()
        mat_a = x_packed.view(torch.float4_e2m1fn_x2)
        mat_b = weight_packed.view(torch.float4_e2m1fn_x2).t()
        bias = self.bias.to(device=device, dtype=input.dtype) if self.bias is not None else None

        try:
            out = F.scaled_mm(
                mat_a, mat_b,
                [x_block.contiguous(), x_global], [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
                [w_block.contiguous(), w_global], [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
                swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
                swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
                output_dtype=input.dtype,
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            logger.debug("nvfp4 matmul: _scaled_mm_v2 rejected operands; using dequant path", exc_info=True)
            return None
        if self.lora_deltas:
            if _lora_fused_enabled():
                _note_lora_path(self, _with_lokr("fused-fast", self.lora_deltas))
                out = _add_lora_output_branch(out, x2d, self.lora_deltas, self.out_features, owner=self)
            else:
                _note_lora_path(self, "unfused-fast")
                out = out + _lora_output_branch(x2d, self.lora_deltas, out.dtype, self.out_features)
        if bias is not None:
            out = out + bias
        return out.reshape(*orig_shape[:-1], -1)
