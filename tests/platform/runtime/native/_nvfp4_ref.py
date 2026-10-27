"""Shared nvfp4 reference quantiser for tests.

Round-to-nearest port of comfy/float.py's nvfp4 quantiser (non-stochastic, for
determinism). Used to build synthetic nvfp4 tensors so dequant can be validated
without the compiled comfy_kitchen kernel or a real checkpoint.
"""

from __future__ import annotations

import torch

from src.platform.runtime.native.io.quant_formats import E2M1_LUT, NVFP4_BLOCK, to_blocked, unblock_scale

F4_MAX = 6.0
F8_MAX = 448.0


def default_tensor_scale(w: torch.Tensor) -> torch.Tensor:
    return (w.abs().amax() / (F4_MAX * F8_MAX)).detach()


def quantize_nvfp4(w: torch.Tensor, tensor_scale: torch.Tensor):
    """Quantise fp32 ``w`` [out, in] to nvfp4.

    Returns ``(packed_u8 [out, in//2], block_swizzled_fp8, codes_natural [out,in],
    block_natural_fp8 [out, in//16])``.
    """
    out, inn = w.shape
    xb = w.reshape(out, -1, NVFP4_BLOCK)
    block = torch.clamp((xb.abs().amax(-1) / F4_MAX) / tensor_scale, max=F8_MAX).to(torch.float8_e4m3fn)
    xn = xb / (tensor_scale * block.to(torch.float32)).unsqueeze(-1)
    sign = torch.signbit(xn).to(torch.uint8)
    ax = xn.abs()
    exp = torch.floor(torch.log2(ax) + 1.1925).clamp(0, 3)
    mant = torch.where(exp > 0, (ax / (2.0 ** (exp - 1)) - 1.0) * 2.0, ax * 2.0).round().clamp(0, 1).to(torch.uint8)
    codes = ((sign << 3) | (exp.to(torch.uint8) << 1) | mant).reshape(out, inn)
    packed = (codes[:, 0::2] << 4) | codes[:, 1::2]
    block_sw = to_blocked(block.to(torch.float32)).to(torch.float8_e4m3fn)
    return packed, block_sw, codes.long(), block


def old_dequantize_nvfp4(
    packed: torch.Tensor, block_scale: torch.Tensor, tensor_scale: torch.Tensor,
    out_features: int, in_features: int,
) -> torch.Tensor:
    device = packed.device
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    codes = torch.empty(out_features, in_features, dtype=torch.long, device=device)
    codes[:, 0::2] = high.long()
    codes[:, 1::2] = low.long()
    values = E2M1_LUT.to(device)[codes]
    num_blocks = in_features // NVFP4_BLOCK
    block_nat = unblock_scale(block_scale, out_features, num_blocks).to(device)
    scale = (block_nat * tensor_scale.to(torch.float32).to(device)).repeat_interleave(NVFP4_BLOCK, dim=1)
    return values * scale
