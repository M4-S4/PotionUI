from __future__ import annotations

import torch
from vendor.gpl.comfyui.ops import disable_weight_init, manual_cast
from src.platform.runtime.native.kernels import (
    QUANT_FP8_SCALED,
    detect_quant_format,
    fp8_ops,
    pick_operations,
)


def test_detect_quant_format():
    assert detect_quant_format(
        {"_quantization_metadata": "{}"},
        {"blk.0.weight": torch.zeros(2, dtype=torch.float8_e4m3fn)},
    ) == QUANT_FP8_SCALED
    assert detect_quant_format({}, {"scaled_fp8": torch.zeros(0)}) == QUANT_FP8_SCALED
    assert detect_quant_format({}, {"blk.0.weight_scale": torch.zeros(())}) == QUANT_FP8_SCALED
    assert detect_quant_format({}, {"blk.0.scale_weight": torch.zeros(())}) == QUANT_FP8_SCALED
    assert detect_quant_format({"format": "pt"}, {"blk.0.weight": torch.zeros(2)}) is None


def test_detect_quant_format_ignores_file_level_marker_for_bf16_component():
    metadata = {"_quantization_metadata": '{"format_version": 3, "layers": {"model.diffusion_model.x": {}}}'}
    bf16_vae_sd = {
        "encoder.conv_in.weight": torch.zeros(4, 4, 3, 3, dtype=torch.bfloat16),
        "encoder.conv_in.bias": torch.zeros(4, dtype=torch.bfloat16),
    }
    assert detect_quant_format(metadata, bf16_vae_sd) is None

    dit_sd = {"transformer_blocks.0.attn.to_k.weight": torch.zeros(4, 4, dtype=torch.float8_e4m3fn)}
    assert detect_quant_format(metadata, dit_sd) == QUANT_FP8_SCALED


def test_detect_quant_format_nvfp4_marker_unaffected():
    sd = {
        "blk.0.weight": torch.zeros(4, 2, dtype=torch.uint8),
        "blk.0.weight_scale": torch.zeros(4, 1, dtype=torch.float8_e4m3fn),
        "blk.0.weight_scale_2": torch.zeros(()),
    }
    assert detect_quant_format({}, sd) == QUANT_FP8_SCALED
    assert detect_quant_format({"_quantization_metadata": "{}"}, sd) == QUANT_FP8_SCALED


def test_pick_operations_selection():
    assert pick_operations(torch.bfloat16, torch.bfloat16) is disable_weight_init
    assert pick_operations(torch.float16, torch.bfloat16) is manual_cast
    assert pick_operations(torch.float8_e4m3fn, torch.bfloat16) is fp8_ops
    assert pick_operations(torch.bfloat16, torch.bfloat16, QUANT_FP8_SCALED) is fp8_ops
