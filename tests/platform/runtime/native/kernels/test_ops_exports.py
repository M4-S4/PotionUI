from __future__ import annotations

import pytest

from src.platform.runtime.native import kernels
from vendor.gpl.comfyui import ops

KERNEL_CLASSES_AND_FUNCTIONS = [
    "CastWeightBiasOp",
    "Fp8ScaledLinear",
    "cast_to",
    "detect_quant_format",
    "disable_weight_init",
    "fp8_ops",
    "invalidate_lora_fused_cache",
    "log_lora_path_summary",
    "manual_cast",
    "partition_output_branch_deltas",
    "pick_operations",
]
KERNEL_PACKAGE = "src.platform.runtime.native.kernels"
MOVED_OUT_OF_VENDOR = [
    "CastWeightBiasOp",
    "Fp8ScaledLinear",
    "GEMM_SUPPORT",
    "Nvfp4Linear",
    "QUANT_FP8_SCALED",
    "cast_bias_weight",
    "cast_to",
    "detect_quant_format",
    "disable_weight_init",
    "fp8_ops",
    "manual_cast",
    "pick_operations",
    "_fp8_matmul_enabled",
    "_nvfp4_matmul_enabled",
    "_scaled_mm_fast_path_reject_reason",
    "_nvfp4_fast_path_reject_reason",
    "_quantize_fp8_dynamic",
    "NATIVE_LORA_FUSED_ENV",
    "RUNTIME_FLAGS",
    "_LokrBranch",
    "_NVFP4_LORA_BRANCH_CHUNK_BYTES",
    "_add_lora_output_branch",
    "_deltas_output_branch_ok",
    "_lora_fused_enabled",
    "_lora_output_branch",
    "apply_lora_deltas",
    "invalidate_lora_fused_cache",
    "linear_with_lora_deltas",
    "log_lora_path_summary",
    "partition_output_branch_deltas",
    "QUANT_NVFP4",
    "_BYTE_ARANGE_I32",
    "_COMFY_QUANT_FP8_FORMATS",
    "_COMFY_QUANT_INT8_TENSORWISE",
    "_CONVROT_DEFAULT_GROUPSIZE",
    "_CONVROT_H4",
    "_CONVROT_HADAMARD_CACHE",
    "_E2M1_LUT",
    "_E2M1_LUT_HIGH",
    "_E2M1_LUT_LOW",
    "_E4M3_MAX",
    "_F4_MAX",
    "_FP8_DTYPES",
    "_INV_BLOCKED_CACHE",
    "_NVFP4_BLOCK",
    "_build_convrot_hadamard",
    "_check_quant_format_supported",
    "_convrot_group_exponent",
    "_convrot_unrotate_weight",
    "_dequant_values",
    "_extract_convrot_config",
    "_inv_blocked_index",
    "_parse_comfy_quant",
    "_quantize_nvfp4_dynamic",
    "_reject_ambiguous_int8",
    "_scale_values",
    "_to_blocked",
    "_unblock_scale",
    "dequantize_nvfp4",
    "precompute_nvfp4_scale",
]


@pytest.mark.parametrize("name", KERNEL_CLASSES_AND_FUNCTIONS)
def test_the_layer_classes_and_op_selection_are_defined_in_the_kernels_package(name):
    assert name in kernels.__all__
    assert getattr(kernels, name).__module__.startswith(KERNEL_PACKAGE + ".")


def test_the_quant_format_marker_is_exported_from_the_kernels_package():
    assert "QUANT_FP8_SCALED" in kernels.__all__
    assert kernels.QUANT_FP8_SCALED == "fp8_scaled"


@pytest.mark.parametrize("name", MOVED_OUT_OF_VENDOR)
def test_the_moved_names_are_gone_from_the_vendored_ops_module(name):
    assert not hasattr(ops, name)


def test_no_kernel_export_is_an_underscore_name():
    assert [name for name in kernels.__all__ if name.startswith("_")] == []
