from __future__ import annotations

import pytest

from src.platform.runtime.native import kernels

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


@pytest.mark.parametrize("name", KERNEL_CLASSES_AND_FUNCTIONS)
def test_the_layer_classes_and_op_selection_are_defined_in_the_kernels_package(name):
    assert name in kernels.__all__
    assert getattr(kernels, name).__module__.startswith(KERNEL_PACKAGE + ".")


def test_the_quant_format_marker_is_exported_from_the_kernels_package():
    assert "QUANT_FP8_SCALED" in kernels.__all__
    assert kernels.QUANT_FP8_SCALED == "fp8_scaled"


def test_no_kernel_export_is_an_underscore_name():
    assert [name for name in kernels.__all__ if name.startswith("_")] == []
