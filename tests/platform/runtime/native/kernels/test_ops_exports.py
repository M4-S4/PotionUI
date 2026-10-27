from __future__ import annotations

import pytest

from src.platform.runtime.native import kernels
from vendor.gpl.comfyui import ops

NATIVE_CODE_NEEDS = ["CastWeightBiasOp", "QUANT_FP8_SCALED", "invalidate_lora_fused_cache"]


@pytest.mark.parametrize("name", NATIVE_CODE_NEEDS)
def test_the_ops_names_native_code_needs_are_public_kernel_exports(name):
    assert name in kernels.__all__
    assert getattr(kernels, name) is getattr(ops, name)


def test_no_kernel_export_is_an_underscore_name():
    assert [name for name in kernels.__all__ if name.startswith("_")] == []
