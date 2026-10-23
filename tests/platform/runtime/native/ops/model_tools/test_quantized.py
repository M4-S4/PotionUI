from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from src.platform.runtime.native.ops.model_tools import quantized
from src.platform.runtime.native.ops.model_tools.spec import TensorSpec


def plan_of(*items):
    return [TensorSpec(key, dtype, shape) for key, dtype, shape in items]


@pytest.mark.parametrize(
    "items, expected",
    [
        ([("a.weight", "BF16", (4, 4))], False),
        ([("a.weight", "F32", (4, 4)), ("a.bias", "F32", (4,))], False),
        ([("a.weight", "F8_E4M3", (4, 4))], True),
        ([("a.weight", "F8_E5M2", (4, 4))], True),
        ([("a.weight", "BF16", (4, 4)), ("a.weight_scale", "F32", ())], True),
        ([("a.weight", "BF16", (4, 4)), ("a.comfy_quant", "U8", (8,))], True),
        ([("a.weight", "BF16", (4, 4)), ("scaled_fp8", "BF16", (2,))], True),
        ([("a.weight", "U8", (4, 4)), ("a.weight.SCB", "F32", (4,))], True),
        ([("a.qweight", "I32", (4, 4)), ("a.scales", "F16", (4,))], True),
    ],
)
def test_plan_classification(items, expected):
    assert quantized.is_quantized_plan(plan_of(*items)) is expected


def test_files_are_classified_by_header_and_cached(tmp_path):
    plain = tmp_path / "plain.safetensors"
    save_file({"a.weight": torch.zeros(4, 4, dtype=torch.bfloat16)}, str(plain))
    assert quantized.is_unquantized_file(plain) is True
    assert quantized.is_unquantized_file(plain) is True

    save_file({"a.weight": torch.zeros(4, 4).to(torch.float8_e4m3fn)}, str(plain))
    assert quantized.is_unquantized_file(plain) is False


@pytest.mark.parametrize("name", ["m.gguf", "m.ckpt", "m.pt"])
def test_other_formats_are_never_offered(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"x")
    assert quantized.is_unquantized_file(path) is False


def test_missing_and_corrupt_files_are_not_offered(tmp_path):
    assert quantized.is_unquantized_file(tmp_path / "missing.safetensors") is False
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"\x01\x02")
    assert quantized.is_unquantized_file(bad) is False
