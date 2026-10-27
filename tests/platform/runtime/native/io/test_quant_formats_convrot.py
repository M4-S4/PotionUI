from __future__ import annotations

import pytest
import torch

from src.platform.runtime.native.io import quant_formats
from src.platform.runtime.native.io.quant_formats import build_convrot_hadamard, convrot_unrotate_weight
from src.platform.runtime.native.kernels import casting, fp8_linear
from src.platform.runtime.native.ops.model_tools import formats

from .._quant_layouts import HADAMARD_SIZES, reference_hadamard, reference_rotate


@pytest.mark.parametrize("size", HADAMARD_SIZES)
def test_hadamard_is_symmetric_and_involutory(size):
    h = build_convrot_hadamard(size, device="cpu", dtype=torch.float32)
    assert torch.allclose(h, h.T)
    assert torch.allclose(h @ h, torch.eye(size), atol=1e-5)
    assert torch.allclose(h, reference_hadamard(size), atol=1e-6)


def test_hadamard_rejects_non_power_of_4_groupsize():
    build_convrot_hadamard(64, device="cpu", dtype=torch.float32)
    with pytest.raises(ValueError, match="power of 4"):
        build_convrot_hadamard(128, device="cpu", dtype=torch.float32)


def test_the_hadamard_is_built_once_per_group_device_and_dtype():
    first = build_convrot_hadamard(64, device="cpu", dtype=torch.float32)
    assert build_convrot_hadamard(64, device="cpu", dtype=torch.float32) is first
    assert build_convrot_hadamard(64, device="cpu", dtype=torch.float64) is not first
    assert build_convrot_hadamard(16, device="cpu", dtype=torch.float32) is not first


def test_unrotating_applies_the_same_hadamard_the_reference_rotation_does():
    torch.manual_seed(40)
    weight = torch.randn(8, 128)
    hadamard = build_convrot_hadamard(64, device="cpu", dtype=torch.float32)
    got = convrot_unrotate_weight(weight, hadamard, 64)
    assert torch.allclose(got, reference_rotate(weight, reference_hadamard(64), 64), atol=1e-5)


def test_unrotating_twice_restores_the_original_basis():
    torch.manual_seed(41)
    weight = torch.randn(8, 256)
    hadamard = build_convrot_hadamard(256, device="cpu", dtype=torch.float32)
    once = convrot_unrotate_weight(weight, hadamard, 256)
    assert not torch.allclose(once, weight, atol=1e-3)
    assert torch.allclose(convrot_unrotate_weight(once, hadamard, 256), weight, atol=1e-5)


def test_a_groupsize_that_does_not_divide_the_in_features_is_rejected():
    hadamard = build_convrot_hadamard(64, device="cpu", dtype=torch.float32)
    with pytest.raises(ValueError, match="in_features 48 not divisible by ConvRot group size 64"):
        convrot_unrotate_weight(torch.zeros(2, 48), hadamard, 64)


@pytest.mark.parametrize("module", [casting, fp8_linear, formats], ids=["casting", "fp8_linear", "model_tools"])
def test_every_consumer_builds_the_hadamard_through_the_one_shared_cache(module):
    assert module.build_convrot_hadamard is quant_formats.build_convrot_hadamard
