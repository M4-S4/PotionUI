from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.platform.runtime.native import kernels
from src.platform.runtime.native.kernels import (
    CastWeightBiasOp,
    cast_to,
    casting,
    disable_weight_init,
    fp8_ops,
    manual_cast,
)
from tests.fixtures.kernel_ops import patch_ops_name

NAMESPACE_LAYERS = ("Linear", "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "GroupNorm", "LayerNorm", "RMSNorm", "Embedding")


def test_disable_weight_init_skips_init_and_runs():
    lin = disable_weight_init.Linear(4, 3)
    lin.weight = nn.Parameter(torch.eye(3, 4))
    lin.bias = nn.Parameter(torch.zeros(3))
    out = lin(torch.ones(1, 4))
    assert out.shape == (1, 3)


def test_manual_cast_linear_fp16_storage_fp32_compute():
    lin = manual_cast.Linear(4, 3, bias=True)
    lin.weight = nn.Parameter(torch.randn(3, 4, dtype=torch.float16))
    lin.bias = nn.Parameter(torch.randn(3, dtype=torch.float16))
    x = torch.randn(2, 4, dtype=torch.float32)
    out = lin(x)
    assert out.dtype == torch.float32
    ref = F.linear(x, lin.weight.float(), lin.bias.float())
    assert torch.allclose(out, ref, atol=1e-3)


def test_manual_cast_stack_two_layers():
    net = nn.Sequential(
        manual_cast.Linear(4, 8, bias=False),
        manual_cast.Linear(8, 2, bias=False),
    )
    for m in net:
        m.weight = nn.Parameter(torch.randn(m.out_features, m.in_features, dtype=torch.float16))
    x = torch.randn(3, 4, dtype=torch.float32)
    out = net(x)
    assert out.shape == (3, 2)
    assert out.dtype == torch.float32


def test_manual_cast_conv2d():
    conv = manual_cast.Conv2d(2, 2, 3, padding=1, bias=True)
    conv.weight = nn.Parameter(torch.randn(2, 2, 3, 3, dtype=torch.float16))
    conv.bias = nn.Parameter(torch.zeros(2, dtype=torch.float16))
    x = torch.randn(1, 2, 4, 4, dtype=torch.float32)
    out = conv(x)
    assert out.dtype == torch.float32
    assert out.shape == (1, 2, 4, 4)


def test_ops_namespace_protocol_surface():
    for ns in (disable_weight_init, manual_cast, fp8_ops):
        for attr in ("Linear", "Conv2d", "Conv3d", "GroupNorm", "LayerNorm", "RMSNorm", "Embedding"):
            assert hasattr(ns, attr), f"{ns.__name__} missing {attr}"


@pytest.mark.parametrize("name", NAMESPACE_LAYERS)
def test_every_layer_class_casts_weights_through_the_cast_base(name):
    layer = getattr(disable_weight_init, name)
    assert issubclass(layer, CastWeightBiasOp)
    assert getattr(manual_cast, name).comfy_cast_weights is True
    assert getattr(disable_weight_init, name).comfy_cast_weights is False


def test_the_casting_names_are_defined_in_the_casting_module():
    for name in ("CastWeightBiasOp", "cast_to", "disable_weight_init", "manual_cast"):
        assert getattr(kernels, name).__module__ == casting.__name__


def test_cast_to_does_not_touch_the_tensor_when_nothing_has_to_change(monkeypatch):
    tensor = torch.randn(3)

    def refuse(*args, **kwargs):
        raise AssertionError("a tensor already on the target dtype and device must not be converted")

    monkeypatch.setattr(torch.Tensor, "to", refuse)

    assert cast_to(tensor, torch.float32, tensor.device) is tensor


def test_cast_to_converts_the_dtype_and_passes_none_through():
    tensor = torch.randn(3)
    out = cast_to(tensor, torch.float16, tensor.device)
    assert out.dtype == torch.float16
    assert out is not tensor
    assert cast_to(None, torch.float32, torch.device("cpu")) is None


def test_cast_bias_weight_takes_dtype_and_device_from_the_activation():
    lin = manual_cast.Linear(4, 3, bias=True)
    lin.weight = nn.Parameter(torch.randn(3, 4, dtype=torch.float16))
    lin.bias = nn.Parameter(torch.randn(3, dtype=torch.float16))
    weight, bias = casting.cast_bias_weight(lin, torch.zeros(1, 4, dtype=torch.bfloat16))
    assert weight.dtype == torch.bfloat16
    assert bias.dtype == torch.bfloat16


def test_cast_bias_weight_without_a_bias_returns_none_for_it():
    lin = manual_cast.Linear(4, 3, bias=False)
    weight, bias = casting.cast_bias_weight(lin, torch.zeros(1, 4))
    assert weight.shape == (3, 4)
    assert bias is None


def test_the_streaming_flag_reaches_every_cast_of_a_cast_mode_forward(monkeypatch):
    seen = []
    real = casting.cast_to

    def spy(tensor, dtype, device, *, non_blocking=False):
        seen.append(non_blocking)
        return real(tensor, dtype, device, non_blocking=non_blocking)

    patch_ops_name(monkeypatch, "cast_to", spy)
    lin = manual_cast.Linear(4, 3, bias=True)
    lin.stream_non_blocking = True

    lin(torch.zeros(1, 4))

    assert seen == [True, True]


def test_a_resident_layer_casts_blocking_by_default(monkeypatch):
    seen = []
    real = casting.cast_to

    def spy(tensor, dtype, device, *, non_blocking=False):
        seen.append(non_blocking)
        return real(tensor, dtype, device, non_blocking=non_blocking)

    patch_ops_name(monkeypatch, "cast_to", spy)
    lin = manual_cast.Linear(4, 3, bias=True)

    lin(torch.zeros(1, 4))

    assert seen == [False, False]


def test_prepared_linear_yields_the_layer_itself():
    lin = disable_weight_init.Linear(4, 3)
    with lin.prepared_linear(torch.zeros(1, 4)) as prepared:
        assert prepared is lin
