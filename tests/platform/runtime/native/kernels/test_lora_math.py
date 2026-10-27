from __future__ import annotations

import pytest
import torch

from src.platform.runtime.native.kernels import lora_math
from src.platform.runtime.native.lora.key_mapping import LoraDelta
from src.platform.settings.runtime_flags import runtime_flag_values
from tests.fixtures.kernel_ops import patch_ops_name

RANK, IN, OUT = 4, 8, 12
CPU = torch.device("cpu")


class _Owner:
    pass


def _delta(seed: int, dtype: torch.dtype = torch.bfloat16, scale: float = 1.0) -> LoraDelta:
    g = torch.Generator().manual_seed(seed)
    return LoraDelta(
        down=torch.randn(RANK, IN, generator=g).to(dtype),
        up=torch.randn(OUT, RANK, generator=g).to(dtype),
        alpha=2.0,
        scale=scale,
    )


def test_the_fused_cache_is_rebuilt_in_the_new_compute_dtype():
    owner, deltas = _Owner(), [_delta(1)]
    bf16 = lora_math._cached_lora_fused(owner, deltas, torch.bfloat16, CPU)

    fp32 = lora_math._cached_lora_fused(owner, deltas, torch.float32, CPU)

    assert fp32 is not bf16
    assert fp32[0][1].dtype is torch.float32
    assert fp32[0][2].dtype is torch.float32
    assert lora_math._cached_lora_fused(owner, deltas, torch.bfloat16, CPU)[0][1].dtype is torch.bfloat16


@pytest.mark.parametrize("value", [True, False])
def test_the_fused_switch_follows_the_live_flag_at_every_call(monkeypatch, value):
    monkeypatch.setitem(runtime_flag_values(), "native_lora_fused", not value)
    assert lora_math._lora_fused_enabled() is (not value)

    monkeypatch.setitem(runtime_flag_values(), "native_lora_fused", value)

    assert lora_math._lora_fused_enabled() is value


def test_a_weight_side_lokr_delta_adds_its_kron_product_to_the_weight_through_the_patched_apply(monkeypatch):
    monkeypatch.setitem(runtime_flag_values(), "native_lora_fused", False)
    g = torch.Generator().manual_seed(5)
    up, down = torch.randn(3, 3, generator=g), torch.randn(4, 4, generator=g)
    lokr = LoraDelta(down=down, up=up, alpha=1.5, scale=0.7, kron=True)
    calls = []
    real = lora_math.apply_lora_deltas

    def spy(weight, deltas):
        calls.append(len(deltas))
        return real(weight, deltas)

    patch_ops_name(monkeypatch, "apply_lora_deltas", spy)
    x, weight = torch.randn(2, 12), torch.randn(12, 12)

    out = lora_math.linear_with_lora_deltas(x, weight, None, [lokr], 12)

    expected = torch.nn.functional.linear(x, weight + torch.kron(up, down).reshape(weight.shape) * 0.7 * 1.5)
    assert calls == [1]
    assert torch.allclose(out, expected, atol=1e-5)
