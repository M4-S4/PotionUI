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


def test_the_fused_cache_is_reused_while_nothing_it_depends_on_changes():
    owner, deltas = _Owner(), [_delta(1)]

    first = lora_math._cached_lora_fused(owner, deltas, torch.bfloat16, CPU)
    second = lora_math._cached_lora_fused(owner, deltas, torch.bfloat16, CPU)

    assert second is first
    assert lora_math._LORA_FUSED_ATTR in owner.__dict__


def test_the_fused_cache_is_rebuilt_in_the_new_compute_dtype():
    owner, deltas = _Owner(), [_delta(1)]
    bf16 = lora_math._cached_lora_fused(owner, deltas, torch.bfloat16, CPU)

    fp32 = lora_math._cached_lora_fused(owner, deltas, torch.float32, CPU)

    assert fp32 is not bf16
    assert fp32[0][1].dtype is torch.float32
    assert fp32[0][2].dtype is torch.float32
    assert lora_math._cached_lora_fused(owner, deltas, torch.bfloat16, CPU)[0][1].dtype is torch.bfloat16


def test_the_fused_cache_is_rebuilt_when_a_strength_changes():
    owner, deltas = _Owner(), [_delta(1, scale=1.0)]
    before = lora_math._cached_lora_fused(owner, deltas, torch.bfloat16, CPU)
    assert before[0][3] == pytest.approx(1.0 * 2.0 / RANK)

    deltas[0].scale = 0.25
    after = lora_math._cached_lora_fused(owner, deltas, torch.bfloat16, CPU)

    assert after is not before
    assert after[0][3] == pytest.approx(0.25 * 2.0 / RANK)


def test_the_fused_cache_is_rebuilt_when_the_adapter_stack_changes():
    owner, deltas = _Owner(), [_delta(1)]
    single = lora_math._cached_lora_fused(owner, deltas, torch.bfloat16, CPU)

    stacked = lora_math._cached_lora_fused(owner, [*deltas, _delta(2)], torch.bfloat16, CPU)

    assert stacked is not single
    assert stacked[0][1].shape[0] == 2 * RANK


def test_invalidating_the_fused_cache_forces_a_rebuild():
    owner, deltas = _Owner(), [_delta(1)]
    built = lora_math._cached_lora_fused(owner, deltas, torch.bfloat16, CPU)

    lora_math.invalidate_lora_fused_cache(owner)

    assert lora_math._LORA_FUSED_ATTR not in owner.__dict__
    assert lora_math._cached_lora_fused(owner, deltas, torch.bfloat16, CPU) is not built


@pytest.mark.parametrize("value", [True, False])
def test_the_fused_switch_follows_the_live_flag_at_every_call(monkeypatch, value):
    monkeypatch.setitem(runtime_flag_values(), "native_lora_fused", not value)
    assert lora_math._lora_fused_enabled() is (not value)

    monkeypatch.setitem(runtime_flag_values(), "native_lora_fused", value)

    assert lora_math._lora_fused_enabled() is value


def test_a_weight_side_delta_reaches_the_module_level_apply_the_spies_patch(monkeypatch):
    monkeypatch.setitem(runtime_flag_values(), "native_lora_fused", False)
    g = torch.Generator().manual_seed(5)
    lokr = LoraDelta(
        down=torch.randn(4, 4, generator=g),
        up=torch.randn(3, 3, generator=g),
        alpha=1.0,
        kron=True,
    )
    calls = []
    real = lora_math.apply_lora_deltas

    def spy(weight, deltas):
        calls.append(len(deltas))
        return real(weight, deltas)

    patch_ops_name(monkeypatch, "apply_lora_deltas", spy)
    x, weight = torch.randn(2, 12), torch.randn(12, 12)

    out = lora_math.linear_with_lora_deltas(x, weight, None, [lokr], 12)

    assert calls == [1]
    assert torch.allclose(out, torch.nn.functional.linear(x, real(weight, [lokr])))
