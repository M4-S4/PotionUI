from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.platform.runtime.native.lora import apply_loras
from src.platform.runtime.native.lora.key_mapping import LoraDelta
from src.platform.runtime.native.ops.model_tools.delta import compute_delta

from .lora_fixtures import cpu_only


def _plain(out=6, inf=5, rank=3, alpha=6.0, scale=1.0, seed=0, target_slice=None):
    gen = torch.Generator().manual_seed(seed)
    up = torch.randn(out, rank, generator=gen)
    down = torch.randn(rank, inf, generator=gen)
    return LoraDelta(down=down, up=up, alpha=alpha, scale=scale, target_slice=target_slice)


def test_plain_delta_scales_by_strength_alpha_over_rank():
    d = _plain(alpha=6.0, scale=0.5)
    expected = (0.5 * 6.0 / 3) * (d.up @ d.down)
    assert torch.allclose(compute_delta([d], (6, 5)), expected, atol=1e-6)


def test_alpha_equal_rank_means_unit_scale():
    d = _plain(alpha=3.0, scale=1.0)
    assert torch.allclose(compute_delta([d], (6, 5)), d.up @ d.down, atol=1e-6)


def test_lokr_is_kron_w1_then_w2_times_scale_and_alpha():
    gen = torch.Generator().manual_seed(1)
    w1 = torch.randn(2, 3, generator=gen)
    w2 = torch.randn(4, 5, generator=gen)
    d = LoraDelta(down=w2, up=w1, alpha=0.25, scale=2.0, kron=True)
    out = compute_delta([d], (8, 15))
    assert torch.allclose(out, torch.kron(w1, w2) * 0.5, atol=1e-6)
    assert not torch.allclose(out, torch.kron(w2, w1).reshape(8, 15) * 0.5, atol=1e-3)


def test_sliced_delta_lands_in_its_rows_only():
    d = _plain(out=2, inf=5, rank=2, alpha=2.0, target_slice=(0, 4, 2))
    out = compute_delta([d], (8, 5))
    assert torch.count_nonzero(out[:4]) == 0 and torch.count_nonzero(out[6:]) == 0
    assert torch.allclose(out[4:6], d.up @ d.down, atol=1e-6)


def test_deltas_accumulate():
    a, b = _plain(seed=1), _plain(seed=2)
    both = compute_delta([a, b], (6, 5))
    assert torch.allclose(both, compute_delta([a], (6, 5)) + compute_delta([b], (6, 5)), atol=1e-6)


@pytest.mark.parametrize("kron", [False, True])
def test_runtime_in_place_equals_compute_delta_bit_for_bit(kron, cpu_only):
    linear = nn.Linear(15 if kron else 5, 8 if kron else 6, bias=False)
    base = linear.weight.detach().clone()
    gen = torch.Generator().manual_seed(3)
    if kron:
        sd = {
            "layer.lokr_w1": torch.randn(2, 3, generator=gen),
            "layer.lokr_w2_a": torch.randn(4, 2, generator=gen),
            "layer.lokr_w2_b": torch.randn(2, 5, generator=gen),
            "layer.alpha": torch.tensor(1.0),
        }
    else:
        sd = {
            "layer.lora_up.weight": torch.randn(6, 3, generator=gen),
            "layer.lora_down.weight": torch.randn(3, 5, generator=gen),
            "layer.alpha": torch.tensor(1.5),
        }

    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = linear

    holder = Holder()
    from src.platform.runtime.native.lora.key_mapping import map_lora_keys

    mapped, unmatched = map_lora_keys(sd, holder)
    assert not unmatched and list(mapped) == ["layer.weight"]
    apply_loras(holder, [(sd, 0.8)])
    scaled = [
        LoraDelta(down=d.down, up=d.up, alpha=d.alpha, scale=d.scale * 0.8, kron=d.kron, kron_factors=d.kron_factors)
        for d in mapped["layer.weight"]
    ]
    expected = base + compute_delta(scaled, tuple(base.shape))
    assert torch.equal(holder.layer.weight.detach(), expected)
