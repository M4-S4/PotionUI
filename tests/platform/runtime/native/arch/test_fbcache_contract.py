from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pytest
import torch

from src.platform.runtime.native.arch.z_image.model import ZImageDiT
from src.platform.runtime.native.sampling.step_cache import FirstBlockCache
from src.platform.runtime.native.kernels import pick_operations

from . import test_anima_model, test_flux_model, test_krea2_model, test_qwen_image21_model
from . import test_qwen_image_model, test_wan_model

_T = torch.tensor([0.5])

ZTINY = {
    "image_model": "lumina2", "in_channels": 16, "dim": 256, "cap_feat_dim": 8,
    "n_layers": 2, "n_refiner_layers": 1, "n_heads": 4, "n_kv_heads": 4,
    "intermediate_size": 128, "axes_dims": (32, 16, 16), "patch_size": 2,
}


def _build_z_image() -> ZImageDiT:
    m = ZImageDiT.from_config(ZTINY, pick_operations(torch.float32, torch.float32))
    sd = {}
    for k, v in m.state_dict().items():
        if not v.is_floating_point():
            sd[k] = v.clone()
        elif "norm" in k and (k.endswith(".weight") or k.endswith(".scale")):
            sd[k] = torch.ones_like(v)
        else:
            sd[k] = torch.randn_like(v) * 0.02
    m.load_state_dict(sd)
    return m.eval()


@dataclass(frozen=True)
class FbcacheArch:
    id: str
    build: Callable[[], torch.nn.Module]
    latent_shape: tuple[int, ...]
    resized_shape: tuple[int, ...]
    make_context: Callable[[], Any]
    forward: Callable[..., torch.Tensor]
    guarded_block: Callable[[torch.nn.Module], torch.nn.Module]


def _call(m, x, t, *args, **kw):
    with torch.no_grad():
        return m(x, t, *args, **kw)


def _anima_forward(m, x, ctx, **kw):
    text, ids, weights = ctx
    return _call(m, x, torch.tensor([0.7]), text, t5xxl_ids=ids, t5xxl_weights=weights, **kw)


def _anima_context():
    return torch.randn(1, 5, 16), torch.randint(0, 100, (1, 7)), torch.ones(1, 7)


def _flux_forward(m, x, ctx, **kw):
    return _call(m, x, _T, ctx, **kw)


def _masked_forward(m, x, ctx, **kw):
    return _call(m, x, _T, ctx, attention_mask=torch.ones(1, 5, dtype=torch.long), **kw)


ARCHS = [
    FbcacheArch(
        "anima", lambda: test_anima_model._build_ready(test_anima_model.TINY),
        (1, 8, 1, 16, 16), (1, 8, 1, 16, 32), _anima_context, _anima_forward, lambda m: m.blocks[-1],
    ),
    FbcacheArch(
        "flux", lambda: test_flux_model._build_ready(test_flux_model.TINY_FLUX2),
        (1, 16, 16, 16), (1, 16, 24, 16), lambda: torch.randn(1, 7, 32), _flux_forward,
        lambda m: m.single_blocks[-1],
    ),
    FbcacheArch(
        "krea2", lambda: test_krea2_model._build_ready(dict(test_krea2_model.TINY, layers=2)),
        (1, 4, 8, 8), (1, 4, 8, 16), lambda: torch.randn(1, 5, 3, 16), _masked_forward,
        lambda m: m.blocks[-1],
    ),
    FbcacheArch(
        "qwen_image", lambda: test_qwen_image_model._build_ready(test_qwen_image_model.TINY),
        (1, 4, 1, 8, 8), (1, 4, 1, 8, 12), lambda: torch.randn(1, 5, 12), _masked_forward,
        lambda m: m.transformer_blocks[-1],
    ),
    FbcacheArch(
        "qwen_image21", lambda: test_qwen_image21_model._build_ready(test_qwen_image21_model.TINY),
        (1, 8, 4, 4), (1, 8, 4, 8), lambda: torch.randn(1, 5, 12), _flux_forward,
        lambda m: m.transformer_blocks[-1],
    ),
    FbcacheArch(
        "wan", lambda: test_wan_model._build(test_wan_model.TINY_T2V),
        (1, 16, 4, 16, 16), (1, 16, 4, 16, 32), lambda: torch.randn(1, 12, 32), _flux_forward,
        lambda m: m.blocks[-1],
    ),
    FbcacheArch(
        "z_image", _build_z_image,
        (1, 16, 16, 16), (1, 16, 16, 32), lambda: torch.randn(1, 4, 8), _flux_forward,
        lambda m: m.layers[-1],
    ),
]

archs = pytest.mark.parametrize("arch", ARCHS, ids=lambda a: a.id)


@archs
def test_step_cache_none_is_byte_identical(arch):
    m = arch.build()
    x = torch.randn(*arch.latent_shape)
    ctx = arch.make_context()
    base = arch.forward(m, x, ctx)
    assert torch.equal(base, arch.forward(m, x, ctx, step_cache=None))


@archs
def test_identical_inputs_skip_returns_cached_output(arch):
    m = arch.build()
    x = torch.randn(*arch.latent_shape)
    ctx = arch.make_context()
    cache = FirstBlockCache(rel_threshold=0.5, warmup_steps=0)
    first = arch.forward(m, x, ctx, step_cache=cache)
    assert cache.stats() == {"computed": 1, "skipped": 0}
    second = arch.forward(m, x, ctx, step_cache=cache)
    assert cache.stats() == {"computed": 1, "skipped": 1}
    assert torch.equal(second, first)


@archs
def test_different_inputs_do_not_skip(arch):
    m = arch.build()
    ctx = arch.make_context()
    cache = FirstBlockCache(rel_threshold=0.01, warmup_steps=0)
    arch.forward(m, torch.randn(*arch.latent_shape), ctx, step_cache=cache)
    arch.forward(m, torch.randn(*arch.latent_shape) * 5.0, ctx, step_cache=cache)
    assert cache.stats() == {"computed": 2, "skipped": 0}


@archs
def test_skip_avoids_later_blocks(arch):
    m = arch.build()
    x = torch.randn(*arch.latent_shape)
    ctx = arch.make_context()
    cache = FirstBlockCache(rel_threshold=0.5, warmup_steps=0)
    last = arch.guarded_block(m)
    original = last.forward
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return original(*a, **k)

    last.forward = counting
    arch.forward(m, x, ctx, step_cache=cache)
    computed_calls = calls["n"]
    assert computed_calls > 0
    arch.forward(m, x, ctx, step_cache=cache)
    assert calls["n"] == computed_calls


@archs
def test_resolution_change_forces_compute(arch):
    m = arch.build()
    ctx = arch.make_context()
    cache = FirstBlockCache(rel_threshold=0.9, warmup_steps=0)
    arch.forward(m, torch.randn(*arch.latent_shape), ctx, step_cache=cache)
    out = arch.forward(m, torch.randn(*arch.resized_shape), ctx, step_cache=cache)
    assert cache.stats()["skipped"] == 0
    assert out.shape == arch.resized_shape
