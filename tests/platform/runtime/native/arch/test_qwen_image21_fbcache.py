"""FBCache integration on the Qwen-Image-2.1 single-stream arch: probe/skip
seam, target-rows-only probe, and byte-identical default."""

from __future__ import annotations

import torch

from src.platform.runtime.native.sampling.step_cache import FirstBlockCache

from .test_qwen_image21_model import TINY, _build_ready

_T = torch.tensor([0.5])


def _ctx():
    return torch.randn(1, 5, 12)


def _fwd(m, x, ctx, **kw):
    with torch.no_grad():
        return m(x, _T, ctx, **kw)


def test_skip_preserves_input_latent_rank():
    m = _build_ready(TINY)
    x = torch.randn(1, 8, 1, 4, 4)  # 5-D latent (extra singleton axis)
    ctx = _ctx()
    cache = FirstBlockCache(rel_threshold=0.5, warmup_steps=0)
    first = _fwd(m, x, ctx, step_cache=cache)
    second = _fwd(m, x, ctx, step_cache=cache)
    assert cache.stats()["skipped"] == 1
    assert second.shape == x.shape == first.shape


def test_probe_is_target_rows_only_not_prefix():
    """The probe fed to the cache must be block-0's TARGET rows
    (hidden_states[:, prefix_len:]) only -- text ("prefix") rows are pinned
    to t=0 every step and would dilute the drift signal if included."""
    m = _build_ready(TINY)
    x = torch.randn(1, 8, 4, 4)  # target: 4*4 = 16 tokens
    ctx = _ctx()  # text: 5 tokens
    cache = FirstBlockCache(rel_threshold=0.0, warmup_steps=0)  # never skips; just records
    _fwd(m, x, ctx, step_cache=cache)
    assert cache.prev_probe.shape == (1, 16, m.config.inner_dim)


def test_with_ref_latents_probe_excludes_prefix_and_reference_rows():
    m = _build_ready(TINY)
    x = torch.randn(1, 8, 4, 4)  # target: 16 tokens
    ref = torch.randn(1, 8, 4, 4)  # reference block: 16 tokens (prefix)
    ctx = _ctx()  # text: 5 tokens (prefix)
    cache = FirstBlockCache(rel_threshold=0.0, warmup_steps=0)
    _fwd(m, x, ctx, step_cache=cache, ref_latents=[ref])
    assert cache.prev_probe.shape == (1, 16, m.config.inner_dim)
