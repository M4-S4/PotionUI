"""FBCache integration on the Wan arch (seam in _forward_orig; SLG interaction)."""

from __future__ import annotations

import torch

from src.platform.runtime.native.sampling.step_cache import FirstBlockCache

from .test_wan_model import TINY_T2V, _build

_T = torch.tensor([0.5])


def _fwd(m, x, ctx, **kw):
    with torch.no_grad():
        return m(x, _T, ctx, **kw)


def test_skip_layers_pass_never_touches_cache():
    # A degraded (SLG) forward must never read or write the cache — the arch
    # guards it even when a cache is handed in alongside skip_layers.
    m = _build(TINY_T2V)
    x = torch.randn(1, 16, 4, 16, 16)
    ctx = torch.randn(1, 12, 32)
    cache = FirstBlockCache(rel_threshold=0.5, warmup_steps=0)
    _fwd(m, x, ctx, step_cache=cache)  # computed=1
    assert cache.stats() == {"computed": 1, "skipped": 0}
    # identical inputs would normally skip; with skip_layers set the cache is
    # bypassed entirely, so counters are unchanged.
    _fwd(m, x, ctx, step_cache=cache, skip_layers={0})
    assert cache.stats() == {"computed": 1, "skipped": 0}

