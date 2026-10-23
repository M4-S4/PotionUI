"""FBCache integration on the Flux arch: probe/skip seam + byte-identical default.

Uses the same tiny-config builder as test_flux_model.py.
"""

from __future__ import annotations

import torch

from src.platform.runtime.native.sampling.step_cache import FirstBlockCache

from .test_flux_model import TINY_FLUX2, _build_ready


def _forward(m, x, t, ctx, **kw):
    with torch.no_grad():
        return m(x, t, ctx, **kw)


# --- (a) None -> byte-identical to pre-change path ------------------------

# --- (b) identical inputs -> second call skips + returns cached output ----

# --- (c) different-enough inputs -> no skip ------------------------------

# --- (d) a skip actually avoids running the later blocks -----------------

# --- ControlNet payload bypasses the cache entirely (roadmap S8/#8) -------

def test_control_payload_bypasses_step_cache():
    """FBCache's probe is captured at block 0; a ControlNet residual applied
    at a LATER block (control["input"][1:], control["output"]) is invisible
    to it. Rather than risk reusing a stale cached output under a changed
    residual, a control payload must bypass the cache entirely: the passed-in
    ``FirstBlockCache`` is never consulted or updated at all (stats stay at
    zero), even across repeated identical-input calls with a threshold that
    would otherwise always skip."""
    m = _build_ready(TINY_FLUX2)
    x = torch.randn(1, 16, 16, 16)
    t = torch.tensor([0.5])
    ctx = torch.randn(1, 7, 32)
    cache = FirstBlockCache(rel_threshold=0.99, warmup_steps=0)
    control = {}  # presence alone (not its content) triggers the bypass

    out1 = _forward(m, x, t, ctx, step_cache=cache, control=control)
    out2 = _forward(m, x, t, ctx, step_cache=cache, control=control)
    assert cache.stats() == {"computed": 0, "skipped": 0}
    # Real (non-cached) forwards both times, so the outputs still agree given
    # identical deterministic inputs — the bypass isn't a behavior change, only
    # a caching one.
    assert torch.equal(out1, out2)


def test_no_control_still_skips_as_before():
    """Sanity check that the control guard doesn't break the ordinary
    no-ControlNet path — same setup as test_control_payload_bypasses_step_cache
    but without ``control``, a skip must still happen."""
    m = _build_ready(TINY_FLUX2)
    x = torch.randn(1, 16, 16, 16)
    t = torch.tensor([0.5])
    ctx = torch.randn(1, 7, 32)
    cache = FirstBlockCache(rel_threshold=0.99, warmup_steps=0)

    _forward(m, x, t, ctx, step_cache=cache)
    _forward(m, x, t, ctx, step_cache=cache)
    assert cache.stats() == {"computed": 1, "skipped": 1}


# --- shape change (resolution) resets, does not crash --------------------
