from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.platform.runtime.native.kernels import stream_wait
from src.platform.runtime.native.sampling import denoise_prenoised, step_profile
from src.platform.runtime.native.sampling.step_profile import STEP_EVENT
from tests.pipelines.pipes.generator.video_ltx.test_video_ltx_generator import (
    _MAIN,
    _RecordingProgress,
    _pipe,
    _pipe_input,
)


@pytest.fixture(autouse=True)
def _no_leftover_recorder():
    stream_wait.stop()
    yield
    stream_wait.stop()


def _profiler(monkeypatch, *, active: bool = True):
    events: list = []
    fake = SimpleNamespace(active=active, mark=lambda event, **fields: events.append((event, fields)))
    monkeypatch.setattr(step_profile, "get_profiler", lambda: fake)
    return events


def _run_through_the_shared_loop(monkeypatch, **config):
    probe = torch.ones(1, 4)
    hooks_seen: list = []

    def cache_driving_forward(x, sigma, cond):
        cache = cond.get("step_cache")
        if cache is not None and cache.should_skip(probe):
            return cache.record_skip()
        out = torch.zeros_like(x)
        if cache is not None:
            cache.record_compute(probe, out)
        return out

    def shared_loop(fwd, x, cond, uncond, **kw):
        hooks_seen.extend(kw["hooks"])
        return denoise_prenoised(cache_driving_forward, x, cond, uncond, **kw)

    monkeypatch.setattr(f"{_MAIN}.denoise_prenoised", shared_loop)
    monkeypatch.setattr(f"{_MAIN}.decode_ltx_video",
                        lambda vae, device, latent, seed, keep_resident=(): np.zeros((9, 2, 2, 3), np.uint8))
    monkeypatch.setattr(f"{_MAIN}.encode_frames_to_mp4", lambda frames, path, fps, audio=None: path)
    monkeypatch.setattr(f"{_MAIN}.restore_dit_best_effort", lambda *a: None)
    pipe = _pipe(steps=4, preview=False, **config)
    ctx = pipe.build_context(_pipe_input())
    pipe.generate_one(ctx, 0, 7, _RecordingProgress())
    return hooks_seen


def test_every_ltx_step_is_profiled_as_dense_attention(monkeypatch):
    events = _profiler(monkeypatch)
    _run_through_the_shared_loop(monkeypatch)

    steps = [fields for event, fields in events if event == STEP_EVENT]
    assert len(steps) == steps[0]["steps"] >= 2
    assert [s["step"] for s in steps] == list(range(1, len(steps) + 1))
    assert {s["attn"] for s in steps} == {"dense"}
    assert all("cache_skip" not in s for s in steps)


def test_ltx_step_cache_skips_reach_the_step_events(monkeypatch):
    events = _profiler(monkeypatch)
    _run_through_the_shared_loop(
        monkeypatch, step_cache_threshold=0.5, step_cache_warmup_steps=1, step_cache_max_skips=99,
    )

    steps = [fields for event, fields in events if event == STEP_EVENT]
    assert [s["cache_skip"] for s in steps] == [False] + [True] * (len(steps) - 2) + [False]
    assert {s["attn"] for s in steps} == {"dense"}


def test_the_ltx_step_plan_rides_the_hooks_it_hands_the_sampler(monkeypatch):
    _profiler(monkeypatch, active=False)
    hooks = _run_through_the_shared_loop(monkeypatch)

    plans = [h for h in hooks if callable(getattr(h, "step_profile_fields", None))]
    assert len(plans) == 1
    assert plans[0].step_profile_fields(0, 4) == {"attn": "dense"}
