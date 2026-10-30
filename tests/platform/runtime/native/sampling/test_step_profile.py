from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from src.platform.observability.profiling.profiler import GenerationProfiler
from src.platform.runtime.native.errors import SamplingCancelled
from src.platform.runtime.native.kernels import stream_wait
from src.platform.runtime.native.sampling import step_profile
from src.platform.runtime.native.sampling.conditioned import denoise_prenoised
from src.platform.runtime.native.sampling.denoise_loop import denoise
from src.platform.runtime.native.sampling.hooks import BaseStepHook, ProgressHook
from src.platform.runtime.native.sampling.step_profile import STEP_EVENT, StepPlanHook, with_step_profile

_SETTINGS = {"shift": 2.02, "guidance": None}


class _FakeEvent:
    def __init__(self, ms: float) -> None:
        self.ms = ms

    def elapsed_time(self, other: "_FakeEvent") -> float:
        return other.ms - self.ms


@pytest.fixture(autouse=True)
def _no_leftover_recorder():
    stream_wait.stop()
    yield
    stream_wait.stop()


@pytest.fixture
def cuda(monkeypatch):
    sync = Mock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    return sync


def _profiler(monkeypatch, *, active: bool):
    events: list = []
    fake = SimpleNamespace(active=active, mark=lambda event, **fields: events.append((event, fields)))
    monkeypatch.setattr(step_profile, "get_profiler", lambda: fake)
    return events


def _clock(monkeypatch, times):
    ticks = iter(times)
    monkeypatch.setattr(step_profile, "_clock", lambda: next(ticks))


def _steps(events):
    return [fields for event, fields in events if event == STEP_EVENT]


def _const_forward(x, sigma, cond):
    return torch.full_like(x, 0.5)


def _denoise(model_forward=_const_forward, *, steps=4, hooks=(), **kwargs):
    return denoise(
        model_forward, torch.zeros(1, 4, 4, 4), {"context": "c"}, None,
        steps=steps, sampling_settings=_SETTINGS, guidance_scale=0.0,
        seed_noise=torch.randn(1, 4, 4, 4), hooks=hooks, **kwargs,
    )


def test_profiling_on_emits_one_timed_step_event_per_step(monkeypatch, cuda):
    events = _profiler(monkeypatch, active=True)
    _clock(monkeypatch, [0.0, 1.5, 2.0, 4.0, 4.5, 7.5, 8.0, 12.0, 12.5])

    _denoise(steps=4)

    steps = _steps(events)
    assert [(s["step"], s["steps"]) for s in steps] == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert [s["step_s"] for s in steps] == pytest.approx([1.5, 2.0, 3.0, 4.0])
    assert cuda.call_count >= 2 * len(steps)
    assert all("attn" not in s and "cache_skip" not in s and "stream_wait_s" not in s for s in steps)


def test_profiling_off_emits_nothing_and_never_synchronizes(monkeypatch, cuda):
    events = _profiler(monkeypatch, active=False)
    seen_recorders = []

    def forward(x, sigma, cond):
        seen_recorders.append(stream_wait.recorder)
        return torch.full_like(x, 0.5)

    hooks = (ProgressHook(lambda *a: None),)
    assert with_step_profile(hooks) == hooks
    _denoise(forward, hooks=hooks)

    assert events == []
    cuda.assert_not_called()
    assert seen_recorders and all(r is None for r in seen_recorders)


def test_an_idle_generation_profiler_adds_no_hooks_and_no_syncs(monkeypatch, cuda):
    idle = GenerationProfiler()
    monkeypatch.setattr(step_profile, "get_profiler", lambda: idle)
    hooks = (ProgressHook(lambda *a: None),)

    assert idle.active is False
    assert with_step_profile(hooks) == hooks
    _denoise(hooks=hooks)
    cuda.assert_not_called()


def test_step_time_excludes_the_hooks_that_observe_the_step(monkeypatch, cuda):
    events = _profiler(monkeypatch, active=True)
    now = {"t": 0.0}
    monkeypatch.setattr(step_profile, "_clock", lambda: now["t"])

    def forward(x, sigma, cond):
        now["t"] += 1.0
        return torch.full_like(x, 0.5)

    class _SlowPreview(BaseStepHook):
        def on_step(self, *args):
            now["t"] += 100.0

    _denoise(forward, steps=3, hooks=(_SlowPreview(),))

    assert [s["step_s"] for s in _steps(events)] == pytest.approx([1.0, 1.0, 1.0])


def test_cache_skip_flag_follows_the_shared_step_cache(monkeypatch, cuda):
    events = _profiler(monkeypatch, active=True)
    probe = torch.ones(1, 4, 4, 4)

    def forward(x, sigma, cond):
        cache = cond.get("step_cache")
        if cache is not None and cache.should_skip(probe):
            return cache.record_skip()
        out = torch.full_like(x, 0.5)
        if cache is not None:
            cache.record_compute(probe, out)
        return out

    _denoise(forward, steps=6, step_cache_options={
        "rel_threshold": 0.5, "warmup_steps": 2, "max_consecutive_skips": 99,
    })

    assert [s["cache_skip"] for s in _steps(events)] == [False, False, True, True, True, False]


def test_step_plan_hook_fields_land_on_their_own_step(monkeypatch, cuda):
    events = _profiler(monkeypatch, active=True)
    plan = StepPlanHook(lambda i, n: {"attn": "sol-attn" if i < n - 2 else "dense"})

    _denoise(steps=4, hooks=(plan,))

    assert [s["attn"] for s in _steps(events)] == ["sol-attn", "sol-attn", "dense", "dense"]


def test_stream_wait_and_bytes_are_accounted_per_step(monkeypatch, cuda):
    events = _profiler(monkeypatch, active=True)
    waits_ms = iter([[300.0, 200.0], [1000.0], [50.0, 50.0, 25.0]])

    def forward(x, sigma, cond):
        recorder = stream_wait.recorder
        for ms in next(waits_ms):
            recorder.span(_FakeEvent(0.0), _FakeEvent(ms))
            recorder.add_bytes(1024 ** 3 // 4)
        return torch.full_like(x, 0.5)

    class _StreamsDuringHooks(BaseStepHook):
        def on_step(self, *args):
            stream_wait.recorder.span(_FakeEvent(0.0), _FakeEvent(9000.0))
            stream_wait.recorder.add_bytes(1024 ** 3)

    _denoise(forward, steps=3, hooks=(_StreamsDuringHooks(),))

    steps = _steps(events)
    assert [s["stream_wait_s"] for s in steps] == pytest.approx([0.5, 1.0, 0.125])
    assert [s["streamed_gb"] for s in steps] == pytest.approx([0.5, 0.25, 0.75])
    assert stream_wait.recorder is None


def test_recorder_is_released_when_sampling_is_cancelled(monkeypatch, cuda):
    _profiler(monkeypatch, active=True)
    calls = {"n": 0}

    def is_cancelled():
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(SamplingCancelled):
        _denoise(steps=5, is_cancelled=is_cancelled)

    assert stream_wait.recorder is None


def test_conditioned_entry_emits_step_events_too(monkeypatch, cuda):
    events = _profiler(monkeypatch, active=True)

    denoise_prenoised(
        _const_forward, torch.randn(1, 4, 4, 4), {"context": "c"}, None,
        steps=3, sampling_settings=_SETTINGS, guidance_scale=0.0,
        hooks=(StepPlanHook(lambda i, n: {"attn": "dense"}),),
    )

    assert [(s["step"], s["attn"]) for s in _steps(events)] == [(1, "dense"), (2, "dense"), (3, "dense")]
