from __future__ import annotations

import time

import torch

from src.platform.observability.profiling import get_profiler

from ..kernels import stream_wait
from .hooks import BaseStepHook

STEP_EVENT = "sampler.step"
_BYTES_PER_GB = 1024 ** 3


def _clock() -> float:
    return time.perf_counter()


def _synchronize() -> None:
    if not torch.cuda.is_available():
        return
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


class StepPlanHook(BaseStepHook):
    def __init__(self, fields) -> None:
        self._fields = fields

    def step_profile_fields(self, step_index: int, total_steps: int) -> dict:
        return self._fields(step_index, total_steps)


class _CacheSkipProbe:
    def __init__(self, skipped) -> None:
        self._skipped = skipped
        self._seen = skipped()

    def step_profile_fields(self, step_index: int, total_steps: int) -> dict:
        now = self._skipped()
        skipped, self._seen = now > self._seen, now
        return {"cache_skip": skipped}


def cache_skip_probe(cache) -> _CacheSkipProbe | None:
    if cache is None:
        return None
    counts = getattr(cache, "totals", None) or cache.stats
    return _CacheSkipProbe(lambda: counts()["skipped"])


class StepProfiler:
    def __init__(self, sources=(), profiler=None) -> None:
        self._sources = tuple(sources)
        self._profiler = profiler if profiler is not None else get_profiler()
        self._recorder = None
        self._started = None
        self.hooks = (_StepEdge(self), _StepResume(self))

    def begin(self) -> None:
        self._recorder = stream_wait.start()
        self.resume()

    def resume(self) -> None:
        _synchronize()
        if self._recorder is not None:
            self._recorder.drain()
        self._started = _clock()

    def close_step(self, step_index: int, total_steps: int) -> None:
        if self._started is None:
            return
        _synchronize()
        fields = {"step": int(step_index) + 1, "steps": int(total_steps), "step_s": _clock() - self._started}
        self._started = None
        for source in self._sources:
            fields.update(source.step_profile_fields(step_index, total_steps) or {})
        if self._recorder is not None:
            wait_s, nbytes = self._recorder.drain()
            if nbytes or wait_s:
                fields["stream_wait_s"] = wait_s
                fields["streamed_gb"] = nbytes / _BYTES_PER_GB
        self._profiler.mark(STEP_EVENT, **fields)

    def end(self) -> None:
        stream_wait.stop(self._recorder)
        self._recorder = None
        self._started = None


class _StepEdge(BaseStepHook):
    priority = 1_000_000

    def __init__(self, owner: StepProfiler) -> None:
        self._owner = owner

    def on_step(self, step_index, total_steps, x, sigma, denoised_x0) -> None:
        self._owner.close_step(step_index, total_steps)


class _StepResume(BaseStepHook):
    priority = -1_000_000

    def __init__(self, owner: StepProfiler) -> None:
        self._owner = owner

    def on_start(self, total_steps: int) -> None:
        self._owner.begin()

    def on_step(self, step_index, total_steps, x, sigma, denoised_x0) -> None:
        self._owner.resume()

    def on_end(self) -> None:
        self._owner.end()


def with_step_profile(hooks, *plans) -> tuple:
    profiler = get_profiler()
    if not profiler.active:
        return tuple(hooks)
    sources = [
        plan for plan in (*hooks, *plans)
        if plan is not None and callable(getattr(plan, "step_profile_fields", None))
    ]
    return tuple(hooks) + StepProfiler(sources, profiler).hooks
