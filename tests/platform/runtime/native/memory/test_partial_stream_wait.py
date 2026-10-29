from __future__ import annotations

import pytest
import torch

from src.platform.runtime.native.kernels import stream_wait
from src.platform.runtime.native.memory.partial import LayerPrefetcher
from tests.platform.runtime.native.memory.test_partial_prefetch import _Tiny, _install_fake_cuda, _streamed_leaves


class _ScriptedEvent:
    def __init__(self, timeline: list, log: list) -> None:
        self._timeline = timeline
        self._log = log
        self.ms = None

    def record(self, stream=None) -> None:
        self.ms = self._timeline.pop(0)
        self._log.append(("mark", getattr(stream, "name", stream)))

    def elapsed_time(self, other: "_ScriptedEvent") -> float:
        return other.ms - self.ms


@pytest.fixture(autouse=True)
def _no_leftover_recorder():
    stream_wait.stop()
    yield
    stream_wait.stop()


def test_prefetch_waits_are_bracketed_on_the_compute_stream_and_summed(monkeypatch):
    fake = _install_fake_cuda(monkeypatch)
    model = _Tiny(n_linear=4, dim=8)
    tokens = torch.randint(0, 16, (2, 5))
    prefetcher = LayerPrefetcher(model, _streamed_leaves(model), "cuda:0", prefetch_depth=1)
    model(tokens)

    timeline = [0.0, 30.0, 100.0, 100.0, 200.0, 245.0, 300.0, 300.0]
    marks: list = []
    recorder = stream_wait.start(new_event=lambda: _ScriptedEvent(timeline, marks), stream_for=None)
    model(tokens)

    assert len(fake["compute"].waited) == 4
    assert marks == [("mark", "compute")] * 8
    staged = sum(leaf.weight.numel() * leaf.weight.element_size() for leaf in _streamed_leaves(model)[1:])
    assert recorder.drain() == (pytest.approx(0.075), staged)
    prefetcher.teardown()


def test_prefetch_records_nothing_without_an_active_recorder(monkeypatch):
    fake = _install_fake_cuda(monkeypatch)
    model = _Tiny(n_linear=4, dim=8)
    tokens = torch.randint(0, 16, (2, 5))
    prefetcher = LayerPrefetcher(model, _streamed_leaves(model), "cuda:0", prefetch_depth=1)
    model(tokens)
    events_before = len(fake["events"])
    model(tokens)

    assert fake["compute"].waited
    assert len(fake["events"]) - events_before == len(fake["compute"].waited)
    assert stream_wait.recorder is None
    prefetcher.teardown()
