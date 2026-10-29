from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from src.platform.runtime.native.kernels import cast_to, stream_wait
from src.platform.runtime.native.kernels.stream_wait import StreamWaitRecorder


class _TimedEvent:
    def __init__(self, clock: list) -> None:
        self._clock = clock
        self.ms = None
        self.stream = None

    def record(self, stream=None) -> None:
        self.stream = stream
        self.ms = self._clock[0]

    def elapsed_time(self, other: "_TimedEvent") -> float:
        return other.ms - self.ms


@pytest.fixture(autouse=True)
def _no_leftover_recorder():
    stream_wait.stop()
    yield
    stream_wait.stop()


def _source(nbytes: int, copy_ms: float, clock: list):
    tensor = Mock()
    tensor.numel.return_value = nbytes
    tensor.element_size.return_value = 1

    def to(**kwargs):
        clock[0] += copy_ms
        return "copied"

    tensor.to.side_effect = to
    return tensor


def test_copies_are_bracketed_on_the_target_stream_and_sum_per_drain():
    clock = [0.0]
    streams = []
    recorder = StreamWaitRecorder(
        new_event=lambda: _TimedEvent(clock), stream_for=lambda device: streams.append(device) or "compute",
    )

    for nbytes, copy_ms in ((100, 40.0), (300, 60.0), (600, 150.0)):
        clock[0] += 5.0
        assert recorder.copy(_source(nbytes, copy_ms, clock), torch.bfloat16, "cuda:0", True) == "copied"

    wait_s, nbytes = recorder.drain()
    assert wait_s == pytest.approx(0.25)
    assert nbytes == 1000
    assert streams == ["cuda:0"] * 3
    assert recorder.drain() == (0.0, 0)


def test_a_copy_to_a_non_cuda_device_is_not_timed():
    recorder = StreamWaitRecorder(new_event=Mock(side_effect=AssertionError("no event")), stream_for=Mock())
    out = recorder.copy(torch.ones(4), torch.float16, "cpu", False)
    assert out.dtype == torch.float16
    assert recorder.drain() == (0.0, 0)


def test_spans_stop_growing_at_the_cap(monkeypatch):
    monkeypatch.setattr(stream_wait, "_MAX_SPANS", 2)
    clock = [0.0]
    recorder = StreamWaitRecorder(new_event=lambda: _TimedEvent(clock), stream_for=lambda device: None)
    for _ in range(5):
        recorder.copy(_source(10, 1.0, clock), torch.bfloat16, "cuda", True)
    wait_s, nbytes = recorder.drain()
    assert wait_s == pytest.approx(0.002)
    assert nbytes == 20


def test_cast_to_routes_a_host_tensor_through_the_active_recorder(monkeypatch):
    fake = Mock()
    fake.copy.return_value = "streamed"
    monkeypatch.setattr(stream_wait, "recorder", fake)
    host = torch.ones(3)

    assert cast_to(host, torch.bfloat16, torch.device("cuda:0"), non_blocking=True) == "streamed"
    fake.copy.assert_called_once_with(host, torch.bfloat16, torch.device("cuda:0"), True)


def test_cast_to_without_a_recorder_never_touches_cuda_events(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", Mock(side_effect=AssertionError("no event")))
    out = cast_to(torch.ones(3), torch.float16, torch.device("cpu"))
    assert out.dtype == torch.float16
    assert stream_wait.recorder is None
