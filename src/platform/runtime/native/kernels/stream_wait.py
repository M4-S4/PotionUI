from __future__ import annotations

import torch

_MAX_SPANS = 1 << 16

recorder: "StreamWaitRecorder | None" = None


def _timing_event():
    return torch.cuda.Event(enable_timing=True)


def _stream_for(device):
    return torch.cuda.current_stream(device)


class StreamWaitRecorder:
    def __init__(self, new_event=_timing_event, stream_for=_stream_for) -> None:
        self._new_event = new_event
        self._stream_for = stream_for
        self._spans: list = []
        self._bytes = 0

    @property
    def full(self) -> bool:
        return len(self._spans) >= _MAX_SPANS

    def mark(self, stream):
        event = self._new_event()
        event.record(stream)
        return event

    def span(self, start, end) -> None:
        if not self.full:
            self._spans.append((start, end))

    def add_bytes(self, nbytes: int) -> None:
        self._bytes += int(nbytes)

    def copy(self, tensor: torch.Tensor, dtype: torch.dtype, device, non_blocking: bool) -> torch.Tensor:
        if self.full or torch.device(device).type != "cuda":
            return tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)
        stream = self._stream_for(device)
        start = self.mark(stream)
        out = tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)
        self.span(start, self.mark(stream))
        self.add_bytes(tensor.numel() * tensor.element_size())
        return out

    def drain(self) -> tuple[float, int]:
        spans, nbytes = self._spans, self._bytes
        self._spans, self._bytes = [], 0
        wait_ms = sum(start.elapsed_time(end) for start, end in spans)
        return wait_ms / 1000.0, nbytes


def start(**kwargs) -> StreamWaitRecorder:
    global recorder
    recorder = StreamWaitRecorder(**kwargs)
    return recorder


def stop(owner: "StreamWaitRecorder | None" = None) -> None:
    global recorder
    if owner is None or recorder is owner:
        recorder = None
