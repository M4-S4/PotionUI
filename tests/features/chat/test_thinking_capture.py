from types import SimpleNamespace
from unittest.mock import MagicMock

from src.features.chat.thinking import (
    ReasoningCapture,
    config_shows_thinking,
    hide_thinking_in_event,
)
from src.features.llm.harness.events import ReasoningDelta


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def capture(enabled=True):
    clock = FakeClock()
    return ReasoningCapture(enabled, clock=clock), clock


def test_reasoning_deltas_stream_as_events_and_close_into_one_timed_segment():
    cap, clock = capture()

    first = cap.on_reasoning(ReasoningDelta(block=0, text="let me "))
    clock.advance(1.5)
    second = cap.on_reasoning(ReasoningDelta(block=0, text="think"))
    clock.advance(0.5)
    closed = cap.close()

    assert first + second == [
        {"event": "reasoning", "data": {"block": 0, "kind": "raw", "text": "let me "}},
        {"event": "reasoning", "data": {"block": 0, "kind": "raw", "text": "think"}},
    ]
    assert closed == [{"event": "reasoning_end", "data": {"block": 0, "duration_ms": 2000}}]
    assert cap.segments() == [{"text": "let me think", "duration_ms": 2000}]
    assert cap.close() == []


def test_a_new_block_closes_the_previous_segment():
    cap, clock = capture()

    cap.on_reasoning(ReasoningDelta(block=0, text="first"))
    clock.advance(1)
    events = cap.on_reasoning(ReasoningDelta(block=3, text="second", summary=True))
    cap.close()

    assert events[0] == {"event": "reasoning_end", "data": {"block": 0, "duration_ms": 1000}}
    assert events[1]["data"] == {"block": 3, "kind": "summary", "text": "second"}
    assert [segment["text"] for segment in cap.segments()] == ["first", "second"]


def test_empty_reasoning_is_reported_as_omitted_and_kept_out_of_the_segments():
    cap, _ = capture()

    events = cap.on_reasoning(ReasoningDelta(block=0, text=""))
    cap.close()

    assert events == [{"event": "reasoning", "data": {"block": 0, "kind": "omitted", "text": ""}}]
    assert cap.segments() == []


def test_a_disabled_capture_emits_and_records_nothing():
    cap, _ = capture(enabled=False)

    assert cap.on_reasoning(ReasoningDelta(block=0, text="secret")) == []
    assert cap.close() == []
    assert cap.segments() == []


def test_hidden_thinking_drops_reasoning_events_from_the_stream():
    assert hide_thinking_in_event({"event": "reasoning", "data": {"text": "x"}}) is None
    assert hide_thinking_in_event({"event": "reasoning_end", "data": {"block": 0}}) is None
    assert hide_thinking_in_event({"event": "token", "data": {"content": "x"}}) == {"event": "token", "data": {"content": "x"}}


def llm_service(show_thinking=None, raises=False):
    def get_configuration(config_id):
        if raises:
            raise RuntimeError("boom")
        return SimpleNamespace(show_thinking=show_thinking)

    return SimpleNamespace(repository=SimpleNamespace(get_configuration=get_configuration))


def test_thinking_is_visible_only_when_the_config_says_so():
    assert config_shows_thinking(llm_service(True), "cfg") is True
    assert config_shows_thinking(llm_service(False), "cfg") is False


def test_thinking_is_hidden_for_a_missing_config_or_lookup_failure():
    assert config_shows_thinking(llm_service(True), None) is False
    assert config_shows_thinking(llm_service(raises=True), "cfg") is False
    assert config_shows_thinking(MagicMock(), "cfg") is False
