from types import SimpleNamespace
from unittest.mock import MagicMock

from src.features.chat.dto import MessageResponse
from src.features.chat.thinking import (
    ReasoningCapture,
    apply_thinking_visibility,
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


def message(metadata, msg_id="m1"):
    return MessageResponse(id=msg_id, session_id="s", role="assistant", content="hi", metadata=metadata)


STORED = {"segments": [{"text": "x", "duration_ms": 1}]}


def test_hidden_thinking_is_stripped_but_other_metadata_survives():
    original = message({"thinking": STORED, "tokens_used": 5})

    [redacted] = apply_thinking_visibility([original], llm_service(False), "cfg")

    assert redacted.metadata == {"tokens_used": 5}
    assert "thinking" in original.metadata


def test_the_trace_marker_is_stripped_with_the_thinking():
    original = message({
        "thinking": STORED,
        "behavior_trace": {"version": 1, "thinking_visible": True, "mode": "m"},
    })

    [redacted] = apply_thinking_visibility([original], llm_service(False), "cfg")

    assert redacted.metadata == {"behavior_trace": {"version": 1, "mode": "m"}}
    assert original.metadata["behavior_trace"]["thinking_visible"] is True


def test_visible_thinking_is_returned_unchanged():
    messages = [message({"thinking": STORED})]

    assert apply_thinking_visibility(messages, llm_service(True), "cfg") is messages


def test_thinking_reappears_when_the_knob_is_turned_back_on():
    messages = [message({"thinking": STORED})]

    hidden = apply_thinking_visibility(messages, llm_service(False), "cfg")
    shown = apply_thinking_visibility(messages, llm_service(True), "cfg")

    assert "thinking" not in hidden[0].metadata
    assert shown[0].metadata["thinking"] == STORED


def test_messages_without_thinking_are_returned_as_they_are():
    plain = message({"tokens_used": 5})
    bare = message(None, msg_id="m2")
    messages = [plain, bare]

    assert apply_thinking_visibility(messages, llm_service(False), "cfg") is messages


def test_the_config_is_not_even_looked_up_when_no_message_carries_thinking():
    service = MagicMock()

    apply_thinking_visibility([message({"tokens_used": 5})], service, "cfg")

    service.repository.get_configuration.assert_not_called()


def test_a_session_with_no_config_hides_stored_thinking():
    [redacted] = apply_thinking_visibility([message({"thinking": STORED})], llm_service(True), None)

    assert "thinking" not in redacted.metadata
