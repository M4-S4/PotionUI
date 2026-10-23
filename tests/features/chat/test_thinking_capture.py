from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.features.chat.dto import MessageResponse
from src.features.chat.thinking import (
    CLOSED,
    TEXT,
    THINKING,
    Piece,
    ThinkingCapture,
    ThinkTagSplitter,
    apply_thinking_visibility,
    config_shows_thinking,
    visible_text,
)


def split(chunks):
    splitter = ThinkTagSplitter()
    pieces = []
    for chunk in chunks:
        pieces.extend(splitter.feed(chunk))
    pieces.extend(splitter.finish())
    return pieces


def joined(pieces, kind):
    return "".join(piece.text for piece in pieces if piece.kind == kind)


def kinds_collapsed(pieces):
    order = []
    for piece in pieces:
        if not order or order[-1] != piece.kind:
            order.append(piece.kind)
    return order


def every_split(text):
    yield [text]
    for cut in range(1, len(text)):
        yield [text[:cut], text[cut:]]
    yield list(text)


SAMPLES = [
    ("<think>reason</think>answer", "answer", "reason"),
    ("before<think>reason</think>after", "beforeafter", "reason"),
    ("<think>a</think>\n\nanswer", "answer", "a"),
    ("<thinking>deep</thinking>answer", "answer", "deep"),
    ("<thought>hm</thought>answer", "answer", "hm"),
    ("<THINK>loud</THINK>answer", "answer", "loud"),
    ('<think style="x">attr</think>answer', "answer", "attr"),
    ("<think>a</think>mid<think>b</think>end", "midend", "ab"),
    ("<think>a < b and c > d</think>x", "x", "a < b and c > d"),
    ("<think>keep</thin</think>x", "x", "keep</thin"),
    ("plain text only", "plain text only", ""),
    ("a < b and <b>bold</b>", "a < b and <b>bold</b>", ""),
    ("<thinker>no</thinker>", "<thinker>no</thinker>", ""),
    ("stray </think> close", "stray </think> close", ""),
    ("<think>İ</think>answer", "answer", "İ"),
    ("<think>aİb</think>x", "x", "aİb"),
    ("İ<think>İİ</think>İ", "İİ", "İİ"),
    ("<THINK>İ</THINK>x", "x", "İ"),
    ("<think>İİ</thin", "", "İİ</thin"),
    ("<think>a</think>İ<think>İ</think>b", "İb", "aİ"),
    ("<thİnk>x</thİnk>y", "<thİnk>x</thİnk>y", ""),
    ("<thin\u212a>x</thin\u212a>y", "<thin\u212a>x</thin\u212a>y", ""),
]


@pytest.mark.parametrize("raw,text,thinking", SAMPLES)
def test_a_stream_splits_the_same_whichever_way_it_is_chunked(raw, text, thinking):
    for chunks in every_split(raw):
        pieces = split(chunks)

        assert joined(pieces, TEXT) == text, chunks
        assert joined(pieces, THINKING) == thinking, chunks


def test_visible_text_survives_characters_whose_lowercase_is_longer():
    assert visible_text("<think>İ</think>answer") == "answer"
    assert visible_text("<think>İİİ</think>İanswer") == "İanswer"


def test_a_whole_block_yields_thinking_then_text():
    pieces = split(["<think>reason</think>answer"])

    assert kinds_collapsed(pieces) == [THINKING, CLOSED, TEXT]


def test_text_without_tags_is_passed_through_untouched():
    pieces = split(["hello ", "world"])

    assert joined(pieces, TEXT) == "hello world"
    assert all(piece.kind == TEXT for piece in pieces)


def test_leading_whitespace_after_the_closing_tag_is_dropped_across_chunks():
    pieces = split(["<think>a</think>", "\n", "\n  ", "answer"])

    assert joined(pieces, TEXT) == "answer"


def test_leading_whitespace_inside_the_block_is_dropped():
    pieces = split(["<think>", "\n", "reason</think>x"])

    assert joined(pieces, THINKING) == "reason"


def test_an_unclosed_block_at_the_end_of_the_stream_is_all_thinking():
    pieces = split(["<think>still ", "going"])

    assert joined(pieces, THINKING) == "still going"
    assert joined(pieces, TEXT) == ""
    assert pieces[-1].kind == CLOSED


def test_a_held_partial_tag_at_the_end_of_the_stream_is_released_as_text():
    pieces = split(["answer <thi"])

    assert joined(pieces, TEXT) == "answer <thi"


def test_a_partial_closing_tag_at_the_end_of_an_unclosed_block_stays_thinking():
    pieces = split(["<think>abc</thi"])

    assert joined(pieces, THINKING) == "abc</thi"


def test_a_tag_split_inside_its_name_is_not_emitted_early():
    splitter = ThinkTagSplitter()

    assert splitter.feed("hi <thi") == [Piece(TEXT, "hi ")]
    assert splitter.feed("nk>x") == [Piece(THINKING, "x")]


def test_a_closing_tag_must_match_the_opening_name():
    pieces = split(["<think>a</thinking>b</think>c"])

    assert joined(pieces, THINKING) == "a</thinking>b"
    assert joined(pieces, TEXT) == "c"


def test_a_non_ascii_lookalike_of_a_closing_tag_is_not_held_back():
    splitter = ThinkTagSplitter()

    pieces = splitter.feed("<think>abc</thin\u212a")

    assert joined(pieces, THINKING) == "abc</thin\u212a"


def test_an_overlong_unfinished_tag_is_released_as_text():
    junk = "<think " + "x" * 400

    assert joined(split([junk]), TEXT) == junk


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def capture(enabled=True):
    clock = FakeClock()
    return ThinkingCapture(enabled, clock=clock), clock


def texts(pieces):
    return [piece.text for piece in pieces if piece.kind == TEXT]


def thoughts(pieces):
    return [(piece.text, piece.segment) for piece in pieces if piece.kind == THINKING]


def test_provider_thinking_then_an_answer_is_one_timed_segment():
    cap, clock = capture()

    first = cap.on_thinking("let me ")
    clock.advance(1.5)
    second = cap.on_thinking("think")
    clock.advance(0.5)
    answer = cap.on_token("done")

    assert thoughts(first + second) == [("let me ", 0), ("think", 0)]
    assert texts(answer) == ["done"]
    assert cap.segments() == [{"text": "let me think", "duration_ms": 2000}]


def test_inline_thinking_is_captured_and_removed_from_the_visible_text():
    cap, clock = capture()

    pieces = cap.on_token("<think>reason")
    clock.advance(3)
    pieces += cap.on_token("</think>answer")

    assert thoughts(pieces) == [("reason", 0)]
    assert texts(pieces) == ["answer"]
    assert cap.segments() == [{"text": "reason", "duration_ms": 3000}]


def test_each_model_turn_gets_its_own_segment():
    cap, clock = capture()

    cap.on_thinking("first turn")
    clock.advance(1)
    cap.flush()
    clock.advance(10)
    second = cap.on_thinking("second turn")
    clock.advance(2)
    cap.on_token("answer")

    assert thoughts(second) == [("second turn", 1)]
    assert cap.segments() == [
        {"text": "first turn", "duration_ms": 1000},
        {"text": "second turn", "duration_ms": 2000},
    ]


def test_whitespace_only_text_does_not_end_a_segment():
    cap, _ = capture()

    cap.on_thinking("a")
    cap.on_token("\n")
    again = cap.on_thinking("b")

    assert thoughts(again) == [("b", 0)]
    assert cap.segments()[0]["text"] == "ab"


def test_provider_and_inline_thinking_in_one_turn_share_a_segment():
    cap, _ = capture()

    cap.on_thinking("one ")
    cap.on_token("<think>two</think>")
    cap.on_token("answer")

    assert cap.segments()[0]["text"] == "one two"
    assert len(cap.segments()) == 1


def test_two_inline_blocks_in_one_turn_are_two_segments():
    cap, _ = capture()

    cap.on_token("<think>a</think><think>b</think>x")

    assert [segment["text"] for segment in cap.segments()] == ["a", "b"]


def test_an_unclosed_inline_block_is_flushed_into_a_closed_segment():
    cap, clock = capture()

    cap.on_token("<think>never closed")
    clock.advance(4)
    pieces = cap.flush()

    assert texts(pieces) == []
    assert cap.segments() == [{"text": "never closed", "duration_ms": 4000}]


def test_flush_releases_a_held_partial_tag_as_text():
    cap, _ = capture()

    cap.on_token("see <thi")
    pieces = cap.flush()

    assert texts(pieces) == ["<thi"]


def test_blank_segments_are_not_kept():
    cap, _ = capture()

    cap.on_token("<think>   </think>answer")

    assert cap.segments() == []


def test_a_disabled_capture_passes_text_through_and_records_nothing():
    cap, _ = capture(enabled=False)

    assert texts(cap.on_token("<think>raw</think>answer")) == ["<think>raw</think>answer"]
    assert cap.on_thinking("secret") == []
    assert cap.flush() == []
    assert cap.segments() == []


def test_a_disabled_capture_passes_even_an_empty_token_through():
    cap, _ = capture(enabled=False)

    assert texts(cap.on_token("")) == [""]


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
