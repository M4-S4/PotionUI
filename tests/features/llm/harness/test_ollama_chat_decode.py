import json
from typing import List

import pytest

from src.features.llm.harness.adapter import RenderRequest, ToolSpec
from src.features.llm.harness.adapters.ollama_chat import OllamaChatAdapter
from src.features.llm.harness.errors import InvalidRequest, ServerFailed
from src.features.llm.harness.events import (
    Finish,
    FinishReason,
    ReasoningDelta,
    TextDelta,
    ToolArgsDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageReported,
)
from src.features.llm.harness.messages import Message, ReasoningPart, Role, TextPart, ToolCallPart
from tests.features.llm.harness import cassettes
from tests.features.llm.harness.history import make_config, profile_with, tool_schemas
from tests.fixtures.llm_harness import assemble, decode_chunks

USER = Message("user_1", Role.USER, (TextPart("hello"),))
TOOL_CALLS = [{"function": {"name": "get_form_state", "arguments": {"scope": "all"}}}]


def line(obj) -> bytes:
    return (json.dumps(obj) + "\n").encode()


def token(content: str) -> bytes:
    return line({"message": {"content": content}})


def thought(text: str) -> bytes:
    return line({"message": {"content": "", "thinking": text}})


def done(**extra) -> bytes:
    return line({"message": {"content": ""}, "done": True, **extra})


def split(raw: bytes, *cuts: int) -> List[bytes]:
    pieces, previous = [], 0
    for cut in cuts:
        pieces.append(raw[previous:cut])
        previous = cut
    pieces.append(raw[previous:])
    return pieces


def decode(chunks, *, with_tools=False, messages=(USER,), **profile):
    tools = tuple(ToolSpec.from_schema(schema) for schema in tool_schemas()) if with_tools else ()
    request = RenderRequest(config=make_config("ollama"), system="SYS", messages=tuple(messages), tools=tools)
    decoder = OllamaChatAdapter().decoder(request, profile_with("ollama-chat", **profile))
    return decode_chunks(decoder, chunks if isinstance(chunks, list) else [chunks])


def texts(events) -> List[str]:
    return [event.text for event in events if isinstance(event, TextDelta)]


def thoughts(events) -> List[str]:
    return [event.text for event in events if isinstance(event, ReasoningDelta)]


def test_a_line_split_across_chunk_boundaries_is_reassembled():
    raw = token("hel") + token("lo") + done()

    assert texts(decode(split(raw, 5, 20, 40))) == ["hel", "lo"]


def test_several_lines_in_one_chunk_all_decode():
    assert texts(decode(token("a") + token("b") + token("c") + done())) == ["a", "b", "c"]


def test_the_done_object_carries_the_token_counts():
    events = decode(token("hi") + done(prompt_eval_count=20, eval_count=10))

    assert events[0] == TextDelta(block=1, text="hi")
    assert isinstance(events[1], UsageReported)
    usage = events[1].usage
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (20, 10, 30)
    assert events[2] == Finish(reason=FinishReason.UNKNOWN, raw_reason=None)


def test_a_done_object_without_counts_reports_no_numbers():
    usage = next(e.usage for e in decode(token("hi") + done()) if isinstance(e, UsageReported))

    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (None, None, None)


def test_one_missing_count_still_produces_a_total():
    usage = next(e.usage for e in decode(done(eval_count=10)) if isinstance(e, UsageReported))

    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (None, 10, 10)


def test_cached_prompt_tokens_and_the_timing_fields_are_kept():
    usage = next(
        e.usage for e in decode(done(prompt_eval_count=9, prompt_eval_cached_count=4, total_duration=77))
        if isinstance(e, UsageReported)
    )

    assert usage.cached_input_tokens == 4
    assert usage.raw["total_duration"] == 77


def test_the_done_reason_is_kept_raw_next_to_the_normalised_one():
    assert decode(token("hi") + done(done_reason="unload"))[-1] == Finish(reason=FinishReason.UNKNOWN, raw_reason="unload")


def test_done_stops_reading_the_body():
    assert texts(decode(token("kept") + done() + token("dropped"))) == ["kept"]
    assert texts(decode([token("kept") + done(), token("dropped")])) == ["kept"]


def test_a_malformed_line_is_skipped_and_the_stream_continues():
    assert texts(decode(token("a") + b"{not json\n" + token("b") + done())) == ["a", "b"]


def test_empty_content_yields_no_text():
    raw = token("") + line({}) + line({"message": {}}) + token("only") + done()

    assert texts(decode(raw)) == ["only"]


def test_a_stream_that_ends_without_a_done_object_finishes_unknown():
    events = decode(token("a") + b'{"message": {"content": "b"')

    assert texts(events) == ["a"]
    assert events[-1] == Finish(reason=FinishReason.UNKNOWN, raw_reason=None)


def test_tool_calls_arrive_complete_with_object_arguments_and_a_minted_id():
    raw = line({"message": {"content": "", "tool_calls": TOOL_CALLS}}) + done(prompt_eval_count=2, eval_count=1, done_reason="stop")

    events = decode(raw, with_tools=True)

    start = events[0]
    assert isinstance(start, ToolCallStart)
    assert (start.name, start.wire_id) == ("get_form_state", None)
    assert start.call_id.startswith("call_")
    assert events[1] == ToolCallEnd(block=2, args={"scope": "all"})
    assert events[-1] == Finish(reason=FinishReason.TOOL_CALLS, raw_reason="stop")


def test_a_server_supplied_tool_call_id_is_not_used_ids_are_always_minted():
    call = {"id": "call_xyz", "function": {"name": "t", "arguments": {}}}

    start = decode(line({"message": {"tool_calls": [call]}}) + done())[0]

    assert start.call_id != "call_xyz" and start.wire_id is None
    assert len(start.call_id) == len("call_") + 24


def test_parallel_calls_in_one_chunk_get_distinct_ids_and_blocks():
    calls = [{"function": {"name": "a", "arguments": {}}}, {"function": {"name": "b", "arguments": {"x": 1}}}]

    message = assemble(decode(line({"message": {"tool_calls": calls}}) + done()))

    parts = message.tool_calls()
    assert [(p.name, p.args) for p in parts] == [("a", {}), ("b", {"x": 1})]
    assert parts[0].call_id != parts[1].call_id


def test_calls_in_later_chunks_are_added_not_replaced():
    later = [{"function": {"name": "other", "arguments": {}}}]
    raw = line({"message": {"tool_calls": TOOL_CALLS}}) + line({"message": {"tool_calls": later}}) + done()

    assert [c.name for c in assemble(decode(raw)).tool_calls()] == ["get_form_state", "other"]


def test_minted_ids_are_stable_for_the_same_request_and_differ_across_turns():
    raw = line({"message": {"tool_calls": TOOL_CALLS}}) + done()

    first, again = decode(raw)[0], decode(raw)[0]
    later = decode(raw, messages=(Message("user_2", Role.USER, (TextPart("again"),)),))[0]

    assert first == again
    assert first.call_id != later.call_id


def test_string_arguments_are_parsed_and_unparseable_ones_leave_the_call_unrunnable():
    good = {"function": {"name": "a", "arguments": '{"x": 1}'}}
    bad = {"function": {"name": "b", "arguments": "{oops"}}

    calls = assemble(decode(line({"message": {"tool_calls": [good, bad]}}) + done())).tool_calls()

    assert [(c.name, c.args) for c in calls] == [("a", {"x": 1}), ("b", None)]
    assert calls[1].args_raw == "{oops"


def test_a_turn_with_no_tool_calls_has_no_tool_events():
    events = decode(token("plain") + done(), with_tools=True)

    assert not [e for e in events if isinstance(e, (ToolCallStart, ToolCallEnd, ToolArgsDelta))]


def test_thinking_chunks_stream_as_reasoning_and_never_as_text():
    raw = thought("let me ") + thought("think") + token("answer") + done()

    events = decode(raw)

    assert thoughts(events) == ["let me ", "think"]
    assert texts(events) == ["answer"]
    assert [type(e) for e in events[:3]] == [ReasoningDelta, ReasoningDelta, TextDelta]


def test_a_thinking_line_split_across_chunks_is_reassembled():
    raw = thought("hello there") + token("x") + done()

    assert thoughts(decode(split(raw, 10, 30))) == ["hello there"]


def test_empty_and_non_text_thinking_emit_nothing():
    raw = thought("") + line({"message": {"content": "a", "thinking": {"nested": 1}}}) + done()

    events = decode(raw)

    assert thoughts(events) == []
    assert texts(events) == ["a"]


def test_a_chunk_with_thinking_and_content_yields_both():
    events = decode(line({"message": {"content": "a", "thinking": "b"}}) + done())

    assert thoughts(events) == ["b"] and texts(events) == ["a"]


def test_reasoning_text_and_calls_assemble_in_stream_order():
    raw = thought("plan") + token("ok") + line({"message": {"tool_calls": TOOL_CALLS}}) + done()

    message = assemble(decode(raw))

    assert [type(part) for part in message.parts] == [ReasoningPart, TextPart, ToolCallPart]


def test_inline_think_tags_become_reasoning():
    events = decode(token("<think>plan</think>answer") + done())

    assert thoughts(events) == ["plan"] and texts(events) == ["answer"]


def test_an_error_line_raises_the_matching_error():
    with pytest.raises(ServerFailed) as caught:
        decode(token("part") + line({"error": "model runner has unexpectedly stopped"}))

    assert "unexpectedly stopped" in caught.value.detail["body"]


def test_a_missing_model_error_is_an_invalid_request():
    with pytest.raises(InvalidRequest):
        decode(line({"error": "model 'x' not found"}))


CALL_TEXT = '<tool_call>{"name": "get_form_state", "arguments": {"scope": "all"}}</tool_call>'


@pytest.mark.parametrize("prompt_tools", [False, True])
def test_inline_tool_call_blocks_are_never_parsed_in_the_decoder(prompt_tools):
    raw = token("Let me look. ") + token(CALL_TEXT) + done()

    events = decode(raw, with_tools=True, prompt_tools=prompt_tools)

    assert "".join(texts(events)) == "Let me look. " + CALL_TEXT
    assert not assemble(events).tool_calls()
    assert events[-1].reason is FinishReason.UNKNOWN


def test_blocks_are_fixed_reasoning_zero_text_one_and_calls_from_two():
    calls = [{"function": {"name": "a", "arguments": {}}}, {"function": {"name": "b", "arguments": {}}}]
    raw = token("hi") + thought("plan") + line({"message": {"tool_calls": calls}}) + done()

    events = decode(raw)

    assert [(type(e).__name__, e.block) for e in events if hasattr(e, "block") and not isinstance(e, ToolCallEnd)] == [
        ("TextDelta", 1),
        ("ReasoningDelta", 0),
        ("ToolCallStart", 2),
        ("ToolCallStart", 3),
    ]


def test_a_tool_call_inside_a_think_block_stays_reasoning():
    raw = token("<think>maybe " + CALL_TEXT + "</think>no call") + done()

    events = decode(raw, with_tools=True, prompt_tools=True)

    assert not assemble(events).tool_calls()
    assert texts(events) == ["no call"]
    assert thoughts(events) == ["maybe " + CALL_TEXT]


@pytest.mark.parametrize("scenario", cassettes.scenario_dirs("ollama-chat"), ids=lambda path: path.name)
def test_the_stored_provider_streams_replay_through_the_adapter(scenario):
    cassettes.run_scenario(OllamaChatAdapter(), scenario)
