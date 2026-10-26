import json
from typing import List

import pytest

from src.features.llm.harness.adapter import RenderRequest
from src.features.llm.harness.adapters.openai_chat import OpenAIChatAdapter
from src.features.llm.harness.errors import InvalidRequest, ServerFailed
from src.features.llm.harness.events import (
    Finish,
    FinishReason,
    ReasoningDelta,
    TextDelta,
    ToolArgsDelta,
    ToolCallEnd,
    ToolCallStart,
    Usage,
    UsageReported,
)
from src.features.llm.harness.messages import Message, ReasoningPart, Role, TextPart, ToolCallPart
from src.features.llm.harness.profile import ToolIdRule
from tests.features.llm.harness import cassettes
from tests.features.llm.harness.history import make_config, profile_with
from tests.fixtures.llm_harness import assemble, decode_chunks

USER = Message("user_1", Role.USER, (TextPart("hello"),))


def frame(obj, *, no_space=False, ending="\n\n") -> bytes:
    body = obj if isinstance(obj, str) else json.dumps(obj)
    return f"data{':' if no_space else ': '}{body}{ending}".encode()


def delta(content: str) -> bytes:
    return frame({"choices": [{"delta": {"content": content}}]})


def reasoning(text: str, key="reasoning_content") -> bytes:
    return frame({"choices": [{"delta": {key: text}}]})


def finish_chunk(reason: str) -> bytes:
    return frame({"choices": [{"delta": {}, "finish_reason": reason}]})


def usage_chunk(**usage) -> bytes:
    return frame({"choices": [], "usage": usage or {"total_tokens": 30, "prompt_tokens": 20, "completion_tokens": 10}})


def call_chunk(index=0, call_id=None, name=None, arguments=None) -> bytes:
    fragment = {"index": index}
    if index is None:
        del fragment["index"]
    if call_id is not None:
        fragment["id"] = call_id
        fragment["type"] = "function"
    function = {}
    if name is not None:
        function["name"] = name
    if arguments is not None:
        function["arguments"] = arguments
    fragment["function"] = function
    return frame({"choices": [{"delta": {"tool_calls": [fragment]}}]})


DONE = frame("[DONE]")


def split(raw: bytes, *cuts: int) -> List[bytes]:
    pieces, previous = [], 0
    for cut in cuts:
        pieces.append(raw[previous:cut])
        previous = cut
    pieces.append(raw[previous:])
    return pieces


def decode(chunks, *, tools=False, messages=(USER,), **profile):
    request = RenderRequest(config=make_config("openai"), system="SYS", messages=tuple(messages))
    decoder = OpenAIChatAdapter().decoder(request, profile_with("openai-chat", **profile))
    return decode_chunks(decoder, chunks if isinstance(chunks, list) else [chunks])


def texts(events) -> List[str]:
    return [event.text for event in events if isinstance(event, TextDelta)]


def thoughts(events) -> List[str]:
    return [event.text for event in events if isinstance(event, ReasoningDelta)]


def test_a_frame_split_across_chunk_boundaries_is_reassembled():
    raw = delta("hel") + delta("lo") + DONE

    assert texts(decode(split(raw, 7, 25, 44, 60))) == ["hel", "lo"]


def test_several_frames_in_one_chunk_all_decode():
    assert texts(decode(delta("a") + delta("b") + delta("c") + DONE)) == ["a", "b", "c"]


def test_a_plain_answer_is_text_then_usage_then_finish_with_one_block():
    events = decode(delta("hi") + usage_chunk() + DONE)

    assert events == [
        TextDelta(block=0, text="hi"),
        UsageReported(Usage(
            input_tokens=20, output_tokens=10, total_tokens=30,
            raw={"total_tokens": 30, "prompt_tokens": 20, "completion_tokens": 10},
        )),
        Finish(reason=FinishReason.UNKNOWN, raw_reason=None),
    ]


@pytest.mark.parametrize(
    "raw, reason",
    [
        ("stop", FinishReason.STOP),
        ("length", FinishReason.LENGTH),
        ("tool_calls", FinishReason.TOOL_CALLS),
        ("content_filter", FinishReason.CONTENT_FILTER),
        ("something_new", FinishReason.UNKNOWN),
    ],
)
def test_the_finish_reason_arrives_before_usage_and_done_and_is_remembered(raw, reason):
    events = decode(delta("hi") + finish_chunk(raw) + usage_chunk() + DONE)

    assert events[-1] == Finish(reason=reason, raw_reason=raw)
    assert isinstance(events[-2], UsageReported)


def test_a_stream_without_usage_still_ends_with_a_finish():
    events = decode(delta("hi") + DONE)

    assert [type(event) for event in events] == [TextDelta, Finish]


def test_done_stops_reading_the_body():
    assert texts(decode(delta("kept") + DONE + delta("dropped"))) == ["kept"]
    assert texts(decode([delta("kept") + DONE, delta("dropped")])) == ["kept"]


def test_a_malformed_frame_is_skipped_and_the_stream_continues():
    raw = delta("a") + b"data: {not json\n\n" + delta("b") + DONE

    assert texts(decode(raw)) == ["a", "b"]


def test_non_data_lines_and_empty_deltas_yield_nothing():
    raw = (
        b": heartbeat\n\nevent: ping\n\n"
        + frame({"choices": [{"delta": {}}]})
        + delta("")
        + frame({"choices": []})
        + delta("only")
        + DONE
    )

    assert texts(decode(raw)) == ["only"]
    assert len(decode(raw)) == 2


def test_no_space_after_the_data_colon_is_still_parsed():
    raw = frame({"choices": [{"delta": {"content": "hello"}}]}, no_space=True) + frame("[DONE]", no_space=True)

    assert texts(decode(raw)) == ["hello"]


def test_json_split_across_two_data_lines_of_one_record_is_joined():
    payload = json.dumps({"choices": [{"delta": {"content": "hello"}}]})
    cut = payload.index('"content"') + len('"content"')
    raw = f"data: {payload[:cut]}\n: keep-alive\ndata: {payload[cut:]}\n\n".encode() + DONE

    assert texts(decode(raw)) == ["hello"]


@pytest.mark.parametrize("ending", ["\n\n", "\r\n\r\n", "\r\r"], ids=["lf", "crlf", "cr"])
def test_every_line_ending_style_decodes_identically(ending):
    raw = frame({"choices": [{"delta": {"content": "hi"}}]}, ending=ending) + frame("[DONE]", ending=ending)

    assert texts(decode(raw)) == ["hi"]


def test_multibyte_text_cut_between_its_bytes_survives():
    payload = json.dumps({"choices": [{"delta": {"content": "café"}}]}, ensure_ascii=False)
    raw = f"data: {payload}\n\n".encode() + DONE
    cut = raw.index(b"caf") + 4

    assert texts(decode(split(raw, cut))) == ["café"]


def test_an_unfinished_record_at_eof_emits_nothing_and_the_finish_stays_unknown():
    raw = delta("kept") + b"data: " + json.dumps({"choices": [{"delta": {"content": "lost"}}]}).encode()

    events = decode(raw)

    assert texts(events) == ["kept"]
    assert events[-1] == Finish(reason=FinishReason.UNKNOWN, raw_reason=None)


def test_an_incomplete_done_sentinel_is_never_treated_as_done():
    events = decode(delta("kept") + b"data: [DON")

    assert texts(events) == ["kept"]
    assert events[-1].reason is FinishReason.UNKNOWN


def test_an_empty_body_is_just_an_unknown_finish():
    assert decode([]) == [Finish(reason=FinishReason.UNKNOWN, raw_reason=None)]


def test_tool_call_fragments_are_assembled_by_index():
    raw = (
        call_chunk(0, "call_a", "get_form", '{"sc')
        + call_chunk(0, arguments='ope": 1}')
        + call_chunk(1, "call_b", "other", "{}")
        + finish_chunk("tool_calls")
        + DONE
    )

    events = decode(split(raw, 90, 200, 330))

    assert events == [
        ToolCallStart(block=0, call_id="call_a", name="get_form", wire_id="call_a"),
        ToolArgsDelta(block=0, text='{"sc'),
        ToolArgsDelta(block=0, text='ope": 1}'),
        ToolCallStart(block=1, call_id="call_b", name="other", wire_id="call_b"),
        ToolArgsDelta(block=1, text="{}"),
        ToolCallEnd(block=0),
        ToolCallEnd(block=1),
        Finish(reason=FinishReason.TOOL_CALLS, raw_reason="tool_calls"),
    ]
    message = assemble(events)
    assert [(c.call_id, c.name, c.args, c.args_raw) for c in message.tool_calls()] == [
        ("call_a", "get_form", {"scope": 1}, '{"scope": 1}'),
        ("call_b", "other", {}, "{}"),
    ]


def test_interleaved_fragments_of_parallel_calls_land_on_their_own_call():
    raw = (
        call_chunk(0, "a", "one", '{"x":')
        + call_chunk(1, "b", "two", '{"y":')
        + call_chunk(0, arguments="1}")
        + call_chunk(1, arguments="2}")
        + DONE
    )

    calls = assemble(decode(raw)).tool_calls()

    assert [(c.name, c.args) for c in calls] == [("one", {"x": 1}), ("two", {"y": 2})]


def test_a_turn_with_no_tool_calls_has_no_tool_events():
    events = decode(delta("plain") + DONE)

    assert not [e for e in events if isinstance(e, (ToolCallStart, ToolCallEnd, ToolArgsDelta))]


def test_the_name_may_arrive_in_pieces_before_the_arguments():
    raw = call_chunk(0, "c1", "get_") + call_chunk(0, name="form") + call_chunk(0, arguments="{}") + DONE

    calls = assemble(decode(raw)).tool_calls()

    assert [(c.name, c.args) for c in calls] == [("get_form", {})]


def test_a_call_without_arguments_still_ends_with_empty_arguments():
    raw = call_chunk(0, "c1", "ping", "") + finish_chunk("tool_calls") + DONE

    calls = assemble(decode(raw)).tool_calls()

    assert [(c.name, c.args, c.args_raw) for c in calls] == [("ping", {}, None)]


def test_deltas_without_an_index_each_start_a_call_when_they_carry_an_id():
    raw = call_chunk(None, "c1", "one", '{"a":1}') + call_chunk(None, "c2", "two", '{"b":2}') + DONE

    calls = assemble(decode(raw)).tool_calls()

    assert [(c.call_id, c.name, c.args) for c in calls] == [("c1", "one", {"a": 1}), ("c2", "two", {"b": 2})]


def test_deltas_without_an_index_or_an_id_each_start_a_call_when_they_carry_a_name():
    raw = call_chunk(None, None, "one", '{"a":1}') + call_chunk(None, None, "two", '{"b":2}') + DONE

    calls = assemble(decode(raw)).tool_calls()

    assert [(c.name, c.args) for c in calls] == [("one", {"a": 1}), ("two", {"b": 2})]


def test_an_argument_only_delta_without_an_index_continues_the_latest_call():
    raw = call_chunk(None, "c1", "t", '{"a":') + call_chunk(None, arguments="1}") + DONE

    calls = assemble(decode(raw)).tool_calls()

    assert [(c.name, c.args) for c in calls] == [("t", {"a": 1})]


def test_a_reused_index_with_a_new_id_is_a_new_call():
    raw = call_chunk(0, "c1", "one", "{}") + call_chunk(0, "c2", "two", "{}") + DONE

    calls = assemble(decode(raw)).tool_calls()

    assert [c.call_id for c in calls] == ["c1", "c2"]


def test_dict_arguments_are_serialised_to_text():
    raw = call_chunk(0, "c1", "t", {"a": 1}) + DONE

    assert assemble(decode(raw)).tool_calls()[0].args == {"a": 1}


def test_a_done_marker_ends_open_calls_but_a_cut_stream_leaves_them_unended():
    started = call_chunk(0, "c1", "t", '{"a":1}')

    ended = assemble(decode(started + DONE)).tool_calls()[0]
    cut = assemble(decode(started)).tool_calls()[0]

    assert ended.args == {"a": 1}
    assert cut.args is None and cut.args_raw == '{"a":1}'


def test_stop_with_tool_calls_is_reported_as_tool_calls_and_keeps_the_raw_reason():
    raw = call_chunk(0, "c1", "t", "{}") + finish_chunk("stop") + DONE

    assert decode(raw)[-1] == Finish(reason=FinishReason.TOOL_CALLS, raw_reason="stop")


def test_parts_come_out_in_the_order_the_stream_produced_them():
    raw = reasoning("think ") + reasoning("more") + delta("answer") + call_chunk(0, "c1", "t", "{}") + DONE

    message = assemble(decode(raw))

    assert [type(part) for part in message.parts] == [ReasoningPart, TextPart, ToolCallPart]
    assert message.parts[0].text == "think more"


@pytest.mark.parametrize("key", ["reasoning_content", "reasoning"])
def test_reasoning_deltas_stream_as_reasoning_never_as_text(key):
    raw = reasoning("let me ", key) + reasoning("think", key) + delta("answer") + DONE

    events = decode(raw)

    assert thoughts(events) == ["let me ", "think"]
    assert texts(events) == ["answer"]
    assert [type(e) for e in events[:3]] == [ReasoningDelta, ReasoningDelta, TextDelta]


def test_a_reasoning_frame_split_across_chunks_is_reassembled():
    raw = reasoning("hello there") + delta("x") + DONE

    assert thoughts(decode(split(raw, 12, 40))) == ["hello there"]


def test_null_empty_and_structured_reasoning_emit_nothing():
    raw = (
        frame({"choices": [{"delta": {"reasoning_content": None, "content": "a"}}]})
        + reasoning("")
        + frame({"choices": [{"delta": {"reasoning": [{"text": "x"}]}}]})
        + DONE
    )

    events = decode(raw)

    assert thoughts(events) == []
    assert texts(events) == ["a"]


def test_inline_think_tags_become_reasoning_even_when_split_across_chunks():
    raw = delta("<thi") + delta("nk>plan it</th") + delta("ink>\n\nthe answer") + DONE

    events = decode(raw)

    assert thoughts(events) == ["plan it"]
    assert texts(events) == ["the answer"]


def test_an_unclosed_inline_think_block_is_all_reasoning():
    events = decode(delta("<think>still going") + DONE)

    assert thoughts(events) == ["still going"]
    assert texts(events) == []


def test_usage_carries_reasoning_and_cached_tokens():
    chunk = usage_chunk(
        prompt_tokens=100,
        completion_tokens=40,
        total_tokens=140,
        completion_tokens_details={"reasoning_tokens": 25},
        prompt_tokens_details={"cached_tokens": 64},
    )

    usage = next(e.usage for e in decode(delta("x") + chunk + DONE) if isinstance(e, UsageReported))

    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (100, 40, 140)
    assert (usage.reasoning_tokens, usage.cached_input_tokens) == (25, 64)
    assert usage.raw["prompt_tokens_details"] == {"cached_tokens": 64}


def test_a_deepseek_style_cache_hit_count_is_read_as_cached_input():
    chunk = usage_chunk(prompt_tokens=10, completion_tokens=1, total_tokens=11, prompt_cache_hit_tokens=8, prompt_cache_miss_tokens=2)

    usage = next(e.usage for e in decode(chunk + DONE) if isinstance(e, UsageReported))

    assert usage.cached_input_tokens == 8


def test_an_error_object_in_the_stream_raises_the_matching_error():
    raw = delta("partial") + frame({"error": {"message": "engine died", "code": 500}})

    with pytest.raises(ServerFailed) as caught:
        decode(raw)

    assert caught.value.message == "engine died"


def test_an_in_band_bad_request_is_classified_by_its_code():
    with pytest.raises(InvalidRequest):
        decode(frame({"error": {"message": "bad field", "code": 400}}))


def test_server_ids_pass_through_by_default_and_are_kept_as_wire_ids():
    events = decode(call_chunk(0, "srv_9", "t", "{}") + DONE)

    start = next(e for e in events if isinstance(e, ToolCallStart))
    assert (start.call_id, start.wire_id) == ("srv_9", "srv_9")


def test_a_missing_server_id_is_replaced_by_a_stable_minted_one():
    raw = call_chunk(0, None, "t", "{}") + call_chunk(1, None, "u", "{}") + DONE

    first = decode(raw)
    second = decode(raw)
    other_turn = decode(raw, messages=(Message("user_2", Role.USER, (TextPart("again"),)),))

    starts = [e for e in first if isinstance(e, ToolCallStart)]
    assert [s.wire_id for s in starts] == [None, None]
    assert len({s.call_id for s in starts}) == 2
    assert first == second
    assert [e.call_id for e in other_turn if isinstance(e, ToolCallStart)] != [s.call_id for s in starts]


def test_the_mint_rule_ignores_server_ids_but_remembers_them():
    events = decode(call_chunk(0, "dup", "t", "{}") + DONE, tool_id_rule=ToolIdRule.MINT)

    start = next(e for e in events if isinstance(e, ToolCallStart))
    assert start.call_id != "dup" and start.wire_id == "dup"


@pytest.mark.parametrize("scenario", cassettes.scenario_dirs("openai-chat"), ids=lambda path: path.name)
def test_the_stored_provider_streams_replay_through_the_adapter(scenario):
    cassettes.run_scenario(OpenAIChatAdapter(), scenario)
