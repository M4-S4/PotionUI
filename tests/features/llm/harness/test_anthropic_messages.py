import dataclasses
import json
from typing import Any, Dict, List

import pytest

from src.features.llm.harness.adapter import Adapter, RenderRequest, Settings, ToolChoice, ToolSpec
from src.features.llm.harness.adapters import anthropic_messages
from src.features.llm.harness.adapters.anthropic_messages import AnthropicMessagesAdapter
from src.features.llm.harness.assembler import StepAssembler
from src.features.llm.harness.errors import (
    AuthFailed,
    ConnectionFailed,
    ContextOverflow,
    InvalidRequest,
    MalformedStream,
    OpaqueViolation,
    Overloaded,
    QuotaExhausted,
    RateLimited,
    ServerFailed,
    WireFailure,
)
from src.features.llm.harness.events import (
    Finish,
    FinishReason,
    OpaqueDelta,
    ReasoningDelta,
    TextDelta,
    ToolArgsDelta,
    ToolCallEnd,
    ToolCallStart,
    Usage,
    UsageReported,
)
from src.features.llm.harness.messages import (
    CompactionPart,
    ContextPart,
    FilePart,
    ImagePart,
    JsonPart,
    Origin,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    message,
)
from src.features.llm.harness.profile import ProfileSource
from src.features.llm.model_listing import ModelListingError
from tests.features.llm.harness.anthropic_helpers import (
    API_KEY,
    MODEL,
    WEATHER,
    bare_profile,
    make_config,
    make_profile,
    make_request,
    own_origin,
    redacted_block,
    render_body,
    sse_bytes,
    stream,
    text_block,
    thinking_block,
    tool_block,
)
from tests.fixtures.llm_harness import decode_chunks

TOOL = WEATHER
USER = message("u1", "user", TextPart("Hello"))


def decode(payload: bytes, profile=None, request=None):
    adapter = AnthropicMessagesAdapter()
    request = request or make_request([USER])
    return decode_chunks(adapter.decoder(request, profile or make_profile()), [payload])


def test_adapter_satisfies_the_adapter_protocol():
    adapter = AnthropicMessagesAdapter()
    assert isinstance(adapter, Adapter)
    assert adapter.name == "anthropic-messages"


@pytest.mark.parametrize("base_url", [
    "https://api.anthropic.com",
    "https://api.anthropic.com/",
    "https://api.anthropic.com/v1",
    "https://api.anthropic.com/v1/",
])
def test_request_line_and_headers(base_url):
    wire = AnthropicMessagesAdapter().render(make_request([USER], base_url=base_url), make_profile())
    assert (wire.method, wire.url) == ("POST", "https://api.anthropic.com/v1/messages")
    assert wire.headers["anthropic-version"] == "2026-10-21"
    assert wire.headers["x-api-key"] == API_KEY
    assert wire.headers["content-type"] == "application/json"
    assert "authorization" not in {name.lower() for name in wire.headers}
    assert wire.stream is True


def test_a_proxy_base_url_without_a_key_sends_no_key_header():
    wire = AnthropicMessagesAdapter().render(make_request([USER], base_url="http://localhost:4000", api_key=None), make_profile())
    assert wire.url == "http://localhost:4000/v1/messages"
    assert "x-api-key" not in wire.headers


def test_body_carries_model_system_and_streams_even_when_not_asked():
    request = RenderRequest(config=make_config(), system="Be brief.", messages=(USER,), stream=False)
    body = render_body(request)
    assert body["model"] == MODEL
    assert body["system"] == "Be brief."
    assert body["stream"] is True
    assert "system" not in render_body(make_request([USER]))


@pytest.mark.parametrize("settings", [
    Settings(temperature=0.2),
    Settings(top_p=0.9),
    Settings(top_k=40),
    Settings(temperature=0.0, top_p=1.0, top_k=1, min_p=0.1),
])
def test_sampling_parameters_are_never_sent(settings):
    body = render_body(make_request([USER], settings=settings), make_profile())
    assert not {"temperature", "top_p", "top_k", "min_p"} & set(body)
    assert not {"temperature", "top_p", "top_k", "min_p"} & set(render_body(make_request([USER], settings=settings), bare_profile()))


@pytest.mark.parametrize("fields, reasoning, thinking, output_config", [
    ({}, "high", {"type": "adaptive"}, {"effort": "high"}),
    ({}, "low", {"type": "adaptive"}, {"effort": "low"}),
    ({}, "extreme", {"type": "adaptive"}, None),
    ({}, True, {"type": "adaptive"}, None),
    ({}, None, None, None),
    ({"reasoning_toggle": "thinking.type"}, False, {"type": "disabled"}, None),
    ({"reasoning_toggle": None}, False, None, None),
    ({"reasoning": "none"}, "high", None, None),
    ({"reasoning": "none"}, False, None, None),
    ({"reasoning": "toggle", "reasoning_levels": []}, "high", {"type": "adaptive"}, None),
    ({"reasoning": "toggle", "reasoning_levels": [], "reasoning_toggle": "thinking.type"}, False, {"type": "disabled"}, None),
    ({"reasoning": "toggle", "reasoning_levels": [], "reasoning_toggle": None}, False, None, None),
    ({"reasoning": "always_on"}, False, None, None),
    ({"reasoning": "always_on", "reasoning_toggle": "thinking.type"}, False, None, None),
    ({"reasoning": "always_on"}, None, {"type": "adaptive"}, None),
    ({"reasoning": "budget"}, "high", None, None),
    ({"reasoning": "budget", "reasoning_toggle": "thinking.type"}, False, {"type": "disabled"}, None),
    ({"reasoning": "budget", "reasoning_toggle": None}, False, None, None),
])
def test_thinking_and_effort_come_from_settings_and_profile(fields, reasoning, thinking, output_config):
    body = render_body(make_request([USER], settings=Settings(reasoning=reasoning)), make_profile(**fields))
    assert body.get("thinking") == thinking
    assert body.get("output_config") == output_config


@pytest.mark.parametrize("choice, thinking_on, expected", [
    (ToolChoice("auto"), False, None),
    (ToolChoice("none"), True, {"type": "none"}),
    (ToolChoice("required"), False, {"type": "any"}),
    (ToolChoice("required"), True, None),
    (ToolChoice("named", "get_weather"), False, {"type": "tool", "name": "get_weather"}),
    (ToolChoice("named", "get_weather"), True, None),
    (ToolChoice("named"), False, None),
    (None, False, None),
])
def test_tool_choice_stays_inside_the_profile_restrictions(choice, thinking_on, expected):
    settings = Settings(tool_choice=choice, reasoning="medium" if thinking_on else None)
    body = render_body(make_request([USER], tools=[TOOL], settings=settings), make_profile())
    assert body.get("tool_choice") == expected


def test_a_profile_that_forbids_forcing_never_gets_a_forced_choice():
    profile = make_profile(tool_choice=["auto", "none"])
    settings = Settings(tool_choice=ToolChoice("required"))
    assert "tool_choice" not in render_body(make_request([USER], tools=[TOOL], settings=settings), profile)


@pytest.mark.parametrize("settings", [
    Settings(),
    Settings(tool_choice=ToolChoice("required")),
    Settings(reasoning="medium"),
])
def test_tool_history_without_tools_sends_minimal_definitions_and_forbids_new_calls(settings):
    assistant = message(
        "a1",
        "assistant",
        tool_call("call_1"),
        ToolCallPart(call_id="call_2", name="get_time", args={"city": "Paris"}),
        tool_call("call_3"),
        origin=FOREIGN,
    )
    results = message(
        "t1",
        "tool",
        ToolResultPart("call_1", "get_weather", (TextPart("a"),)),
        ToolResultPart("call_2", "get_time", (TextPart("b"),)),
        ToolResultPart("call_3", "get_weather", (TextPart("c"),)),
    )
    body = render_body(make_request([USER, assistant, results], settings=settings), make_profile())
    assert body["tools"] == [
        {"name": "get_weather", "input_schema": {"type": "object"}},
        {"name": "get_time", "input_schema": {"type": "object"}},
    ]
    assert body["tool_choice"] == {"type": "none"}


def test_real_tools_are_never_replaced_by_the_history_fallback():
    assistant = message("a1", "assistant", ToolCallPart(call_id="call_1", name="get_time", args={}), origin=FOREIGN)
    results = message("t1", "tool", ToolResultPart("call_1", "get_time", (TextPart("a"),)))
    body = render_body(make_request([USER, assistant, results], tools=[TOOL]), make_profile())
    assert [tool["name"] for tool in body["tools"]] == ["get_weather"] and "tool_choice" not in body


def test_a_history_without_tool_calls_and_without_tools_sends_neither():
    body = render_body(make_request([USER, message("a1", "assistant", TextPart("Hi."), origin=FOREIGN), USER]), make_profile())
    assert "tools" not in body and "tool_choice" not in body


@pytest.mark.parametrize("kind, reasoning, choice, expected", [
    ("always_on", False, ToolChoice("required"), None),
    ("always_on", False, ToolChoice("none"), {"type": "none"}),
    ("always_on", None, ToolChoice("required"), None),
    ("effort", False, ToolChoice("required"), {"type": "any"}),
    ("toggle", False, ToolChoice("named", "get_weather"), {"type": "tool", "name": "get_weather"}),
])
def test_tool_choice_limits_follow_whether_the_model_thinks_not_whether_thinking_was_asked_for(kind, reasoning, choice, expected):
    body = render_body(
        make_request([USER], tools=[TOOL], settings=Settings(reasoning=reasoning, tool_choice=choice)),
        make_profile(reasoning=kind),
    )
    assert body.get("tool_choice") == expected


def test_tool_choice_is_dropped_without_tools():
    settings = Settings(tool_choice=ToolChoice("none"))
    body = render_body(make_request([USER], settings=settings), make_profile())
    assert "tool_choice" not in body and "tools" not in body


def test_tools_use_input_schema_and_strict_follows_the_profile():
    strict = ToolSpec(name="lookup", description="d", parameters={"properties": {}}, strict=True)
    bare = ToolSpec(name="ping", description="d", parameters={})
    request = make_request([USER], tools=[TOOL, strict, bare])
    tools = render_body(request, make_profile(strict_tools=True))["tools"]
    assert tools[0] == {"name": "get_weather", "description": "Current weather for a city.", "input_schema": TOOL.parameters}
    assert tools[1]["input_schema"] == {"type": "object", "properties": {}} and tools[1]["strict"] is True
    assert tools[2]["input_schema"] == {"type": "object", "properties": {}} and "strict" not in tools[2]
    assert "strict" not in render_body(request, make_profile(strict_tools=False))["tools"][1]


@pytest.mark.parametrize("cache, expected", [
    ("explicit", {"type": "ephemeral"}),
    ("none", None),
    ("automatic", None),
    ("local_prefix", None),
])
def test_top_level_cache_control_follows_the_profile(cache, expected):
    assert render_body(make_request([USER]), make_profile(cache=cache)).get("cache_control") == expected


def test_an_unsupported_cache_ttl_is_refused():
    with pytest.raises(InvalidRequest, match="5m or 1h"):
        render_body(make_request([USER], settings=Settings(cache_ttl="2h")), make_profile())


@pytest.mark.parametrize("settings_tokens, cap, expected", [
    (4096, None, 4096),
    (None, None, 2000),
    (100000, 64000, 64000),
    (1000, 64000, 1000),
])
def test_max_tokens_comes_from_settings_then_config_and_is_capped(settings_tokens, cap, expected):
    profile = make_profile(max_output_tokens=cap) if cap else bare_profile()
    request = make_request([USER], settings=Settings(max_tokens=settings_tokens), max_tokens=2000)
    assert render_body(request, profile)["max_tokens"] == expected


def test_user_message_blocks_keep_media_before_text_and_wrap_context():
    user = message(
        "u1",
        "user",
        TextPart("Describe this."),
        ImagePart(media_type="image/png", data="AAAA"),
        ImagePart(media_type="image/jpeg", ref="https://example.com/a.jpg"),
        FilePart(media_type="application/pdf", name="spec.pdf", data="JVBERg=="),
        FilePart(media_type="text/plain", data="aGVsbG8="),
        ContextPart(kind="memory", text="Likes cats."),
        CompactionPart(summary="Earlier they planned a trip."),
    )
    blocks = render_body(make_request([user]))["messages"][0]["content"]
    assert [block["type"] for block in blocks] == ["image", "image", "document", "document", "text", "text"]
    assert blocks[0]["source"] == {"type": "base64", "media_type": "image/png", "data": "AAAA"}
    assert blocks[1]["source"] == {"type": "url", "url": "https://example.com/a.jpg"}
    assert blocks[2] == {"type": "document", "title": "spec.pdf", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERg=="}}
    assert blocks[3]["source"] == {"type": "text", "media_type": "text/plain", "data": "hello"}
    assert blocks[4]["text"] == "Earlier they planned a trip."
    assert blocks[5]["text"] == "Describe this.\n\n<context>\nLikes cats.\n</context>"


def test_whitespace_only_user_text_is_not_sent_and_empty_messages_vanish():
    blank = message("u1", "user", TextPart("  \n"))
    body = render_body(make_request([blank, message("u2", "user", TextPart("Hi"))]))
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]


@pytest.mark.parametrize("part", [
    ImagePart(media_type="image/png", ref="image-1"),
    FilePart(media_type="application/zip", name="a.zip", data="AAAA"),
    FilePart(media_type="text/plain", data="not base64!"),
    FilePart(media_type="application/pdf", ref="file:///a.pdf"),
])
def test_attachments_that_cannot_be_sent_are_refused(part):
    with pytest.raises(InvalidRequest):
        render_body(make_request([message("u1", "user", part)]))


def test_system_messages_belong_in_the_system_field():
    with pytest.raises(InvalidRequest):
        render_body(make_request([message("s1", "system", TextPart("Be brief.")), USER]))


FOREIGN = Origin(adapter="openai-chat", provider="openai", model="gpt-5", credential="0f3a9c1d2e4b5a6c")


def tool_call(call_id: str = "call_1", args=None, **fields) -> ToolCallPart:
    return ToolCallPart(call_id=call_id, name="get_weather", args={"city": "Paris"} if args is None else args, **fields)


def test_results_of_one_batch_share_one_user_message_with_results_first():
    assistant = message("a1", "assistant", TextPart("Checking."), tool_call("call_1"), tool_call("call_2"), origin=FOREIGN)
    results = message(
        "t1",
        "tool",
        ToolResultPart("call_1", "get_weather", (TextPart("sunny"),)),
        ToolResultPart("call_2", "get_weather", (TextPart("skipped: the call was not approved"),), is_error=True),
    )
    nudge = message("u2", "user", ContextPart(kind="note", text="Wrap up."))
    messages = render_body(make_request([USER, assistant, results, nudge]))["messages"]
    assert [item["role"] for item in messages] == ["user", "assistant", "user"]
    assert [block["type"] for block in messages[2]["content"]] == ["tool_result", "tool_result", "text"]
    assert messages[2]["content"][0] == {
        "type": "tool_result", "tool_use_id": "call_1", "content": [{"type": "text", "text": "sunny"}],
    }
    assert messages[2]["content"][1]["is_error"] is True
    assert messages[2]["content"][2]["text"] == "Wrap up."


def test_consecutive_tool_messages_still_form_one_user_message():
    assistant = message("a1", "assistant", tool_call("call_1"), tool_call("call_2"), origin=FOREIGN)
    first = message("t1", "tool", ToolResultPart("call_1", "get_weather", (TextPart("a"),)))
    second = message("t2", "tool", ToolResultPart("call_2", "get_weather", (TextPart("b"),)))
    messages = render_body(make_request([USER, assistant, first, second]))["messages"]
    assert [len(item["content"]) for item in messages] == [1, 2, 2]
    assert [block["tool_use_id"] for block in messages[2]["content"]] == ["call_1", "call_2"]


def test_tool_results_carry_images_and_json_and_omit_empty_content():
    assistant = message("a1", "assistant", tool_call("call_1"), tool_call("call_2"), tool_call("call_3"), origin=FOREIGN)
    results = message(
        "t1",
        "tool",
        ToolResultPart("call_1", "get_weather", (TextPart("map"), ImagePart(media_type="image/png", data="AAAA"))),
        ToolResultPart("call_2", "get_weather", (JsonPart({"temp": 14, "city": "Zürich"}),)),
        ToolResultPart("call_3", "get_weather", ()),
    )
    blocks = render_body(make_request([USER, assistant, results]))["messages"][2]["content"]
    assert [item["type"] for item in blocks[0]["content"]] == ["text", "image"]
    assert blocks[1]["content"] == [{"type": "text", "text": '{"temp": 14, "city": "Zürich"}'}]
    assert "content" not in blocks[2]


def test_foreign_and_legacy_assistant_messages_render_from_parts_without_reasoning():
    reasoning = ReasoningPart(summary="hidden", opaque={"signature": "sig"})
    foreign = message("a1", "assistant", reasoning, TextPart("Checking."), tool_call("call_1"), origin=FOREIGN, raw={"content": []})
    legacy = message("a2", "assistant", reasoning, TextPart("Done."), origin=Origin("legacy", "legacy", "legacy"))
    results = message("t1", "tool", ToolResultPart("call_1", "get_weather", (TextPart("sunny"),)))
    messages = render_body(make_request([USER, foreign, results, legacy, message("u2", "user", TextPart("Thanks"))]))["messages"]
    assert messages[1]["content"] == [
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Paris"}},
    ]
    assert messages[3]["content"] == [{"type": "text", "text": "Done."}]


def test_same_model_under_another_credential_drops_reasoning():
    stored = own_origin(api_key="old-key")
    assistant = message("a1", "assistant", ReasoningPart(summary="x", opaque={"signature": "s"}), TextPart("Hi."), origin=stored, raw={"content": [{"type": "thinking"}], "sha256": "0"})
    messages = render_body(make_request([USER, assistant, message("u2", "user", TextPart("Again"))]))["messages"]
    assert messages[1]["content"] == [{"type": "text", "text": "Hi."}]


def test_a_same_origin_reply_with_no_content_is_dropped():
    empty = message("a1", "assistant", origin=own_origin(), raw={"content": [], "sha256": anthropic_messages.digest([]), "prefix": ""})
    messages = render_body(make_request([USER, empty, message("u2", "user", TextPart("Again"))]))["messages"]
    assert messages == [{"role": "user", "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": "Again"}]}]


def test_assistant_messages_without_renderable_content_are_dropped():
    only_reasoning = message("a1", "assistant", ReasoningPart(summary="x"), TextPart(""), origin=FOREIGN)
    messages = render_body(make_request([USER, only_reasoning, message("u2", "user", TextPart("Again"))]))["messages"]
    assert [item["role"] for item in messages] == ["user"]
    assert [block["text"] for block in messages[0]["content"]] == ["Hello", "Again"]


def test_tool_ids_are_made_valid_and_stay_paired():
    ident = "call.1:" + "x" * 80
    assistant = message("a1", "assistant", tool_call(ident), origin=FOREIGN)
    results = message("t1", "tool", ToolResultPart(ident, "get_weather", (TextPart("ok"),)))
    messages = render_body(make_request([USER, assistant, results]))["messages"]
    use_id = messages[1]["content"][0]["id"]
    assert use_id == messages[2]["content"][0]["tool_use_id"]
    assert len(use_id) == 64 and use_id.startswith("call_1_xxx")


def test_unresolved_tool_pairs_are_refused_before_sending():
    assistant = message("a1", "assistant", tool_call("call_1"), tool_call("call_2"), origin=FOREIGN)
    partial = message("t1", "tool", ToolResultPart("call_1", "get_weather", (TextPart("a"),)))
    stray = message("t2", "tool", ToolResultPart("call_9", "get_weather", (TextPart("a"),)))
    full = message("t3", "tool", ToolResultPart("call_1", "get_weather", (TextPart("a"),)), ToolResultPart("call_2", "get_weather", (TextPart("b"),)))
    chatter = message("a2", "assistant", TextPart("Anyway."), origin=FOREIGN)
    for history in (
        [USER, assistant],
        [USER, assistant, chatter, message("u2", "user", TextPart("hi"))],
        [USER, assistant, partial],
        [USER, assistant, stray],
        [USER, assistant, message("u2", "user", TextPart("hi")), full],
        [USER, full],
    ):
        with pytest.raises(OpaqueViolation):
            render_body(make_request(history))
    assert render_body(make_request([USER, assistant, full]))


def test_foreign_tool_calls_without_parsed_arguments_are_sent_with_empty_input():
    cut_off = message("a1", "assistant", ToolCallPart(call_id="call_1", name="get_weather", args=None, args_raw='{"city": "Pa'), origin=FOREIGN)
    results = message("t1", "tool", ToolResultPart("call_1", "get_weather", (TextPart("cut off"),), is_error=True))
    assert render_body(make_request([USER, cut_off, results]))["messages"][1]["content"][0]["input"] == {}


def test_decoder_turns_a_thinking_tool_stream_into_events_and_a_replayable_message():
    payload = stream(
        thinking_block(0, "Plan it.", "sig-a"),
        redacted_block(1, "red-1"),
        text_block(2, "Checking ", "now."),
        tool_block(3, "toolu_1", "get_weather", "", '{"city"', ': "Paris"}'),
        stop="tool_use",
    )
    events = decode(payload)
    assert events[1:3] == [
        ReasoningDelta(block=0, text="", summary=True),
        ReasoningDelta(block=0, text="Plan it.", summary=True),
    ]
    assert OpaqueDelta(block=0, key="signature", value="sig-a") in events
    assert ReasoningDelta(block=1, text="", redacted=True) in events and OpaqueDelta(block=1, key="data", value="red-1") in events
    assert [event for event in events if isinstance(event, TextDelta)] == [TextDelta(2, "Checking "), TextDelta(2, "now.")]
    assert ToolCallStart(block=3, call_id="toolu_1", name="get_weather", wire_id="toolu_1") in events
    assert [event.text for event in events if isinstance(event, ToolArgsDelta)] == ["", '{"city"', ': "Paris"}']
    assert ToolCallEnd(block=3, args={"city": "Paris"}) in events
    finish = events[-1]
    assert isinstance(finish, Finish) and finish.reason is FinishReason.TOOL_CALLS and finish.raw_reason == "tool_use"
    assert [block["type"] for block in finish.raw["content"]] == ["thinking", "redacted_thinking", "text", "tool_use"]
    assert finish.raw["content"][0] == {"type": "thinking", "thinking": "Plan it.", "signature": "sig-a"}
    assert finish.raw["content"][3] == {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}
    assembler = StepAssembler("a1", own_origin())
    assembler.feed_all(events)
    step = assembler.message()
    assert step.parts[0] == ReasoningPart(summary="Plan it.", opaque={"signature": "sig-a"})
    assert step.parts[1] == ReasoningPart(redacted=True, opaque={"data": "red-1"})
    assert step.tool_calls()[0].args == {"city": "Paris"}


@pytest.mark.parametrize("stop, reason", [
    ("end_turn", FinishReason.STOP),
    ("stop_sequence", FinishReason.STOP),
    ("max_tokens", FinishReason.LENGTH),
    ("tool_use", FinishReason.TOOL_CALLS),
    ("pause_turn", FinishReason.PAUSE),
    ("refusal", FinishReason.REFUSAL),
    ("model_context_window_exceeded", FinishReason.CONTEXT_WINDOW),
    ("something_new", FinishReason.UNKNOWN),
])
def test_stop_reasons_map_to_finish_reasons(stop, reason):
    finish = decode(stream(text_block(0, "Hi"), stop=stop))[-1]
    assert (finish.reason, finish.raw_reason) == (reason, stop)


def test_a_stream_without_a_stop_reason_finishes_as_unknown():
    payload = sse_bytes([
        ("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 1}}}),
        ("message_stop", {"type": "message_stop"}),
    ])
    finish = decode(payload)[-1]
    assert (finish.reason, finish.raw_reason, finish.raw["content"]) == (FinishReason.UNKNOWN, None, [])


def test_usage_counts_cached_and_written_tokens_inside_the_prompt_total():
    usage = {"input_tokens": 100, "cache_creation_input_tokens": 300, "cache_read_input_tokens": 1000, "output_tokens": 1}
    events = decode(stream(text_block(0, "Hi"), usage=usage))
    reported = [event.usage for event in events if isinstance(event, UsageReported)]
    assert reported[0] == Usage(
        input_tokens=1400, output_tokens=1, total_tokens=1401, cached_input_tokens=1000, cache_write_tokens=300, raw=usage,
    )
    assembler = StepAssembler("a1", own_origin())
    assembler.feed_all(events)
    assert assembler.usage.input_tokens == 1400
    assert assembler.usage.output_tokens == 5
    assert assembler.usage.total_tokens == 1405
    assert assembler.usage.raw == {**usage, "output_tokens": 5}


def test_decoding_does_not_depend_on_chunk_boundaries():
    payload = stream(
        thinking_block(0, "Plan — 東京", "sig-a"),
        text_block(1, "Zażółć ", "gęślą jaźń"),
        tool_block(2, "toolu_1", "get_weather", '{"city": "Zürich"}'),
        stop="tool_use",
    ).replace(b"\n", b"\r\n")
    adapter = AnthropicMessagesAdapter()
    whole = decode_chunks(adapter.decoder(make_request([USER]), make_profile()), [payload])
    bytewise = decode_chunks(adapter.decoder(make_request([USER]), make_profile()), [payload[i:i + 1] for i in range(len(payload))])
    assert bytewise == whole
    assert whole[-1].raw["content"][1]["text"] == "Zażółć gęślą jaźń"


def test_pings_comments_and_unknown_events_are_ignored():
    payload = b": keep-alive\n\n" + sse_bytes([("ping", {"type": "ping"}), ("future_event", {"type": "future_event"})]) + stream(text_block(0, "Hi"))
    events = decode(payload)
    assert [event.text for event in events if isinstance(event, TextDelta)] == ["Hi"]


def test_empty_text_blocks_are_not_kept_and_unknown_blocks_stay_verbatim():
    unknown = {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"q": "x"}}
    payload = stream(
        text_block(0),
        [
            ("content_block_start", {"type": "content_block_start", "index": 1, "content_block": unknown}),
            ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ],
        text_block(2, "Done."),
    )
    events = decode(payload)
    assert events[-1].raw["content"] == [unknown, {"type": "text", "text": "Done."}]
    step = StepAssembler("a1", own_origin())
    step.feed_all(events)
    assert [part.text for part in step.message().parts] == ["Done."]


def test_whitespace_only_text_blocks_stay_out_of_the_stored_payload_and_do_not_break_replay():
    payload = stream(text_block(0, "\n\n"), tool_block(1, "toolu_1", "get_weather", '{"city": "Paris"}'), text_block(2, "  "), stop="tool_use")
    events = decode(payload)
    assert [block["type"] for block in events[-1].raw["content"]] == ["tool_use"]
    assembler = StepAssembler("a1", own_origin())
    assembler.feed_all(events)
    step = assembler.message()
    assert [part.text for part in step.parts if isinstance(part, TextPart)] == ["\n\n", "  "]
    results = message("t1", "tool", ToolResultPart("toolu_1", "get_weather", (TextPart("ok"),)))
    sent = render_body(make_request([USER, step, results]))["messages"]
    assert [block["type"] for block in sent[1]["content"]] == ["tool_use"]


def test_a_tool_call_with_cut_off_arguments_is_kept_unexecutable_and_still_replays():
    payload = stream(tool_block(0, "toolu_1", "get_weather", '{"city": "Pa'), stop="max_tokens")
    events = decode(payload)
    assert ToolCallEnd(block=0, args=None) in events
    assert events[-1].raw["content"][0]["input"] == {}
    assembler = StepAssembler("a1", own_origin())
    assembler.feed_all(events)
    step = assembler.message()
    assert step.tool_calls()[0].args is None
    results = message("t1", "tool", ToolResultPart("toolu_1", "get_weather", (TextPart("The call was cut off."),), is_error=True))
    sent = render_body(make_request([USER, step, results]))["messages"]
    assert sent[1]["content"][0] == {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}


@pytest.mark.parametrize("parts", [(), ("",)])
def test_a_tool_without_arguments_gets_empty_arguments(parts):
    events = decode(stream(tool_block(0, "toolu_1", "get_time", *parts), stop="tool_use"))
    assert ToolCallEnd(block=0, args={}) in events
    assert events[-1].raw["content"][0]["input"] == {}
    assembler = StepAssembler("a1", own_origin())
    assembler.feed_all(events)
    assert assembler.message().tool_calls()[0].args == {}


def test_tool_arguments_that_are_not_an_object_are_unusable():
    events = decode(stream(tool_block(0, "toolu_1", "get_time", "[1, 2]"), stop="tool_use"))
    assert ToolCallEnd(block=0, args=None) in events
    assert events[-1].raw["content"][0]["input"] == {}


@pytest.mark.parametrize("signature_events", [[], [("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": ""}})]])
def test_a_thinking_block_without_a_signature_cannot_be_replayed(signature_events):
    payload = stream([
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
        *signature_events,
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        *tool_block(1, "toolu_1", "get_weather", '{"city": "Paris"}'),
    ], stop="tool_use")
    assembler = StepAssembler("a1", own_origin())
    assembler.feed_all(decode(payload))
    results = message("t1", "tool", ToolResultPart("toolu_1", "get_weather", (TextPart("ok"),)))
    with pytest.raises(OpaqueViolation, match="signature"):
        render_body(make_request([USER, assembler.message(), results]))


def test_a_redacted_block_without_data_cannot_be_replayed():
    payload = stream([
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "redacted_thinking", "data": ""}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        *tool_block(1, "toolu_1", "get_weather", '{"city": "Paris"}'),
    ], stop="tool_use")
    assembler = StepAssembler("a1", own_origin())
    assembler.feed_all(decode(payload))
    results = message("t1", "tool", ToolResultPart("toolu_1", "get_weather", (TextPart("ok"),)))
    with pytest.raises(OpaqueViolation, match="data"):
        render_body(make_request([USER, assembler.message(), results]))


def test_results_find_their_calls_when_the_internal_id_differs_from_the_wire_id():
    events = decode(stream(tool_block(0, "toolu_1", "get_weather", '{"city": "Paris"}'), stop="tool_use"))
    assembler = StepAssembler("a1", own_origin())
    assembler.feed_all(events)
    step = assembler.message()
    renamed = dataclasses.replace(step, parts=tuple(dataclasses.replace(part, call_id="internal-7") for part in step.parts))
    results = message("t1", "tool", ToolResultPart("internal-7", "get_weather", (TextPart("ok"),)))
    sent = render_body(make_request([USER, renamed, results]))["messages"]
    assert sent[1]["content"][0]["id"] == "toolu_1" and sent[2]["content"][0]["tool_use_id"] == "toolu_1"


def test_the_rendered_body_does_not_alias_the_stored_payload():
    events = decode(stream(thinking_block(0, "Plan.", "sig-a"), tool_block(1, "toolu_1", "get_weather", '{"city": "Paris"}'), stop="tool_use"))
    assembler = StepAssembler("a1", own_origin())
    assembler.feed_all(events)
    step = assembler.message()
    results = message("t1", "tool", ToolResultPart("toolu_1", "get_weather", (TextPart("ok"),)))
    first = render_body(make_request([USER, step, results]))
    first["messages"][1]["content"][0]["thinking"] = "changed"
    first["messages"][1]["content"][1]["input"]["city"] = "Rome"
    assert step.raw["content"][0]["thinking"] == "Plan." and step.raw["content"][1]["input"] == {"city": "Paris"}
    assert render_body(make_request([USER, step, results]))["messages"][1]["content"][0]["thinking"] == "Plan."


@pytest.mark.parametrize("raw", [
    ["not", "a", "dict"],
    {"content": "text"},
    {"sha256": "x"},
    {"content": ["oops"], "sha256": anthropic_messages.digest(["oops"]), "prefix": ""},
    {"content": [{"type": "text", "text": "Hi"}], "sha256": "wrong", "prefix": ""},
])
def test_a_malformed_stored_payload_is_refused(raw):
    stored = message("a1", "assistant", TextPart("Hi"), origin=own_origin(), raw=raw)
    with pytest.raises(OpaqueViolation):
        render_body(make_request([USER, stored, message("u2", "user", TextPart("Again"))]))


def test_message_start_usage_is_reported_before_any_content():
    events = decode(stream(text_block(0, "Hi")))
    assert isinstance(events[0], UsageReported) and events[0].usage.input_tokens == 9


@pytest.mark.parametrize("error_type, text, expected", [
    ("overloaded_error", "Overloaded", Overloaded),
    ("rate_limit_error", "Slow down", RateLimited),
    ("api_error", "Internal server error", ServerFailed),
    ("authentication_error", "invalid x-api-key", AuthFailed),
    ("invalid_request_error", "messages.1.content.0: Invalid `signature` in `thinking` block", OpaqueViolation),
    ("invalid_request_error", "prompt is too long: 213462 tokens > 200000 maximum", ContextOverflow),
])
def test_an_error_event_mid_stream_raises_the_typed_error(error_type, text, expected):
    payload = stream(text_block(0, "Hel"), extra=[("error", {"type": "error", "error": {"type": error_type, "message": text}, "request_id": "req_1"})])
    with pytest.raises(expected) as caught:
        decode(payload)
    assert caught.value.code == error_type and caught.value.status is None
    assert caught.value.detail == {"request_id": "req_1"}


def test_an_error_event_stops_the_stream_even_inside_one_chunk():
    payload = sse_bytes([("error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})]) + stream(text_block(0, "late"))
    with pytest.raises(Overloaded):
        decode(payload)


@pytest.mark.parametrize("payload, message_part", [
    (b'event: message_start\ndata: {"type": "message_start", "message": {}}\n\n', "before message_stop"),
    (b"data: not json\n\n", "not valid JSON"),
    (b"data: [1, 2]\n\n", "not an object"),
    (sse_bytes([("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}})]), "not open"),
    (sse_bytes([("content_block_start", {"type": "content_block_start", "content_block": {"type": "text", "text": ""}})]), "no index"),
    (
        sse_bytes([
            ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ]),
        "cannot be started",
    ),
    (
        sse_bytes([
            ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            ("message_stop", {"type": "message_stop"}),
        ]),
        "was not closed",
    ),
    (
        sse_bytes([
            *text_block(0, "x"),
            ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "late"}}),
        ]),
        "not open",
    ),
    (sse_bytes([("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {}})]), "has no type"),
])
def test_malformed_streams_are_refused(payload, message_part):
    with pytest.raises(MalformedStream, match=message_part):
        decode(payload)


def test_events_after_message_stop_are_ignored():
    events = decode(stream(text_block(0, "Hi")) + stream(text_block(0, "again")))
    assert [event.text for event in events if isinstance(event, TextDelta)] == ["Hi"]


def test_finish_raw_is_sealed_for_the_replay_check():
    finish = decode(stream(text_block(0, "Hi")))[-1]
    assert set(finish.raw) == {"content", "sha256", "prefix"}
    assert finish.raw["sha256"] == anthropic_messages.digest(finish.raw["content"])


def test_prefix_digest_tracks_the_messages_sent():
    first = decode(stream(text_block(0, "Hi")), request=make_request([USER]))[-1].raw["prefix"]
    again = decode(stream(text_block(0, "Hi")), request=make_request([USER]))[-1].raw["prefix"]
    other = decode(stream(text_block(0, "Hi")), request=make_request([message("u1", "user", TextPart("Bye"))]))[-1].raw["prefix"]
    assert first == again != other


def failure(status, body, **headers):
    payload = body if isinstance(body, bytes) else json.dumps(body).encode()
    return WireFailure(status=status, body=payload, headers=headers)


def error_body(error_type, text, request_id="req_9"):
    return {"type": "error", "error": {"type": error_type, "message": text}, "request_id": request_id}


@pytest.mark.parametrize("status, body, expected, retryable", [
    (429, error_body("rate_limit_error", "Number of request tokens has exceeded your per-minute rate limit"), RateLimited, True),
    (529, error_body("overloaded_error", "Overloaded"), Overloaded, True),
    (500, error_body("api_error", "Internal server error"), ServerFailed, True),
    (504, error_body("timeout_error", "Request timed out"), ServerFailed, True),
    (401, error_body("authentication_error", "invalid x-api-key"), AuthFailed, False),
    (403, error_body("permission_error", "Not allowed"), AuthFailed, False),
    (402, error_body("billing_error", "Billing problem"), QuotaExhausted, False),
    (400, error_body("invalid_request_error", "Your credit balance is too low to access the Anthropic API."), QuotaExhausted, False),
    (400, error_body("invalid_request_error", "You have reached your specified API usage limits."), QuotaExhausted, False),
    (404, error_body("not_found_error", "model: claude-nope"), InvalidRequest, False),
    (413, error_body("request_too_large", "Request exceeds the maximum allowed number of bytes."), InvalidRequest, False),
    (400, error_body("invalid_request_error", "max_tokens: Field required"), InvalidRequest, False),
    (400, error_body("invalid_request_error", "prompt is too long: 213462 tokens > 200000 maximum"), ContextOverflow, False),
    (400, error_body("invalid_request_error", "input length and `max_tokens` exceed context limit: 188240 + 20000 > 200000, decrease input length or `max_tokens` and try again"), ContextOverflow, False),
    (400, error_body("invalid_request_error", "messages.1.content.0: Invalid `signature` in `thinking` block"), OpaqueViolation, False),
    (400, error_body("invalid_request_error", "`thinking` or `redacted_thinking` blocks in the latest assistant message cannot be modified. These blocks must remain as they were in the original response."), OpaqueViolation, False),
    (400, error_body("invalid_request_error", "messages.1.content.0.thinking: Expected `thinking` or `redacted_thinking`, but found `tool_use`. When `thinking` is enabled, a final `assistant` message must start with a thinking block"), OpaqueViolation, False),
    (400, error_body("invalid_request_error", "messages.2: `tool_use` ids were found without `tool_result` blocks immediately after: toolu_01. Each `tool_use` block must have a corresponding `tool_result` block in the next message."), OpaqueViolation, False),
    (400, error_body("invalid_request_error", "messages.1.content.0: unexpected `tool_use_id` found in `tool_result` blocks: toolu_02. Each `tool_result` block must have a corresponding `tool_use` block in the previous message."), OpaqueViolation, False),
    (400, error_body("invalid_request_error", "thinking.budget_tokens: Input should be greater than or equal to 1024"), InvalidRequest, False),
    (502, b"<html>Bad gateway</html>", ServerFailed, True),
    (529, b"", Overloaded, True),
    (503, b"{}", ServerFailed, True),
    (429, b"{}", RateLimited, True),
    (401, b"", AuthFailed, False),
    (402, b"", QuotaExhausted, False),
    (418, b"teapot", InvalidRequest, False),
])
def test_http_failures_are_classified(status, body, expected, retryable):
    error = AnthropicMessagesAdapter().classify_error(failure(status, body))
    assert type(error) is expected
    assert error.retryable is retryable
    assert error.status == status


def test_classified_errors_keep_the_provider_message_code_and_request_id():
    error = AnthropicMessagesAdapter().classify_error(failure(400, error_body("invalid_request_error", "max_tokens: Field required")))
    assert (error.message, error.code, error.detail) == ("max_tokens: Field required", "invalid_request_error", {"request_id": "req_9"})


def test_rate_limits_and_overload_carry_retry_after():
    adapter = AnthropicMessagesAdapter()
    limited = adapter.classify_error(failure(429, error_body("rate_limit_error", "slow"), **{"Retry-After": "12"}))
    busy = adapter.classify_error(failure(529, error_body("overloaded_error", "busy"), **{"retry-after": "3.5"}))
    unknown = adapter.classify_error(failure(429, error_body("rate_limit_error", "slow")))
    assert (limited.retry_after, busy.retry_after, unknown.retry_after) == (12.0, 3.5, None)


def test_a_failure_without_a_status_is_a_connection_problem():
    error = AnthropicMessagesAdapter().classify_error(WireFailure(status=None, cause=ConnectionResetError("reset by peer")))
    assert type(error) is ConnectionFailed and error.retryable and "reset by peer" in error.message
    assert type(AnthropicMessagesAdapter().classify_error(WireFailure(status=None))) is ConnectionFailed


def test_an_unparseable_error_body_is_kept_in_the_detail():
    error = AnthropicMessagesAdapter().classify_error(failure(502, b"<html>Bad gateway</html>"))
    assert error.detail == {"body": "<html>Bad gateway</html>"} and error.message == "HTTP 502"


MODELS = {
    "data": [
        {
            "id": "claude-sonnet-5-5",
            "type": "model",
            "display_name": "Claude Sonnet 5.5",
            "created_at": "2026-09-01T00:00:00Z",
            "max_input_tokens": 1000000,
            "max_tokens": 64000,
            "capabilities": {
                "image_input": {"supported": True},
                "pdf_input": {"supported": True},
                "structured_outputs": {"supported": True},
                "thinking": {"supported": True, "types": {"adaptive": {"supported": True}, "enabled": {"supported": False}}},
                "effort": {"supported": True, "low": {"supported": True}, "medium": {"supported": True}, "high": {"supported": True}, "max": {"supported": False}},
            },
        },
        {
            "id": "claude-legacy-4-5",
            "type": "model",
            "display_name": "Claude Legacy 4.5",
            "max_input_tokens": 200000,
            "max_tokens": 32000,
            "capabilities": {
                "image_input": {"supported": False},
                "structured_outputs": {"supported": False},
                "thinking": {"supported": True, "types": {"adaptive": {"supported": False}, "enabled": {"supported": True}}},
                "effort": {"supported": False},
            },
        },
        {"id": "claude-plain", "capabilities": {"thinking": {"supported": False}}},
        {
            "id": "claude-toggle",
            "capabilities": {
                "thinking": {"supported": True, "types": {"adaptive": {"supported": True}, "enabled": {"supported": False}}},
                "effort": {"supported": False},
            },
        },
        {"id": "claude-typeless", "capabilities": {"thinking": {"supported": True}}},
    ],
    "has_more": False,
}


@pytest.fixture
def models_api(monkeypatch):
    calls: List[Dict[str, Any]] = []

    async def fake(provider, base_url, path, headers=None, **kwargs):
        calls.append({"provider": provider, "base_url": base_url, "path": path, "headers": headers})
        return MODELS

    monkeypatch.setattr(anthropic_messages, "fetch_json", fake)
    return calls


async def test_discovery_reads_limits_and_capabilities_into_a_discovered_layer(models_api):
    layer = await AnthropicMessagesAdapter().discover(make_config())
    assert layer.source is ProfileSource.DISCOVERED
    assert dict(layer.fields) == {
        "context_window": 1000000,
        "max_output_tokens": 64000,
        "vision": True,
        "structured_output": "native",
        "reasoning": "effort",
        "reasoning_levels": ["low", "medium", "high"],
    }
    assert models_api == [{
        "provider": "Anthropic",
        "base_url": "https://api.anthropic.com",
        "path": "/v1/models?limit=1000",
        "headers": {"anthropic-version": "2026-10-21", "x-api-key": API_KEY},
    }]
    profile = bare_profile().layered([layer])
    assert profile.context_window == 1000000 and profile.reasoning_levels == ("low", "medium", "high")


async def test_discovery_distinguishes_manual_thinking_and_no_thinking(models_api):
    adapter = AnthropicMessagesAdapter()
    legacy = (await adapter.discover(make_config(model="claude-legacy-4-5"))).fields
    plain = (await adapter.discover(make_config(model="claude-plain"))).fields
    toggle = (await adapter.discover(make_config(model="claude-toggle"))).fields
    typeless = (await adapter.discover(make_config(model="claude-typeless"))).fields
    assert legacy["reasoning"] == "budget" and "reasoning_levels" not in legacy
    assert legacy["vision"] is False and legacy["structured_output"] == "prompted"
    assert plain == {"reasoning": "none"}
    assert toggle == {"reasoning": "toggle"} and typeless == {"reasoning": "toggle"}


async def test_discovery_fails_loudly_for_an_unlisted_model(models_api):
    with pytest.raises(ModelListingError, match="claude-missing"):
        await AnthropicMessagesAdapter().discover(make_config(model="claude-missing"))


async def test_model_listing_maps_entries_and_sorts_them(models_api):
    models = await AnthropicMessagesAdapter().list_models("https://api.anthropic.com/v1", API_KEY)
    assert [model.id for model in models] == [
        "claude-legacy-4-5", "claude-plain", "claude-sonnet-5-5", "claude-toggle", "claude-typeless",
    ]
    sonnet = models[2]
    assert (sonnet.label, sonnet.modified_at) == ("Claude Sonnet 5.5", "2026-09-01T00:00:00Z")
    assert sonnet.details == {"context_window": 1000000, "max_output_tokens": 64000}
    assert models[1].details is None and models[1].label is None
    assert models_api[0]["base_url"] == "https://api.anthropic.com"


async def test_a_model_list_in_an_unexpected_shape_is_an_error(monkeypatch):
    async def fake(*args, **kwargs):
        return {"models": []}

    monkeypatch.setattr(anthropic_messages, "fetch_json", fake)
    with pytest.raises(ModelListingError):
        await AnthropicMessagesAdapter().list_models("https://api.anthropic.com", API_KEY)


@pytest.mark.parametrize("cap, expected", [(64000, 64000), (None, 4096)])
def test_an_unset_max_tokens_falls_back_to_the_models_output_limit_because_the_api_requires_one(cap, expected):
    profile = make_profile(max_output_tokens=cap) if cap else bare_profile()
    request = make_request([USER], max_tokens=None)

    assert render_body(request, profile)["max_tokens"] == expected
