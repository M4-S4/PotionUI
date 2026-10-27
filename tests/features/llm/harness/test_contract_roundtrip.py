import json
from dataclasses import fields
from typing import Tuple

import httpx
import pytest

from src.features.llm.harness import codec
from src.features.llm.harness.adapter import Adapter, Settings, ToolChoice, ToolSpec, WireRequest
from src.features.llm.harness.assembler import StepAssembler, parse_args
from src.features.llm.harness.context import user_text
from src.features.llm.harness.errors import (
    ERROR_KINDS,
    HarnessError,
    MalformedStream,
    RateLimited,
    WireFailure,
)
from src.features.llm.harness.events import (
    Finish,
    FinishReason,
    OpaqueDelta,
    ReasoningDelta,
    StreamEvent,
    TextDelta,
    ToolArgsDelta,
    ToolCallEnd,
    ToolCallStart,
    Usage,
    UsageReported,
)
from src.features.llm.harness.messages import (
    LEGACY_ORIGIN,
    CompactionPart,
    ContextPart,
    FilePart,
    ImagePart,
    JsonPart,
    Message,
    Origin,
    Part,
    ReasoningPart,
    Role,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    credential_fingerprint,
    message,
)
from src.features.llm.harness.profile import (
    CacheMode,
    CapabilityProfile,
    Prices,
    ProfileLayer,
    ProfileSource,
    ReasoningKind,
    ReplayPolicy,
    Verification,
)
from src.features.llm.harness.turn_state import (
    DecisionKind,
    Notice,
    NoticeKind,
    PendingCall,
    StepRecord,
    StepStatus,
    TurnSnapshot,
    TurnState,
    TurnStatus,
)
from tests.fixtures.llm_harness import ScriptedAdapter, assemble, decode_chunks, text_step, tool_step

ANTHROPIC = Origin(adapter="anthropic-messages", provider="anthropic", model="claude-sonnet-5-5", credential="0f3a9c1d2e4b5a6c")

RAW_BLOCKS = [
    {"type": "thinking", "thinking": "", "signature": "EqQBCkYIBxgCKkA="},
    {"type": "redacted_thinking", "data": "EmwKAhgB"},
    {"type": "text", "text": "Checking both.", "citations": None},
    {"type": "tool_use", "id": "toolu_01", "name": "search", "input": {"q": "ż", "n": 2, "deep": {"x": [1, 2.5, None, True]}}},
]

ALL_PARTS = [
    TextPart(text="héllo 🌍 世界\n\"quoted\"\t\u0000"),
    TextPart(text="final", phase="final_answer"),
    ReasoningPart(),
    ReasoningPart(text="raw chain", summary="short", redacted=False, opaque={"signature": "sig=="}),
    ReasoningPart(redacted=True, opaque={"data": "EmwKAhgB"}),
    ReasoningPart(text="timed", duration_ms=1840),
    ToolCallPart(call_id="call_1", name="search", args={"q": "x", "nested": {"k": [1, 2]}}, args_raw='{"q": "x", "nested": {"k": [1,2]}}', wire_id="toolu_01", opaque={"thought_signature": "abc"}),
    ToolCallPart(call_id="call_2", name="broken", args=None, args_raw='{"q": "trunc'),
    ToolCallPart(call_id="call_3", name="ollama_style", args={"a": 1}),
    ToolCallPart(call_id="call_4", name="empty", args={}),
    ToolResultPart(call_id="call_1", name="search", content=(TextPart(text="ok"), ImagePart(media_type="image/png", ref="media://42"), JsonPart(value={"rows": [1, None, "x"]}))),
    ToolResultPart(call_id="call_2", name="broken", content=(), is_error=True),
    ImagePart(media_type="image/jpeg", data="aGVsbG8="),
    FilePart(media_type="application/pdf", name="spec.pdf", ref="files/spec.pdf"),
    ContextPart(kind="memory", text="<memory>likes cats</memory>"),
    CompactionPart(summary="earlier: talked about cats", opaque={"encrypted_content": "gAAAA"}),
]


def roundtrip(hint, value):
    encoded = codec.dumps(value)
    decoded = codec.loads(hint, encoded)
    assert decoded == value
    assert codec.dumps(decoded) == encoded
    return decoded


@pytest.mark.parametrize("part", ALL_PARTS, ids=lambda part: f"{type(part).__name__}")
def test_every_part_round_trips_losslessly(part):
    decoded = roundtrip(Part, part)
    assert type(decoded) is type(part)


def test_a_full_conversation_round_trips_with_raw_and_origin():
    conversation = (
        message("m0", "system", TextPart(text="You are terse.")),
        message("m1", "user", TextPart(text="find cats"), ImagePart(media_type="image/jpeg", data="aGk="), ContextPart(kind="workspace", text="tab: 3")),
        message("m2", "assistant", ReasoningPart(opaque={"signature": "EqQBCkYIBxgCKkA="}), ReasoningPart(redacted=True, opaque={"data": "EmwKAhgB"}), TextPart(text="Checking both."), ToolCallPart(call_id="c1", wire_id="toolu_01", name="search", args=RAW_BLOCKS[3]["input"], args_raw=json.dumps(RAW_BLOCKS[3]["input"])), origin=ANTHROPIC, raw=RAW_BLOCKS),
        message("m3", "tool", ToolResultPart(call_id="c1", name="search", content=(TextPart(text="2 hits"),))),
        message("m4", "assistant", TextPart(text="Done."), origin=LEGACY_ORIGIN),
    )
    decoded = roundtrip(Tuple[Message, ...], conversation)
    assert decoded[2].raw == RAW_BLOCKS
    assert decoded[2].replays_raw_to(ANTHROPIC)
    assert not decoded[2].replays_raw_to(Origin(adapter="anthropic-messages", provider="anthropic", model="claude-sonnet-5-5", credential="ffffffffffffffff"))
    assert decoded[1].role is Role.USER


def test_a_reasoning_part_stored_before_it_had_a_duration_still_loads():
    stored = {"type": "reasoning", "text": "x", "summary": "", "redacted": False, "opaque": None}

    loaded = codec.from_json(Part, stored)

    assert loaded == ReasoningPart(text="x") and loaded.duration_ms is None
    assert codec.to_json(ReasoningPart(text="x", duration_ms=12))["duration_ms"] == 12
    with pytest.raises(ValueError):
        codec.from_json(Part, {**stored, "duration_ms": "12"})


def test_stored_json_is_ascii_and_keeps_the_part_tag_first():
    encoded = codec.dumps(message("m1", "user", TextPart(text="zażółć 🌍")))
    assert encoded.isascii()
    assert json.loads(encoded)["parts"][0] == {"type": "text", "text": "zażółć 🌍", "phase": None}


@pytest.mark.parametrize(
    "payload, error",
    [
        ({"type": "nope", "text": "x"}, ValueError),
        ({"type": "text", "text": "x", "phase": None, "extra": 1}, ValueError),
        ({"type": "text"}, TypeError),
        ({"type": "image", "media_type": "image/png", "data": "x", "ref": "y"}, ValueError),
        ({"type": "tool_call", "call_id": "c", "name": "n", "args": "not-an-object", "args_raw": None, "wire_id": None, "opaque": None}, ValueError),
        ({"type": "text", "text": 5, "phase": None}, ValueError),
    ],
)
def test_decoding_is_strict(payload, error):
    with pytest.raises(error):
        codec.from_json(Part, payload)


@pytest.mark.parametrize(
    "role, part",
    [
        ("user", ToolCallPart(call_id="c", name="n", args={})),
        ("user", ReasoningPart(text="x")),
        ("tool", TextPart(text="x")),
        ("system", ImagePart(media_type="image/png", ref="r")),
        ("assistant", ToolResultPart(call_id="c", name="n", content=())),
    ],
)
def test_a_message_refuses_parts_its_role_cannot_carry(role, part):
    with pytest.raises(ValueError):
        message("m", role, part, origin=ANTHROPIC)


def test_an_assistant_message_needs_an_origin():
    with pytest.raises(ValueError):
        message("m", "assistant", TextPart(text="x"))


def every_event():
    return [
        ReasoningDelta(block=0, text="", summary=False),
        OpaqueDelta(block=0, key="signature", value="sig"),
        ReasoningDelta(block=1, text="sum", summary=True, redacted=False),
        TextDelta(block=2, text="hi", phase="commentary"),
        ToolCallStart(block=3, call_id="c1", name="search", wire_id="call_abc"),
        ToolArgsDelta(block=3, text='{"q":'),
        ToolArgsDelta(block=3, text='"x"}'),
        ToolCallEnd(block=3),
        ToolCallStart(block=4, call_id="c2", name="obj"),
        ToolCallEnd(block=4, args={"a": 1}),
        UsageReported(Usage(input_tokens=100, cached_input_tokens=80, raw={"input_tokens": 100})),
        UsageReported(Usage(output_tokens=7, reasoning_tokens=3, cost=0.0004)),
        Finish(reason=FinishReason.TOOL_CALLS, raw_reason="tool_use", raw=RAW_BLOCKS),
    ]


def test_stream_events_round_trip_for_cassettes():
    roundtrip(Tuple[StreamEvent, ...], tuple(every_event()))


def test_the_assembler_builds_parts_in_block_order_and_merges_usage():
    assembler = StepAssembler("m9", ANTHROPIC)
    events = every_event()
    assembler.feed_all([events[4], events[0], *events[1:4], *events[5:]])
    built = assembler.message()
    assert [type(part).__name__ for part in built.parts] == ["ReasoningPart", "ReasoningPart", "TextPart", "ToolCallPart", "ToolCallPart"]
    assert built.parts[0] == ReasoningPart(text="", opaque={"signature": "sig"})
    assert built.parts[1].summary == "sum"
    assert built.parts[2] == TextPart(text="hi", phase="commentary")
    assert built.parts[3] == ToolCallPart(call_id="c1", name="search", args={"q": "x"}, args_raw='{"q":"x"}', wire_id="call_abc")
    assert built.parts[4] == ToolCallPart(call_id="c2", name="obj", args={"a": 1})
    assert built.raw == RAW_BLOCKS
    assert assembler.usage == Usage(input_tokens=100, output_tokens=7, reasoning_tokens=3, cached_input_tokens=80, cost=0.0004, raw={"input_tokens": 100})
    assert assembler.finish.reason is FinishReason.TOOL_CALLS
    roundtrip(Message, built)


def test_a_cut_off_tool_call_never_gets_empty_arguments():
    built = assemble([
        ToolCallStart(block=0, call_id="c", name="n"),
        ToolArgsDelta(block=0, text='{"q": "unterminated'),
        Finish(reason=FinishReason.LENGTH),
    ])
    assert built.parts[0].args is None
    assert built.parts[0].args_raw == '{"q": "unterminated'
    unended = assemble([
        ToolCallStart(block=0, call_id="c", name="n"),
        ToolArgsDelta(block=0, text='{"q": "x"}'),
        Finish(reason=FinishReason.LENGTH),
    ])
    assert unended.parts[0].args is None
    assert parse_args("") == {}
    assert parse_args("[1]") is None


@pytest.mark.parametrize(
    "events",
    [
        [OpaqueDelta(block=0, key="signature", value="x")],
        [TextDelta(block=0, text="a"), ReasoningDelta(block=0, text="b")],
        [ToolArgsDelta(block=0, text="{}")],
        [Finish(reason=FinishReason.STOP), TextDelta(block=0, text="late")],
    ],
)
def test_the_assembler_refuses_malformed_streams(events):
    with pytest.raises(MalformedStream):
        assemble(events)


def full_profile():
    return CapabilityProfile(
        adapter="ollama-chat",
        model="qwen3.8:14b",
        context_window=65536,
        tool_choice=frozenset({"auto", "none"}),
        reasoning=ReasoningKind.TOGGLE,
        reasoning_levels=("low", "high"),
        replay=ReplayPolicy.PRESERVE,
        cache=CacheMode.LOCAL_PREFIX,
        prices=Prices(input=0.0, output=0),
        verified=Verification(suite="chat_eval:v3", passed_at="2026-10-27", runtime="ollama 0.13.4", gpu_tier="16gb"),
        sources={"context_window": ProfileSource.DISCOVERED},
    )


def test_a_turn_state_with_its_snapshot_round_trips():
    state = TurnState(
        turn_id="t1",
        session_id="s1",
        status=TurnStatus.AWAITING_DECISION,
        snapshot=TurnSnapshot(
            llm_config_id="cfg",
            system="You are terse.",
            tools=(ToolSpec(name="search", description="find", parameters={"type": "object", "properties": {}}),),
            settings=Settings(temperature=0.2, max_tokens=900, reasoning="high", tool_choice=ToolChoice(mode="named", name="search")),
            profile=full_profile(),
            mode="assistant",
            deferred_tools={"gallery": ("search_gallery", "organize_gallery")},
        ),
        first_seq=7,
        rounds_used=2,
        pending=(PendingCall(call_id="c2", name="delete_prompt", decision=DecisionKind.APPROVAL, preview={"before": ["a"]}),),
        gathered=(ToolResultPart(call_id="c1", name="search", content=(TextPart(text="2 hits"),)),),
        failed_calls=frozenset({"search:{\"q\":\"x\"}"}),
        tool_failures={"search": 1},
        applied_calls=frozenset({"a", "b"}),
        loaded_tool_groups=("gallery",),
        notices=(Notice(kind=NoticeKind.REASONING_DROPPED, message="Reasoning from another key was left out.", step_seq=9, detail={"parts": 2}),),
        error=RateLimited("slow down", status=429, retry_after=2.5).to_json(),
    )
    decoded = roundtrip(TurnState, state)
    assert decoded.snapshot.profile.prices.output == 0.0
    assert HarnessError.from_json(decoded.error) == RateLimited("slow down", status=429, retry_after=2.5)


def test_a_step_record_round_trips():
    step = StepRecord(
        id="st1", session_id="s1", turn_id="t1", seq=3,
        message=message("m2", "assistant", TextPart(text="x"), origin=ANTHROPIC, raw=RAW_BLOCKS),
        status=StepStatus.CANCELLED, usage=Usage(input_tokens=1), finish=FinishReason.LENGTH, finish_raw="max_tokens",
        created_at="2026-10-26T19:00:00Z",
    )
    roundtrip(StepRecord, step)


def test_profile_layers_apply_in_order_and_record_their_source():
    base = CapabilityProfile(adapter="ollama-chat", model="qwen3.8:14b")
    profile = base.layered([
        ProfileLayer(ProfileSource.SEED, {"context_window": 32768, "replay": "drop_earlier", "tool_choice": ["auto"]}),
        ProfileLayer(ProfileSource.DISCOVERED, {"context_window": 65536, "vision": True}),
        ProfileLayer(ProfileSource.OVERRIDE, {"replay": "preserve", "prices": {"input": 1, "output": 2, "cached_input": None, "cache_write": None, "reasoning": None}}),
    ])
    assert profile.context_window == 65536
    assert profile.replay is ReplayPolicy.PRESERVE
    assert profile.tool_choice == frozenset({"auto"})
    assert profile.prices == Prices(input=1.0, output=2.0)
    assert profile.sources == {
        "context_window": ProfileSource.DISCOVERED,
        "replay": ProfileSource.OVERRIDE,
        "tool_choice": ProfileSource.SEED,
        "vision": ProfileSource.DISCOVERED,
        "prices": ProfileSource.OVERRIDE,
    }
    roundtrip(CapabilityProfile, profile)
    with pytest.raises(ValueError):
        base.layered([ProfileLayer(ProfileSource.OVERRIDE, {"adapter": "other"})])
    with pytest.raises(ValueError):
        base.layered([ProfileLayer(ProfileSource.OVERRIDE, {"replay": "sometimes"})])


@pytest.mark.parametrize("kind", sorted(ERROR_KINDS))
def test_every_error_kind_round_trips(kind):
    error = ERROR_KINDS[kind]("boom", status=503, code="x", retry_after=1.0, detail={"body": "y"})
    assert HarnessError.from_json(json.loads(json.dumps(error.to_json()))) == error
    assert ERROR_KINDS[kind].kind == kind


def test_retry_after_is_read_case_insensitively():
    assert WireFailure(status=429, headers={"Retry-After": "3"}).retry_after_seconds() == 3.0
    assert WireFailure(status=429, headers={"retry-after": "Wed"}).retry_after_seconds() is None


@pytest.mark.parametrize(
    "parts, rendered",
    [
        ((TextPart(text="hello"),), "hello"),
        (
            (TextPart(text="hello"), ContextPart(kind="memory", text="m"), ContextPart(kind="workspace", text="w")),
            "hello\n\n<context>\nm\n</context>\n\n<context>\nw\n</context>",
        ),
        ((ContextPart(kind="memory", text="m"), ContextPart(kind="workspace", text="w")), "m\n\n<context>\nw\n</context>"),
        ((TextPart(text=""), ContextPart(kind="nudge", text="n")), "\n\n<context>\nn\n</context>"),
    ],
)
def test_user_text_renders_context_blocks_like_the_pre_harness_wire(parts, rendered):
    assert user_text(message("m", "user", *parts)) == rendered


def test_wire_content_is_byte_identical_to_httpx_json():
    body = {"model": "m", "messages": [{"role": "user", "content": "zażółć 🌍 \"q\""}], "temperature": 0.7, "tools": [ToolSpec(name="t", description="d", parameters={"type": "object"}).function_schema()]}
    wire = WireRequest(method="POST", url="http://x.invalid/v1/chat/completions", headers={"Content-Type": "application/json"}, body=body)
    assert wire.content() == httpx.Request("POST", wire.url, json=body).content


def test_tool_spec_reproduces_the_registry_schema():
    schema = {"type": "function", "function": {"name": "t", "description": "d", "parameters": {"type": "object"}}}
    assert ToolSpec.from_schema(schema).function_schema() == schema


def test_settings_map_todays_options_override():
    settings = Settings.from_options({"max_tokens": 24, "temperature": 0.3, "think": False})
    assert (settings.max_tokens, settings.temperature, settings.reasoning) == (24, 0.3, False)
    roundtrip(Settings, settings)


def test_credential_fingerprint_is_stable_and_hides_the_key():
    first = credential_fingerprint("https://api.openai.com/v1/", "sk-secret")
    assert first == credential_fingerprint("https://api.openai.com/v1", "sk-secret")
    assert first != credential_fingerprint("https://api.openai.com/v1", "sk-other")
    assert "sk-secret" not in first and len(first) == 16


async def test_the_scripted_double_satisfies_the_adapter_protocol():
    adapter = ScriptedAdapter(tool_step({"name": "search", "args": {"q": "x"}}, reasoning="think"), text_step("done"))
    assert isinstance(adapter, Adapter)
    profile = CapabilityProfile(adapter="scripted", model="scripted-model")
    built = []
    for _ in range(2):
        wire = adapter.render(None, profile)
        chunks = [chunk async for chunk in adapter.send(wire, None)]
        built.append(assemble(decode_chunks(adapter.decoder(None, profile), chunks)))
    assert built[0].reasoning()[0].text == "think"
    assert built[0].tool_calls()[0].args == {"q": "x"}
    assert built[1].text() == "done"
    assert adapter.exhausted


def test_every_contract_dataclass_is_frozen():
    for cls in (Message, Origin, TextPart, ReasoningPart, ToolCallPart, ToolResultPart, ImagePart, FilePart, ContextPart, CompactionPart, Usage, CapabilityProfile, TurnState, StepRecord):
        assert cls.__dataclass_params__.frozen, cls.__name__
        assert fields(cls)
