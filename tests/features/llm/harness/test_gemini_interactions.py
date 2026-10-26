import json
from typing import Any, Dict, List, Optional

import pytest

from src.features.llm.harness import codec
from src.features.llm.harness.adapter import Adapter, RenderRequest, Settings, ToolChoice, ToolSpec
from src.features.llm.harness.adapters import gemini_interactions as gemini
from src.features.llm.harness.adapters.gemini_interactions import (
    ADAPTER_NAME,
    GeminiInteractionsAdapter,
    discovered_fields,
)
from src.features.llm.harness.errors import (
    AuthFailed,
    ConnectionFailed,
    ContextOverflow,
    HarnessError,
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
    ContextPart,
    FilePart,
    ImagePart,
    JsonPart,
    Message,
    Origin,
    ReasoningPart,
    Role,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    message,
)
from src.features.llm.harness.profile import CapabilityProfile, ProfileLayer, ProfileSource
from src.features.llm.model_listing import ModelListingError
from src.features.llm.repository import LLMConfig
from tests.features.llm.harness.wire_peer import chunked as peer_chunks
from tests.features.llm.harness.wire_peer import install_peer
from tests.fixtures.llm_harness import assemble, decode_chunks

SIGNATURE = "EpoGCpcGAXLI2nx/sig-one"
SUMMARY_ONE = {"type": "text", "text": "**Reading the request**\n\nThe user wants two actions."}
SUMMARY_TWO = {"type": "text", "text": "**Planning**\n\nCall both tools together."}

LIGHT_TOOL = ToolSpec(
    name="dim_lights",
    description="Dim the lights.",
    parameters={"type": "object", "properties": {"level": {"type": "number"}}, "required": ["level"]},
)
MUSIC_TOOL = ToolSpec(name="start_music", description="Start music.", parameters={"type": "object", "properties": {}})


def config(**overrides: Any) -> LLMConfig:
    fields = dict(
        id="c1",
        name="Gemini",
        type="gemini",
        enabled=True,
        base_url="https://generativelanguage.googleapis.com",
        api_key="test-key",
        model="gemini-3.8-flash",
        system_message="",
        temperature=0.7,
        max_tokens=2000,
    )
    fields.update(overrides)
    return LLMConfig(**fields)


def profile(**fields: Any) -> CapabilityProfile:
    base = CapabilityProfile(adapter=ADAPTER_NAME, model="gemini-3.8-flash")
    return base.layered([ProfileLayer(ProfileSource.OVERRIDE, fields)]) if fields else base


THINKING = dict(reasoning="effort", reasoning_levels=["minimal", "low", "medium", "high"])


def request(messages, *, tools=(), settings: Optional[Settings] = None, system="", **config_fields) -> RenderRequest:
    return RenderRequest(
        config=config(**config_fields),
        system=system,
        messages=tuple(messages),
        tools=tuple(tools),
        settings=settings or Settings(),
    )


def origin(model: str = "gemini-3.8-flash", adapter: str = ADAPTER_NAME, **config_fields: Any) -> Origin:
    return Origin.for_config(adapter, config(model=model, **config_fields))


def user(text: str = "hi", *parts: Any) -> Message:
    return message("u1", Role.USER, TextPart(text), *parts)


def assistant_message(*, raw: Any, signature: str = SIGNATURE, source: Optional[Origin] = None) -> Message:
    return message(
        "a1",
        Role.ASSISTANT,
        ReasoningPart(
            summary=SUMMARY_ONE["text"] + "\n\n" + SUMMARY_TWO["text"],
            opaque={"signature": signature, "summary": [SUMMARY_ONE, SUMMARY_TWO]},
        ),
        TextPart("On it."),
        ToolCallPart(call_id="fc_1", name="dim_lights", args={"level": 20}, args_raw='{"level": 20}', wire_id="fc_1"),
        ToolCallPart(call_id="fc_2", name="start_music", args={}, args_raw="{}", wire_id="fc_2"),
        origin=source or origin(),
        raw=raw,
    )


RAW_STEPS = [
    {"type": "thought", "signature": SIGNATURE, "summary": [SUMMARY_ONE, SUMMARY_TWO]},
    {"type": "model_output", "content": [{"type": "text", "text": "On it."}]},
    {"type": "function_call", "id": "fc_1", "name": "dim_lights", "arguments": {"level": 20}},
    {"type": "function_call", "id": "fc_2", "name": "start_music", "arguments": {}},
]

RESULTS = message(
    "t1",
    Role.TOOL,
    ToolResultPart(call_id="fc_1", name="dim_lights", content=(TextPart("dimmed to 20"),)),
    ToolResultPart(call_id="fc_2", name="start_music", content=(TextPart("denied"),), is_error=True),
)


def history(raw: Any = RAW_STEPS, **kwargs: Any) -> List[Message]:
    return [user("lights and music"), assistant_message(raw=raw, **kwargs), RESULTS]


def render(messages, *, profile_fields: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    adapter = GeminiInteractionsAdapter()
    wire = adapter.render(request(messages, **kwargs), profile(**(profile_fields or {})))
    return wire.body


def event(kind: str, **payload: Any) -> bytes:
    return f"event: {kind}\ndata: {json.dumps({**payload, 'event_type': kind})}\n\n".encode()


def stream(*records: bytes, done: bool = True) -> List[bytes]:
    return [*records, *([b"event: done\ndata: [DONE]\n\n"] if done else [])]


def delta(index: int, **payload: Any) -> bytes:
    return event("step.delta", index=index, delta=payload)


def start(index: int, **step: Any) -> bytes:
    return event("step.start", index=index, step=step)


def stop(index: int) -> bytes:
    return event("step.stop", index=index)


def completed(status: str = "completed", **interaction: Any) -> bytes:
    return event("interaction.completed", interaction={"id": "v1_x", "status": status, **interaction})


def decode(records: List[bytes]) -> list:
    return decode_chunks(gemini.GeminiDecoder(), records)


TOOL_STREAM = stream(
    event("interaction.created", interaction={"id": "v1_x", "status": "in_progress", "model": "gemini-3.8-flash"}),
    event("interaction.status_update", interaction_id="v1_x", status="in_progress"),
    start(0, type="thought"),
    delta(0, type="thought_summary", content=SUMMARY_ONE),
    delta(0, type="thought_summary", content=SUMMARY_TWO),
    delta(0, type="thought_signature", signature=SIGNATURE),
    stop(0),
    start(1, type="model_output"),
    delta(1, type="text", text="On "),
    delta(1, type="text", text="it."),
    stop(1),
    start(2, type="function_call", id="fc_1", name="dim_lights", arguments={}),
    delta(2, type="arguments_delta", arguments='{"level"'),
    delta(2, type="arguments_delta", arguments=": 20}"),
    stop(2),
    start(3, type="function_call", id="fc_2", name="start_music", arguments={}),
    stop(3),
    event("interaction.status_update", interaction_id="v1_x", status="requires_action"),
    completed(
        "requires_action",
        usage={"total_input_tokens": 100, "total_output_tokens": 20, "total_thought_tokens": 30,
               "total_cached_tokens": 64, "total_tokens": 150},
    ),
)


class TestRender:
    def test_wire_shape(self):
        wire = GeminiInteractionsAdapter().render(
            request([user("hello")], system="Be brief.", tools=[LIGHT_TOOL]), profile()
        )
        assert wire.method == "POST"
        assert wire.url == "https://generativelanguage.googleapis.com/v1beta/interactions"
        assert wire.headers == {"x-goog-api-key": "test-key", "content-type": "application/json"}
        assert wire.stream is True
        assert wire.body == {
            "model": "gemini-3.8-flash",
            "input": [{"type": "user_input", "content": [{"type": "text", "text": "hello"}]}],
            "store": False,
            "stream": True,
            "system_instruction": "Be brief.",
            "tools": [{
                "type": "function",
                "name": "dim_lights",
                "description": "Dim the lights.",
                "parameters": LIGHT_TOOL.parameters,
            }],
            "generation_config": {"temperature": 0.7, "max_output_tokens": 2000},
        }

    @pytest.mark.parametrize("base_url", [
        "https://generativelanguage.googleapis.com/",
        "https://generativelanguage.googleapis.com/v1beta",
        "https://generativelanguage.googleapis.com/v1beta/",
        "",
    ])
    def test_base_url_forms_reach_one_endpoint(self, base_url):
        wire = GeminiInteractionsAdapter().render(request([user()], base_url=base_url), profile())
        assert wire.url == "https://generativelanguage.googleapis.com/v1beta/interactions"

    def test_model_resource_prefix_is_stripped(self):
        assert render([user()], model="models/gemini-3.8-flash")["model"] == "gemini-3.8-flash"

    def test_missing_key_fails_before_sending(self):
        with pytest.raises(AuthFailed):
            GeminiInteractionsAdapter().render(request([user()], api_key=None), profile())

    def test_system_messages_are_refused(self):
        with pytest.raises(InvalidRequest):
            render([message("s1", Role.SYSTEM, TextPart("rules"))])

    def test_response_schema_becomes_response_format(self):
        schema = {"type": "object", "properties": {"title": {"type": "string"}}}
        body = render([user()], settings=Settings(response_schema=schema))
        assert body["response_format"] == {"type": "text", "mime_type": "application/json", "schema": schema}

    def test_settings_override_config_and_rejected_sampling_is_dropped(self):
        body = render([user()], settings=Settings(temperature=0.2, top_p=0.9, max_tokens=64, top_k=5, min_p=0.1))
        assert body["generation_config"] == {"temperature": 0.2, "top_p": 0.9, "max_output_tokens": 64}
        body = render(
            [user()], settings=Settings(top_p=0.9), profile_fields={"sampling_rejected": ["temperature", "top_p"]},
        )
        assert body["generation_config"] == {"max_output_tokens": 2000}

    def test_user_media_context_and_unloaded_images(self):
        content = render([message(
            "u1", Role.USER,
            TextPart("look"),
            ImagePart(media_type="image/png", data="QUJD"),
            ImagePart(media_type="image/png", ref="uploads/a.png"),
            FilePart(media_type="application/pdf", data="UERG"),
            ContextPart(kind="memory", text="remember this"),
        )])["input"][0]["content"]
        assert content == [
            {"type": "image", "data": "QUJD", "mime_type": "image/png"},
            {"type": "document", "data": "UERG", "mime_type": "application/pdf"},
            {"type": "text", "text": "look\n\n<context>\nremember this\n</context>"},
        ]

    def test_message_without_sendable_content_is_skipped(self):
        body = render([message("u0", Role.USER, ImagePart(media_type="image/png", ref="a.png")), user("hi")])
        assert [step["type"] for step in body["input"]] == ["user_input"]


class TestThinkingConfig:
    def generation(self, settings: Optional[Settings] = None, fields: Optional[Dict[str, Any]] = None, **cfg: Any):
        return render([user()], settings=settings, profile_fields=fields, **cfg)["generation_config"]

    def test_non_thinking_model_gets_no_thinking_fields(self):
        generation = self.generation(Settings(reasoning="high"), show_thinking=True)
        assert "thinking_level" not in generation and "thinking_summaries" not in generation

    def test_level_is_passed_when_the_profile_lists_it(self):
        assert self.generation(Settings(reasoning="medium"), THINKING)["thinking_level"] == "medium"

    def test_level_outside_the_profile_is_left_to_the_model_default(self):
        fields = dict(reasoning="effort", reasoning_levels=["low", "high"])
        assert "thinking_level" not in self.generation(Settings(reasoning="minimal"), fields)

    def test_reasoning_off_picks_the_lowest_level(self):
        generation = self.generation(Settings(reasoning=False), THINKING, show_thinking=True)
        assert generation["thinking_level"] == "minimal"
        assert "thinking_summaries" not in generation

    def test_reasoning_off_on_a_model_that_cannot_reach_minimal_keeps_summaries(self):
        fields = dict(reasoning="effort", reasoning_levels=["low", "high"])
        generation = self.generation(Settings(reasoning=False), fields, show_thinking=True)
        assert generation["thinking_level"] == "low"
        assert generation["thinking_summaries"] == "auto"

    def test_always_on_model_ignores_reasoning_off(self):
        fields = dict(reasoning="always_on", reasoning_levels=["low", "high"])
        assert "thinking_level" not in self.generation(Settings(reasoning=False), fields)

    def test_summaries_follow_show_thinking(self):
        assert self.generation(None, THINKING, show_thinking=True)["thinking_summaries"] == "auto"
        assert "thinking_summaries" not in self.generation(None, THINKING, show_thinking=False)


class TestToolChoice:
    FIELDS = dict(tool_choice=["auto", "none", "required", "named"])

    def choice(self, settings_choice, fields=None, settings_reasoning=None):
        body = render(
            [user()], tools=[LIGHT_TOOL], profile_fields=fields,
            settings=Settings(tool_choice=settings_choice, reasoning=settings_reasoning),
        )
        return body["generation_config"].get("tool_choice")

    def test_modes_map_to_the_wire_vocabulary(self):
        assert self.choice(ToolChoice("required"), self.FIELDS) == "any"
        assert self.choice(ToolChoice("none"), self.FIELDS) == "none"
        assert self.choice(ToolChoice("auto"), self.FIELDS) is None
        assert self.choice(ToolChoice("named", "dim_lights"), self.FIELDS) == {
            "allowed_tools": {"mode": "any", "tools": ["dim_lights"]}
        }

    def test_modes_the_profile_does_not_allow_are_not_forced(self):
        assert self.choice(ToolChoice("required")) is None

    def test_reasoning_uses_its_own_allowed_set(self):
        fields = {**THINKING, **self.FIELDS, "tool_choice_with_reasoning": ["auto"]}
        assert self.choice(ToolChoice("required"), fields) is None
        assert self.choice(ToolChoice("required"), fields, settings_reasoning=False) == "any"

    def test_named_choice_needs_a_name(self):
        with pytest.raises(InvalidRequest):
            self.choice(ToolChoice("named"), self.FIELDS)

    def test_no_tools_means_no_tool_choice(self):
        body = render([user()], settings=Settings(tool_choice=ToolChoice("required")), profile_fields=self.FIELDS)
        assert "tool_choice" not in body["generation_config"]


class TestHistory:
    def test_same_origin_history_is_replayed_exactly(self):
        body = render(history(), profile_fields=THINKING, tools=[LIGHT_TOOL, MUSIC_TOOL])
        assert body["input"] == [
            {"type": "user_input", "content": [{"type": "text", "text": "lights and music"}]},
            *RAW_STEPS,
            {"type": "function_result", "call_id": "fc_1", "name": "dim_lights",
             "result": [{"type": "text", "text": "dimmed to 20"}]},
            {"type": "function_result", "call_id": "fc_2", "name": "start_music",
             "result": [{"type": "text", "text": "denied"}], "is_error": True},
        ]

    def test_rendering_from_parts_matches_rendering_from_raw(self):
        from_raw = GeminiInteractionsAdapter().render(request(history()), profile(**THINKING))
        from_parts = GeminiInteractionsAdapter().render(request(history(raw=None)), profile(**THINKING))
        assert from_parts.content() == from_raw.content()

    def test_switching_gemini_models_keeps_the_thoughts(self):
        other = origin(model="gemini-3.1-pro-preview")
        from_other = GeminiInteractionsAdapter().render(
            request(history(source=other, raw=RAW_STEPS)), profile(**THINKING)
        )
        from_same = GeminiInteractionsAdapter().render(request(history()), profile(**THINKING))
        assert from_other.content() == from_same.content()
        assert from_other.body["input"][1]["signature"] == SIGNATURE

    def test_foreign_tool_history_gets_the_placeholder_signature_and_loses_foreign_reasoning(self):
        foreign = Origin(adapter="anthropic-messages", provider="anthropic", model="claude", credential="x")
        assistant = message(
            "a1", Role.ASSISTANT,
            ReasoningPart(text="secret", opaque={"signature": "anthropic-sig"}),
            TextPart("Checking."),
            ToolCallPart(call_id="toolu_1", name="dim_lights", args={"level": 5}, args_raw='{"level":5}'),
            ToolCallPart(call_id="toolu_2", name="start_music", args={}, args_raw="{}"),
            origin=foreign, raw=[{"type": "thinking", "signature": "anthropic-sig"}],
        )
        results = message("t1", Role.TOOL, ToolResultPart(call_id="toolu_1", name="dim_lights", content=(TextPart("ok"),)))
        steps = render([user(), assistant, results], profile_fields=THINKING)["input"]
        assert steps[1:5] == [
            {"type": "model_output", "content": [{"type": "text", "text": "Checking."}]},
            {"type": "thought", "signature": "skip_thought_signature_validator"},
            {"type": "function_call", "id": "toolu_1", "name": "dim_lights", "arguments": {"level": 5}},
            {"type": "function_call", "id": "toolu_2", "name": "start_music", "arguments": {}},
        ]
        assert steps[5]["call_id"] == "toolu_1"
        assert "anthropic-sig" not in json.dumps(steps)
        assert "secret" not in json.dumps(steps)

    def test_models_without_thinking_get_no_placeholder(self):
        assistant = message(
            "a1", Role.ASSISTANT,
            ToolCallPart(call_id="c1", name="dim_lights", args={}, args_raw="{}"),
            origin=Origin(adapter="legacy", provider="legacy", model="legacy"),
        )
        steps = render([user(), assistant])["input"]
        assert [step["type"] for step in steps] == ["user_input", "function_call"]

    def test_calls_flagged_for_a_placeholder_signature_get_one_even_without_a_thinking_profile(self):
        flagged = {"placeholder_signature": True}
        assistant = message(
            "a1", Role.ASSISTANT,
            TextPart("Checking."),
            ToolCallPart(call_id="c1", name="dim_lights", args={}, args_raw="{}", opaque=flagged),
            ToolCallPart(call_id="c2", name="start_music", args={}, args_raw="{}", opaque=flagged),
            origin=Origin(adapter="legacy", provider="legacy", model="legacy"),
        )
        steps = render([user(), assistant])["input"]
        assert [step["type"] for step in steps] == [
            "user_input", "model_output", "thought", "function_call", "function_call",
        ]
        assert steps[2] == {"type": "thought", "signature": "skip_thought_signature_validator"}

    def test_a_flagged_call_after_a_real_thought_adds_no_placeholder(self):
        assistant = message(
            "a1", Role.ASSISTANT,
            ReasoningPart(opaque={"signature": SIGNATURE}),
            ToolCallPart(call_id="c1", name="dim_lights", args={}, args_raw="{}", opaque={"placeholder_signature": True}),
            origin=origin(model="gemini-3.1-pro-preview"),
        )
        steps = render([user(), assistant], profile_fields=THINKING)["input"]
        assert [step["signature"] for step in steps if step["type"] == "thought"] == [SIGNATURE]

    def test_tool_call_with_unparseable_arguments_is_sent_with_empty_arguments(self):
        assistant = message(
            "a1", Role.ASSISTANT,
            ToolCallPart(call_id="c1", name="dim_lights", args=None, args_raw='{"level": '),
            origin=Origin(adapter="legacy", provider="legacy", model="legacy"),
        )
        assert render([user(), assistant])["input"][1]["arguments"] == {}

    def test_wire_id_is_what_the_result_points_at(self):
        assistant = message(
            "a1", Role.ASSISTANT,
            ToolCallPart(call_id="internal_1", name="dim_lights", args={}, args_raw="{}", wire_id="gth23981"),
            origin=Origin(adapter="legacy", provider="legacy", model="legacy"),
        )
        results = message("t1", Role.TOOL, ToolResultPart(call_id="internal_1", name="dim_lights", content=(TextPart("ok"),)))
        steps = render([user(), assistant, results])["input"]
        assert steps[1]["id"] == "gth23981"
        assert steps[2]["call_id"] == "gth23981"

    def test_tool_result_content_kinds(self):
        results = message("t1", Role.TOOL, ToolResultPart(
            call_id="c1", name="dim_lights",
            content=(TextPart("a"), JsonPart({"k": "é"}), ImagePart(media_type="image/png", data="QUJD")),
        ))
        plain = render([results])["input"][0]["result"]
        assert plain == [{"type": "text", "text": "a"}, {"type": "text", "text": '{"k": "é"}'}]
        media = render([results], profile_fields={"tool_result_media": True})["input"][0]["result"]
        assert media[-1] == {"type": "image", "data": "QUJD", "mime_type": "image/png"}

    def test_empty_tool_result_still_has_content(self):
        results = message("t1", Role.TOOL, ToolResultPart(call_id="c1", name="dim_lights", content=()))
        assert render([results])["input"][0]["result"] == [{"type": "text", "text": ""}]


class TestOpaqueMutations:
    def raw_with(self, mutate) -> list:
        steps = json.loads(json.dumps(RAW_STEPS))
        mutate(steps)
        return steps

    def test_dropping_the_signature_of_a_stored_thought_is_refused(self):
        steps = self.raw_with(lambda raw: raw[0].pop("signature"))
        with pytest.raises(OpaqueViolation):
            render(history(raw=steps), profile_fields=THINKING)

    def test_emptying_the_signature_of_a_stored_thought_is_refused(self):
        steps = self.raw_with(lambda raw: raw[0].update(signature=""))
        with pytest.raises(OpaqueViolation):
            render(history(raw=steps), profile_fields=THINKING)

    def test_dropping_the_signature_from_the_reasoning_part_is_refused(self):
        stripped = message(
            "a1", Role.ASSISTANT,
            ReasoningPart(summary="x", opaque={"summary": [SUMMARY_ONE]}),
            TextPart("On it."),
            origin=origin(), raw=None,
        )
        with pytest.raises(OpaqueViolation):
            render([user(), stripped], profile_fields=THINKING)

    def test_a_reasoning_part_without_opaque_data_is_refused_on_the_raw_path_too(self):
        bare = history()
        parts = tuple(ReasoningPart(summary="x") if isinstance(p, ReasoningPart) else p for p in bare[1].parts)
        tampered = message("a1", Role.ASSISTANT, *parts, origin=bare[1].origin, raw=RAW_STEPS)
        with pytest.raises(OpaqueViolation):
            render([bare[0], tampered, bare[2]], profile_fields=THINKING)

    def test_reordering_stored_thoughts_is_refused(self):
        raw = self.raw_with(lambda steps: steps.insert(1, {"type": "thought", "signature": "second", "summary": []}))
        parts = (
            ReasoningPart(opaque={"signature": "second"}),
            ReasoningPart(opaque={"signature": SIGNATURE}),
            *history()[1].parts[1:],
        )
        tampered = message("a1", Role.ASSISTANT, *parts, origin=origin(), raw=raw)
        with pytest.raises(OpaqueViolation):
            render([user(), tampered, RESULTS], profile_fields=THINKING)

    def test_editing_tool_call_arguments_is_refused(self):
        steps = self.raw_with(lambda raw: raw[2].update(arguments={"level": 99}))
        with pytest.raises(OpaqueViolation):
            render(history(raw=steps), profile_fields=THINKING)

    def test_renaming_a_stored_tool_call_id_is_refused(self):
        steps = self.raw_with(lambda raw: raw[2].update(id="other"))
        with pytest.raises(OpaqueViolation):
            render(history(raw=steps), profile_fields=THINKING)

    def test_dropping_a_tool_call_from_the_steps_is_refused(self):
        steps = self.raw_with(lambda raw: raw.pop(3))
        with pytest.raises(OpaqueViolation):
            render(history(raw=steps), profile_fields=THINKING)

    def test_stored_steps_that_are_not_steps_are_refused(self):
        with pytest.raises(OpaqueViolation):
            render(history(raw={"steps": RAW_STEPS}), profile_fields=THINKING)

class TestDecoder:
    def test_thought_becomes_summary_deltas_then_signature_and_summary_opaque_data(self):
        events = decode(TOOL_STREAM)
        assert events[0] == ReasoningDelta(block=0, text="", summary=True)
        assert events[1:3] == [
            ReasoningDelta(block=0, text=SUMMARY_ONE["text"], summary=True),
            ReasoningDelta(block=0, text="\n\n" + SUMMARY_TWO["text"], summary=True),
        ]
        assert events[3:5] == [
            OpaqueDelta(block=0, key="signature", value=SIGNATURE),
            OpaqueDelta(block=0, key="summary", value=[SUMMARY_ONE, SUMMARY_TWO]),
        ]

    def test_text_deltas_keep_the_block_of_their_step(self):
        assert [e for e in decode(TOOL_STREAM) if isinstance(e, TextDelta)] == [
            TextDelta(block=1, text="On "), TextDelta(block=1, text="it."),
        ]

    def test_calls_start_with_their_wire_id_and_stream_their_arguments_in_pieces(self):
        events = decode(TOOL_STREAM)
        assert [e for e in events if isinstance(e, ToolCallStart)] == [
            ToolCallStart(block=2, call_id="fc_1", name="dim_lights", wire_id="fc_1"),
            ToolCallStart(block=3, call_id="fc_2", name="start_music", wire_id="fc_2"),
        ]
        assert [e for e in events if isinstance(e, ToolArgsDelta)] == [
            ToolArgsDelta(block=2, text='{"level"'), ToolArgsDelta(block=2, text=": 20}"),
        ]
        assert [e for e in events if isinstance(e, ToolCallEnd)] == [
            ToolCallEnd(block=2), ToolCallEnd(block=3, args={}),
        ]

    def test_usage_is_reported_before_the_finish_that_carries_the_assembled_steps(self):
        events = decode(TOOL_STREAM)
        assert events[-2] == UsageReported(Usage(
            input_tokens=100, output_tokens=50, total_tokens=150, reasoning_tokens=30, cached_input_tokens=64,
            raw={"total_input_tokens": 100, "total_output_tokens": 20, "total_thought_tokens": 30,
                 "total_cached_tokens": 64, "total_tokens": 150},
        ))
        assert events[-1] == Finish(reason=FinishReason.TOOL_CALLS, raw_reason="requires_action", raw=RAW_STEPS)

    def test_assembled_message_holds_signature_summary_and_calls(self):
        assembled = assemble(decode(TOOL_STREAM), origin=origin())
        reasoning = assembled.reasoning()[0]
        assert reasoning.opaque == {"signature": SIGNATURE, "summary": [SUMMARY_ONE, SUMMARY_TWO]}
        assert reasoning.summary == SUMMARY_ONE["text"] + "\n\n" + SUMMARY_TWO["text"]
        assert assembled.text() == "On it."
        calls = assembled.tool_calls()
        assert [(c.call_id, c.wire_id, c.args, c.args_raw) for c in calls] == [
            ("fc_1", "fc_1", {"level": 20}, '{"level": 20}'),
            ("fc_2", "fc_2", {}, None),
        ]
        assert assembled.raw == RAW_STEPS

    def test_assembled_history_renders_back_to_the_same_steps(self):
        assembled = assemble(decode(TOOL_STREAM), origin=origin())
        body = render([user(), assembled, RESULTS], profile_fields=THINKING)
        assert body["input"][1:5] == RAW_STEPS

    def test_byte_by_byte_feed_matches_whole_chunks(self):
        whole = decode(TOOL_STREAM)
        blob = b"".join(TOOL_STREAM)
        assert decode_chunks(gemini.GeminiDecoder(), [blob[i:i + 1] for i in range(len(blob))]) == whole
        assert decode_chunks(gemini.GeminiDecoder(), [blob]) == whole

    def test_crlf_framing(self):
        blob = b"".join(TOOL_STREAM).replace(b"\n", b"\r\n")
        assert decode_chunks(gemini.GeminiDecoder(), [blob]) == decode(TOOL_STREAM)

    def test_thought_without_summaries_still_carries_its_signature(self):
        events = decode(stream(
            start(0, type="thought"), delta(0, type="thought_signature", signature="sig"), stop(0),
            start(1, type="model_output"), delta(1, type="text", text="Hi"), stop(1),
            completed(),
        ))
        assembled = assemble(events, origin=origin())
        assert assembled.reasoning()[0].opaque == {"signature": "sig"}
        assert assembled.raw == [
            {"type": "thought", "signature": "sig"},
            {"type": "model_output", "content": [{"type": "text", "text": "Hi"}]},
        ]
        assert events[-1].reason is FinishReason.STOP

    def test_function_call_arguments_may_arrive_whole_in_the_start_step(self):
        events = decode(stream(
            start(0, type="function_call", id="fc", name="dim_lights", arguments={"level": 3}), stop(0), completed(),
        ))
        assert ToolCallEnd(block=0, args={"level": 3}) in events
        assert events[-1].raw == [{"type": "function_call", "id": "fc", "name": "dim_lights", "arguments": {"level": 3}}]

    def test_unparseable_arguments_leave_the_call_unexecutable(self):
        events = decode(stream(
            start(0, type="function_call", id="fc", name="dim_lights", arguments={}),
            delta(0, type="arguments_delta", arguments='{"level": '), stop(0), completed("requires_action"),
        ))
        call = assemble(events, origin=origin()).tool_calls()[0]
        assert call.args is None
        assert events[-1].raw[0]["arguments"] == {}

    @pytest.mark.parametrize("status, reason", [
        ("completed", FinishReason.STOP),
        ("incomplete", FinishReason.LENGTH),
        ("cancelled", FinishReason.UNKNOWN),
    ])
    def test_statuses_map_to_finish_reasons(self, status, reason):
        events = decode(stream(
            start(0, type="model_output"), delta(0, type="text", text="x"), stop(0), completed(status),
        ))
        assert events[-1].reason is reason
        assert events[-1].raw_reason == status

    def test_completed_status_with_pending_calls_still_reports_tool_calls(self):
        events = decode(stream(
            start(0, type="function_call", id="fc", name="dim_lights", arguments={}), stop(0), completed("completed"),
        ))
        assert events[-1].reason is FinishReason.TOOL_CALLS

    def test_usage_falls_back_to_the_last_cumulative_report(self):
        events = decode(stream(
            start(0, type="model_output"),
            event("step.delta", index=0, delta={"type": "text", "text": "x"},
                  metadata={"total_usage": {"total_input_tokens": 5, "total_output_tokens": 1}}),
            stop(0), completed(),
        ))
        assert events[-2].usage.input_tokens == 5
        assert events[-2].usage.output_tokens == 1

    def test_thought_tokens_are_folded_into_the_output_count_and_kept_as_the_reasoning_subset(self):
        usage = gemini.usage_from({"total_input_tokens": 10, "total_output_tokens": 4, "total_thought_tokens": 6})

        assert (usage.output_tokens, usage.reasoning_tokens) == (10, 6)

    def test_an_unreported_output_count_stays_unknown_even_with_thought_tokens(self):
        assert gemini.usage_from({"total_input_tokens": 10, "total_thought_tokens": 6}).output_tokens is None
        assert gemini.usage_from({"total_output_tokens": 4}).output_tokens == 4

    def test_no_usage_means_no_usage_event(self):
        assert not any(isinstance(e, UsageReported) for e in decode(stream(completed())))

    def test_status_update_can_finish_a_stream_that_lost_its_completed_event(self):
        events = decode(stream(
            start(0, type="model_output"), delta(0, type="text", text="x"), stop(0),
            event("interaction.status_update", interaction_id="v1", status="completed"),
            done=False,
        ))
        assert events[-1].reason is FinishReason.STOP

    @pytest.mark.parametrize("records", [
        stream(start(0, type="model_output"), delta(0, type="text", text="x")),
        stream(event("interaction.status_update", interaction_id="v1", status="in_progress")),
        [],
    ])
    def test_stream_ending_early_is_malformed(self, records):
        with pytest.raises(MalformedStream):
            decode(records)

    def test_thought_without_a_signature_is_malformed(self):
        with pytest.raises(MalformedStream, match="signature"):
            decode(stream(start(0, type="thought"), delta(0, type="thought_summary", content=SUMMARY_ONE), stop(0)))

    def test_completing_with_an_open_step_is_malformed(self):
        with pytest.raises(MalformedStream, match="never finished"):
            decode(stream(start(0, type="model_output"), completed()))

    @pytest.mark.parametrize("records", [
        [start(0, type="google_search_call")],
        [start(0, type="model_output"), delta(0, type="image", data="x")],
        [delta(4, type="text", text="x")],
        [start(0, type="model_output"), start(0, type="model_output")],
        [stop(2)],
        [start(0, type="function_call", id="", name="dim_lights", arguments={})],
        [b"data: not json\n\n"],
        [b"data: [1]\n\n"],
    ])
    def test_unreplayable_or_out_of_order_records_are_malformed(self, records):
        with pytest.raises(MalformedStream):
            decode(records)

    def test_records_after_the_interaction_completed_are_ignored(self):
        events = decode(stream(completed(), start(0, type="model_output")))
        assert [type(e) for e in events] == [Finish]

    @pytest.mark.parametrize("code, kind, message", [
        ("gateway_timeout", ServerFailed, "Deadline expired before operation could complete."),
        ("rate_limit_exceeded", RateLimited, "slow down"),
        ("quota_exceeded", QuotaExhausted, "daily quota"),
        ("service_unavailable", Overloaded, "overloaded"),
        ("authentication", AuthFailed, "bad key"),
        ("invalid_request", InvalidRequest, "bad body"),
        ("invalid_request", OpaqueViolation, "Function call is missing a thought_signature"),
        ("invalid_request", ContextOverflow, "The input token count (9) exceeds the maximum number of tokens allowed (4)."),
    ])
    def test_in_band_errors_raise_typed_failures(self, code, kind, message):
        records = stream(start(0, type="thought"), event("error", error={"code": code, "message": message}))
        with pytest.raises(kind) as caught:
            decode(records)
        assert caught.value.message == message
        assert caught.value.code == code

    def test_in_band_error_that_merely_mentions_a_signature_stays_an_invalid_request(self):
        error = {"code": "invalid_request", "message": "Tool verify_signature has an invalid parameter schema."}
        with pytest.raises(InvalidRequest) as caught:
            decode(stream(event("error", error=error)))
        assert type(caught.value) is InvalidRequest

    def test_failed_status_raises_the_recorded_error(self):
        errors = [{"code": "quota_exceeded", "message": "out of quota"}]
        with pytest.raises(QuotaExhausted):
            decode(stream(completed("failed", errors=errors)))
        with pytest.raises(ServerFailed):
            decode(stream(completed("failed")))


def failure(status: Optional[int], body: Any = None, headers: Optional[dict] = None, **kwargs: Any) -> WireFailure:
    raw = json.dumps(body).encode() if body is not None else b""
    return WireFailure(status=status, body=raw, headers=headers or {}, **kwargs)


def classic(status: int, grpc: str, message: str, *details: dict) -> dict:
    return {"error": {"code": status, "message": message, "status": grpc, "details": list(details)}}


def quota_detail(quota_id: str, value: str = "20") -> dict:
    return {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaMetric": "m", "quotaId": quota_id, "quotaValue": value}]}


class TestClassifyError:
    adapter = GeminiInteractionsAdapter()

    def classify(self, *args: Any, **kwargs: Any) -> HarnessError:
        return self.adapter.classify_error(failure(*args, **kwargs))

    def test_adapter_satisfies_the_contract(self):
        assert isinstance(self.adapter, Adapter)
        assert self.adapter.name == "gemini-interactions"

    def test_interactions_style_429_separates_rate_from_quota(self):
        rate = self.classify(429, {"error": {"code": "rate_limit_exceeded", "message": "too fast"}}, {"retry-after": "7"})
        assert type(rate) is RateLimited
        assert rate.retry_after == 7.0
        quota = self.classify(429, {"error": {"code": "quota_exceeded", "message": "daily limit"}})
        assert type(quota) is QuotaExhausted
        assert quota.retryable is False

    def test_resource_exhausted_per_minute_is_a_rate_limit_with_the_server_delay(self):
        body = classic(
            429, "RESOURCE_EXHAUSTED", "You exceeded your current quota",
            quota_detail("GenerateRequestsPerMinutePerProjectPerModel"),
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34.5s"},
        )
        error = self.classify(429, body)
        assert type(error) is RateLimited
        assert error.retry_after == 34.5
        assert error.code == "RESOURCE_EXHAUSTED"
        assert error.status == 429

    def test_error_body_may_be_a_json_array(self):
        body = classic(429, "RESOURCE_EXHAUSTED", "limit", quota_detail("GenerateRequestsPerDayPerProjectPerModel"))
        assert type(self.adapter.classify_error(failure(429, [body]))) is QuotaExhausted

    def test_an_unparsable_server_delay_is_ignored(self):
        body = classic(429, "RESOURCE_EXHAUSTED", "slow", {"retryDelay": "soons"})
        error = self.classify(429, body)
        assert type(error) is RateLimited
        assert error.retry_after is None

    def test_a_body_that_is_not_json_still_classifies_by_status(self):
        error = self.adapter.classify_error(WireFailure(status=429, body=b"upstream busy"))
        assert type(error) is RateLimited
        assert error.retry_after is None

    def test_header_wins_over_the_server_delay(self):
        body = classic(429, "RESOURCE_EXHAUSTED", "slow", {"retryDelay": "34s"})
        assert self.classify(429, body, {"Retry-After": "3"}).retry_after == 3.0

    @pytest.mark.parametrize("quota_id, value", [
        ("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "20"),
        ("GenerateContentInputTokensPerModelPerMinute-FreeTier", "0"),
    ])
    def test_resource_exhausted_daily_or_zero_limit_is_quota(self, quota_id, value):
        body = classic(429, "RESOURCE_EXHAUSTED", "You exceeded your current quota", quota_detail(quota_id, value))
        assert type(self.classify(429, body)) is QuotaExhausted

    @pytest.mark.parametrize("message", [
        "Function call is missing a thought_signature in functionCall parts.",
        "Corrupted thought signature.",
        "Invalid thoughtSignature on content block 2.",
    ])
    def test_missing_thought_signature_is_an_opaque_violation(self, message):
        error = self.classify(400, classic(400, "INVALID_ARGUMENT", message))
        assert type(error) is OpaqueViolation
        assert error.retryable is False

    def test_a_400_that_merely_mentions_a_signature_stays_an_invalid_request(self):
        body = classic(400, "INVALID_ARGUMENT", "Tool verify_signature has an invalid parameter schema.")
        assert type(self.classify(400, body)) is InvalidRequest

    def test_oversized_prompt_is_a_context_overflow(self):
        body = classic(400, "INVALID_ARGUMENT", "The input token count (1200000) exceeds the maximum number of tokens allowed (1048575).")
        assert type(self.classify(400, body)) is ContextOverflow

    def test_other_bad_requests_stay_invalid_requests(self):
        assert type(self.classify(400, {"error": {"code": "parameter_unknown", "message": "no such field"}})) is InvalidRequest

    @pytest.mark.parametrize("status, kind", [
        (401, AuthFailed),
        (403, AuthFailed),
        (402, QuotaExhausted),
        (404, InvalidRequest),
        (500, ServerFailed),
        (503, Overloaded),
        (504, ServerFailed),
        (408, ConnectionFailed),
    ])
    def test_status_mapping(self, status, kind):
        assert type(self.classify(status, {"error": {"code": "x", "message": "m"}})) is kind

    def test_retryability(self):
        assert self.classify(503, None).retryable and self.classify(500, None).retryable
        assert not self.classify(401, None).retryable

    def test_unreachable_server_is_a_connection_failure(self):
        error = self.adapter.classify_error(WireFailure(status=None, cause=OSError("refused")))
        assert type(error) is ConnectionFailed

    def test_detail_keeps_the_body(self):
        error = self.classify(500, {"error": {"code": "api_error", "message": "boom"}})
        assert "boom" in error.detail["body"]


class TestSend:
    async def collect(self, req: RenderRequest) -> list:
        adapter = GeminiInteractionsAdapter()
        wire = adapter.render(req, profile(**THINKING))
        return [chunk async for chunk in adapter.send(wire, req)]

    async def test_request_goes_out_with_the_key_header_and_the_rendered_bytes(self, monkeypatch):
        peer = install_peer(monkeypatch, peer_chunks(TOOL_STREAM))
        req = request([user("lights and music")], tools=[LIGHT_TOOL, MUSIC_TOOL])
        received = await self.collect(req)
        sent = peer.requests[0]
        wire = GeminiInteractionsAdapter().render(req, profile(**THINKING))
        assert str(sent.url) == "https://generativelanguage.googleapis.com/v1beta/interactions"
        assert sent.method == "POST"
        assert sent.headers["x-goog-api-key"] == "test-key"
        assert sent.headers["content-type"] == "application/json"
        assert sent.content == wire.content()
        assert decode_chunks(gemini.GeminiDecoder(), received) == decode(TOOL_STREAM)

    async def test_resource_exhausted_reaches_the_caller_as_a_rate_limit_with_the_delay(self, monkeypatch):
        body = classic(
            429, "RESOURCE_EXHAUSTED", "You exceeded your current quota",
            quota_detail("GenerateRequestsPerMinutePerProjectPerModel"),
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12s"},
        )
        install_peer(monkeypatch, peer_chunks([json.dumps(body).encode()], status=429))
        with pytest.raises(RateLimited) as caught:
            await self.collect(request([user()]))
        assert caught.value.retry_after == 12.0

    async def test_missing_signature_rejection_reaches_the_caller_as_an_opaque_violation(self, monkeypatch):
        body = classic(400, "INVALID_ARGUMENT", "Function call is missing a thought_signature in functionCall parts.")
        install_peer(monkeypatch, peer_chunks([json.dumps(body).encode()], status=400))
        with pytest.raises(OpaqueViolation):
            await self.collect(request([user()]))


class TestDiscovery:
    def test_discovered_fields_are_minimal(self):
        assert discovered_fields({"inputTokenLimit": 1048576, "outputTokenLimit": 65536, "thinking": True}) == {
            "context_window": 1048576, "max_output_tokens": 65536,
        }

    def test_a_model_without_thinking_says_so(self):
        assert discovered_fields({"thinking": False})["reasoning"] == "none"

    def test_junk_values_are_ignored(self):
        assert discovered_fields({"inputTokenLimit": 0, "outputTokenLimit": "x", "thinking": "yes"}) == {}

    def test_layer_applies_to_a_profile(self):
        layer = ProfileLayer(ProfileSource.DISCOVERED, discovered_fields({"inputTokenLimit": 1000000, "thinking": False}))
        layered = profile(**THINKING).layered([layer])
        assert layered.context_window == 1000000
        assert layered.reasoning.value == "none"
        assert layered.sources["context_window"] == ProfileSource.DISCOVERED

    async def test_discover_reads_models_get_with_the_key_in_a_header(self, monkeypatch):
        calls = []

        async def fake(provider, base_url, path, headers=None, **kwargs):
            calls.append((base_url, path, headers))
            return {"name": "models/gemini-3.8-flash", "inputTokenLimit": 1048576, "outputTokenLimit": 65536}

        monkeypatch.setattr(gemini, "fetch_json", fake)
        layer = await GeminiInteractionsAdapter().discover(config(model="models/gemini-3.8-flash"))
        assert layer.source is ProfileSource.DISCOVERED
        assert layer.fields == {"context_window": 1048576, "max_output_tokens": 65536}
        assert calls == [(
            "https://generativelanguage.googleapis.com",
            "/v1beta/models/gemini-3.8-flash",
            {"x-goog-api-key": "test-key"},
        )]

    async def test_list_models_pages_and_filters(self, monkeypatch):
        pages = [
            {"models": [
                {"name": "models/gemini-3.8-flash", "displayName": "Gemini 3.8 Flash", "inputTokenLimit": 1048576,
                 "supportedGenerationMethods": ["generateContent"], "thinking": True},
                {"name": "models/text-embedding-9", "supportedGenerationMethods": ["embedContent"]},
            ], "nextPageToken": "next page"},
            {"models": [{"name": "models/gemini-3.1-pro-preview"}]},
        ]
        paths = []

        async def fake(provider, base_url, path, headers=None, **kwargs):
            paths.append((path, headers))
            return pages[len(paths) - 1]

        monkeypatch.setattr(gemini, "fetch_json", fake)
        models = await GeminiInteractionsAdapter().list_models("https://generativelanguage.googleapis.com/v1beta", "k")
        assert [m.id for m in models] == ["gemini-3.1-pro-preview", "gemini-3.8-flash"]
        assert models[1].label == "Gemini 3.8 Flash"
        assert models[1].details == {"inputTokenLimit": 1048576, "thinking": True}
        assert paths == [
            ("/v1beta/models?pageSize=1000", {"x-goog-api-key": "k"}),
            ("/v1beta/models?pageSize=1000&pageToken=next%20page", {"x-goog-api-key": "k"}),
        ]

    async def test_list_models_without_a_models_key_fails(self, monkeypatch):
        async def fake(provider, base_url, path, headers=None, **kwargs):
            return {"unexpected": []}

        monkeypatch.setattr(gemini, "fetch_json", fake)
        with pytest.raises(ModelListingError):
            await GeminiInteractionsAdapter().list_models("https://generativelanguage.googleapis.com", "k")

    async def test_list_models_stops_after_the_page_limit(self, monkeypatch):
        calls = []

        async def fake(provider, base_url, path, headers=None, **kwargs):
            calls.append(path)
            return {"models": [{"name": f"models/m{len(calls)}"}], "nextPageToken": "again"}

        monkeypatch.setattr(gemini, "fetch_json", fake)
        models = await GeminiInteractionsAdapter().list_models("https://generativelanguage.googleapis.com", "k")
        assert len(calls) == gemini.LISTING_MAX_PAGES
        assert len(models) == gemini.LISTING_MAX_PAGES

    async def test_a_model_that_only_supports_interactions_is_listed(self, monkeypatch):
        async def fake(provider, base_url, path, headers=None, **kwargs):
            return {"models": [{"name": "models/agent-1", "supportedGenerationMethods": ["createInteraction"]}]}

        monkeypatch.setattr(gemini, "fetch_json", fake)
        models = await GeminiInteractionsAdapter().list_models("https://generativelanguage.googleapis.com", "k")
        assert [m.id for m in models] == ["agent-1"]


def test_assembled_message_survives_the_codec():
    assembled = assemble(decode(TOOL_STREAM), origin=origin())
    assert codec.from_json(Message, codec.to_json(assembled)) == assembled
