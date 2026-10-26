import pytest

from src.features.llm.harness.adapter import Settings, ToolChoice, ToolSpec
from src.features.llm.harness.errors import InvalidRequest
from src.features.llm.harness.messages import (
    CompactionPart,
    ContextPart,
    FilePart,
    ImagePart,
    JsonPart,
    Message,
    Role,
    TextPart,
    ToolResultPart,
)
from tests.features.llm.harness.adapters.openai_responses_support import (
    ADAPTER,
    REASONING_PROFILE,
    make_config,
    make_profile,
    make_request,
    user_message,
)

WEATHER = ToolSpec(
    name="get_weather",
    description="Current weather.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    strict=True,
)


def body_of(request=None, profile=None):
    return ADAPTER.render(request or make_request([user_message()]), profile or make_profile()).body


class TestEnvelope:
    def test_the_request_is_stateless_and_streamed(self):
        wire = ADAPTER.render(make_request([user_message()]), make_profile())

        assert wire.method == "POST"
        assert wire.url == "https://api.openai.com/v1/responses"
        assert wire.stream is True
        assert wire.body["store"] is False
        assert wire.body["stream"] is True
        assert "previous_response_id" not in wire.body
        assert wire.body["model"] == "gpt-5.6"
        assert wire.body["instructions"] == "Be brief."
        assert wire.body["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}]

    def test_headers_carry_the_key_and_ask_for_an_event_stream(self):
        wire = ADAPTER.render(make_request([user_message()]), make_profile())

        assert wire.headers == {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": "Bearer test-key-a",
        }

    def test_a_config_without_a_key_sends_no_authorization(self):
        request = make_request([user_message()], config=make_config(api_key=None))

        assert "Authorization" not in ADAPTER.render(request, make_profile()).headers

    def test_the_base_url_may_end_in_a_slash(self):
        request = make_request([user_message()], config=make_config(base_url="https://proxy.example/v1/"))

        assert ADAPTER.render(request, make_profile()).url == "https://proxy.example/v1/responses"

    def test_a_buffered_request_drops_the_event_stream_header(self):
        wire = ADAPTER.render(make_request([user_message()], stream=False), make_profile())

        assert wire.stream is False
        assert wire.body["stream"] is False
        assert "Accept" not in wire.headers

    def test_an_empty_system_prompt_sends_no_instructions(self):
        assert "instructions" not in body_of(make_request([user_message()], system=""))

    def test_the_body_serialises_to_the_same_bytes_every_time(self):
        request = make_request([user_message("Héllo")], tools=[WEATHER], settings=Settings(max_tokens=100))

        first = ADAPTER.render(request, make_profile(**REASONING_PROFILE)).content()
        second = ADAPTER.render(request, make_profile(**REASONING_PROFILE)).content()

        assert first == second
        assert "Héllo".encode("utf-8") in first


class TestReasoningSettings:
    def test_a_model_without_reasoning_gets_no_include_and_no_reasoning_block(self):
        body = body_of(make_request([user_message()], settings=Settings(reasoning="high")), make_profile())

        assert "include" not in body
        assert "reasoning" not in body

    def test_a_reasoning_model_asks_for_the_encrypted_reasoning(self):
        body = body_of(profile=make_profile(**REASONING_PROFILE))

        assert body["include"] == ["reasoning.encrypted_content"]
        assert "reasoning" not in body

    def test_an_effort_inside_the_profile_levels_is_sent(self):
        request = make_request([user_message()], settings=Settings(reasoning="high"))

        assert body_of(request, make_profile(**REASONING_PROFILE))["reasoning"] == {"effort": "high"}

    def test_an_effort_outside_the_profile_levels_is_left_out(self):
        request = make_request([user_message()], settings=Settings(reasoning="extreme"))

        assert "reasoning" not in body_of(request, make_profile(**REASONING_PROFILE))

    def test_summaries_are_requested_only_when_thinking_is_shown(self):
        request = make_request([user_message()], config=make_config(show_thinking=True), settings=Settings(reasoning="low"))

        assert body_of(request, make_profile(**REASONING_PROFILE))["reasoning"] == {"effort": "low", "summary": "auto"}

    def test_turning_reasoning_off_sends_none_and_no_summary(self):
        request = make_request([user_message()], config=make_config(show_thinking=True), settings=Settings(reasoning=False))

        assert body_of(request, make_profile(**REASONING_PROFILE))["reasoning"] == {"effort": "none"}

    def test_turning_reasoning_off_without_a_none_level_sends_nothing(self):
        request = make_request([user_message()], settings=Settings(reasoning=False))
        profile = make_profile(**{**REASONING_PROFILE, "reasoning_levels": ["low", "high"]})

        assert "reasoning" not in body_of(request, profile)


class TestSampling:
    def test_temperature_and_limits_are_sent_when_the_profile_allows(self):
        request = make_request(
            [user_message()], settings=Settings(temperature=0.2, top_p=0.9, max_tokens=500, cache_key="conv-9"),
        )

        body = body_of(request, make_profile())

        assert body["temperature"] == 0.2
        assert body["top_p"] == 0.9
        assert body["max_output_tokens"] == 500
        assert body["prompt_cache_key"] == "conv-9"

    def test_rejected_parameters_are_never_sent(self):
        request = make_request([user_message()], settings=Settings(temperature=0.2, top_p=0.9))

        body = body_of(request, make_profile(sampling_rejected=["top_p"]))

        assert body["temperature"] == 0.2
        assert "top_p" not in body

    def test_parameters_ignored_while_reasoning_are_left_out_only_then(self):
        thinking = make_request([user_message()], settings=Settings(temperature=0.2, reasoning="low"))
        quiet = make_request([user_message()], settings=Settings(temperature=0.2, reasoning=False))
        profile = make_profile(**REASONING_PROFILE)

        assert "temperature" not in body_of(thinking, profile)
        assert body_of(quiet, profile)["temperature"] == 0.2

    def test_top_k_and_min_p_have_no_responses_equivalent(self):
        request = make_request([user_message()], settings=Settings(top_k=40, min_p=0.05))

        body = body_of(request, make_profile())

        assert "top_k" not in body
        assert "min_p" not in body

    def test_max_tokens_is_capped_by_the_profile_ceiling(self):
        request = make_request([user_message()], settings=Settings(max_tokens=9000))

        assert body_of(request, make_profile(max_output_tokens=4096))["max_output_tokens"] == 4096

    def test_a_response_schema_becomes_a_strict_json_schema_format(self):
        schema = {"title": "Plan: v2", "type": "object", "properties": {"steps": {"type": "array"}}}
        request = make_request([user_message()], settings=Settings(response_schema=schema))

        assert body_of(request)["text"] == {
            "format": {"type": "json_schema", "name": "Plan_v2", "schema": schema, "strict": True}
        }

    def test_an_untitled_schema_is_named_response(self):
        request = make_request([user_message()], settings=Settings(response_schema={"type": "object"}))

        assert body_of(request)["text"]["format"]["name"] == "response"


class TestTools:
    def test_function_tools_are_flat_and_strict_only_when_both_sides_agree(self):
        request = make_request([user_message()], tools=[WEATHER])

        strict = body_of(request, make_profile(strict_tools=True))["tools"]
        loose = body_of(request, make_profile(strict_tools=False))["tools"]

        assert strict == [{
            "type": "function",
            "name": "get_weather",
            "description": "Current weather.",
            "parameters": WEATHER.parameters,
            "strict": True,
        }]
        assert loose[0]["strict"] is False

    def test_no_tools_means_no_tool_keys(self):
        body = body_of(make_request([user_message()], settings=Settings(tool_choice=ToolChoice("required"))))

        assert "tools" not in body
        assert "tool_choice" not in body
        assert "parallel_tool_calls" not in body

    def test_parallel_calls_are_switched_off_unless_the_profile_allows_them(self):
        request = make_request([user_message()], tools=[WEATHER])

        assert body_of(request, make_profile())["parallel_tool_calls"] is False
        assert "parallel_tool_calls" not in body_of(request, make_profile(parallel_tools=True))

    @pytest.mark.parametrize("choice, expected", [
        (ToolChoice("auto"), None),
        (ToolChoice("none"), "none"),
        (ToolChoice("required"), "required"),
        (ToolChoice("named", "get_weather"), {"type": "function", "name": "get_weather"}),
    ])
    def test_allowed_tool_choices_are_sent(self, choice, expected):
        request = make_request([user_message()], tools=[WEATHER], settings=Settings(tool_choice=choice))
        profile = make_profile(tool_choice=["auto", "none", "required", "named"])

        assert body_of(request, profile).get("tool_choice") == expected

    def test_a_tool_choice_the_profile_bans_falls_back_to_the_default(self):
        request = make_request([user_message()], tools=[WEATHER], settings=Settings(tool_choice=ToolChoice("required")))

        assert "tool_choice" not in body_of(request, make_profile(tool_choice=["auto"]))

    def test_the_reasoning_limits_apply_only_while_reasoning(self):
        profile = make_profile(**REASONING_PROFILE)
        thinking = make_request(
            [user_message()], tools=[WEATHER], settings=Settings(tool_choice=ToolChoice("required"), reasoning="low"),
        )
        quiet = make_request(
            [user_message()], tools=[WEATHER], settings=Settings(tool_choice=ToolChoice("required"), reasoning=False),
        )

        assert "tool_choice" not in body_of(thinking, profile)
        assert body_of(quiet, profile)["tool_choice"] == "required"

    def test_a_named_choice_needs_a_name(self):
        request = make_request([user_message()], tools=[WEATHER], settings=Settings(tool_choice=ToolChoice("named")))

        with pytest.raises(InvalidRequest):
            ADAPTER.render(request, make_profile(tool_choice=["named"]))


class TestUserAndResultMessages:
    def test_context_blocks_render_like_the_attached_context_block(self):
        message = Message(
            id="u", role=Role.USER, parts=(TextPart("Question"), ContextPart(kind="memory", text="likes tea")),
        )

        item = body_of(make_request([message]))["input"][0]

        assert item == {
            "role": "user",
            "content": [{"type": "input_text", "text": "Question\n\n<context>\nlikes tea\n</context>"}],
        }

    def test_images_and_files_follow_the_text(self):
        message = Message(id="u", role=Role.USER, parts=(
            ImagePart(media_type="image/png", data="aW1n"),
            FilePart(media_type="application/pdf", name="a.pdf", data="cGRm"),
            TextPart("Look"),
        ))

        item = body_of(make_request([message]))["input"][0]

        assert item["content"] == [
            {"type": "input_text", "text": "Look"},
            {"type": "input_image", "image_url": "data:image/png;base64,aW1n"},
            {"type": "input_file", "filename": "a.pdf", "file_data": "data:application/pdf;base64,cGRm"},
        ]

    def test_an_image_that_was_not_loaded_cannot_be_rendered(self):
        message = Message(id="u", role=Role.USER, parts=(TextPart("Look"), ImagePart(media_type="image/png", ref="up/1.png")))

        with pytest.raises(InvalidRequest):
            ADAPTER.render(make_request([message]), make_profile())

    def test_a_compaction_summary_leads_the_user_content(self):
        message = Message(id="u", role=Role.USER, parts=(TextPart("Go on"), CompactionPart(summary="Earlier: we met.")))

        content = body_of(make_request([message]))["input"][0]["content"]

        assert [entry["text"] for entry in content] == ["Earlier: we met.", "Go on"]

    def test_a_mid_conversation_system_message_is_a_developer_message(self):
        system = Message(id="s", role=Role.SYSTEM, parts=(TextPart("Be formal."),))

        assert body_of(make_request([user_message(), system]))["input"][1] == {"role": "developer", "content": "Be formal."}

    def test_each_tool_result_is_its_own_output_item_in_order(self):
        batch = Message(id="t", role=Role.TOOL, parts=(
            ToolResultPart(call_id="c1", name="a", content=(TextPart("one"), TextPart(" two"))),
            ToolResultPart(call_id="c2", name="b", content=(JsonPart({"ok": True}),), is_error=True),
        ))

        items = body_of(make_request([batch]))["input"]

        assert items == [
            {"type": "function_call_output", "call_id": "c1", "output": "one two"},
            {"type": "function_call_output", "call_id": "c2", "output": "{\"ok\":true}"},
        ]

    def test_tool_result_images_travel_only_when_the_profile_takes_media(self):
        batch = Message(id="t", role=Role.TOOL, parts=(ToolResultPart(
            call_id="c1", name="shot", content=(TextPart("Screenshot"), ImagePart(media_type="image/png", data="aW1n")),
        ),))

        with_media = body_of(make_request([batch]), make_profile(tool_result_media=True))["input"][0]
        without = body_of(make_request([batch]), make_profile(tool_result_media=False))["input"][0]

        assert with_media["output"] == [
            {"type": "input_text", "text": "Screenshot"},
            {"type": "input_image", "image_url": "data:image/png;base64,aW1n"},
        ]
        assert without["output"] == "Screenshot"
