import json
from pathlib import Path

import pytest

from src.features.llm.harness.adapter import Adapter, RenderRequest, Settings, ToolChoice, ToolSpec
from src.features.llm.harness.adapters.openai_chat import OpenAIChatAdapter
from src.features.llm.harness.errors import InvalidRequest
from src.features.llm.harness.messages import (
    ImagePart,
    Message,
    Origin,
    ReasoningPart,
    Role,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from src.features.llm.harness.profile import ToolIdRule
from tests.features.llm.harness.cassettes import request_bytes
from tests.features.llm.harness.history import (
    IMAGE_B64,
    make_config,
    messages_from_legacy,
    profile_with,
    tool_schemas,
)
from tests.features.llm.harness.openai_chat_cases import CASES, HISTORY

GOLDEN = json.loads(
    (Path(__file__).resolve().parents[3] / "fixtures" / "llm_goldens" / "openai_chat.json").read_text(encoding="utf-8")
)
ORIGIN = Origin(adapter="openai-chat", provider="openai-chat", model="golden-model")


def request_for(case) -> RenderRequest:
    return RenderRequest(
        config=make_config("openai", **case.config),
        system=case.system,
        messages=messages_from_legacy(case.history, ORIGIN, case.image),
        tools=tuple(ToolSpec.from_schema(schema) for schema in case.tools or []),
        settings=Settings.from_options(case.override),
    )


def simple(history=HISTORY, **kwargs) -> RenderRequest:
    return RenderRequest(
        config=make_config("openai"),
        system="SYS",
        messages=messages_from_legacy(history, ORIGIN),
        **kwargs,
    )


def render(request: RenderRequest, **profile):
    return OpenAIChatAdapter().render(request, profile_with("openai-chat", **profile))


def test_the_adapter_satisfies_the_adapter_protocol():
    assert isinstance(OpenAIChatAdapter(), Adapter)


def test_every_golden_case_is_covered_by_a_twin():
    assert set(GOLDEN) == {case.id for case in CASES}


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_the_new_adapter_renders_the_legacy_request_byte_for_byte(case):
    expected = GOLDEN[case.id]

    wire = OpenAIChatAdapter().render(request_for(case), profile_with("openai-chat"))

    assert wire.url == expected["url"]
    assert wire.headers.get("Authorization") == expected["authorization"]
    assert wire.headers["Content-Type"] == "application/json"
    assert wire.content() == request_bytes(expected["body"])


def test_a_trailing_slash_on_the_base_url_does_not_double_up():
    request = RenderRequest(config=make_config("openai", base_url="http://peer.invalid:1234/"), system="", messages=())

    assert render(request).url == "http://peer.invalid:1234/chat/completions"


def test_an_unresolved_image_reference_is_refused_rather_than_dropped():
    message = Message("u", Role.USER, (TextPart("look"), ImagePart(media_type="image/png", ref="image-1")))
    request = RenderRequest(config=make_config("openai"), system="", messages=(message,))

    with pytest.raises(InvalidRequest):
        render(request)


def test_reasoning_uses_the_field_name_the_profile_names():
    request = simple(
        [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "reasoning": "why", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "t", "arguments": {}}}]},
        ]
    )

    body = render(request, reasoning_field="reasoning").body

    assert body["messages"][2]["reasoning"] == "why"
    assert "reasoning_content" not in body["messages"][2]


def _reasoned_call(*parts):
    call = ToolCallPart(call_id="c1", name="t", args={}, wire_id="c1")
    assistant = Message("a", Role.ASSISTANT, (*parts, call), origin=ORIGIN)
    return RenderRequest(config=make_config("openai"), system="", messages=(assistant,))


def test_redacted_and_empty_reasoning_parts_are_not_sent():
    request = _reasoned_call(ReasoningPart(text="secret", redacted=True), ReasoningPart(text=""), TextPart("hi"))

    assert "reasoning_content" not in render(request).body["messages"][1]


def test_several_reasoning_parts_join_into_one_field():
    request = _reasoned_call(ReasoningPart(text="one "), ReasoningPart(text="two"))

    assert render(request).body["messages"][1]["reasoning_content"] == "one two"


def test_reasoning_goes_back_only_on_assistant_messages_that_have_tool_calls():
    assistant = Message("a", Role.ASSISTANT, (ReasoningPart(text="why"), TextPart("hi")), origin=ORIGIN)
    request = RenderRequest(config=make_config("openai"), system="", messages=(assistant,))

    assert render(request).body["messages"][1] == {"role": "assistant", "content": "hi"}


def _call_message(call_id="local_1", wire_id="srv_1", args_raw=None, args=None):
    call = ToolCallPart(call_id=call_id, name="t", args={"a": 1} if args is None else args, args_raw=args_raw, wire_id=wire_id)
    result = ToolResultPart(call_id=call_id, name="t", content=(TextPart("done"),))
    return (
        Message("u", Role.USER, (TextPart("go"),)),
        Message("a", Role.ASSISTANT, (call,), origin=ORIGIN),
        Message("t", Role.TOOL, (result,)),
    )


def test_the_server_issued_id_goes_back_on_the_call_and_on_its_result():
    request = RenderRequest(config=make_config("openai"), system="", messages=_call_message())

    messages = render(request).body["messages"]

    assert messages[2]["tool_calls"][0]["id"] == "srv_1"
    assert messages[3]["tool_call_id"] == "srv_1"


def test_a_nine_digit_rule_rewrites_ids_consistently_on_both_sides():
    request = RenderRequest(config=make_config("openai"), system="", messages=_call_message())

    messages = render(request, tool_id_rule=ToolIdRule.NINE_DIGITS).body["messages"]

    sent = messages[2]["tool_calls"][0]["id"]
    assert len(sent) == 9 and sent.isdigit()
    assert messages[3]["tool_call_id"] == sent
    assert render(request, tool_id_rule=ToolIdRule.NINE_DIGITS).body["messages"][2]["tool_calls"][0]["id"] == sent


def test_the_stored_raw_arguments_are_replayed_verbatim_when_they_match_the_parsed_ones():
    request = RenderRequest(config=make_config("openai"), system="", messages=_call_message(args_raw='{"a":1}'))

    assert render(request).body["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"a":1}'


def test_raw_arguments_that_disagree_with_the_parsed_ones_are_not_replayed():
    request = RenderRequest(config=make_config("openai"), system="", messages=_call_message(args_raw='{"a":2}'))

    assert render(request).body["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


def test_a_call_that_never_parsed_goes_back_with_empty_arguments():
    user, _, result = _call_message()
    broken = Message(
        "a", Role.ASSISTANT,
        (ToolCallPart(call_id="local_1", name="t", args=None, args_raw='{"a":', wire_id="srv_1"),),
        origin=ORIGIN,
    )
    request = RenderRequest(config=make_config("openai"), system="", messages=(user, broken, result))

    assert render(request).body["messages"][2]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_a_tool_message_with_several_results_becomes_several_wire_messages():
    first = ToolResultPart(call_id="c1", name="a", content=(TextPart("one"),))
    second = ToolResultPart(call_id="c2", name="b", content=(TextPart("two"),))
    request = RenderRequest(config=make_config("openai"), system="", messages=(Message("t", Role.TOOL, (first, second)),))

    assert render(request).body["messages"][1:] == [
        {"role": "tool", "content": "one", "tool_call_id": "c1", "name": "a"},
        {"role": "tool", "content": "two", "tool_call_id": "c2", "name": "b"},
    ]


def test_a_tool_result_renders_its_text_parts_only():
    result = ToolResultPart(
        call_id="c1", name="t", content=(TextPart("a "), ImagePart(media_type="image/png", data=IMAGE_B64), TextPart("z")),
    )
    request = RenderRequest(config=make_config("openai"), system="", messages=(Message("t", Role.TOOL, (result,)),))

    assert render(request).body["messages"][1]["content"] == "a z"


def test_context_parts_render_the_way_the_legacy_context_block_did():
    from src.features.llm.harness.messages import ContextPart

    message = Message("u", Role.USER, (TextPart("question"), ContextPart(kind="memory", text="notes")))
    request = RenderRequest(config=make_config("openai"), system="", messages=(message,))

    assert render(request).body["messages"][1]["content"] == "question\n\n<context>\nnotes\n</context>"


def _image(media_type="image/png", data=IMAGE_B64):
    return ImagePart(media_type=media_type, data=data)


def _wire_image(data=IMAGE_B64, media_type="image/png"):
    return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}


def test_the_users_image_goes_before_the_text_of_the_last_user_message():
    message = Message("u", Role.USER, (TextPart("q"), _image()))
    request = RenderRequest(config=make_config("openai"), system="", messages=(message,))

    assert render(request).body["messages"][1]["content"] == [_wire_image(), {"type": "text", "text": "q"}]


def test_only_the_newest_user_image_is_sent_and_it_lands_on_the_last_user_message():
    older = Message("u1", Role.USER, (TextPart("first"), _image(data="b2xk")))
    newer = Message("u2", Role.USER, (TextPart("second"), _image(data="bmV3")))
    last = Message("u3", Role.USER, (TextPart("third"),))
    request = RenderRequest(config=make_config("openai"), system="", messages=(older, newer, last))

    messages = render(request).body["messages"]

    assert [m["content"] for m in messages[1:3]] == ["first", "second"]
    assert messages[3]["content"] == [_wire_image("bmV3"), {"type": "text", "text": "third"}]


def test_an_image_a_tool_returned_wins_when_the_request_ends_with_that_tool_message():
    call = ToolCallPart(call_id="c1", name="t", args={}, wire_id="c1")
    result = ToolResultPart(call_id="c1", name="t", content=(TextPart("see"), _image(data="dG9vbA==")))
    messages = (
        Message("u", Role.USER, (TextPart("go"), _image())),
        Message("a", Role.ASSISTANT, (call,), origin=ORIGIN),
        Message("t", Role.TOOL, (result,)),
    )
    request = RenderRequest(config=make_config("openai"), system="", messages=messages)

    wire = render(request).body["messages"]

    assert wire[1]["content"] == [_wire_image("dG9vbA=="), {"type": "text", "text": "go"}]
    assert wire[3]["content"] == "see"


def test_a_tool_image_from_an_earlier_round_is_not_sent_again():
    call = ToolCallPart(call_id="c1", name="t", args={}, wire_id="c1")
    result = ToolResultPart(call_id="c1", name="t", content=(_image(data="dG9vbA=="),))
    messages = (
        Message("u", Role.USER, (TextPart("go"),)),
        Message("a", Role.ASSISTANT, (call,), origin=ORIGIN),
        Message("t", Role.TOOL, (result,)),
        Message("a2", Role.ASSISTANT, (TextPart("done"),), origin=ORIGIN),
        Message("u2", Role.USER, (TextPart("thanks"),)),
    )
    request = RenderRequest(config=make_config("openai"), system="", messages=messages)

    assert all("image" not in json.dumps(m) for m in render(request).body["messages"])


def test_a_request_with_no_user_message_sends_no_image():
    result = ToolResultPart(call_id="c1", name="t", content=(_image(),))
    request = RenderRequest(config=make_config("openai"), system="", messages=(Message("t", Role.TOOL, (result,)),))

    assert "image_url" not in json.dumps(render(request).body)


def test_tools_are_left_out_when_the_profile_says_the_model_has_none():
    request = simple(tools=tuple(ToolSpec.from_schema(schema) for schema in tool_schemas()))

    assert "tools" not in render(request, tools=False).body
    assert "tools" in render(request).body


def test_tool_choice_is_sent_only_in_a_mode_the_profile_allows():
    tools = tuple(ToolSpec.from_schema(schema) for schema in tool_schemas())

    def body(choice, **profile):
        return render(simple(tools=tools, settings=Settings(tool_choice=choice)), **profile).body

    assert "tool_choice" not in body(ToolChoice("required"))
    assert body(ToolChoice("required"), tool_choice=["auto", "required"])["tool_choice"] == "required"
    assert body(ToolChoice("named", "get_form_state"), tool_choice=["auto", "named"])["tool_choice"] == {
        "type": "function",
        "function": {"name": "get_form_state"},
    }
    assert "tool_choice" not in body(ToolChoice("auto"), tool_choice=["auto", "required"])


def test_tool_choice_with_reasoning_uses_its_own_allowed_modes():
    tools = tuple(ToolSpec.from_schema(schema) for schema in tool_schemas())
    request = simple(tools=tools, settings=Settings(tool_choice=ToolChoice("required"), reasoning=True))

    assert "tool_choice" not in render(request, tool_choice=["auto", "required"]).body
    assert render(request, tool_choice=["auto", "required"], tool_choice_with_reasoning=["auto", "required"]).body["tool_choice"] == "required"


@pytest.mark.parametrize(
    "toggle, reasoning, expected",
    [
        ("reasoning_effort", False, {"reasoning_effort": "none"}),
        ("reasoning_effort", "low", {"reasoning_effort": "low"}),
        ("reasoning_effort", True, {}),
        ("thinking", False, {"thinking": {"type": "disabled"}}),
        ("thinking", True, {"thinking": {"type": "enabled"}}),
        ("chat_template_kwargs.enable_thinking", False, {"chat_template_kwargs": {"enable_thinking": False}}),
        ("chat_template_kwargs.enable_thinking", "high", {"chat_template_kwargs": {"enable_thinking": True}}),
    ],
)
def test_the_thinking_toggle_lands_where_the_profile_says(toggle, reasoning, expected):
    body = render(simple(settings=Settings(reasoning=reasoning)), reasoning_toggle=toggle).body
    base = render(simple()).body

    assert {key: value for key, value in body.items() if key not in base} == expected


def test_the_thinking_toggle_is_absent_without_a_profile_location_or_a_setting():
    assert render(simple(settings=Settings(reasoning=False))).body == render(simple()).body
    assert render(simple(), reasoning_toggle="thinking").body == render(simple()).body


def test_sampling_the_model_rejects_never_reaches_the_wire():
    request = simple(settings=Settings(top_p=0.4))

    body = render(request, sampling_rejected=["temperature", "top_p"]).body

    assert "temperature" not in body and "top_p" not in body


def test_sampling_ignored_with_reasoning_is_dropped_only_while_reasoning():
    on = simple(settings=Settings(reasoning=True))
    off = simple(settings=Settings(reasoning=False))

    assert "temperature" not in render(on, sampling_ignored_with_reasoning=["temperature"]).body
    assert "temperature" in render(off, sampling_ignored_with_reasoning=["temperature"]).body


def test_a_settings_value_replaces_the_configured_one_in_place():
    request = simple(settings=Settings(temperature=0.2, max_tokens=99))

    body = render(request).body

    assert list(body)[2:4] == ["temperature", "max_completion_tokens"]
    assert (body["temperature"], body["max_completion_tokens"]) == (0.2, 99)
