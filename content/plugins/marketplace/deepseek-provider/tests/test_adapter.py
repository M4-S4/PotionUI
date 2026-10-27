import json

import httpx
import pytest

from src.plugin_api.llm import (
    CapabilityProfile,
    LLMConfig,
    Message,
    OpenAIChatAdapter,
    Origin,
    ReasoningDelta,
    ReasoningPart,
    RenderRequest,
    Role,
    Settings,
    TextDelta,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    ToolSpec,
)

from backend import adapter as adapter_module
from backend.adapter import PLUGIN_ID, DeepSeekAdapter

BASE_URL = "https://api.deepseek.com"
AUTO = {"thinking": "auto", "reasoning_effort": "high"}
OFF = {"thinking": "off", "reasoning_effort": "high"}
ORIGIN = Origin(adapter="openai-chat", provider="deepseek", model="deepseek-flash")
TOOL = ToolSpec(name="list_presets", description="", parameters={"type": "object", "properties": {}})
REASONING = "The user wants the presets, so I should list them."

USER = Message("u1", Role.USER, (TextPart("hello"),))
CALL = ToolCallPart(call_id="call_1", name="list_presets", args={}, wire_id="call_1")
RESULT = Message("t1", Role.TOOL, (ToolResultPart(call_id="call_1", name="", content=(TextPart("[]"),)),))
UNREASONED_ROUND = (USER, Message("a1", Role.ASSISTANT, (CALL,), origin=ORIGIN), RESULT)
REASONED_ROUND = (USER, Message("a1", Role.ASSISTANT, (ReasoningPart(text=REASONING), CALL), origin=ORIGIN), RESULT)
EARLIER_ANSWER = (
    USER,
    Message("a1", Role.ASSISTANT, (TextPart("done"),), origin=ORIGIN),
    Message("u2", Role.USER, (TextPart("and now?"),)),
)
PROFILE = CapabilityProfile(adapter="openai-chat", model="deepseek-flash")


def config(**overrides):
    values = dict(
        id="cfg-1", name="DeepSeek", type="deepseek", enabled=True, base_url=BASE_URL, api_key="sk-test",
        model="deepseek-flash", system_message="SYS", temperature=0.7, max_tokens=4096,
    )
    values.update(overrides)
    return LLMConfig(**values)


def render(settings, messages, tools=(), **config_overrides):
    request = RenderRequest(
        config=config(**config_overrides), system="SYS", messages=tuple(messages), tools=tuple(tools),
        settings=Settings(temperature=0.7, max_tokens=4096),
    )
    return DeepSeekAdapter(lambda: settings).render(request, PROFILE)


def test_a_request_without_tools_lets_deepseek_think_at_the_chosen_effort():
    body = render(AUTO, [USER]).body

    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "high"


def test_a_first_tool_round_thinks_at_the_chosen_effort():
    body = render(AUTO, [USER], tools=[TOOL]).body

    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "high"


def test_a_tool_round_whose_earlier_assistant_turns_all_carry_reasoning_thinks():
    body = render(AUTO, REASONED_ROUND, tools=[TOOL]).body

    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "high"
    assert body["messages"][2]["reasoning_content"] == REASONING


@pytest.mark.parametrize("messages", [UNREASONED_ROUND, EARLIER_ANSWER])
def test_thinking_is_off_for_a_tool_request_that_holds_an_assistant_turn_without_reasoning(messages):
    body = render(AUTO, messages, tools=[TOOL]).body

    assert body["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in body


def test_one_assistant_turn_without_reasoning_among_reasoned_ones_turns_thinking_off():
    retry = Message("a2", Role.ASSISTANT, (TextPart("retry"),), origin=ORIGIN)

    assert render(AUTO, REASONED_ROUND + (retry,), tools=[TOOL]).body["thinking"] == {"type": "disabled"}


def test_a_tool_history_without_the_tools_parameter_follows_the_same_rule():
    assert render(AUTO, UNREASONED_ROUND).body["thinking"] == {"type": "disabled"}
    assert render(AUTO, REASONED_ROUND).body["thinking"] == {"type": "enabled"}


def test_a_request_without_tools_thinks_whatever_the_earlier_turns_carried():
    body = render(AUTO, EARLIER_ANSWER).body

    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "high"


@pytest.mark.parametrize("messages, tools", [([USER], []), ([USER], [TOOL]), (REASONED_ROUND, [TOOL])])
def test_the_off_setting_never_thinks(messages, tools):
    body = render(OFF, messages, tools=tools).body

    assert body["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in body


def test_a_reasoning_effort_in_the_config_gives_way_to_the_setting():
    options = {"reasoning_effort": "low"}

    assert render(AUTO, [USER], provider_options=options).body["reasoning_effort"] == "high"
    assert render(AUTO, [USER], tools=[TOOL], provider_options=options).body["reasoning_effort"] == "high"
    assert "reasoning_effort" not in render(AUTO, UNREASONED_ROUND, tools=[TOOL], provider_options=options).body


def test_max_tokens_go_out_under_the_name_deepseek_reads():
    body = render(AUTO, [USER]).body

    assert body["max_tokens"] == 4096
    assert "max_completion_tokens" not in body


def test_settings_the_registry_cannot_resolve_fall_back_to_auto_and_high():
    body = render({}, [USER]).body

    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "high"


def test_the_rest_of_the_request_is_left_as_built():
    wire = render(AUTO, [USER], tools=[TOOL])

    assert wire.url == f"{BASE_URL}/chat/completions"
    assert wire.headers["Authorization"] == "Bearer sk-test"
    assert wire.body["model"] == "deepseek-flash"
    assert wire.body["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hello"}]
    assert wire.body["tools"] == [TOOL.function_schema()]
    assert wire.body["temperature"] == 0.7


def test_the_bytes_are_the_openai_request_with_the_deepseek_changes_applied_in_the_old_order():
    wire = render(AUTO, [USER], tools=[TOOL])

    assert list(wire.body) == [
        "model", "messages", "temperature", "stream", "stream_options", "tools",
        "max_tokens", "thinking", "reasoning_effort",
    ]


def test_a_per_call_thinking_override_does_not_change_what_the_plugin_decides():
    request = RenderRequest(
        config=config(), system="SYS", messages=(USER,), settings=Settings.from_options({"think": False})
    )

    assert DeepSeekAdapter(lambda: AUTO).render(request, PROFILE).body["thinking"] == {"type": "enabled"}


def test_the_default_adapter_reads_this_plugins_settings(monkeypatch):
    asked = []
    monkeypatch.setattr(adapter_module, "plugin_settings", lambda plugin_id: asked.append(plugin_id) or OFF)
    request = RenderRequest(config=config(), system="SYS", messages=(USER,))

    body = DeepSeekAdapter().render(request, PROFILE).body

    assert asked == [PLUGIN_ID]
    assert body["thinking"] == {"type": "disabled"}


def test_it_is_an_openai_chat_adapter_so_the_stream_decodes_like_any_compatible_server():
    assert isinstance(DeepSeekAdapter(lambda: AUTO), OpenAIChatAdapter)


async def test_a_streamed_answer_carries_deepseeks_reasoning(monkeypatch):
    sent = []
    events = [
        {"choices": [{"delta": {"reasoning_content": "The user greets me."}}]},
        {"choices": [{"delta": {"content": "Hello!"}, "finish_reason": "stop"}]},
    ]
    stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    real_client = httpx.AsyncClient

    def handle(request):
        sent.append(request)
        return httpx.Response(200, content=stream.encode(), headers={"content-type": "text/event-stream"})

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handle)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    adapter = DeepSeekAdapter(lambda: AUTO)
    request = RenderRequest(config=config(), system="SYS", messages=(USER,))
    wire = adapter.render(request, PROFILE)
    decoder = adapter.decoder(request, PROFILE)

    decoded = []
    async for chunk in adapter.send(wire, request):
        decoded.extend(decoder.feed(chunk))
    decoded.extend(decoder.close())

    assert json.loads(sent[0].content)["thinking"] == {"type": "enabled"}
    assert [e.text for e in decoded if isinstance(e, ReasoningDelta)] == ["The user greets me."]
    assert [e.text for e in decoded if isinstance(e, TextDelta)] == ["Hello!"]
