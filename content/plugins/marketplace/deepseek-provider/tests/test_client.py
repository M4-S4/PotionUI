import json

import httpx
import pytest

from src.plugin_api.llm import LLMConfig, OpenAIRequest

from backend import client as client_module
from backend.client import PLUGIN_ID, DeepSeekClient

BASE_URL = "https://api.deepseek.com"
AUTO = {"thinking": "auto", "reasoning_effort": "high"}
OFF = {"thinking": "off", "reasoning_effort": "high"}
TOOL = {"type": "function", "function": {"name": "list_presets", "parameters": {"type": "object", "properties": {}}}}
TOOL_CALL = {"id": "call_1", "type": "function", "function": {"name": "list_presets", "arguments": "{}"}}
PLAIN_HISTORY = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hello"}]
TOOL_HISTORY = PLAIN_HISTORY + [
    {"role": "assistant", "content": "", "tool_calls": [TOOL_CALL]},
    {"role": "tool", "content": "[]", "tool_call_id": "call_1"},
]
REASONING = "The user wants the presets, so I should list them."
REASONED_TOOL_HISTORY = PLAIN_HISTORY + [
    {"role": "assistant", "content": "", "tool_calls": [TOOL_CALL], "reasoning_content": REASONING},
    {"role": "tool", "content": "[]", "tool_call_id": "call_1"},
]
EARLIER_ANSWER_HISTORY = PLAIN_HISTORY + [
    {"role": "assistant", "content": "done"},
    {"role": "user", "content": "and now?"},
]
REPLY = {
    "choices": [{"message": {"content": "two presets"}, "finish_reason": "stop"}],
    "usage": {"total_tokens": 30, "prompt_tokens": 20, "completion_tokens": 10},
}


def request_with(messages, tools=None, **extra):
    params = {"temperature": 0.7, "max_completion_tokens": 4096, **extra}
    payload = {"model": "deepseek-flash", "messages": messages, **params}
    if tools:
        payload["tools"] = tools
    return OpenAIRequest(payload=payload, messages=messages, sampling_params=dict(params), headers={})


def config(**overrides):
    values = dict(
        id="cfg-1", name="DeepSeek", type="deepseek", enabled=True, base_url=BASE_URL, api_key="sk-test",
        model="deepseek-flash", system_message="SYS", max_tokens=4096, provider_options={"context_window": 1048576},
    )
    values.update(overrides)
    return LLMConfig(**values)


def adapt(settings, request):
    return DeepSeekClient(lambda: settings).adapt_request(request, config())


def test_a_request_without_tools_lets_deepseek_think_at_the_chosen_effort():
    request = adapt(AUTO, request_with(PLAIN_HISTORY))

    assert request.payload["thinking"] == {"type": "enabled"}
    assert request.payload["reasoning_effort"] == "high"


def test_a_first_tool_round_thinks_at_the_chosen_effort():
    request = adapt(AUTO, request_with(PLAIN_HISTORY, tools=[TOOL]))

    assert request.payload["thinking"] == {"type": "enabled"}
    assert request.payload["reasoning_effort"] == "high"


def test_a_tool_round_whose_earlier_assistant_turns_all_carry_reasoning_thinks():
    request = adapt(AUTO, request_with(REASONED_TOOL_HISTORY, tools=[TOOL]))

    assert request.payload["thinking"] == {"type": "enabled"}
    assert request.payload["reasoning_effort"] == "high"


@pytest.mark.parametrize("history", [TOOL_HISTORY, EARLIER_ANSWER_HISTORY])
def test_thinking_is_off_for_a_tool_request_that_holds_an_assistant_turn_without_reasoning(history):
    request = adapt(AUTO, request_with(history, tools=[TOOL]))

    assert request.payload["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in request.payload


def test_one_assistant_turn_without_reasoning_among_reasoned_ones_turns_thinking_off():
    history = REASONED_TOOL_HISTORY + [{"role": "assistant", "content": "retry"}]

    assert adapt(AUTO, request_with(history, tools=[TOOL])).payload["thinking"] == {"type": "disabled"}


def test_a_tool_history_without_the_tools_parameter_follows_the_same_rule():
    assert adapt(AUTO, request_with(TOOL_HISTORY)).payload["thinking"] == {"type": "disabled"}
    assert adapt(AUTO, request_with(REASONED_TOOL_HISTORY)).payload["thinking"] == {"type": "enabled"}


def test_a_request_without_tools_thinks_whatever_the_earlier_turns_carried():
    request = adapt(AUTO, request_with(EARLIER_ANSWER_HISTORY))

    assert request.payload["thinking"] == {"type": "enabled"}
    assert request.payload["reasoning_effort"] == "high"


@pytest.mark.parametrize("request_factory", [
    lambda: request_with(PLAIN_HISTORY),
    lambda: request_with(PLAIN_HISTORY, tools=[TOOL]),
    lambda: request_with(REASONED_TOOL_HISTORY, tools=[TOOL]),
])
def test_the_off_setting_never_thinks(request_factory):
    request = adapt(OFF, request_factory())

    assert request.payload["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in request.payload


def test_a_reasoning_effort_already_in_the_request_gives_way_to_the_setting():
    assert adapt(AUTO, request_with(PLAIN_HISTORY, reasoning_effort="low")).payload["reasoning_effort"] == "high"
    assert adapt(AUTO, request_with(PLAIN_HISTORY, [TOOL], reasoning_effort="low")).payload["reasoning_effort"] == "high"
    assert "reasoning_effort" not in adapt(AUTO, request_with(TOOL_HISTORY, [TOOL], reasoning_effort="low")).payload


def test_max_tokens_go_out_under_the_name_deepseek_reads():
    request = adapt(AUTO, request_with(PLAIN_HISTORY))

    assert request.payload["max_tokens"] == 4096
    assert "max_completion_tokens" not in request.payload


def test_settings_the_registry_cannot_resolve_fall_back_to_auto_and_high():
    request = adapt({}, request_with(PLAIN_HISTORY))

    assert request.payload["thinking"] == {"type": "enabled"}
    assert request.payload["reasoning_effort"] == "high"


def test_the_rest_of_the_request_is_left_as_built():
    request = adapt(AUTO, request_with(PLAIN_HISTORY, tools=[TOOL]))

    assert (request.payload["model"], request.payload["messages"], request.payload["tools"]) == (
        "deepseek-flash", PLAIN_HISTORY, [TOOL],
    )
    assert request.payload["temperature"] == 0.7


def test_the_default_client_reads_this_plugins_settings(monkeypatch):
    asked = []
    monkeypatch.setattr(client_module, "plugin_settings", lambda plugin_id: asked.append(plugin_id) or OFF)

    request = DeepSeekClient().adapt_request(request_with(PLAIN_HISTORY), config())

    assert asked == [PLUGIN_ID]
    assert request.payload["thinking"] == {"type": "disabled"}


@pytest.fixture
def wire(monkeypatch):
    sent = []
    replies = []
    real_client = httpx.AsyncClient

    def handle(request):
        sent.append(request)
        return replies.pop(0)()

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handle)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return sent, replies


async def test_a_tool_turn_whose_history_lacks_reasoning_reaches_deepseek_with_thinking_off(wire):
    sent, replies = wire
    replies.append(lambda: httpx.Response(200, json=REPLY))

    response = await DeepSeekClient(lambda: AUTO).generate_with_tools(TOOL_HISTORY[1:], config(), "SYS", [TOOL])

    body = json.loads(sent[0].content)
    assert str(sent[0].url) == f"{BASE_URL}/chat/completions"
    assert sent[0].headers["authorization"] == "Bearer sk-test"
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 4096
    assert response.content == "two presets"


async def test_a_streamed_answer_without_tools_carries_deepseeks_reasoning(wire):
    sent, replies = wire
    events = [
        {"choices": [{"delta": {"reasoning_content": "The user greets me."}}]},
        {"choices": [{"delta": {"content": "Hello!"}, "finish_reason": "stop"}]},
    ]
    stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    replies.append(lambda: httpx.Response(200, content=stream.encode(), headers={"content-type": "text/event-stream"}))

    received = [event async for event in DeepSeekClient(lambda: AUTO).stream_with_tools(PLAIN_HISTORY[1:], config(), "SYS")]

    assert json.loads(sent[0].content)["thinking"] == {"type": "enabled"}
    assert {"type": "thinking", "content": "The user greets me."} in received
    assert {"type": "token", "content": "Hello!"} in received


async def test_a_tool_turn_sends_the_earlier_rounds_reasoning_back_with_thinking_on(wire):
    sent, replies = wire
    replies.append(lambda: httpx.Response(200, json=REPLY))
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "", "tool_calls": [{**TOOL_CALL, "function": {"name": "list_presets", "arguments": {}}}], "reasoning": REASONING},
        {"role": "tool", "content": "[]", "tool_call_id": "call_1"},
    ]

    await DeepSeekClient(lambda: AUTO).generate_with_tools(history, config(), "SYS", [TOOL])

    body = json.loads(sent[0].content)
    assistant = body["messages"][2]
    assert assistant["reasoning_content"] == REASONING
    assert "reasoning" not in assistant
    assert assistant["tool_calls"][0]["function"]["arguments"] == "{}"
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "high"


async def test_a_streamed_first_tool_round_thinks_and_surfaces_its_reasoning(wire):
    sent, replies = wire
    events = [
        {"choices": [{"delta": {"reasoning_content": "I should list them."}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "list_presets", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}]},
    ]
    stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    replies.append(lambda: httpx.Response(200, content=stream.encode(), headers={"content-type": "text/event-stream"}))

    received = [
        event async for event in DeepSeekClient(lambda: AUTO).stream_with_tools(PLAIN_HISTORY[1:], config(), "SYS", [TOOL])
    ]

    body = json.loads(sent[0].content)
    assert body["thinking"] == {"type": "enabled"}
    assert {"type": "thinking", "content": "I should list them."} in received
    assert any(event["type"] == "tool_calls" for event in received)
