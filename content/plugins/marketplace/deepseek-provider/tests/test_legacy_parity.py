import httpx
import pytest

from src.plugin_api.llm import (
    CapabilityProfile,
    LLMConfig,
    Message,
    Origin,
    ReasoningPart,
    RenderRequest,
    Role,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    ToolSpec,
)

from backend.adapter import DeepSeekAdapter
from backend.client import DeepSeekClient

AUTO = {"thinking": "auto", "reasoning_effort": "high"}
OFF = {"thinking": "off", "reasoning_effort": "high"}
ORIGIN = Origin(adapter="openai-chat", provider="deepseek", model="deepseek-flash")
SCHEMA = {"type": "function", "function": {"name": "list_presets", "description": "", "parameters": {"type": "object", "properties": {}}}}
REASONING = "The user wants the presets."
TOOL_CALL = {"id": "call_1", "type": "function", "function": {"name": "list_presets", "arguments": {}}}
USER = {"role": "user", "content": "hello"}
CALL_TURN = {"role": "assistant", "content": "", "tool_calls": [TOOL_CALL]}
RESULT = {"role": "tool", "content": "[]", "tool_call_id": "call_1", "name": "list_presets"}

CASES = {
    "plain": (AUTO, [USER], False, {}),
    "first_tool_round": (AUTO, [USER], True, {}),
    "reasoned_round": (AUTO, [USER, {**CALL_TURN, "reasoning": REASONING}, RESULT], True, {}),
    "unreasoned_round": (AUTO, [USER, CALL_TURN, RESULT], True, {}),
    "off": (OFF, [USER], True, {}),
    "configured_effort_gives_way": (AUTO, [USER], True, {"provider_options": {"reasoning_effort": "low", "seed": 3}}),
}


def canonical(history):
    messages = []
    for index, entry in enumerate(history):
        if entry["role"] == "user":
            messages.append(Message(f"m{index}", Role.USER, (TextPart(entry["content"]),)))
        elif entry["role"] == "assistant":
            parts = []
            if entry.get("reasoning"):
                parts.append(ReasoningPart(text=entry["reasoning"]))
            for call in entry["tool_calls"]:
                parts.append(ToolCallPart(call_id=call["id"], name="list_presets", args={}, wire_id=call["id"]))
            messages.append(Message(f"m{index}", Role.ASSISTANT, tuple(parts), origin=ORIGIN))
        else:
            result = ToolResultPart(call_id=entry["tool_call_id"], name=entry["name"], content=(TextPart(entry["content"]),))
            messages.append(Message(f"m{index}", Role.TOOL, (result,)))
    return tuple(messages)


def config(**overrides):
    values = dict(
        id="cfg-1", name="DeepSeek", type="deepseek", enabled=True, base_url="https://api.deepseek.com",
        api_key="sk-test", model="deepseek-flash", system_message="SYS", max_tokens=4096,
        provider_options={"context_window": 1048576},
    )
    values.update(overrides)
    return LLMConfig(**values)


@pytest.mark.parametrize("name", list(CASES))
async def test_the_adapter_sends_the_bytes_the_client_did(monkeypatch, name):
    settings, history, with_tools, overrides = CASES[name]
    sent = []
    real_client = httpx.AsyncClient

    def handle(request):
        sent.append(request)
        return httpx.Response(200, content=b"data: [DONE]\n\n")

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handle)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    stream = DeepSeekClient(lambda: settings).stream_with_tools(
        history, config(**overrides), "SYS", tools=[SCHEMA] if with_tools else None
    )
    async for _ in stream:
        pass
    request = RenderRequest(
        config=config(**overrides),
        system="SYS",
        messages=canonical(history),
        tools=(ToolSpec.from_schema(SCHEMA),) if with_tools else (),
    )

    wire = DeepSeekAdapter(lambda: settings).render(request, CapabilityProfile(adapter="openai-chat", model="deepseek-flash"))

    assert wire.content() == sent[0].content
    assert wire.url == str(sent[0].url)
    assert wire.headers["Authorization"] == sent[0].headers["authorization"]
