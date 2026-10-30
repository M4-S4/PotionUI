import json
from typing import Any, Dict, List, Optional, Sequence

from src.features.llm.harness.adapter import RenderRequest, Settings, ToolSpec
from src.features.llm.harness.adapters.openrouter import DETAILS_KEY, DIGEST_KEY, OpenRouterAdapter, details_digest
from src.features.llm.harness.messages import (
    Message,
    Origin,
    ReasoningPart,
    Role,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from src.features.llm.harness.profile import CapabilityProfile, ProfileLayer, ProfileSource
from src.features.llm.repository import LLMConfig

ADAPTER = OpenRouterAdapter()
MODEL = "anthropic/claude-sonnet-5.5"
BASE_URL = "https://openrouter.ai/api/v1"

ENCRYPTED = {
    "type": "reasoning.encrypted",
    "id": "rd-1",
    "format": "anthropic-claude-v1",
    "index": 0,
    "data": "Zm9v",
}
TEXT_BLOCK = {
    "type": "reasoning.text",
    "id": "rd-2",
    "format": "anthropic-claude-v1",
    "index": 1,
    "text": "Both cities need a lookup.",
    "signature": "sig-abc",
}
SUMMARY = {"type": "reasoning.summary", "id": "rd-3", "format": "openai-responses-v1", "index": 2, "summary": "Checks both."}


def make_config(**fields: Any) -> LLMConfig:
    values: Dict[str, Any] = {
        "id": "cfg_1",
        "name": "Router",
        "type": "openrouter",
        "enabled": True,
        "base_url": BASE_URL,
        "api_key": "or-key-a",
        "model": MODEL,
        "system_message": "",
        "show_thinking": False,
    }
    values.update(fields)
    return LLMConfig(**values)


def make_profile(**fields: Any) -> CapabilityProfile:
    return CapabilityProfile(adapter="openrouter", model=MODEL).layered(
        [ProfileLayer(ProfileSource.OVERRIDE, fields)] if fields else []
    )


def make_request(
    messages: Sequence[Message] = (),
    *,
    config: Optional[LLMConfig] = None,
    tools: Sequence[ToolSpec] = (),
    settings: Optional[Settings] = None,
) -> RenderRequest:
    return RenderRequest(
        config=config or make_config(),
        system="Be brief.",
        messages=tuple(messages),
        tools=tuple(tools),
        settings=settings or Settings(),
        stream=True,
    )


def lookup_tool() -> ToolSpec:
    return ToolSpec(name="lookup", description="Look a city up.", parameters={"type": "object", "properties": {}})


def user(text: str = "Weather in Paris and Zurich?", message_id: str = "user_1") -> Message:
    return Message(id=message_id, role=Role.USER, parts=(TextPart(text),))


def reasoned_assistant(
    details: Sequence[Dict[str, Any]],
    *,
    config: Optional[LLMConfig] = None,
    provider: Optional[str] = "Anthropic",
    message_id: str = "asst_1",
    thinking: str = "Both cities need a lookup.",
    calls: bool = True,
    origin: Optional[Origin] = None,
) -> Message:
    parts: List[Any] = [
        ReasoningPart(text=thinking, opaque={DIGEST_KEY: details_digest(details)}),
        TextPart("Looking."),
    ]
    if calls:
        parts += [
            ToolCallPart(call_id="call_a", name="lookup", args={"city": "Paris"}, args_raw='{"city":"Paris"}', wire_id="toolu_a"),
            ToolCallPart(call_id="call_b", name="lookup", args={"city": "Zurich"}, args_raw='{"city":"Zurich"}', wire_id="toolu_b"),
        ]
    return Message(
        id=message_id,
        role=Role.ASSISTANT,
        parts=tuple(parts),
        origin=origin or Origin.for_config(ADAPTER.name, config or make_config()),
        raw={"provider": provider, DETAILS_KEY: [dict(item) for item in details]},
    )


def tool_results(message_id: str = "tool_1") -> Message:
    return Message(
        id=message_id,
        role=Role.TOOL,
        parts=(
            ToolResultPart(call_id="call_a", name="lookup", content=(TextPart("Sunny"),)),
            ToolResultPart(call_id="call_b", name="lookup", content=(TextPart("Rain"),)),
        ),
    )


def sse(*events: Any, comments: Sequence[str] = ()) -> bytes:
    frames = [f": {comment}\n\n" for comment in comments]
    for event in events:
        frames.append("data: [DONE]\n\n" if event == "[DONE]" else f"data: {json.dumps(event, ensure_ascii=False)}\n\n")
    return "".join(frames).encode("utf-8")


def chunk(delta: Dict[str, Any], finish: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    return {
        "id": "gen-1",
        "object": "chat.completion.chunk",
        "model": MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        **extra,
    }
