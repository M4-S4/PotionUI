import json
from dataclasses import replace
from typing import Any, Dict, Optional, Sequence, Tuple

from src.features.llm.harness import sampling
from src.features.llm.harness.adapter import RenderRequest, Settings
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
from src.features.llm.harness.profile import LAYERED_FIELDS, CapabilityProfile, ProfileLayer, ProfileSource
from src.features.llm.repository import LLMConfig

IMAGE_B64 = "aW1hZ2UtYnl0ZXM="


def resolved(request: RenderRequest, profile: CapabilityProfile) -> RenderRequest:
    return replace(request, settings=sampling.resolve_settings(request.config, profile, request.settings))


def make_config(provider_type: str, **overrides: Any) -> LLMConfig:
    defaults: Dict[str, Any] = dict(
        id="cfg-1",
        name="Golden Config",
        type=provider_type,
        enabled=True,
        base_url="http://peer.invalid:1234",
        api_key=None,
        model="golden-model",
        system_message="stored system message",
        temperature=0.7,
        max_tokens=512,
        timeout=30,
        supports_vision=False,
        provider_options=None,
    )
    defaults.update(overrides)
    return LLMConfig(**defaults)


def tool_history() -> list:
    return [
        {"role": "user", "content": "use the tool"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_form_state", "arguments": {"scope": "all"}},
                }
            ],
        },
        {"role": "tool", "content": '{"ok": true}', "tool_call_id": "call_1", "name": "get_form_state"},
        {"role": "user", "content": "thanks"},
    ]


def tool_schemas() -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_form_state",
                "description": "Read the current form",
                "parameters": {
                    "type": "object",
                    "properties": {"scope": {"type": "string", "description": "what to read"}},
                    "required": ["scope"],
                },
            },
        }
    ]


def reasoned_history(reasoning: Optional[str]) -> list:
    history = tool_history()
    history[1]["reasoning"] = reasoning
    return history


def profile_with(adapter: str, model: str = "golden-model", **fields: Any) -> CapabilityProfile:
    profile = CapabilityProfile(adapter=adapter, model=model)
    request = fields.pop("context_request", None)
    if fields:
        profile = profile.layered([ProfileLayer(ProfileSource.OVERRIDE, fields)])
    if request is not None:
        if "context_request" in LAYERED_FIELDS:
            profile = profile.layered([ProfileLayer(ProfileSource.OVERRIDE, {"context_request": request})])
        else:
            object.__setattr__(profile, "context_request", request)
    return profile


def settings_from(options: Optional[Dict[str, Any]]) -> Settings:
    return Settings.from_options(options)


def messages_from_legacy(
    history: Sequence[Dict[str, Any]], origin: Origin, image_data: Optional[str] = None,
) -> Tuple[Message, ...]:
    messages = []
    last_user = max((i for i, entry in enumerate(history) if entry["role"] == "user"), default=None)
    for index, entry in enumerate(history):
        message_id = f"m{index}"
        role = entry["role"]
        if role == "user":
            parts: list = [TextPart(entry["content"])]
            if image_data and index == last_user:
                parts.append(ImagePart(media_type="image/jpeg", data=image_data))
            messages.append(Message(message_id, Role.USER, tuple(parts)))
        elif role == "assistant":
            parts = []
            calls = entry.get("tool_calls") or []
            if calls and entry.get("reasoning"):
                parts.append(ReasoningPart(text=entry["reasoning"]))
            if entry.get("content"):
                parts.append(TextPart(entry["content"]))
            for call in calls:
                function = call["function"]
                arguments = function["arguments"]
                raw = arguments if isinstance(arguments, str) else None
                args = json.loads(arguments) if isinstance(arguments, str) else arguments
                parts.append(ToolCallPart(
                    call_id=call["id"], name=function["name"], args=args, args_raw=raw, wire_id=call["id"],
                ))
            messages.append(Message(message_id, Role.ASSISTANT, tuple(parts), origin=origin))
        else:
            result = ToolResultPart(
                call_id=entry["tool_call_id"], name=entry.get("name", ""), content=(TextPart(entry["content"]),),
            )
            messages.append(Message(message_id, Role.TOOL, (result,)))
    return tuple(messages)
