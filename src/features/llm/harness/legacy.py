import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.features.llm.harness.build import new_id
from src.features.llm.harness.messages import (
    LEGACY_ORIGIN,
    Message,
    Origin,
    Part,
    ReasoningPart,
    Role,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)


def _arguments(value: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    if isinstance(value, dict):
        return value, None
    if isinstance(value, str):
        try:
            decoded = json.loads(value) if value.strip() else {}
        except ValueError:
            return {}, value
        return (decoded if isinstance(decoded, dict) else {}), value
    return {}, None


def _tool_call(call: Dict[str, Any]) -> ToolCallPart:
    function = call.get("function") or {}
    args, args_raw = _arguments(function.get("arguments"))
    call_id = call.get("id") or new_id("call")
    return ToolCallPart(call_id=call_id, name=function.get("name", ""), args=args, args_raw=args_raw, wire_id=call.get("id"))


def _assistant(entry: Dict[str, Any], origin: Origin) -> Message:
    parts: List[Part] = []
    if entry.get("reasoning"):
        parts.append(ReasoningPart(text=entry["reasoning"]))
    if entry.get("content"):
        parts.append(TextPart(text=entry["content"]))
    parts.extend(_tool_call(call) for call in entry.get("tool_calls") or [])
    return Message(id=new_id("msg"), role=Role.ASSISTANT, parts=tuple(parts), origin=origin)


def _tool_result(entry: Dict[str, Any]) -> ToolResultPart:
    return ToolResultPart(
        call_id=entry.get("tool_call_id") or "",
        name=entry.get("name") or "",
        content=(TextPart(text=entry.get("content") or ""),),
    )


def messages_from_dicts(
    entries: Sequence[Dict[str, Any]], origin: Origin = LEGACY_ORIGIN,
) -> Tuple[List[Message], List[List[Dict[str, Any]]]]:
    messages: List[Message] = []
    sources: List[List[Dict[str, Any]]] = []
    for entry in entries:
        role = entry.get("role")
        if role == "tool":
            if messages and messages[-1].role is Role.TOOL:
                previous = messages[-1]
                messages[-1] = Message(id=previous.id, role=Role.TOOL, parts=(*previous.parts, _tool_result(entry)))
                sources[-1].append(entry)
                continue
            messages.append(Message(id=new_id("msg"), role=Role.TOOL, parts=(_tool_result(entry),)))
        elif role == "assistant":
            messages.append(_assistant(entry, origin))
        elif role == "system":
            messages.append(Message(id=new_id("msg"), role=Role.SYSTEM, parts=(TextPart(text=entry.get("content") or ""),)))
        else:
            messages.append(Message(id=new_id("msg"), role=Role.USER, parts=(TextPart(text=entry.get("content") or ""),)))
        sources.append([entry])
    return messages, sources
