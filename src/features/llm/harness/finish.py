from typing import Optional

from src.features.llm.harness.events import FinishReason

_OPENAI_COMPAT = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "function_call": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTER,
}

_OLLAMA = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
}


def from_openai_compat(raw: Optional[str]) -> FinishReason:
    return _OPENAI_COMPAT.get(raw, FinishReason.UNKNOWN) if raw is not None else FinishReason.UNKNOWN


def from_ollama(raw: Optional[str]) -> FinishReason:
    return _OLLAMA.get(raw, FinishReason.UNKNOWN) if raw is not None else FinishReason.UNKNOWN


def with_tool_calls(reason: FinishReason, has_tool_calls: bool) -> FinishReason:
    if has_tool_calls and reason in (FinishReason.STOP, FinishReason.UNKNOWN):
        return FinishReason.TOOL_CALLS
    return reason
