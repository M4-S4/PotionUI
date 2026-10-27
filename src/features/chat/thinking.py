import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from src.features.chat.projection import metadata_without_reasoning
from src.features.llm.harness.events import ReasoningDelta

REASONING_EVENTS = frozenset({"reasoning", "reasoning_end"})


class ReasoningCapture:
    def __init__(self, enabled: bool, clock: Callable[[], float] = time.monotonic) -> None:
        self.enabled = enabled
        self._clock = clock
        self._segments: List[Dict[str, Any]] = []
        self._open_block: Optional[int] = None
        self._started = 0.0

    def on_reasoning(self, delta: ReasoningDelta) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        events: List[Dict[str, Any]] = []
        if self._open_block is not None and delta.block != self._open_block:
            events.extend(self.close())
        if self._open_block is None:
            self._open_block = delta.block
            self._started = self._clock()
            self._segments.append({"text": "", "duration_ms": 0})
        self._segments[-1]["text"] += delta.text
        kind = "summary" if delta.summary else ("raw" if delta.text else "omitted")
        events.append({"event": "reasoning", "data": {"block": delta.block, "kind": kind, "text": delta.text}})
        return events

    def close(self) -> List[Dict[str, Any]]:
        if self._open_block is None:
            return []
        duration_ms = int((self._clock() - self._started) * 1000)
        self._segments[-1]["duration_ms"] = duration_ms
        block, self._open_block = self._open_block, None
        return [{"event": "reasoning_end", "data": {"block": block, "duration_ms": duration_ms}}]

    def segments(self) -> List[Dict[str, Any]]:
        return [
            {"text": segment["text"], "duration_ms": segment["duration_ms"]}
            for segment in self._segments
            if segment["text"].strip()
        ]


def config_shows_thinking(llm_service: Any, llm_config_id: Optional[str]) -> bool:
    if not llm_config_id:
        return False
    try:
        config = llm_service.repository.get_configuration(llm_config_id)
    except Exception:
        return False
    return getattr(config, "show_thinking", False) is True


def hide_thinking_in_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    kind = event.get("event")
    data = event.get("data")
    if kind in REASONING_EVENTS:
        return None
    if not isinstance(data, dict):
        return event
    if kind == "status":
        detail = data.get("detail")
        if not isinstance(detail, dict) or "thinking_visible" not in detail:
            return event
        remaining = {key: value for key, value in detail.items() if key != "thinking_visible"}
        cleaned = {key: value for key, value in data.items() if key != "detail"}
        if remaining:
            cleaned["detail"] = remaining
        return {**event, "data": cleaned}
    if kind == "done":
        message = data.get("assistant_message")
        if isinstance(message, dict) and isinstance(message.get("metadata"), dict):
            redacted = {**message, "metadata": metadata_without_reasoning(message["metadata"])}
            return {**event, "data": {**data, "assistant_message": redacted}}
    return event


async def hide_thinking_stream(events: AsyncIterator[Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:
    async for event in events:
        shown = hide_thinking_in_event(event)
        if shown is not None:
            yield shown
