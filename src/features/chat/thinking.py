import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from src.features.chat.dto import MessageResponse
from src.features.llm.harness.events import ReasoningDelta

REASONING_EVENTS = frozenset({"reasoning", "reasoning_end"})


class ReasoningCapture:
    def __init__(self, enabled: bool, clock: Callable[[], float] = time.monotonic) -> None:
        self.enabled = enabled
        self._clock = clock
        self._segments: List[Dict[str, Any]] = []
        self._open_block: Optional[int] = None
        self._started = 0.0
        self._kept = 0

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

    def keep(self) -> None:
        self._kept = len(self._segments)

    def discard(self) -> None:
        self._open_block = None
        del self._segments[self._kept:]

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


def _carries_thinking(message: MessageResponse) -> bool:
    metadata = message.metadata or {}
    trace = metadata.get("behavior_trace") or {}
    return "thinking" in metadata or "thinking_visible" in trace


def _metadata_without_thinking(metadata: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {key: value for key, value in metadata.items() if key != "thinking"}
    trace = cleaned.get("behavior_trace")
    if isinstance(trace, dict) and "thinking_visible" in trace:
        cleaned["behavior_trace"] = {key: value for key, value in trace.items() if key != "thinking_visible"}
    return cleaned


def _without_thinking(message: MessageResponse) -> MessageResponse:
    if not _carries_thinking(message):
        return message
    return message.model_copy(update={"metadata": _metadata_without_thinking(message.metadata)})


def without_streamed_thinking(message: Dict[str, Any]) -> Dict[str, Any]:
    metadata = message.get("metadata")
    if not isinstance(metadata, dict) or "thinking" not in metadata:
        return message
    return {**message, "metadata": {key: value for key, value in metadata.items() if key != "thinking"}}


def _carries_replay_reasoning(metadata: Optional[Dict[str, Any]]) -> bool:
    transcript = (metadata or {}).get("tool_transcript")
    return isinstance(transcript, list) and any(
        isinstance(round_, dict) and "reasoning" in round_ for round_ in transcript
    )


def _metadata_without_replay_reasoning(metadata: Dict[str, Any]) -> Dict[str, Any]:
    transcript = [
        {key: value for key, value in round_.items() if key != "reasoning"} if isinstance(round_, dict) else round_
        for round_ in metadata["tool_transcript"]
    ]
    return {**metadata, "tool_transcript": transcript}


def message_without_replay_reasoning(message: MessageResponse) -> MessageResponse:
    if not _carries_replay_reasoning(message.metadata):
        return message
    return message.model_copy(update={"metadata": _metadata_without_replay_reasoning(message.metadata)})


def dict_without_replay_reasoning(message: Dict[str, Any]) -> Dict[str, Any]:
    metadata = message.get("metadata")
    if not isinstance(metadata, dict) or not _carries_replay_reasoning(metadata):
        return message
    return {**message, "metadata": _metadata_without_replay_reasoning(metadata)}


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
            redacted = {**message, "metadata": _metadata_without_thinking(message["metadata"])}
            return {**event, "data": {**data, "assistant_message": redacted}}
    return event


async def hide_thinking_stream(events: AsyncIterator[Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:
    async for event in events:
        shown = hide_thinking_in_event(event)
        if shown is not None:
            yield shown


def apply_thinking_visibility(
    messages: List[MessageResponse], llm_service: Any, llm_config_id: Optional[str],
) -> List[MessageResponse]:
    if any(_carries_replay_reasoning(message.metadata) for message in messages):
        messages = [message_without_replay_reasoning(message) for message in messages]
    if not any(_carries_thinking(message) for message in messages):
        return messages
    if config_shows_thinking(llm_service, llm_config_id):
        return messages
    return [_without_thinking(message) for message in messages]
