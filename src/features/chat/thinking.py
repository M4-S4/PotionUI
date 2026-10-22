import re
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, Iterable, List, Optional

from src.features.chat.dto import MessageResponse

TEXT = "text"
THINKING = "thinking"
CLOSED = "closed"

_TAG_NAMES = ("think", "thinking", "thought")
_TAG_FLAGS = re.IGNORECASE | re.ASCII
_OPEN_TAG = re.compile(r"<(think|thinking|thought)(?:\s[^<>]*)?>", _TAG_FLAGS)
_TAG_HEAD = re.compile(r"<([a-z]*)", _TAG_FLAGS)
_CLOSE_TAGS = {name: re.compile(re.escape(f"</{name}>"), _TAG_FLAGS) for name in _TAG_NAMES}
_TAG_SPACE = " \t\r\n\f\v"
_MAX_HELD_TAG_CHARS = 256


@dataclass(frozen=True)
class Piece:
    kind: str
    text: str = ""
    segment: int = 0


def _could_open_a_tag(tail: str) -> bool:
    if len(tail) > _MAX_HELD_TAG_CHARS or ">" in tail:
        return False
    head = _TAG_HEAD.match(tail)
    name = head.group(1).lower()
    rest = tail[head.end():]
    if not rest:
        return any(candidate.startswith(name) for candidate in _TAG_NAMES)
    return name in _TAG_NAMES and rest[0] in _TAG_SPACE and "<" not in rest


def _held_open_tag_start(buffer: str) -> int:
    start = buffer.rfind("<")
    if start != -1 and _could_open_a_tag(buffer[start:]):
        return start
    return len(buffer)


def _held_close_tag_start(buffer: str, closing: str) -> int:
    for size in range(min(len(closing) - 1, len(buffer)), 0, -1):
        suffix = buffer[-size:]
        if suffix.isascii() and closing.startswith(suffix.lower()):
            return len(buffer) - size
    return len(buffer)


class ThinkTagSplitter:
    def __init__(self) -> None:
        self._open: Optional[str] = None
        self._held = ""
        self._trim_text = False
        self._trim_thinking = False

    def feed(self, text: str) -> List[Piece]:
        buffer = self._held + text
        self._held = ""
        pieces: List[Piece] = []
        while buffer:
            if self._open is None:
                match = _OPEN_TAG.search(buffer)
                if match:
                    self._emit_text(pieces, buffer[:match.start()])
                    self._open = match.group(1).lower()
                    self._trim_thinking = True
                    buffer = buffer[match.end():]
                    continue
                cut = _held_open_tag_start(buffer)
                self._emit_text(pieces, buffer[:cut])
                self._held = buffer[cut:]
                break
            closing = f"</{self._open}>"
            match = _CLOSE_TAGS[self._open].search(buffer)
            if match:
                self._emit_thinking(pieces, buffer[:match.start()])
                pieces.append(Piece(CLOSED))
                self._open = None
                self._trim_text = True
                buffer = buffer[match.end():]
                continue
            cut = _held_close_tag_start(buffer, closing)
            self._emit_thinking(pieces, buffer[:cut])
            self._held = buffer[cut:]
            break
        return pieces

    def finish(self) -> List[Piece]:
        held, self._held = self._held, ""
        pieces: List[Piece] = []
        if self._open is None:
            self._emit_text(pieces, held)
        else:
            self._emit_thinking(pieces, held)
            pieces.append(Piece(CLOSED))
            self._open = None
        return pieces

    def _emit_text(self, pieces: List[Piece], text: str) -> None:
        if self._trim_text:
            text = text.lstrip()
        if text:
            self._trim_text = False
            pieces.append(Piece(TEXT, text))

    def _emit_thinking(self, pieces: List[Piece], text: str) -> None:
        if self._trim_thinking:
            text = text.lstrip()
        if text:
            self._trim_thinking = False
            pieces.append(Piece(THINKING, text))


class ThinkingCapture:
    def __init__(self, enabled: bool, clock: Callable[[], float] = time.monotonic) -> None:
        self.enabled = enabled
        self._clock = clock
        self._splitter = ThinkTagSplitter()
        self._segments: List[Dict[str, Any]] = []
        self._started: Optional[float] = None
        self._open = False

    def on_token(self, text: str) -> List[Piece]:
        if not self.enabled:
            return [Piece(TEXT, text)]
        return self._route(self._splitter.feed(text))

    def on_thinking(self, text: str) -> List[Piece]:
        if not self.enabled or not text:
            return []
        return [self._append(text)]

    def flush(self) -> List[Piece]:
        if not self.enabled:
            return []
        pieces = self._route(self._splitter.finish())
        self._close()
        return pieces

    def segments(self) -> List[Dict[str, Any]]:
        return [
            {"text": segment["text"], "duration_ms": segment["duration_ms"]}
            for segment in self._segments
            if segment["text"].strip()
        ]

    def _route(self, pieces: Iterable[Piece]) -> List[Piece]:
        routed: List[Piece] = []
        for piece in pieces:
            if piece.kind == THINKING:
                routed.append(self._append(piece.text))
            elif piece.kind == CLOSED:
                self._close()
            else:
                if piece.text.strip():
                    self._close()
                routed.append(piece)
        return routed

    def _append(self, text: str) -> Piece:
        if not self._open:
            self._segments.append({"text": "", "duration_ms": 0})
            self._started = self._clock()
            self._open = True
        self._segments[-1]["text"] += text
        return Piece(THINKING, text, len(self._segments) - 1)

    def _close(self) -> None:
        if not self._open:
            return
        self._segments[-1]["duration_ms"] = int((self._clock() - self._started) * 1000)
        self._open = False


def visible_text(raw: str) -> str:
    splitter = ThinkTagSplitter()
    pieces = splitter.feed(raw) + splitter.finish()
    return "".join(piece.text for piece in pieces if piece.kind == TEXT)


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


def hide_thinking_in_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    kind = event.get("event")
    data = event.get("data")
    if kind == "thinking":
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
    if not any(_carries_thinking(message) for message in messages):
        return messages
    if config_shows_thinking(llm_service, llm_config_id):
        return messages
    return [_without_thinking(message) for message in messages]
