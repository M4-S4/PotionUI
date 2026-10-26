import re
from dataclasses import dataclass
from typing import List, Optional

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


class ThinkTagStage:
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
