from typing import List, Optional, Set, Tuple

from src.features.llm.tools import tool_call_rescue

TEXT = "text"
BLOCK = "block"


class ToolCallTextFilter:
    _OPEN = "<tool_call>"
    _CLOSE = "</tool_call>"
    _ACTION_PREFIX = "<tool_action"

    def __init__(self, registered_tool_names: Optional[Set[str]] = None) -> None:
        self._buf = ""
        self._registered = set(registered_tool_names or ())
        self._state = "idle"

    @property
    def suppressing(self) -> bool:
        return self._state != "idle"

    @classmethod
    def _partial_open_suffix_len(cls, buf: str) -> int:
        prefixes = (cls._OPEN, cls._ACTION_PREFIX)
        max_len = min(len(buf), max(len(prefix) for prefix in prefixes) - 1)
        for length in range(max_len, 0, -1):
            suffix = buf[-length:]
            if any(prefix.startswith(suffix) for prefix in prefixes):
                return length
        return 0

    def feed(self, text: str) -> List[Tuple[str, str]]:
        self._buf += text
        segments: List[Tuple[str, str]] = []
        while True:
            if self._state == "idle":
                idx_call = self._buf.find(self._OPEN)
                idx_action = self._buf.find(self._ACTION_PREFIX)
                candidates = [index for index in (idx_call, idx_action) if index != -1]
                if not candidates:
                    keep = self._partial_open_suffix_len(self._buf)
                    if len(self._buf) > keep:
                        cut = len(self._buf) - keep
                        segments.append((TEXT, self._buf[:cut]))
                        self._buf = self._buf[cut:]
                    break
                idx = min(candidates)
                if idx:
                    segments.append((TEXT, self._buf[:idx]))
                self._buf = self._buf[idx:]
                self._state = "suppress_call" if self._buf.startswith(self._OPEN) else "peek_action"
                continue

            if self._state == "peek_action":
                match = tool_call_rescue.match_tool_action_open(self._buf)
                if match is None:
                    break
                if tool_call_rescue.tool_action_type(match) in self._registered:
                    self._state = "suppress_action"
                    continue
                segments.append((TEXT, match.group(0)))
                self._buf = self._buf[match.end():]
                self._state = "idle"
                continue

            if self._state == "suppress_call":
                idx = self._buf.find(self._CLOSE)
                if idx == -1:
                    break
                end = idx + len(self._CLOSE)
                segments.append((BLOCK, self._buf[:end]))
                self._buf = self._buf[end:]
                self._state = "idle"
                continue

            close = tool_call_rescue.find_tool_action_close(self._buf)
            if close is None:
                break
            segments.append((BLOCK, self._buf[:close.end()]))
            self._buf = self._buf[close.end():]
            self._state = "idle"
        return segments

    def flush(self) -> str:
        if self._state != "idle":
            return ""
        text, self._buf = self._buf, ""
        return text
