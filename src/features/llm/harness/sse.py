import logging
from dataclasses import dataclass
from typing import List, Optional

from src.features.llm.harness.lines import LineBuffer

DONE_SENTINEL = "[DONE]"


@dataclass(frozen=True)
class SSERecord:
    data: str
    event: Optional[str] = None


class SSEFramer:
    MAX_RECORD_BYTES = 1 * 1024 * 1024

    def __init__(self, label: str = "sse") -> None:
        self._label = label
        self._lines = LineBuffer()
        self._data: List[str] = []
        self._event: Optional[str] = None
        self._record_size = 0
        self._discarding = False
        self.dropped = 0

    def feed(self, chunk: bytes) -> List[SSERecord]:
        return self._consume(self._lines.feed(chunk))

    def close(self) -> List[SSERecord]:
        records = self._consume(self._lines.close())
        if self._data:
            logging.warning(
                f"[{self._label}] stream ended mid-record; discarding {len(self._data)} pending data line(s)"
            )
        self._reset()
        self._discarding = False
        return records

    def _consume(self, lines: List[str]) -> List[SSERecord]:
        records: List[SSERecord] = []
        for line in lines:
            record = self._line(line)
            if record is not None:
                records.append(record)
        return records

    def _reset(self) -> None:
        self._data = []
        self._event = None
        self._record_size = 0

    def _line(self, line: str) -> Optional[SSERecord]:
        if line == "":
            if self._discarding:
                self._discarding = False
                return None
            return self._dispatch()
        if self._discarding or line.startswith(":"):
            return None
        name, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if name == "event":
            self._event = value
            return None
        if name != "data":
            return None
        cost = max(len(value.encode("utf-8")), 1)
        if self._record_size + cost > self.MAX_RECORD_BYTES:
            logging.warning(
                f"[{self._label}] SSE record exceeded {self.MAX_RECORD_BYTES} bytes; dropping it"
            )
            self.dropped += 1
            self._reset()
            self._discarding = True
            return None
        self._data.append(value)
        self._record_size += cost
        return None

    def _dispatch(self) -> Optional[SSERecord]:
        if not self._data:
            self._reset()
            return None
        record = SSERecord(data="\n".join(self._data), event=self._event)
        self._reset()
        return record
