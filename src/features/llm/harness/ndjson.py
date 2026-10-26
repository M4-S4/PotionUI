import json
import logging
from typing import Any, List

from src.features.llm.harness.lines import LineBuffer


class NDJSONFramer:
    def __init__(self, label: str = "ndjson") -> None:
        self._label = label
        self._lines = LineBuffer()

    def feed(self, chunk: bytes) -> List[Any]:
        return self._parse(self._lines.feed(chunk))

    def close(self) -> List[Any]:
        return self._parse(self._lines.close())

    def _parse(self, lines: List[str]) -> List[Any]:
        values: List[Any] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except ValueError:
                logging.warning(f"[{self._label}] Failed to parse line: {line[:200]}")
        return values
