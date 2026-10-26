import codecs
import re
from typing import List

_LINE_END = re.compile(r"\r\n|\n|\r")


class LineBuffer:
    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8-sig")(errors="replace")
        self._tail = ""
        self._skip_lf = False

    def feed(self, chunk: bytes) -> List[str]:
        return self._split(self._decoder.decode(chunk))

    def close(self) -> List[str]:
        lines = self._split(self._decoder.decode(b"", final=True))
        tail, self._tail = self._tail, ""
        if tail:
            lines.append(tail)
        return lines

    def _split(self, text: str) -> List[str]:
        buffer = self._tail + text
        if self._skip_lf and buffer:
            if buffer.startswith("\n"):
                buffer = buffer[1:]
            self._skip_lf = False
        lines: List[str] = []
        start = 0
        for match in _LINE_END.finditer(buffer):
            lines.append(buffer[start:match.start()])
            start = match.end()
            self._skip_lf = match.group() == "\r" and start == len(buffer)
        self._tail = buffer[start:]
        return lines
