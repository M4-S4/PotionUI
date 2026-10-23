from typing import List

from src.features.prompt import conditions

HEAD_READ_LIMIT = conditions.MAX_HEAD_SCAN_CHARS + conditions.MAX_HEAD_CHARS + 2


class _CountingHead:
    def __init__(self, head, reads: List[int]):
        self.head = head
        self.reads = reads

    def match(self, text, pos, endpos):
        found = self.head.match(text, pos, endpos)
        closed = text.startswith(">>", found.end()) and found.end() - pos <= conditions.MAX_HEAD_CHARS
        if not self.reads:
            self.reads.append(0)
        self.reads[-1] += found.end() - pos if closed else endpos - pos
        return found


def count_head_reads(monkeypatch) -> List[int]:
    reads: List[int] = []
    tokenize = conditions._tokenize

    def counted(text):
        reads.append(0)
        return tokenize(text)

    monkeypatch.setattr(conditions, "_HEAD", _CountingHead(conditions._HEAD, reads))
    monkeypatch.setattr(conditions, "_tokenize", counted)
    return reads
