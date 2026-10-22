import hashlib
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Optional, Tuple

from src.features.models.read_throttle import ReadThrottle

SIGNATURE_VERSION = b"qs1"
HEAD_BYTES = 1024 * 1024
TAIL_BYTES = 1024 * 1024
WINDOW_BYTES = 64 * 1024
WINDOW_COUNT = 14
WINDOW_DIVISIONS = WINDOW_COUNT + 1
INLINE_HASH_LIMIT = 1024 * 1024
DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024


class DigestCancelled(Exception):
    pass


@dataclass(frozen=True)
class FileDigest:
    sha256: str
    quick_sig: str
    size: int


def signature_ranges(size: int) -> List[Tuple[int, int]]:
    if size <= 0:
        return []
    head = min(HEAD_BYTES, size)
    tail = min(TAIL_BYTES, size)
    window = min(WINDOW_BYTES, size)
    ranges = [(0, head)]
    for i in range(1, WINDOW_COUNT + 1):
        ranges.append((min(size * i // WINDOW_DIVISIONS, size - window), window))
    ranges.append((size - tail, tail))
    return ranges


def _signature(size: int, pieces: List[bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(SIGNATURE_VERSION)
    digest.update(size.to_bytes(8, "little"))
    for piece in pieces:
        digest.update(piece)
    return digest.hexdigest()


def quick_signature(path: str, size: Optional[int] = None) -> Optional[str]:
    try:
        total = os.stat(path).st_size if size is None else size
        pieces = []
        with open(path, "rb") as handle:
            for start, length in signature_ranges(total):
                handle.seek(start)
                piece = handle.read(length)
                if len(piece) != length:
                    return None
                pieces.append(piece)
        return _signature(total, pieces)
    except OSError:
        return None


class QuickSignatureCollector:
    def __init__(self, size: int):
        self._size = size
        self._ranges = signature_ranges(size)
        self._buffers = [bytearray() for _ in self._ranges]
        self._offset = 0

    def feed(self, chunk: bytes) -> None:
        end = self._offset + len(chunk)
        for index, (start, length) in enumerate(self._ranges):
            lo = max(start, self._offset)
            hi = min(start + length, end)
            if lo < hi:
                self._buffers[index] += chunk[lo - self._offset:hi - self._offset]
        self._offset = end

    def hexdigest(self) -> Optional[str]:
        if self._offset != self._size:
            return None
        if any(len(buffer) != length for buffer, (_, length) in zip(self._buffers, self._ranges)):
            return None
        return _signature(self._size, [bytes(buffer) for buffer in self._buffers])


def read_chunks(
    path: str,
    chunk_size: int = DEFAULT_CHUNK_BYTES,
    cancel_check: Optional[Callable[[], bool]] = None,
    pacing: Optional[ReadThrottle] = None,
) -> Iterator[bytes]:
    if pacing is not None and not pacing.wait(0, cancel_check):
        raise DigestCancelled(path)
    evict = pacing is not None and pacing.evict_page_cache and hasattr(os, "posix_fadvise")
    offset = 0
    handle = open(path, "rb")
    try:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return
            if cancel_check is not None and cancel_check():
                raise DigestCancelled(path)
            yield chunk
            if evict:
                os.posix_fadvise(handle.fileno(), offset, len(chunk), os.POSIX_FADV_DONTNEED)
            offset += len(chunk)
            if pacing is not None:
                if pacing.holding():
                    handle.close()
                    if not pacing.wait(0, cancel_check):
                        raise DigestCancelled(path)
                    handle = open(path, "rb")
                    handle.seek(offset)
                if not pacing.wait(len(chunk), cancel_check):
                    raise DigestCancelled(path)
    finally:
        handle.close()


def digest_file(
    path: str,
    size: int,
    cancel_check: Optional[Callable[[], bool]] = None,
    pacing: Optional[ReadThrottle] = None,
    chunk_size: int = DEFAULT_CHUNK_BYTES,
    on_progress: Optional[Callable[[int], Any]] = None,
) -> FileDigest:
    sha256 = hashlib.sha256()
    collector = QuickSignatureCollector(size)
    offset = 0
    for chunk in read_chunks(path, chunk_size, cancel_check, pacing):
        sha256.update(chunk)
        collector.feed(chunk)
        offset += len(chunk)
        if on_progress is not None:
            on_progress(offset)
    quick_sig = collector.hexdigest()
    if quick_sig is None:
        raise OSError(f"'{path}' changed size while it was being read")
    return FileDigest(sha256.hexdigest(), quick_sig, offset)
