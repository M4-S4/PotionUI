import hashlib
import os

import pytest

from src.features.models.file_digest import (
    WINDOW_BYTES,
    DigestCancelled,
    QuickSignatureCollector,
    digest_file,
    quick_signature,
    read_chunks,
    signature_ranges,
)
from src.features.models import file_digest as file_digest_module
from src.features.models.read_throttle import ReadThrottle
from tests.features.models.fakes import FakeClock

MIB = 1024 * 1024


def _content(size, seed=7):
    block = bytes((seed + i * 31) % 251 for i in range(251))
    return (block * (size // len(block) + 1))[:size]


def _file(tmp_path, size, name="weights.safetensors", seed=7):
    path = tmp_path / name
    path.write_bytes(_content(size, seed))
    return path


class TestRanges:
    def test_a_large_file_has_a_head_a_tail_and_fourteen_windows_in_order(self):
        size = 100 * MIB

        ranges = signature_ranges(size)

        assert len(ranges) == 16
        assert ranges[0] == (0, 1048576)
        assert ranges[-1] == (size - 1048576, 1048576)
        windows = ranges[1:-1]
        assert all(length == 65536 for _, length in windows)
        assert [start for start, _ in windows] == sorted(start for start, _ in windows)
        assert windows[0][0] == size // 15
        assert windows[-1][0] == size * 14 // 15

    @pytest.mark.parametrize("size", [1, 2, 100, WINDOW_BYTES - 1, WINDOW_BYTES, MIB + 3])
    def test_every_range_lies_inside_the_file(self, size):
        for start, length in signature_ranges(size):
            assert 0 <= start and length > 0 and start + length <= size

    def test_an_empty_file_has_no_ranges(self):
        assert signature_ranges(0) == []


class TestQuickSignature:
    def test_the_same_bytes_give_the_same_signature_regardless_of_the_path(self, tmp_path):
        first = _file(tmp_path, 5 * MIB, "a.safetensors")
        second = _file(tmp_path, 5 * MIB, "b.safetensors")

        assert quick_signature(str(first)) == quick_signature(str(second))

    def test_a_different_byte_in_the_head_changes_it(self, tmp_path):
        first = _file(tmp_path, 5 * MIB, "a.safetensors")
        data = bytearray(first.read_bytes())
        data[100] ^= 1
        second = tmp_path / "b.safetensors"
        second.write_bytes(bytes(data))

        assert quick_signature(str(first)) != quick_signature(str(second))

    def test_a_different_byte_in_the_tail_changes_it(self, tmp_path):
        first = _file(tmp_path, 5 * MIB, "a.safetensors")
        data = bytearray(first.read_bytes())
        data[-5] ^= 1
        second = tmp_path / "b.safetensors"
        second.write_bytes(bytes(data))

        assert quick_signature(str(first)) != quick_signature(str(second))

    def test_a_different_byte_inside_a_sampled_window_changes_it(self, tmp_path):
        size = 10 * MIB
        first = _file(tmp_path, size, "a.safetensors")
        offset = signature_ranges(size)[5][0] + 10
        data = bytearray(first.read_bytes())
        data[offset] ^= 1
        second = tmp_path / "b.safetensors"
        second.write_bytes(bytes(data))

        assert quick_signature(str(first)) != quick_signature(str(second))

    def test_a_different_byte_between_the_samples_does_not_change_it(self, tmp_path):
        size = 10 * MIB
        first = _file(tmp_path, size, "a.safetensors")
        sampled = signature_ranges(size)
        gap = sampled[2][0] + WINDOW_BYTES + 100
        assert all(not (s <= gap < s + n) for s, n in sampled)
        data = bytearray(first.read_bytes())
        data[gap] ^= 1
        second = tmp_path / "b.safetensors"
        second.write_bytes(bytes(data))

        assert quick_signature(str(first)) == quick_signature(str(second))
        assert hashlib.sha256(first.read_bytes()).digest() != hashlib.sha256(second.read_bytes()).digest()

    def test_the_size_is_part_of_the_signature(self, tmp_path):
        first = tmp_path / "a.bin"
        second = tmp_path / "b.bin"
        for path, size in ((first, 10 * MIB), (second, 11 * MIB)):
            with open(path, "wb") as handle:
                handle.truncate(size)

        assert quick_signature(str(first)) != quick_signature(str(second))

    def test_a_missing_file_has_no_signature(self, tmp_path):
        assert quick_signature(str(tmp_path / "missing.bin")) is None

    def test_a_declared_size_larger_than_the_file_has_no_signature(self, tmp_path):
        path = _file(tmp_path, 1000)

        assert quick_signature(str(path), size=2000) is None


class TestCollector:
    @pytest.mark.parametrize(
        "size,chunk",
        [(0, 1), (1, 1), (700, 1), (700, 7), (MIB + 17, 251), (MIB + 17, 70000), (MIB + 17, 5 * MIB), (3 * MIB, 4096), (3 * MIB, 5 * MIB)],
    )
    def test_collecting_from_a_stream_equals_reading_the_ranges(self, tmp_path, size, chunk):
        path = _file(tmp_path, size)
        data = path.read_bytes()
        collector = QuickSignatureCollector(size)

        for start in range(0, len(data), chunk):
            collector.feed(data[start:start + chunk])

        assert collector.hexdigest() == quick_signature(str(path))

    def test_a_stream_longer_than_declared_has_no_signature(self):
        collector = QuickSignatureCollector(100)
        collector.feed(b"x" * 150)

        assert collector.hexdigest() is None

    def test_a_short_stream_has_no_signature(self):
        collector = QuickSignatureCollector(100)
        collector.feed(b"x" * 99)

        assert collector.hexdigest() is None


class TestDigestFile:
    @pytest.mark.parametrize("size", [0, 1, 4095, 4096, 4097, 3 * MIB + 5])
    def test_returns_the_sha256_and_the_signature_in_one_pass(self, tmp_path, size):
        path = _file(tmp_path, size)

        digest = digest_file(str(path), size, chunk_size=4096)

        assert digest.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest.quick_sig == quick_signature(str(path))
        assert digest.size == size

    def test_progress_is_cumulative_and_ends_at_the_size(self, tmp_path):
        path = _file(tmp_path, 10_000)
        seen = []

        digest_file(str(path), 10_000, chunk_size=4096, on_progress=seen.append)

        assert seen == [4096, 8192, 10_000]

    def test_a_cancel_request_stops_the_read(self, tmp_path):
        path = _file(tmp_path, 10_000)
        reads = []

        with pytest.raises(DigestCancelled):
            digest_file(
                str(path), 10_000, cancel_check=lambda: len(reads) >= 1, chunk_size=1000, on_progress=reads.append
            )

        assert len(reads) == 1

    def test_a_file_that_is_smaller_than_declared_is_an_error(self, tmp_path):
        path = _file(tmp_path, 1000)

        with pytest.raises(OSError):
            digest_file(str(path), 2000, chunk_size=256)

    def test_a_pause_that_is_cancelled_stops_before_reading(self, tmp_path):
        path = _file(tmp_path, 1000)
        throttle = ReadThrottle(pauses_for_generation=True, busy=lambda: True, sleep=lambda seconds: None)
        opened = []

        with pytest.raises(DigestCancelled):
            digest_file(str(path), 1000, cancel_check=lambda: True, pacing=throttle, on_progress=opened.append)

        assert opened == []

    def test_hashed_ranges_are_dropped_from_the_page_cache_only_when_asked(self, tmp_path, monkeypatch):
        path = _file(tmp_path, 2500)
        advised = []
        monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
        monkeypatch.setattr(
            os, "posix_fadvise", lambda fd, offset, length, advice: advised.append((offset, length, advice)), raising=False
        )

        digest_file(str(path), 2500, pacing=ReadThrottle(), chunk_size=1024)
        assert advised == []
        digest_file(str(path), 2500, pacing=ReadThrottle(evict_page_cache=True), chunk_size=1024)

        assert advised == [(0, 1024, 4), (1024, 1024, 4), (2048, 452, 4)]

    def test_a_generation_that_is_already_running_holds_the_first_read(self, tmp_path):
        path = _file(tmp_path, 3000)
        clock = FakeClock()
        running = [True]
        throttle = ReadThrottle(pauses_for_generation=True, busy=lambda: running[0], sleep=clock.sleep)
        clock.on_sleep = lambda: running.__setitem__(0, len(clock.sleeps) < 2)
        first_read = []

        digest_file(str(path), 3000, pacing=throttle, chunk_size=1024, on_progress=lambda done: first_read.append(len(clock.sleeps)))

        assert first_read[0] == 2

    def test_the_cap_spaces_the_chunks(self, tmp_path):
        path = _file(tmp_path, 4096)
        sleeps = []
        now = {"t": 0.0}

        def sleep(seconds):
            sleeps.append(seconds)
            now["t"] += seconds

        throttle = ReadThrottle(bytes_per_second=1024, clock=lambda: now["t"], sleep=sleep)

        digest = digest_file(str(path), 4096, pacing=throttle, chunk_size=1024)

        assert digest.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        assert sum(sleeps) == 3.0


class TrackedFile:
    def __init__(self, handle, log, clock):
        self._handle = handle
        self._log = log
        self._clock = clock
        log.append(("open", len(clock.sleeps)))

    def read(self, size=-1):
        return self._handle.read(size)

    def seek(self, offset):
        self._log.append(("seek", offset))
        return self._handle.seek(offset)

    def fileno(self):
        return self._handle.fileno()

    def close(self):
        self._log.append(("close", len(self._clock.sleeps)))
        self._handle.close()


@pytest.fixture
def opened(monkeypatch):
    log = []
    clock = FakeClock()
    real_open = open

    def tracking_open(path, mode="rb"):
        return TrackedFile(real_open(path, mode), log, clock)

    monkeypatch.setattr(file_digest_module, "open", tracking_open, raising=False)
    return log, clock


class TestReadChunks:
    def test_a_running_generation_holds_the_read_before_the_file_is_opened(self, tmp_path, opened):
        log, clock = opened
        path = _file(tmp_path, 3000)
        running = [True]
        throttle = ReadThrottle(pauses_for_generation=True, busy=lambda: running[0], sleep=clock.sleep)
        clock.on_sleep = lambda: running.__setitem__(0, len(clock.sleeps) < 2)

        data = b"".join(read_chunks(str(path), 1024, pacing=throttle))

        assert data == path.read_bytes()
        assert log[0] == ("open", 2)

    def test_a_pause_in_the_middle_of_a_file_closes_the_handle_and_resumes_at_the_same_offset(self, tmp_path, opened):
        log, clock = opened
        path = _file(tmp_path, 3000)
        running = [False]
        throttle = ReadThrottle(pauses_for_generation=True, busy=lambda: running[0], sleep=clock.sleep)
        clock.on_sleep = lambda: running.__setitem__(0, len(clock.sleeps) < 2)

        digest = digest_file(
            str(path), 3000, pacing=throttle, chunk_size=1024,
            on_progress=lambda done: running.__setitem__(0, True) if done == 1024 else None,
        )

        expected = QuickSignatureCollector(3000)
        expected.feed(path.read_bytes())
        assert digest.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest.quick_sig == expected.hexdigest()
        assert log == [("open", 0), ("close", 0), ("open", 2), ("seek", 1024), ("close", 2)]

    def test_cancelling_while_parked_in_the_middle_of_a_file_stops_without_reopening(self, tmp_path, opened):
        log, clock = opened
        path = _file(tmp_path, 3000)
        running = [False]
        throttle = ReadThrottle(pauses_for_generation=True, busy=lambda: running[0], sleep=clock.sleep)
        chunks = read_chunks(str(path), 1024, cancel_check=lambda: len(clock.sleeps) >= 3, pacing=throttle)

        with pytest.raises(DigestCancelled):
            for _ in chunks:
                running[0] = True

        assert [entry[0] for entry in log].count("open") == 1
        assert len(clock.sleeps) == 3

    def test_without_a_pause_the_file_is_opened_once(self, tmp_path, opened):
        log, clock = opened
        path = _file(tmp_path, 3000)

        data = b"".join(read_chunks(str(path), 1024, pacing=ReadThrottle(pauses_for_generation=True, busy=lambda: False)))

        assert data == path.read_bytes()
        assert log == [("open", 0), ("close", 0)]

    def test_the_handle_is_closed_when_the_consumer_stops_early(self, tmp_path, opened):
        log, clock = opened
        path = _file(tmp_path, 3000)

        chunks = read_chunks(str(path), 1024)
        next(chunks)
        chunks.close()

        assert log == [("open", 0), ("close", 0)]
