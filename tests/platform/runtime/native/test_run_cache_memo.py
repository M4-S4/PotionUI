from __future__ import annotations

import torch

from src.platform.runtime.native.cache_identity import UNIDENTIFIABLE
from src.platform.runtime.native.engine import RunCache


def _builder():
    calls = []

    def build():
        calls.append(1)
        return object()

    return calls, build


def test_memo_builds_once_per_key_and_returns_the_same_entry():
    cache = RunCache(revision=1)
    t = torch.zeros(3)
    calls, build = _builder()

    first = cache.memo("tag", (t,), ("a",), build)
    second = cache.memo("tag", (t,), ("a",), build)

    assert first is second
    assert len(calls) == 1
    assert len(cache) == 1


def test_memo_separates_extra_keys_tags_and_tensors():
    cache = RunCache(revision=1, max_entries=8)
    t, u = torch.zeros(3), torch.zeros(3)
    calls, build = _builder()

    cache.memo("tag", (t,), ("a",), build)
    cache.memo("tag", (t,), ("b",), build)
    cache.memo("other", (t,), ("a",), build)
    cache.memo("tag", (u,), ("a",), build)
    cache.memo("tag", (None,), ("a",), build)

    assert len(calls) == 5


def test_memo_rebuilds_after_an_in_place_write():
    cache = RunCache(revision=1)
    t = torch.zeros(3)
    calls, build = _builder()

    cache.memo("tag", (t,), (), build)
    t.add_(1.0)
    cache.memo("tag", (t,), (), build)

    assert len(calls) == 2


def test_memo_reads_the_live_revision_on_every_lookup():
    holder = [1]
    cache = RunCache(lambda: holder[0])
    t = torch.zeros(3)
    calls, build = _builder()

    cache.memo("tag", (t,), (), build)
    cache.memo("tag", (t,), (), build)
    holder[0] = 2
    cache.memo("tag", (t,), (), build)

    assert len(calls) == 2
    assert len(cache) == 1


def test_memo_bypasses_the_cache_for_an_unidentifiable_tensor(monkeypatch):
    from src.platform.runtime.native import engine

    monkeypatch.setattr(engine, "tensor_identity", lambda t: UNIDENTIFIABLE)
    cache = RunCache(revision=1)
    calls, build = _builder()

    cache.memo("tag", (torch.zeros(1),), (), build)
    cache.memo("tag", (torch.zeros(1),), (), build)

    assert len(calls) == 2
    assert len(cache) == 0


def test_memo_evicts_the_oldest_entry_at_capacity():
    cache = RunCache(revision=1, max_entries=2)
    tensors = [torch.zeros(i + 1) for i in range(3)]
    calls, build = _builder()

    for t in tensors:
        cache.memo("tag", (t,), (), build)
    cache.memo("tag", (tensors[2],), (), build)
    cache.memo("tag", (tensors[0],), (), build)

    assert len(calls) == 4
    assert len(cache) == 2
