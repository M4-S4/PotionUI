from __future__ import annotations

from collections import Counter

from src.platform.runtime.native.engine import RunCache


def count_hits(cache: RunCache) -> Counter:
    hits: Counter = Counter()
    real_get = cache.get

    def get(key):
        value = real_get(key)
        if value is not None:
            hits[key[0]] += 1
        return value

    cache.get = get
    return hits
