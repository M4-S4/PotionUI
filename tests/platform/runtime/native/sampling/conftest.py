"""Shared fixtures for the sampling suite."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace

import pytest

from src.platform.runtime.native.sampling.registry import sampler_registry


@pytest.fixture
def swap_sampler():
    """Stand a spy callable in front of a registered core algorithm for the
    duration of one test, restoring the real definition afterwards."""

    @contextmanager
    def _swap(key: str, fn):
        order = sampler_registry.definitions()
        original = sampler_registry.get(key)
        sampler_registry.unregister(key)
        sampler_registry.register(replace(original, sample=fn))
        try:
            yield
        finally:
            for definition in order:
                sampler_registry.unregister(definition.key)
            for definition in order:
                sampler_registry.register(definition)

    return _swap
