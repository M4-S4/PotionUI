from __future__ import annotations

import pytest
import torch

from src.platform.runtime.native.engine import NativeModel
from src.platform.runtime.native.text_encoders import NativeTextEncoder
from src.platform.runtime.native.text_encoders.embed_cache import get_prompt_embed_cache


class FakeModule:
    def named_modules(self):
        return iter(())


class FakeModels:
    def __init__(self):
        self.calls = []

    def acquire(self, key, fingerprint, loader, estimated_vram_gb=None):
        self.calls.append((key, fingerprint))
        return NativeModel("fake", FakeModule(), spec=None, estimated_vram_gb=1.0)


class CachingFakeModels:
    def __init__(self):
        self.calls = []
        self._entries = {}

    def acquire(self, key, fingerprint, loader, estimated_vram_gb=None):
        self.calls.append((key, fingerprint))
        entry = self._entries.get(key)
        if entry is not None and entry[0] == fingerprint:
            return entry[1]
        value = loader()
        self._entries[key] = (fingerprint, value)
        return value


class RecordingEncoder(NativeTextEncoder):
    role = "fake"

    def __init__(self, pooled=False):
        self.calls: list[list[str]] = []
        self.image_calls: list[list] = []
        self._pooled = pooled

    def encode(self, texts, images=None, grounding_px=768, system_prompt=None, keep_vision=False):
        self.calls.append(list(texts))
        self.image_calls.append(list(images) if images else [])
        seq = max(1, len(texts[0]))
        out = {"context": torch.ones(1, seq, 4), "attention_mask": torch.ones(1, seq)}
        if self._pooled:
            out["pooled"] = torch.ones(1, 8)
        return out


class CountingLoader:
    def __init__(self, encoder):
        self._encoder = encoder
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self._encoder


@pytest.fixture
def fresh_embed_cache():
    get_prompt_embed_cache().clear()
    yield
    get_prompt_embed_cache().clear()
