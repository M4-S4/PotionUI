"""Tests for the model_loader/krea2 pipe.

Mirrors the flux loader test: a fake MODELS service records the three
independent acquires (TE / VAE / DiT) so the cache-key scheme + fingerprints
are asserted without touching real checkpoints.

The fake MODELS service below implements REAL hit/miss cache
semantics (key+fingerprint -> cached value, matching
``ModelLifecycle.acquire``) rather than just recording calls, because
the fix under test only shows up across two acquires of the SAME key: a
LoRA-set change must be a cache HIT (same fingerprint, same object, no
``loader()`` re-run) that ``_sync_loras`` reconciles in place, not a
fingerprint-bust that reloads the checkpoint from disk.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.pipelines.contracts import PipeInput
from src.pipelines.pipes._shared.generation.loader_helpers import (
    windowed_stack_fingerprint as _window_fingerprint,
)
from src.platform.runtime.native.lora import LoraStepWindow
from src.pipelines.pipes.model_loader.krea2.main import ModelLoaderKrea2Pipe
from src.platform.runtime.native.engine import NativeEngineLoader, NativeModel


class _FakeModule:
    """Stand-in for an ``nn.Module``: just enough surface (an empty
    ``named_modules()``) for the REAL ``remove_loras`` to run as a genuine
    no-op instead of needing a real Krea-2 arch instance."""

    def named_modules(self):
        return iter(())


class _FakeModels:
    """Real hit/miss cache semantics (key + fingerprint -> cached value),
    matching ``ModelLifecycle.acquire``'s contract: a fingerprint match
    returns the SAME cached object without re-running ``loader()``; a
    mismatch (new key, or a busted fingerprint) runs ``loader()`` and caches
    the result. ``self.calls`` still records every acquire() invocation
    (hit or miss) for the existing key/fingerprint assertions."""

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


def _config(loras=None, **over):
    cfg = ModelLoaderKrea2Pipe.get_default_config()
    cfg.update({
        "diffusion_model": {"file_path": "/m/krea2_dit.safetensors", "name": "dit"},
        "text_encoder": {"file_path": "/m/qwen3vl.safetensors", "name": "te"},
        "vae": {"file_path": "/m/qwen_vae.safetensors", "name": "vae"},
        "loras": loras or [],
    })
    cfg.update(over)
    return cfg


def _run(pipe, models=None):
    models = models if models is not None else _FakeModels()
    out = pipe.process(PipeInput(input={"MODELS": models}), lambda o: None)
    return models, out


@pytest.fixture(autouse=True)
def _fake_engine(monkeypatch):
    """Replace real checkpoint I/O (``NativeEngineLoader.load``) with a cheap
    fake, and count invocations per component kind -- the DiT count is the
    load-bearing assertion (must stay 1 across a LoRA-set change on
    a warm model, proving no re-read of the checkpoint from disk)."""
    counts = {"diffusion_model": 0, "text_encoder": 0, "vae": 0}

    def _fake_load(self, path, kind, **kwargs):
        counts[kind] = counts.get(kind, 0) + 1
        # A real wrapper, not a stand-in: the loaders revise its weight identity.
        return NativeModel(kind, _FakeModule(), spec=None, estimated_vram_gb=1.0)

    monkeypatch.setattr(NativeEngineLoader, "load", _fake_load)
    return counts


@pytest.fixture(autouse=True)
def _apply_spy(monkeypatch):
    """Replace the pipe's own ``_apply_loras`` (which would otherwise read a
    real LoRA file off disk via ``load_torch_file``) with a spy recording the
    (possibly empty) LoRA list it was asked to apply."""
    calls = []

    def _fake_apply(dit_model, loras):
        calls.append(list(loras))

    monkeypatch.setattr(ModelLoaderKrea2Pipe, "_apply_loras", staticmethod(_fake_apply))
    return calls


def test_windowed_lora_rides_the_bundle_and_stays_out_of_the_baked_stack(_apply_spy):
    loras = [
        {"model": "/m/style.safetensors", "strength": 0.8},
        {"model": "/m/turbo-sda.safetensors", "strength": 1.0, "step_start": 1, "step_end": 4},
    ]
    models, out = _run(ModelLoaderKrea2Pipe(config=_config(loras=loras)))
    bundle = out.output["model"]
    assert [(w["file_path"], w["window"]) for w in bundle.windowed_loras] == [
        ("/m/turbo-sda.safetensors", LoraStepWindow(1, 4)),
    ]
    assert [c for c in _apply_spy if c] == [[{"file_path": "/m/style.safetensors", "weight": 0.8, "window": None}]]
    assert "turbo-sda" not in bundle.dit._active_lora_fp
    assert "style.safetensors" in bundle.dit._active_lora_fp
    assert all("turbo-sda" not in fp for _, fp in models.calls)


def test_no_windowed_lora_passes_an_empty_tuple_through():
    _, out = _run(ModelLoaderKrea2Pipe(config=_config(loras=[{"model": "/m/style.safetensors", "strength": 0.8}])))
    assert out.output["model"].windowed_loras == ()


class TestWindowedStackRevision:
    """``_sync_lora_windows`` revises a cache-HIT DiT's weight identity when the
    step-windowed request changes. Windowed entries are never patched into the
    cached module, so the stamp is the only thing that notices."""

    def _dit(self):
        return SimpleNamespace(bumps=[], bump_weight_revision=lambda reason: None)

    def _model(self):
        model = SimpleNamespace()
        model.bumps = []
        model.bump_weight_revision = model.bumps.append
        return model

    def _entry(self, start, end, path="/m/turbo-sda.safetensors", weight=1.0):
        return {"file_path": path, "weight": weight, "window": LoraStepWindow(start, end)}

    def test_an_unchanged_windowed_request_does_not_revise(self):
        model = self._model()
        stamp = _window_fingerprint([self._entry(1, 2)])

        ModelLoaderKrea2Pipe._sync_lora_windows(model, stamp)
        ModelLoaderKrea2Pipe._sync_lora_windows(model, stamp)

        assert len(model.bumps) == 1

    def test_a_changed_window_revises_even_at_the_same_file_and_weight(self):
        model = self._model()

        ModelLoaderKrea2Pipe._sync_lora_windows(model, _window_fingerprint([self._entry(1, 2)]))
        ModelLoaderKrea2Pipe._sync_lora_windows(model, _window_fingerprint([self._entry(1, 5)]))

        assert len(model.bumps) == 2
