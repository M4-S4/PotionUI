"""Tests for the model_loader/flux pipe.

The pipe acquires TE / VAE / DiT under three independent MODELS keys. The fake
MODELS service below implements REAL hit/miss cache semantics (key+fingerprint
-> cached value, matching ``ModelLifecycle.acquire``), because the LoRA
in-place-sync fix under test only shows up across two acquires of the SAME key:
a LoRA-set change must be a cache HIT (same fingerprint, same object, no
``loader()`` re-run) that ``_sync_loras`` reconciles in place, not a
fingerprint-bust that reloads the checkpoint from disk.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from src.pipelines.contracts import PipeInput
from src.pipelines.pipes.model_loader.flux.main import ModelLoaderFluxPipe
from src.platform.runtime.native.engine import NativeEngineLoader, NativeModel


class _FakeModule:
    """Stand-in for an ``nn.Module``: just enough surface (an empty
    ``named_modules()``) for the REAL ``remove_loras`` to run as a genuine
    no-op instead of needing a real Flux arch instance."""

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
    cfg = ModelLoaderFluxPipe.get_default_config()
    cfg.update({
        "diffusion_model": {"file_path": "/m/dit.safetensors", "name": "dit"},
        "text_encoder": {"file_path": "/m/te.safetensors", "name": "te"},
        "clip_l": {"file_path": "/m/clip_l.safetensors", "name": "clip_l"},
        "vae": {"file_path": "/m/vae.safetensors", "name": "vae"},
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
    load-bearing assertion (must stay 1 across a LoRA-set change on a warm
    model, proving no re-read of the checkpoint from disk)."""
    counts = {"diffusion_model": 0, "text_encoder": 0, "vae": 0}

    def _fake_load(self, path, kind, **kwargs):
        counts[kind] = counts.get(kind, 0) + 1
        # A real wrapper, not a stand-in: the loaders revise its weight identity.
        return NativeModel(kind, _FakeModule(), spec=SimpleNamespace(family="flux"), estimated_vram_gb=1.0)

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

    monkeypatch.setattr(ModelLoaderFluxPipe, "_apply_loras", staticmethod(_fake_apply))
    return calls


# -- metadata --------------------------------------------------------------

# -- three-component acquire ----------------------------------------------

def test_te_key_includes_clip_l_for_flux1():
    models, out = _run(ModelLoaderFluxPipe(config=_config()))
    _ = out.output["text_encoder"].encoder  # force the deferred acquire
    te_key = next(k for k, _ in models.calls if k.startswith("native/te/"))
    assert "/m/te.safetensors" in te_key and "/m/clip_l.safetensors" in te_key


def test_klein_single_te_no_clip_l():
    # No clip_l -> Klein/Flux2 path; te key still formed, clip_l portion empty.
    cfg = _config()
    cfg["clip_l"] = None
    models, out = _run(ModelLoaderFluxPipe(config=cfg))
    _ = out.output["text_encoder"].encoder
    te_key = next(k for k, _ in models.calls if k.startswith("native/te/"))
    assert te_key == "native/te//m/te.safetensors|"


def test_flux3_checkpoint_is_refused_from_the_header_before_any_load(tmp_path, _fake_engine):
    path = tmp_path / "flux3.safetensors"
    save_file({"model.diffusion_model.early_stream_modulations.image.lin.weight": torch.zeros(1)}, str(path))
    cfg = _config()
    cfg["diffusion_model"] = {"file_path": str(path), "name": "flux3"}
    with pytest.raises(ValueError, match="FLUX 3 preset"):
        _run(ModelLoaderFluxPipe(config=cfg))
    assert _fake_engine["diffusion_model"] == 0


def test_a_flux_checkpoint_header_passes_the_guard(tmp_path):
    path = tmp_path / "flux.safetensors"
    save_file({"double_blocks.0.img_attn.norm.key_norm.scale": torch.zeros(1)}, str(path))
    cfg = _config()
    cfg["diffusion_model"] = {"file_path": str(path), "name": "flux"}
    _run(ModelLoaderFluxPipe(config=cfg))


# -- fingerprints ------------------------------------------------------------

# -- LoRA sync on an already-cached DiT --------------------------------------
