from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from src.pipelines.contracts import IOType, PipeInput
from src.pipelines.pipes.model_loader.moss_image.bundle import MossImageModelBundle
from src.pipelines.pipes.model_loader.moss_image.main import ModelLoaderMossImagePipe
from src.pipelines.pipes.model_loader.moss_image.moss_image_clip import MossImageClipTextEncoder
from src.pipelines.pipes._shared.generation.loader_helpers import lora_stack_fingerprint
from src.platform.runtime.native.engine import NativeModel
from src.platform.runtime.native.text_encoders import prompt_embed_key


class _FakeModule:
    def named_modules(self):
        return iter(())


class _FakeModels:
    def __init__(self):
        self.calls = []
        self.loaders = {}
        self.entries = {}

    def acquire(self, key, fingerprint, loader, estimated_vram_gb=None):
        self.calls.append((key, fingerprint))
        self.loaders[key] = loader
        entry = self.entries.get(key)
        if entry is not None and entry[0] == fingerprint:
            return entry[1]
        value = NativeModel("fake", _FakeModule(), spec=None, estimated_vram_gb=1.0)
        self.entries[key] = (fingerprint, value)
        return value


@pytest.fixture(autouse=True)
def _apply_spy(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ModelLoaderMossImagePipe, "_apply_loras", staticmethod(lambda dit_model, loras: calls.append(list(loras))),
    )
    return calls


def _config(**over):
    cfg = ModelLoaderMossImagePipe.get_default_config()
    cfg.update({
        "diffusion_model": {"file_path": "/m/dit.safetensors", "name": "dit"},
        "text_encoder": {"file_path": "/m/te.safetensors", "name": "te"},
        "vae": {"file_path": "/m/vae.safetensors", "name": "vae"},
    })
    cfg.update(over)
    return cfg


def _run(pipe, models=None):
    models = models if models is not None else _FakeModels()
    out = pipe.process(PipeInput(input={"MODELS": models}), lambda o: None)
    return models, out


def test_name_and_outputs():
    assert ModelLoaderMossImagePipe.name == "model_loader"
    out_names = {o.name: o.io_type for o in ModelLoaderMossImagePipe.outputs()}
    assert out_names["model"] == IOType.MODEL
    assert out_names["text_encoder"] == IOType.TEXT_ENCODER


def test_config_takes_loras_but_no_vision_knobs():
    names = {s.name for s in ModelLoaderMossImagePipe.configuration()}
    assert {"diffusion_model", "text_encoder", "vae", "loras", "dtype", "device"} <= names
    assert not {"vision", "control_model"} & names


def test_step_windowed_lora_is_refused():
    pipe = ModelLoaderMossImagePipe(config=_config(loras=[
        {"file_path": "/m/style.safetensors", "weight": 0.8, "step_start": 1, "step_end": 4},
    ]))
    with pytest.raises(ValueError, match="step window"):
        _run(pipe)


def test_lora_is_applied_and_stamped_when_the_dit_loads(_apply_spy):
    loras = [{"file_path": "/m/style.safetensors", "weight": 0.8}]
    loaded = SimpleNamespace(module=_FakeModule())

    class _Loader:
        def __init__(self, **_):
            pass

        def load(self, path, kind, **kw):
            return loaded

    with patch("src.pipelines.pipes.model_loader.moss_image.main.NativeEngineLoader", _Loader):
        models, _out = _run(ModelLoaderMossImagePipe(config=_config(loras=loras)))
        applied_before = len(_apply_spy)
        models.loaders["native/dit//m/dit.safetensors"]()
    expanded = [{"file_path": "/m/style.safetensors", "weight": 0.8, "window": None}]
    assert _apply_spy[applied_before:] == [expanded]
    assert loaded._active_lora_fp == lora_stack_fingerprint(expanded)


def test_lora_change_on_a_warm_dit_does_not_bust_the_dit_fingerprint(_apply_spy):
    models = _FakeModels()
    _run(ModelLoaderMossImagePipe(config=_config()), models)
    _run(ModelLoaderMossImagePipe(config=_config(loras=[{"model": "/m/style.safetensors", "strength": 0.8}])), models)
    dit_fps = {fp for key, fp in models.calls if key == "native/dit//m/dit.safetensors"}
    assert len(dit_fps) == 1
    assert [c for c in _apply_spy if c] == [[{"file_path": "/m/style.safetensors", "weight": 0.8, "window": None}]]


def test_zero_weight_lora_is_ignored(_apply_spy):
    _run(ModelLoaderMossImagePipe(config=_config(loras=[{"model": "/m/off.safetensors", "strength": 0.0}])))
    assert [c for c in _apply_spy if c] == []


def test_lora_models_are_reported_for_history():
    pipe = ModelLoaderMossImagePipe(config=_config(loras=[{"model": "/m/style.safetensors", "strength": 0.8}]))
    lora = [m for m in pipe.describe_models() if m.type == "lora"]
    assert [(m.name, m.weight) for m in lora] == [("style", 0.8)]


def test_text_encoder_is_acquired_lazily_under_the_moss_key():
    models, out = _run(ModelLoaderMossImagePipe(config=_config()))
    assert [k for k, _ in models.calls] == ["native/vae//m/vae.safetensors", "native/dit//m/dit.safetensors"]
    _ = out.output["text_encoder"].encoder
    assert models.calls[-1][0] == "native/te//m/te.safetensors|moss_image"


def test_bundle_remembers_the_text_encoder_key():
    _models, out = _run(ModelLoaderMossImagePipe(config=_config()))
    bundle = out.output["model"]
    assert isinstance(bundle, MossImageModelBundle)
    assert bundle.te_cache_key == "native/te//m/te.safetensors|moss_image"


def test_each_component_loads_from_its_own_path():
    calls = []

    class _Loader:
        def __init__(self, **_):
            pass

        def load(self, path, kind, **kw):
            calls.append((path, kind, kw))
            return SimpleNamespace(module=object())

    with patch("src.pipelines.pipes.model_loader.moss_image.main.NativeEngineLoader", _Loader):
        models, out = _run(ModelLoaderMossImagePipe(config=_config()))
        _ = out.output["text_encoder"].encoder
        for loader in models.loaders.values():
            loader()
    assert ("/m/te.safetensors", "text_encoder", {}) in calls
    assert ("/m/dit.safetensors", "diffusion_model", {}) in calls
    assert ("/m/vae.safetensors", "vae", {}) in calls


def test_output_is_a_moss_clip():
    _models, out = _run(ModelLoaderMossImagePipe(config=_config()))
    assert isinstance(out.output["text_encoder"], MossImageClipTextEncoder)


class _RecordingEncoder:
    def __init__(self):
        self.texts = []

    def encode(self, texts):
        self.texts.append(list(texts))
        return {"context": torch.zeros(1, 2, 4), "attention_mask": None}


def test_clip_encodes_one_prompt_per_pass_without_weighting():
    encoder = _RecordingEncoder()
    clip = MossImageClipTextEncoder(encoder, device="cpu", model_fingerprint="fp-one-prompt")
    result = clip.encode_prompts([{
        "prompt": "a (red:1.3) fox", "negative_prompt": "blurry", "do_classifier_free_guidance": True,
    }])[0]
    assert encoder.texts == [["a (red:1.3) fox"], ["blurry"]]
    assert result.embeds["attention_mask"] is None
    assert result.n_embeds["context"].shape == (1, 2, 4)


def test_clip_skips_the_negative_pass_without_cfg():
    encoder = _RecordingEncoder()
    clip = MossImageClipTextEncoder(encoder, device="cpu", model_fingerprint="fp-no-cfg")
    result = clip.encode_prompts([{
        "prompt": "a fox", "negative_prompt": "blurry", "do_classifier_free_guidance": False,
    }])[0]
    assert encoder.texts == [["a fox"]]
    assert result.n_embeds == {}


def test_clip_cache_key_is_namespaced_per_family():
    clip = MossImageClipTextEncoder(_RecordingEncoder(), device="cpu", model_fingerprint="fp-shared")
    _fn, key = clip._encode_fn_and_key({"prompt": "p", "negative_prompt": "n"})
    assert key == prompt_embed_key("fp-shared", "moss_image_te", "p", "n", True)
    assert key != prompt_embed_key("fp-shared", "flux3_te", "p", "n", True)
