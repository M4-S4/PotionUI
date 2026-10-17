from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.pipelines.contracts import IOType, PipeInput
from src.pipelines.pipes.model_loader.flux3.bundle import Flux3ModelBundle
from src.pipelines.pipes.model_loader.flux3.flux3_clip import Flux3ClipTextEncoder
from src.pipelines.pipes.model_loader.flux3.main import ModelLoaderFlux3Pipe
from src.platform.runtime.native.text_encoders import prompt_embed_key


class _FakeModels:
    def __init__(self):
        self.calls = []
        self.loaders = {}

    def acquire(self, key, fingerprint, loader, estimated_vram_gb=None):
        self.calls.append((key, fingerprint))
        self.loaders[key] = loader
        return SimpleNamespace(module=object(), spec=None, estimated_vram_gb=1.0)


def _config(**over):
    cfg = ModelLoaderFlux3Pipe.get_default_config()
    cfg.update({
        "diffusion_model": {"file_path": "/m/dit.safetensors", "name": "dit"},
        "text_encoder": {"file_path": "/m/te.safetensors", "name": "te"},
        "vae": {"file_path": "/m/vae.safetensors", "name": "vae"},
    })
    cfg.update(over)
    return cfg


def _run(pipe):
    models = _FakeModels()
    out = pipe.process(PipeInput(input={"MODELS": models}), lambda o: None)
    return models, out


def test_name_and_outputs():
    assert ModelLoaderFlux3Pipe.name == "model_loader"
    out_names = {o.name: o.io_type for o in ModelLoaderFlux3Pipe.outputs()}
    assert out_names["model"] == IOType.MODEL
    assert out_names["text_encoder"] == IOType.TEXT_ENCODER


def test_config_has_no_vision_or_control_knobs():
    names = {s.name for s in ModelLoaderFlux3Pipe.configuration()}
    assert {"diffusion_model", "text_encoder", "vae", "loras", "dtype", "device"} <= names
    assert not {"vision", "control_model", "clip_l"} & names


def test_text_encoder_is_acquired_lazily_under_the_flux3_key():
    models, out = _run(ModelLoaderFlux3Pipe(config=_config()))
    assert [k for k, _ in models.calls] == ["native/vae//m/vae.safetensors", "native/dit//m/dit.safetensors"]
    _ = out.output["text_encoder"].encoder
    assert models.calls[-1][0] == "native/te//m/te.safetensors|flux3"


def test_bundle_remembers_the_text_encoder_key():
    _models, out = _run(ModelLoaderFlux3Pipe(config=_config()))
    bundle = out.output["model"]
    assert isinstance(bundle, Flux3ModelBundle)
    assert bundle.te_cache_key == "native/te//m/te.safetensors|flux3"


def test_text_encoder_loads_with_the_flux3_variant():
    calls = []

    class _Loader:
        def __init__(self, **_):
            pass

        def load(self, path, kind, **kwargs):
            calls.append((path, kind, kwargs))
            return SimpleNamespace(module=object(), spec=None, estimated_vram_gb=1.0)

    with patch("src.pipelines.pipes.model_loader.flux3.main.NativeEngineLoader", _Loader):
        models, out = _run(ModelLoaderFlux3Pipe(config=_config()))
        _ = out.output["text_encoder"].encoder
        models.loaders["native/te//m/te.safetensors|flux3"]()
    assert ("/m/te.safetensors", "text_encoder", {"te_variant": "flux3"}) in calls


def test_output_is_a_flux3_clip():
    _models, out = _run(ModelLoaderFlux3Pipe(config=_config()))
    assert isinstance(out.output["text_encoder"], Flux3ClipTextEncoder)


class _RecordingEncoder:
    def __init__(self):
        self.texts = []

    def encode(self, texts):
        self.texts.append(list(texts))
        return {"context": torch.zeros(1, 2, 4), "attention_mask": None}


def test_clip_encodes_one_prompt_per_pass_without_weighting():
    encoder = _RecordingEncoder()
    clip = Flux3ClipTextEncoder(encoder, device="cpu", model_fingerprint="fp-one-prompt")
    result = clip.encode_prompts([{
        "prompt": "a (red:1.3) fox", "negative_prompt": "blurry", "do_classifier_free_guidance": True,
    }])[0]
    assert encoder.texts == [["a (red:1.3) fox"], ["blurry"]]
    assert result.embeds["attention_mask"] is None
    assert result.n_embeds["context"].shape == (1, 2, 4)


def test_clip_skips_the_negative_pass_without_cfg():
    encoder = _RecordingEncoder()
    clip = Flux3ClipTextEncoder(encoder, device="cpu", model_fingerprint="fp-no-cfg")
    result = clip.encode_prompts([{
        "prompt": "a fox", "negative_prompt": "blurry", "do_classifier_free_guidance": False,
    }])[0]
    assert encoder.texts == [["a fox"]]
    assert result.n_embeds == {}


def test_clip_cache_key_is_namespaced_per_family():
    clip = Flux3ClipTextEncoder(_RecordingEncoder(), device="cpu", model_fingerprint="fp-shared")
    _fn, key = clip._encode_fn_and_key({"prompt": "p", "negative_prompt": "n"})
    assert key == prompt_embed_key("fp-shared", "flux3_te", "p", "n", True)
    assert key != prompt_embed_key("fp-shared", "krea2_te", "p", "n", True, 768, "")
