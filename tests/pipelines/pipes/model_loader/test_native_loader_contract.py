from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pytest

from src.pipelines.contracts import IOType, PipeInput
from src.pipelines.pipes._shared.generation.bundle import DitTeVaeBundle
from src.pipelines.pipes.model_loader.anima.anima_clip import AnimaClipTextEncoder
from src.pipelines.pipes.model_loader.anima.main import ModelLoaderAnimaPipe
from src.pipelines.pipes.model_loader.flux.flux_clip import FluxClipTextEncoder
from src.pipelines.pipes.model_loader.flux.main import ModelLoaderFluxPipe
from src.pipelines.pipes.model_loader.flux3.flux3_clip import Flux3ClipTextEncoder
from src.pipelines.pipes.model_loader.flux3.main import ModelLoaderFlux3Pipe
from src.pipelines.pipes.model_loader.krea2.krea2_clip import Krea2ClipTextEncoder
from src.pipelines.pipes.model_loader.krea2.main import ModelLoaderKrea2Pipe
from src.pipelines.pipes.model_loader.qwen.main import ModelLoaderQwenPipe
from src.pipelines.pipes.model_loader.qwen.qwen_clip import QwenClipTextEncoder
from src.pipelines.pipes.model_loader.qwen_image21.main import ModelLoaderQwenImage21Pipe
from src.pipelines.pipes.model_loader.qwen_image21.qwen_image21_clip import QwenImage21ClipTextEncoder
from src.pipelines.pipes.model_loader.wan22.main import ModelLoaderWan22Pipe
from src.pipelines.pipes.model_loader.wan22.wan_clip import WanClipTextEncoder
from src.pipelines.pipes.model_loader.z_image.main import ModelLoaderZImagePipe
from src.pipelines.pipes.model_loader.z_image.z_image_clip import ZImageClipTextEncoder
from src.platform.runtime.native.engine import NativeEngineLoader, NativeModel
from tests.pipelines.pipes.model_loader.conftest import CachingFakeModels, FakeModels, FakeModule

DIT = "native/dit//m/dit.safetensors"
VAE = "native/vae//m/vae.safetensors"
TE = "native/te//m/te.safetensors"


def _standard_config(pipe_cls, **extra):
    def build(loras=None, **over):
        cfg = pipe_cls.get_default_config()
        cfg.update({
            "diffusion_model": {"file_path": "/m/dit.safetensors", "name": "dit"},
            "text_encoder": {"file_path": "/m/te.safetensors", "name": "te"},
            "vae": {"file_path": "/m/vae.safetensors", "name": "vae"},
            "loras": loras or [],
            **extra,
        })
        cfg.update(over)
        return cfg

    return build


def _wan_config(loras=None, **over):
    cfg = ModelLoaderWan22Pipe.get_default_config()
    cfg.update({
        "high_noise_model": {"file_path": "/m/wan_high.safetensors", "name": "high"},
        "low_noise_model": {"file_path": "/m/wan_low.safetensors", "name": "low"},
        "text_encoder": {"file_path": "/m/te.safetensors", "name": "umt5"},
        "vae": {"file_path": "/m/vae.safetensors", "name": "vae"},
    })
    cfg.update(over)
    return cfg


@dataclass(frozen=True)
class LoaderFamily:
    id: str
    pipe_cls: type
    clip_cls: type
    config: Callable[..., dict[str, Any]]
    te_key: str
    eager_keys: tuple[str, ...] = (VAE, DIT)
    bundle_cls: type = DitTeVaeBundle
    lora_in_dit_fingerprint: bool | None = None
    spec_family: str | None = None
    single_te: bool = True


FAMILIES = [
    LoaderFamily(
        "anima", ModelLoaderAnimaPipe, AnimaClipTextEncoder, _standard_config(ModelLoaderAnimaPipe),
        TE, lora_in_dit_fingerprint=True,
    ),
    LoaderFamily(
        "flux", ModelLoaderFluxPipe, FluxClipTextEncoder,
        _standard_config(ModelLoaderFluxPipe, clip_l={"file_path": "/m/clip_l.safetensors", "name": "clip_l"}),
        "native/te//m/te.safetensors|/m/clip_l.safetensors", lora_in_dit_fingerprint=False, spec_family="flux", single_te=False,
    ),
    LoaderFamily(
        "flux3", ModelLoaderFlux3Pipe, Flux3ClipTextEncoder, _standard_config(ModelLoaderFlux3Pipe),
        f"{TE}|flux3",
    ),
    LoaderFamily(
        "krea2", ModelLoaderKrea2Pipe, Krea2ClipTextEncoder, _standard_config(ModelLoaderKrea2Pipe),
        TE, lora_in_dit_fingerprint=False,
    ),
    LoaderFamily(
        "qwen", ModelLoaderQwenPipe, QwenClipTextEncoder, _standard_config(ModelLoaderQwenPipe),
        TE, lora_in_dit_fingerprint=True,
    ),
    LoaderFamily(
        "qwen_image21", ModelLoaderQwenImage21Pipe, QwenImage21ClipTextEncoder,
        _standard_config(ModelLoaderQwenImage21Pipe), TE, lora_in_dit_fingerprint=True,
    ),
    LoaderFamily(
        "z_image", ModelLoaderZImagePipe, ZImageClipTextEncoder, _standard_config(ModelLoaderZImagePipe),
        f"{TE}|zimage", lora_in_dit_fingerprint=True,
    ),
]

WAN = LoaderFamily(
    "wan22", ModelLoaderWan22Pipe, WanClipTextEncoder, _wan_config, TE,
    eager_keys=("native/dit//m/wan_high.safetensors", "native/dit//m/wan_low.safetensors", VAE),
    bundle_cls=None,
)

ALL = [*FAMILIES, WAN]
LORA_BUSTS_DIT = [f for f in FAMILIES if f.lora_in_dit_fingerprint is True]
SINGLE_TE = [f for f in FAMILIES if f.single_te]
LORA_KEEPS_DIT = [f for f in FAMILIES if f.lora_in_dit_fingerprint is False]

all_families = pytest.mark.parametrize("family", ALL, ids=lambda f: f.id)
standard_families = pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.id)
lora_busts_dit = pytest.mark.parametrize("family", LORA_BUSTS_DIT, ids=lambda f: f.id)
single_te_families = pytest.mark.parametrize("family", SINGLE_TE, ids=lambda f: f.id)
lora_keeps_dit = pytest.mark.parametrize("family", LORA_KEEPS_DIT, ids=lambda f: f.id)


def _run(family, cfg=None, models=None):
    models = models if models is not None else FakeModels()
    pipe = family.pipe_cls(config=cfg if cfg is not None else family.config())
    return models, pipe.process(PipeInput(input={"MODELS": models}), lambda o: None)


def _fingerprints(family, loras):
    models, out = _run(family, family.config(loras=loras))
    _ = out.output["text_encoder"].encoder
    return dict(models.calls)


@all_families
def test_name_and_outputs(family):
    assert family.pipe_cls.name == "model_loader"
    outs = {o.name: o.io_type for o in family.pipe_cls.outputs()}
    assert outs["model"] == IOType.MODEL
    assert outs["text_encoder"] == IOType.TEXT_ENCODER


@all_families
def test_process_eagerly_acquires_everything_but_the_text_encoder(family):
    models, _ = _run(family)
    assert tuple(k for k, _ in models.calls) == family.eager_keys


@all_families
def test_te_acquire_deferred_until_encoder_is_read(family):
    models, out = _run(family)
    clip = out.output["text_encoder"]
    _ = clip.encoder
    _ = clip.encoder
    assert [k for k, _ in models.calls if k == family.te_key] == [family.te_key]


@all_families
def test_te_never_acquired_when_encoder_is_never_read(family):
    models, out = _run(family)
    assert not any(k == family.te_key for k, _ in models.calls)
    _ = out.output["text_encoder"].encoder
    assert any(k == family.te_key for k, _ in models.calls)


@all_families
def test_outputs_are_bundle_and_clip(family):
    _, out = _run(family)
    if family.bundle_cls is not None:
        assert isinstance(out.output["model"], family.bundle_cls)
    assert isinstance(out.output["text_encoder"], family.clip_cls)


@all_families
def test_bundle_unload_tolerates_a_never_acquired_te(family):
    _, out = _run(family)
    out.output["model"].unload()


@all_families
def test_bundle_carries_the_te_cache_key(family):
    models, out = _run(family)
    _ = out.output["text_encoder"].encoder
    assert out.output["model"].te_cache_key == family.te_key
    assert any(k == family.te_key for k, _ in models.calls)


@all_families
def test_dtype_in_all_fingerprints(family):
    models, out = _run(family, family.config(dtype="float16"))
    _ = out.output["text_encoder"].encoder
    assert models.calls
    assert all("float16" in fp for _, fp in models.calls)


@standard_families
def test_missing_component_raises(family):
    cfg = family.config()
    cfg["vae"] = None
    with pytest.raises(ValueError, match="requires"):
        _run(family, cfg)


@lora_busts_dit
def test_lora_change_busts_only_dit(family):
    no_lora = _fingerprints(family, [])
    with_lora = _fingerprints(family, [{"model": "/m/style.safetensors", "strength": 0.8}])
    assert no_lora[family.te_key] == with_lora[family.te_key]
    assert no_lora[VAE] == with_lora[VAE]
    assert no_lora[DIT] != with_lora[DIT]
    assert "style.safetensors@0.8" in with_lora[DIT]


@lora_busts_dit
def test_zero_weight_lora_ignored(family):
    fps = _fingerprints(family, [{"model": "/m/off.safetensors", "strength": 0.0}])
    assert "off.safetensors" not in fps[DIT]


@pytest.fixture
def engine_loads(monkeypatch):
    counts = {"diffusion_model": 0, "text_encoder": 0, "vae": 0}

    def fake_load(self, path, kind, **kwargs):
        counts[kind] = counts.get(kind, 0) + 1
        return NativeModel(kind, FakeModule(), spec=None, estimated_vram_gb=1.0)

    monkeypatch.setattr(NativeEngineLoader, "load", fake_load)
    return counts


@pytest.fixture
def applied_loras(monkeypatch, request):
    calls = []
    monkeypatch.setattr(
        request.getfixturevalue("family").pipe_cls, "_apply_loras",
        staticmethod(lambda dit_model, loras: calls.append(list(loras))),
    )
    return calls


@lora_keeps_dit
def test_lora_change_does_not_bust_dit_fingerprint(family, applied_loras):
    no_lora = _fingerprints(family, [])
    with_lora = _fingerprints(family, [{"model": "/m/style.safetensors", "strength": 0.8}])
    assert no_lora == with_lora
    assert "style.safetensors" not in with_lora[DIT]


STYLE = [{"model": "/m/style.safetensors", "strength": 0.8}]
STYLE_APPLIED = [{"file_path": "/m/style.safetensors", "weight": 0.8, "window": None}]


@lora_keeps_dit
def test_zero_weight_lora_never_reaches_apply(family, engine_loads, applied_loras):
    _run(family, family.config(loras=[{"model": "/m/off.safetensors", "strength": 0.0}]), CachingFakeModels())
    assert applied_loras == [[]]


@lora_keeps_dit
def test_lora_change_on_warm_dit_reuses_cached_weights(family, engine_loads, applied_loras):
    models = CachingFakeModels()
    _, out1 = _run(family, family.config(loras=[]), models)
    _, out2 = _run(family, family.config(loras=STYLE), models)
    assert out1.output["model"].dit is out2.output["model"].dit
    assert engine_loads["diffusion_model"] == 1
    assert [c for c in applied_loras if c] == [STYLE_APPLIED]


@lora_keeps_dit
def test_repeat_generation_with_same_loras_is_a_pure_noop(family, engine_loads, applied_loras):
    models = CachingFakeModels()
    _run(family, family.config(loras=STYLE), models)
    after_first = len(applied_loras)
    _run(family, family.config(loras=STYLE), models)
    assert len(applied_loras) == after_first
    assert engine_loads["diffusion_model"] == 1


@lora_keeps_dit
def test_removing_all_loras_on_warm_dit_unpatches_without_reload(family, engine_loads, applied_loras, monkeypatch):
    removed = []
    monkeypatch.setattr(
        "src.pipelines.pipes._shared.generation.loader_lifecycle.remove_loras",
        lambda module: removed.append(module),
    )
    models = CachingFakeModels()
    _run(family, family.config(loras=STYLE), models)
    assert removed == []
    _run(family, family.config(loras=[]), models)
    assert len(removed) == 1
    assert [c for c in applied_loras if c] == [STYLE_APPLIED]
    assert engine_loads["diffusion_model"] == 1


@lora_keeps_dit
def test_different_dit_path_still_forces_a_real_reload(family, engine_loads):
    models = CachingFakeModels()
    _run(family, family.config(diffusion_model={"file_path": "/m/dit.safetensors"}), models)
    _run(family, family.config(diffusion_model={"file_path": "/m/dit_v2.safetensors"}), models)
    assert engine_loads["diffusion_model"] == 2


@single_te_families
def test_single_text_encoder_has_no_clip_l_field(family):
    names = {s.name for s in family.pipe_cls.configuration()}
    assert "clip_l" not in names
    assert {"diffusion_model", "text_encoder", "vae", "loras"} <= names
