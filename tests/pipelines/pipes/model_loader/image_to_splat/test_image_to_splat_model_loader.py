from __future__ import annotations

import pytest

from src.pipelines.contracts import PipeInput
from src.pipelines.outputs import ModelsGenerationOutput
from src.pipelines.pipes._shared.generation import geometry_estimator
from src.pipelines.pipes.model_loader.image_to_splat import main as loader_main
from src.pipelines.pipes.model_loader.image_to_splat.bundle import ImageToSplatModelBundle
from src.pipelines.pipes.model_loader.image_to_splat.main import ModelLoaderImageToSplatPipe
from src.pipelines.pipes.model_loader.pixal3d import main as pixal3d_main
from src.pipelines.pipes.model_loader.pixal3d.main import ModelLoaderPixal3DPipe
from src.pipelines.pipes.model_loader.trellis2 import main as trellis2_main

MOGE = "/m/moge_2_vitl_normal_fp16.safetensors"


class _FakeModels:
    def __init__(self):
        self.calls = []
        self.loads = 0
        self._entries = {}

    def acquire(self, key, fingerprint, loader, estimated_vram_gb=None):
        self.calls.append((key, fingerprint, estimated_vram_gb))
        entry = self._entries.get(key)
        if entry is not None and entry[0] == fingerprint:
            return entry[1]
        self.loads += 1
        value = loader()
        self._entries[key] = (fingerprint, value)
        return value


class _FakeModule:
    def to(self, device):
        return self


class _FakeNative:
    def __init__(self, module):
        self.module = module
        self.unloaded = False

    def unload(self):
        self.unloaded = True


@pytest.fixture
def fake_moge(monkeypatch):
    loaded = []

    def _load(path):
        loaded.append(path)
        return _FakeModule()

    monkeypatch.setattr(geometry_estimator.moge_load, "load_moge2", _load)
    monkeypatch.setattr(geometry_estimator, "file_size_gb", lambda path: 0.66)
    return loaded


def _config(**over):
    config = ModelLoaderImageToSplatPipe.get_default_config()
    config.update({"geometry_estimator": {"file_path": MOGE, "name": "moge"}})
    config.update(over)
    return config


def _process(models=None, outputs=None, **over):
    pipe = ModelLoaderImageToSplatPipe(_config(**over))
    models = models if models is not None else _FakeModels()
    sink = outputs if outputs is not None else []
    out = pipe.process(PipeInput(input={"MODELS": models}), sink.append)
    return models, out.output["model"]


def test_moge_is_acquired_under_the_shared_key_at_its_fp32_size(fake_moge):
    models, bundle = _process()
    assert models.calls == [(f"native/moge/{MOGE}", f"native/moge/{MOGE}|float32", pytest.approx(1.32))]
    assert fake_moge == [MOGE]
    assert isinstance(bundle, ImageToSplatModelBundle)
    assert isinstance(bundle.components(), _FakeModule)
    assert bundle.device == "cuda"


def test_a_second_run_reuses_the_cached_estimator(fake_moge):
    models = _FakeModels()
    _process(models=models)
    _process(models=models)
    assert models.loads == 1


def test_the_cache_entry_is_shared_with_pixal3ds_camera_estimator(fake_moge, monkeypatch):
    for name in ("load_dino_conditioner", "load_ss_vae_decoder", "load_shape_slat_decoder", "load_tex_slat_decoder"):
        monkeypatch.setattr(trellis2_main.trellis2_load, name, lambda *args, **kwargs: _FakeModule())
    for name in ("load_pixal3d_naf", "load_pixal3d_ss_flow", "load_pixal3d_shape_flow", "load_pixal3d_tex_flow"):
        monkeypatch.setattr(pixal3d_main.pixal3d_load, name, lambda *args, **kwargs: _FakeModule())
    monkeypatch.setattr(pixal3d_main, "_load_camera_estimator", lambda path: _FakeModule())
    monkeypatch.setattr(trellis2_main, "prefix_size_gb", lambda path, prefix: 1.0)
    monkeypatch.setattr(pixal3d_main, "prefix_size_gb", lambda path, prefix: 1.0)
    monkeypatch.setattr(trellis2_main, "file_size_gb", lambda path: 1.0)
    monkeypatch.setattr(pixal3d_main, "file_size_gb", lambda path: 0.66)
    models = _FakeModels()
    pixal3d = ModelLoaderPixal3DPipe({
        **ModelLoaderPixal3DPipe.get_default_config(),
        "diffusion_model": {"file_path": "/m/pixal3d_bf16.safetensors"},
        "shape_vae": {"file_path": "/m/shape.safetensors"},
        "texture_vae": {"file_path": "/m/texture.safetensors"},
        "image_encoder": {"file_path": "/m/dino.safetensors"},
        "camera_estimator": {"file_path": MOGE},
        "dtype": "bfloat16",
    })
    pixal3d.process(PipeInput(input={"MODELS": models}), lambda output: None)
    loads_after_pixal3d = models.loads

    _process(models=models)

    [pixal3d_call] = [call for call in models.calls if call[0] == f"native/moge/{MOGE}"][:1]
    assert pixal3d_call[1] == f"native/moge/{MOGE}|float32"
    assert models.loads == loads_after_pixal3d


def test_the_loader_announces_the_model(fake_moge):
    outputs = []
    _process(outputs=outputs)
    [described] = [output for output in outputs if isinstance(output, ModelsGenerationOutput)]
    assert [(model.name, model.type) for model in described.models] == [("moge", "geometry_estimation")]


@pytest.mark.parametrize("value", [None, {"file_path": "", "name": ""}])
def test_a_missing_estimator_is_named(fake_moge, value):
    with pytest.raises(ValueError, match="moge_2_vitl_normal_fp16.safetensors"):
        _process(geometry_estimator=value)
    assert fake_moge == []


def test_a_checkpoint_that_is_not_moge_is_named(monkeypatch):
    def _refuse(path):
        raise ValueError("other.safetensors is not a MoGe-2 checkpoint")

    monkeypatch.setattr(geometry_estimator.moge_load, "load_moge2", _refuse)
    with pytest.raises(ValueError, match="geometry estimator could not be loaded: other.safetensors"):
        geometry_estimator.load_geometry_estimator("/m/other.safetensors")


def test_an_evicted_estimator_is_named_and_unload_is_forwarded():
    native = _FakeNative(_FakeModule())
    bundle = ImageToSplatModelBundle(estimator=native, device="cpu")
    bundle.unload()
    assert native.unloaded
    bundle.estimator = None
    with pytest.raises(ValueError, match="MoGe-2 was evicted"):
        bundle.components()


def test_without_a_lifecycle_service_the_estimator_loads_directly(fake_moge):
    pipe = ModelLoaderImageToSplatPipe(_config(device="cpu"))
    out = pipe.process(PipeInput(input={}), lambda output: None)
    assert out.output["model"].device == "cpu"
    assert fake_moge == [MOGE]


def test_the_loader_module_uses_the_shared_helpers():
    assert loader_main.geometry_estimator_key is geometry_estimator.geometry_estimator_key
    assert loader_main.geometry_estimator_fingerprint is geometry_estimator.geometry_estimator_fingerprint
