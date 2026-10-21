from __future__ import annotations

from types import SimpleNamespace

from src.pipelines.contracts import PipeInput
from src.pipelines.pipes.model_loader.qwen_image21.main import ModelLoaderQwenImage21Pipe


class _FakeModels:
    def __init__(self):
        self.calls = []

    def acquire(self, key, fingerprint, loader, estimated_vram_gb=None):
        self.calls.append((key, fingerprint))
        return SimpleNamespace(module=object(), spec=None, estimated_vram_gb=1.0)


def _config(loras=None, **over):
    cfg = ModelLoaderQwenImage21Pipe.get_default_config()
    cfg.update({
        "diffusion_model": {"file_path": "/m/dit.safetensors", "name": "dit"},
        "text_encoder": {"file_path": "/m/te.safetensors", "name": "te"},
        "vae": {"file_path": "/m/vae.safetensors", "name": "vae"},
        "loras": loras or [],
    })
    cfg.update(over)
    return cfg


def _run(pipe):
    models = _FakeModels()
    out = pipe.process(PipeInput(input={"MODELS": models}), lambda o: None)
    return models, out


def test_no_clip_l_in_config():
    names = {s.name for s in ModelLoaderQwenImage21Pipe.configuration()}
    assert "clip_l" not in names
    assert {"diffusion_model", "text_encoder", "vae", "loras", "vision"} <= names


def _fps(loras):
    models, out = _run(ModelLoaderQwenImage21Pipe(config=_config(loras=loras)))
    _ = out.output["text_encoder"].encoder
    return dict(zip([k for k, _ in models.calls], [f for _, f in models.calls]))


def test_runs_without_models_service():
    pipe = ModelLoaderQwenImage21Pipe(config=_config())
    assert pipe.describe_models()


# -- vision fingerprint hazard -----------------------------


def test_vision_defaults_off_and_absent_from_fingerprint_is_still_distinguishable():
    models, out = _run(ModelLoaderQwenImage21Pipe(config=_config()))
    _ = out.output["text_encoder"].encoder
    te_fp_default = dict(models.calls)["native/te//m/te.safetensors"]
    models2, out2 = _run(ModelLoaderQwenImage21Pipe(config=_config(vision=False)))
    _ = out2.output["text_encoder"].encoder
    te_fp_explicit_false = dict(models2.calls)["native/te//m/te.safetensors"]
    assert te_fp_default == te_fp_explicit_false


def test_vision_true_changes_only_the_te_fingerprint():
    """A text-only and a vision-enabled load of the SAME text-encoder path
    must NOT alias to the same model-lifecycle cache entry."""
    no_vision = _fps([])
    models, out = _run(ModelLoaderQwenImage21Pipe(config=_config(vision=True)))
    _ = out.output["text_encoder"].encoder
    with_vision = dict(models.calls)

    te_key = "native/te//m/te.safetensors"
    vae_key = "native/vae//m/vae.safetensors"
    dit_key = "native/dit//m/dit.safetensors"

    assert no_vision[te_key] != with_vision[te_key]
    assert "vision=True" in with_vision[te_key]
    assert "vision=False" in no_vision[te_key]
    assert no_vision[vae_key] == with_vision[vae_key]
    assert no_vision[dit_key] == with_vision[dit_key]


def test_vision_flag_threaded_to_the_engine_loader(monkeypatch):
    """``load_te``'s closure must actually pass ``vision=`` down to
    ``NativeEngineLoader.load``."""
    received = {}

    class _FakeLoader:
        def __init__(self, *a, **kw):
            pass

        def load(self, path, kind, **kwargs):
            if kind == "text_encoder":
                received.update(kwargs)
            return SimpleNamespace(module=object(), spec=None, estimated_vram_gb=1.0)

    import src.pipelines.pipes._shared.generation.three_component_loader as qi21_main
    monkeypatch.setattr(qi21_main, "NativeEngineLoader", _FakeLoader)

    pipe = ModelLoaderQwenImage21Pipe(config=_config(vision=True))
    pipe.process(PipeInput(input={}), lambda o: None)
    assert received == {"vision": True}


_CONTROL = {"file_path": "/m/fun_control.safetensors", "name": "fun_control"}


def test_a_control_model_gets_its_own_dit_cache_entry():
    plain = _fps([])
    models, out = _run(ModelLoaderQwenImage21Pipe(config=_config(control_model=_CONTROL)))
    _ = out.output["text_encoder"].encoder
    controlled = dict(models.calls)

    plain_key = "native/dit//m/dit.safetensors"
    controlled_key = "native/dit//m/dit.safetensors+control=/m/fun_control.safetensors"
    assert plain_key in plain and plain_key not in controlled
    assert controlled_key in controlled and controlled_key not in plain
    assert "control=/m/fun_control.safetensors" in controlled[controlled_key]
    assert "control=" not in plain[plain_key]
    for key in ("native/te//m/te.safetensors", "native/vae//m/vae.safetensors"):
        assert controlled[key] == plain[key]


def test_an_empty_control_model_leaves_the_dit_fingerprint_alone():
    plain = _fps([])
    models, out = _run(ModelLoaderQwenImage21Pipe(
        config=_config(control_model={"file_path": "", "name": ""})))
    _ = out.output["text_encoder"].encoder
    assert dict(models.calls)["native/dit//m/dit.safetensors"] == plain["native/dit//m/dit.safetensors"]


def test_the_control_model_is_loaded_with_the_dit(monkeypatch):
    received = []

    class _FakeLoader:
        def __init__(self, *a, **kw):
            pass

        def load(self, path, kind, **kwargs):
            received.append((path, kind, kwargs))
            return SimpleNamespace(module=object(), spec=None, estimated_vram_gb=1.0)

    import src.pipelines.pipes._shared.generation.three_component_loader as qi21_main
    monkeypatch.setattr(qi21_main, "NativeEngineLoader", _FakeLoader)

    pipe = ModelLoaderQwenImage21Pipe(config=_config(control_model=_CONTROL))
    pipe.process(PipeInput(input={}), lambda o: None)
    assert ("/m/dit.safetensors", "diffusion_model", {"model_patch": "/m/fun_control.safetensors"}) in received


def test_the_control_model_is_listed_with_the_models():
    pipe = ModelLoaderQwenImage21Pipe(config=_config(control_model=_CONTROL))
    assert ("fun_control", "qwen_image21_fun_control") in {(m.name, m.type) for m in pipe.describe_models()}
