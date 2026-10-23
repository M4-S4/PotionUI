from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.features.generation.engine import validate_pipe_configuration
from src.pipelines.contracts import PipeInput
from src.pipelines.pipes.generator.krea2.main import GeneratorKrea2Pipe


@dataclass(frozen=True)
class _FakeSpec:
    family: str = "krea2"
    variant: str = "krea2_turbo"
    latent_format: dict = field(default_factory=lambda: {"latent_channels": 16})
    sampling_settings: dict = field(default_factory=lambda: {"guidance": "none"})


class _FakeGenerator:
    instances: list["_FakeGenerator"] = []

    def __init__(self, dit, te, vae, device_plan=None, **_):
        self.spec = _FakeSpec()
        self.sample_calls = []
        _FakeGenerator.instances.append(self)

    def snap_resolution(self, width, height):
        from src.platform.runtime.native.resolution import snap_resolution
        return snap_resolution(width, height, 8, 2)

    def latent_shape_for(self, width, height, batch=1):
        return (batch, 16, 1, height // 8, width // 8)

    def sample(self, conditioning, latents_shape, **kw):
        self.sample_calls.append({"conditioning": conditioning, "latents_shape": latents_shape, **kw})
        return torch.zeros(latents_shape)

    def decode(self, latent, **_):
        return np.zeros((1, 8, 8, 3), dtype=np.uint8)


PATCHES = (
    patch("src.pipelines.pipes._shared.generation.flow_generator_pipe.make_device_plan", lambda **_: None),
    patch("src.pipelines.pipes._shared.generation.flow_generator_pipe.NativeGenerator", _FakeGenerator),
)


@pytest.fixture(autouse=True)
def _fakes():
    _FakeGenerator.instances.clear()
    for p in PATCHES:
        p.start()
    yield
    for p in PATCHES:
        p.stop()


def _cond_model(negative=True):
    return SimpleNamespace(
        embeds={"context": torch.ones(1, 4, 8)},
        n_embeds={"context": torch.zeros(1, 3, 8), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
        if negative else {},
    )


def _run(cond_model, **config):
    cfg = GeneratorKrea2Pipe.get_default_config()
    cfg.update(config)
    pipe = GeneratorKrea2Pipe(config=cfg)
    bundle = SimpleNamespace(dit=SimpleNamespace(estimated_vram_gb=26.0), te_encoder=object(), vae=object())
    pipe.process(PipeInput(input={"model": bundle, "conditioning": [cond_model], "seed": [1]}), lambda o: None)
    return _FakeGenerator.instances[-1].sample_calls[0]["conditioning"]


def test_the_defaults_attach_nothing_and_hand_the_embeds_through_untouched():
    cond_model = _cond_model()

    conditioning = _run(cond_model)

    assert "text_fusion" not in conditioning.cond
    assert "text_fusion" not in conditioning.uncond


def test_an_active_enhancer_reaches_both_the_cond_and_the_uncond_dict():
    conditioning = _run(_cond_model(), text_enhancer_strength=1.25, text_scale=1.5)

    assert conditioning.cond["text_fusion"] == {"strength": 1.25, "scale": 1.5}
    assert conditioning.uncond["text_fusion"] == {"strength": 1.25, "scale": 1.5}


def test_the_text_scale_alone_is_enough_to_attach():
    conditioning = _run(_cond_model(), text_scale=1.75)

    assert conditioning.cond["text_fusion"] == {"strength": 0.0, "scale": 1.75}
    assert conditioning.uncond["text_fusion"] == {"strength": 0.0, "scale": 1.75}


def test_the_strength_alone_is_enough_to_attach():
    conditioning = _run(_cond_model(), text_enhancer_strength=0.5)

    assert conditioning.cond["text_fusion"] == {"strength": 0.5, "scale": 1.0}


def test_the_shared_embeds_are_never_mutated():
    cond_model = _cond_model()
    embeds, n_embeds = cond_model.embeds, cond_model.n_embeds

    conditioning = _run(cond_model, text_enhancer_strength=1.0)

    assert "text_fusion" not in embeds and "text_fusion" not in n_embeds
    assert conditioning.cond is not embeds
    assert conditioning.cond["context"] is embeds["context"]
    assert conditioning.uncond["context"] is n_embeds["context"]


def test_without_a_negative_encoding_only_the_cond_carries_it():
    conditioning = _run(_cond_model(negative=False), text_enhancer_strength=1.0)

    assert conditioning.cond["text_fusion"] == {"strength": 1.0, "scale": 1.0}
    assert conditioning.uncond is None


def test_it_composes_with_nag():
    conditioning = _run(_cond_model(), nag_scale=1.5, text_enhancer_strength=1.0, text_scale=2.0)

    assert conditioning.cond["nag"]["scale"] == 1.5
    assert torch.equal(conditioning.cond["nag_context"], torch.zeros(1, 3, 8))
    assert conditioning.cond["text_fusion"] == {"strength": 1.0, "scale": 2.0}
    assert "nag_context" not in conditioning.uncond


def test_the_two_settings_are_declared_with_the_expected_bounds():
    specs = {spec.name: spec for spec in GeneratorKrea2Pipe.configuration()}

    strength, scale = specs["text_enhancer_strength"], specs["text_scale"]
    assert (strength.default, strength.min_value, strength.max_value) == (0.0, 0.0, 2.0)
    assert (scale.default, scale.min_value, scale.max_value) == (1.0, 0.25, 4.0)
    assert not strength.required and not scale.required


@pytest.mark.parametrize("bad", [
    {"text_enhancer_strength": -0.1}, {"text_enhancer_strength": 2.5},
    {"text_scale": 0.1}, {"text_scale": 4.5},
])
def test_out_of_range_settings_are_rejected_by_the_pipe_validation(bad):
    config = {**GeneratorKrea2Pipe.get_default_config(), **bad}

    with pytest.raises(ValueError):
        validate_pipe_configuration(GeneratorKrea2Pipe, config)
