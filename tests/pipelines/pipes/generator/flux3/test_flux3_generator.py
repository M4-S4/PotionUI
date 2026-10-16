from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.pipelines.contracts import IOType, PipeInput
from src.pipelines.outputs import GenerationExecutionError
from src.pipelines.pipes.generator.flux3.main import GeneratorFlux3Pipe

BASE = {"prediction": "const", "shift": 6.93, "guidance": "cfg", "default_steps": 50}
DISTILLED = {"prediction": "const", "shift": 6.93, "guidance": "flow_map", "default_steps": 16}
FLOW = "src.pipelines.pipes._shared.generation.flow_generator_pipe"


@dataclass(frozen=True)
class _FakeSpec:
    family: str = "flux3"
    variant: str = "flux3_klein_base"
    latent_format: dict = field(default_factory=lambda: {"latent_channels": 32})
    sampling_settings: dict = field(default_factory=lambda: dict(BASE))


class _FakeGenerator:
    instances: list["_FakeGenerator"] = []
    sampling_settings: dict = BASE

    def __init__(self, dit, te, vae, device_plan=None, **_):
        self.dit = dit
        self.spec = _FakeSpec(sampling_settings=dict(_FakeGenerator.sampling_settings))
        self.sample_calls = []
        self.encoded = []
        self.encoded_means = []
        self._shape = (1, 128, 64, 64)
        _FakeGenerator.instances.append(self)

    def snap_resolution(self, width, height):
        return (max(16, round(width / 16) * 16), max(16, round(height / 16) * 16))

    def latent_shape_for(self, width, height, batch=1):
        return (batch, 128, height // 16, width // 16)

    def sample(self, conditioning, latents_shape, **kw):
        self.sample_calls.append({"conditioning": conditioning, "latents_shape": latents_shape, **kw})
        self._shape = latents_shape
        return torch.zeros(latents_shape)

    def decode(self, latent, **_):
        return np.zeros((1, self._shape[2] * 16, self._shape[3] * 16, 3), dtype=np.uint8)

    def encode_image(self, pixels, **_):
        self.encoded.append(pixels.shape)
        self.encoded_means.append(float(pixels.mean()))
        return torch.zeros(1, 128, pixels.shape[0] // 16, pixels.shape[1] // 16)


def _cond_model(with_negative=True):
    embeds = {"context": torch.ones(1, 4, 8), "attention_mask": None}
    n_embeds = {"context": torch.ones(1, 4, 8), "attention_mask": None} if with_negative else {}
    return SimpleNamespace(embeds=embeds, n_embeds=n_embeds)


def _bundle(has_image_cond=True):
    return SimpleNamespace(
        dit=SimpleNamespace(estimated_vram_gb=11.0, module=SimpleNamespace(cfg=SimpleNamespace(has_image_cond=has_image_cond))),
        te_encoder=object(),
        vae=object(),
    )


def _pipe(**over):
    cfg = GeneratorFlux3Pipe.get_default_config()
    cfg.update(over)
    return GeneratorFlux3Pipe(config=cfg)


def _input(quantity=1, images=None, with_negative=True, has_image_cond=True):
    inp = {
        "model": _bundle(has_image_cond),
        "conditioning": [_cond_model(with_negative) for _ in range(quantity)],
        "seed": list(range(1, quantity + 1)),
    }
    if images is not None:
        inp["image"] = list(images)
    return PipeInput(input=inp)


def _use(variant):
    _FakeGenerator.sampling_settings = BASE if variant == "base" else DISTILLED


def setup_function(_):
    _FakeGenerator.instances.clear()
    _FakeGenerator.sampling_settings = BASE


@pytest.fixture(autouse=True)
def _patched():
    with patch(f"{FLOW}.make_device_plan", lambda **_: None), patch(f"{FLOW}.NativeGenerator", _FakeGenerator):
        yield


def _last_call():
    return _FakeGenerator.instances[-1].sample_calls[-1]


def test_name_inputs_outputs():
    assert GeneratorFlux3Pipe.name == "generator"
    inputs = {i.name: i for i in GeneratorFlux3Pipe.inputs()}
    assert inputs["conditioning"].io_type == IOType.CONDITIONING
    assert inputs["model"].io_type == IOType.MODEL
    assert GeneratorFlux3Pipe.outputs()[0].io_type == IOType.IMAGE


def test_modes_are_txt2img_and_edit():
    spec = next(s for s in GeneratorFlux3Pipe.configuration() if s.name == "mode")
    assert spec.choices == ["txt2img", "edit"]


@pytest.mark.parametrize("variant", ["base", "distilled"])
def test_explicit_steps_are_kept_on_either_checkpoint(variant):
    _use(variant)
    ctx = _pipe(steps=8).build_context(_input())
    assert ctx.extra["steps"] == 8


def test_base_runs_true_cfg_with_cfg_zero_star_off():
    _use("base")
    _pipe(guidance=3.5).process(_input(), lambda o: None)
    call = _last_call()
    assert call["cfg_scale"] == 3.5
    assert call["guidance_options"] == {"cfg_zero_star": False}
    assert call["conditioning"].uncond is not None
    assert call["steps"] == 50


def test_base_lets_the_preset_override_cfg_zero_star():
    _use("base")
    _pipe(guidance_options={"cfg_zero_star": True}).process(_input(), lambda o: None)
    assert _last_call()["guidance_options"] == {"cfg_zero_star": True}


def test_base_with_other_sampler_is_allowed():
    _use("base")
    _pipe(sampler="dpmpp_2m").process(_input(), lambda o: None)
    assert _last_call()["sampler"] == "dpmpp_2m"


def test_base_without_guidance_warns(caplog):
    _use("base")
    with caplog.at_level(logging.WARNING):
        _pipe(guidance=1.0).build_context(_input())
    assert any("without CFG" in r.getMessage() for r in caplog.records)


def test_distilled_never_gets_an_unconditional_pass():
    _use("distilled")
    _pipe(guidance=3.5, steps=16).process(_input(with_negative=True), lambda o: None)
    call = _last_call()
    assert call["conditioning"].uncond is None
    assert call["cfg_scale"] == 1.0
    assert call["steps"] == 16


def test_distilled_ignores_guidance_and_logs_it(caplog):
    _use("distilled")
    with caplog.at_level(logging.INFO):
        ctx = _pipe(guidance=4.0).build_context(_input())
    assert ctx.extra["guidance"] == 1.0
    assert any("distilled: guidance ignored" in r.getMessage() for r in caplog.records)


def test_distilled_disables_step_cache_and_iterate_mode():
    _use("distilled")
    pipe = _pipe(step_cache={"rel_threshold": 0.2}, iterate_mode=True)
    pipe.process(_input(), lambda o: None)
    call = _last_call()
    assert call["step_cache_options"] is None
    assert call["warm_start"] is False


def test_base_keeps_step_cache():
    _use("base")
    _pipe(step_cache={"rel_threshold": 0.2}).process(_input(), lambda o: None)
    assert _last_call()["step_cache_options"] == {"rel_threshold": 0.2}


def test_distilled_rejects_other_samplers_readably():
    _use("distilled")
    with pytest.raises(GenerationExecutionError, match="only samples with Euler"):
        _pipe(sampler="dpmpp_2m").build_context(_input())


def test_txt2img_uses_the_snapped_resolution():
    _use("base")
    _pipe(resolution="1000x600").process(_input(), lambda o: None)
    assert _last_call()["latents_shape"] == (1, 128, 608 // 16, 992 // 16)


def _edit(**over):
    return _pipe(mode="edit", **over)


def test_edit_single_reference_pads_then_crops_back():
    _use("distilled")
    source = Image.new("RGB", (1000, 600), (200, 10, 10))
    result = _edit().process(_input(images=[source]), lambda o: None)
    assert _last_call()["latents_shape"] == (1, 128, 608 // 16, 1008 // 16)
    assert result.output["image"][0].size == (1000, 600)


def test_edit_explicit_output_size_skips_the_crop():
    _use("distilled")
    source = Image.new("RGB", (1000, 600), (200, 10, 10))
    result = _edit(output_size="512x768").process(_input(images=[source]), lambda o: None)
    assert _last_call()["latents_shape"] == (1, 128, 768 // 16, 512 // 16)
    assert result.output["image"][0].size == (512, 768)


def test_edit_match_keyword_means_no_override():
    _use("distilled")
    source = Image.new("RGB", (1008, 640), (5, 5, 5))
    result = _edit(output_size="match").process(_input(images=[source]), lambda o: None)
    assert result.output["image"][0].size == (1008, 640)


def test_edit_multi_reference_attaches_every_latent_to_both_branches():
    _use("base")
    sources = [Image.new("RGB", (640, 480)), Image.new("RGB", (300, 300)), Image.new("RGB", (2000, 1000))]
    _edit().process(_input(images=sources), lambda o: None)
    conditioning = _last_call()["conditioning"]
    assert len(conditioning.cond["ref_latents"]) == 3
    assert conditioning.uncond["ref_latents"] is conditioning.cond["ref_latents"]
    assert _last_call()["latents_shape"] == (1, 128, 480 // 16, 640 // 16)


def test_edit_multi_reference_never_crops():
    _use("distilled")
    sources = [Image.new("RGB", (1000, 601)), Image.new("RGB", (300, 300))]
    result = _edit().process(_input(images=sources), lambda o: None)
    assert result.output["image"][0].size == (992, 592)


def test_edit_distilled_has_references_but_no_uncond():
    _use("distilled")
    _edit().process(_input(images=[Image.new("RGB", (256, 256))]), lambda o: None)
    conditioning = _last_call()["conditioning"]
    assert conditioning.uncond is None
    assert len(conditioning.cond["ref_latents"]) == 1


def test_edit_encodes_references_once_per_generation():
    _use("distilled")
    _edit(quantity=3).process(_input(quantity=3, images=[Image.new("RGB", (256, 256)), Image.new("RGB", (128, 128))]), lambda o: None)
    gen = _FakeGenerator.instances[-1]
    assert len(gen.sample_calls) == 3
    assert len(gen.encoded) == 2


def test_edit_uses_at_most_ten_references():
    _use("distilled")
    _edit().process(_input(images=[Image.new("RGB", (64, 64)) for _ in range(12)]), lambda o: None)
    assert len(_last_call()["conditioning"].cond["ref_latents"]) == 10


def test_edit_converts_alpha_references_to_rgb():
    _use("distilled")
    _edit().process(_input(images=[Image.new("RGBA", (64, 64), (0, 0, 0, 0))]), lambda o: None)
    gen = _FakeGenerator.instances[-1]
    assert gen.encoded[0][-1] == 3
    assert gen.encoded_means[0] == 255.0


def test_edit_without_images_is_a_readable_error():
    _use("distilled")
    with pytest.raises(GenerationExecutionError, match="at least one reference"):
        _edit().process(_input(images=[]), lambda o: None)


def test_edit_rejects_a_reference_below_the_grid():
    _use("distilled")
    with pytest.raises(GenerationExecutionError, match="Reference image 2 is too small"):
        _edit().process(_input(images=[Image.new("RGB", (64, 64)), Image.new("RGB", (8, 64))]), lambda o: None)


def test_edit_rejects_a_checkpoint_without_reference_weights():
    _use("distilled")
    with pytest.raises(GenerationExecutionError, match="no reference-image weights"):
        _edit().process(_input(images=[Image.new("RGB", (64, 64))], has_image_cond=False), lambda o: None)


def test_txt2img_ignores_attached_images():
    _use("base")
    _pipe().process(_input(images=[Image.new("RGB", (64, 64))]), lambda o: None)
    assert "ref_latents" not in _last_call()["conditioning"].cond
