from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from src.pipelines.outputs import GenerationExecutionError
from src.pipelines.pipes.generator.flux3.main import GeneratorFlux3Pipe
from src.platform.runtime.native.arch.flux3.model import Flux3
from src.platform.runtime.native.detect.registry import match_model_spec
from src.platform.runtime.native.engine import Conditioning, NativeGenerator, NativeModel
from src.platform.runtime.native.text_encoders.base import NativeTextEncoder
from src.platform.runtime.native.kernels import pick_operations

HIDDEN, HEADS, IN_CH, CTX, SEQ = 64, 2, 16, 32, 5

CONFIG = {
    "image_model": "flux3", "hidden_size": HIDDEN, "context_in_dim": CTX, "in_channels": IN_CH,
    "num_heads": HEADS, "head_dim": HIDDEN // HEADS, "depth": 1, "depth_single_blocks": 1,
    "mlp_hidden": HIDDEN * 3, "axes_dim": [8, 8, 8, 8], "theta": 10000,
    "flow_map": False, "has_image_cond": True,
}

class _NoTE(NativeTextEncoder):
    role = "stub"

    def encode(self, texts):
        raise AssertionError("conditioning is supplied directly")

def _generator(**overrides):
    config = {**CONFIG, **overrides}
    torch.manual_seed(0)
    module = Flux3.from_config(config, pick_operations(torch.float32, torch.float32)).eval()
    for param in module.parameters():
        param.data.normal_(0.0, 0.05)
    calls = []

    def record(_module, args, kwargs):
        x, sigma, context = args[:3]
        calls.append({"x": x.shape, "sigma": sigma.clone(), "context": context.shape, **kwargs})

    module.register_forward_pre_hook(record, with_kwargs=True)
    spec = match_model_spec(config)
    dit = NativeModel("diffusion_model", module, spec=spec, compute_dtype=torch.float32)
    vae_module = torch.nn.Identity()
    vae_module.z_channels = IN_CH // 4
    vae = NativeModel("vae", vae_module, spec=spec, compute_dtype=torch.float32)
    return NativeGenerator(dit, _NoTE(), vae), calls

def _cond(refs=None):
    cond = {"context": torch.randn(1, SEQ, CTX), "y": None, "attention_mask": None}
    if refs is not None:
        cond["ref_latents"] = refs
    return cond

SHAPE = (1, IN_CH, 4, 6)

def test_base_cfg_sends_reference_latents_on_cond_and_uncond_every_step():
    gen, calls = _generator()
    refs = [torch.randn(1, IN_CH, 2, 3), torch.randn(1, IN_CH, 3, 2)]
    out = gen.sample(
        Conditioning(_cond(refs), _cond(refs)), SHAPE, steps=3, seed=1, cfg_scale=3.5,
        guidance_options={"cfg_zero_star": False},
    )
    assert out.shape == SHAPE
    assert len(calls) == 6
    for call in calls:
        assert "flow_map_target" not in call
        assert [r.shape for r in call["ref_latents"]] == [r.shape for r in refs]
        assert call["x"][-2:] == SHAPE[-2:]

def test_flow_map_receives_the_next_sigma_as_target_and_references_each_step():
    gen, calls = _generator(flow_map=True)
    refs = [torch.randn(1, IN_CH, 2, 3)]
    steps = 4
    out = gen.sample(Conditioning(_cond(refs), None), SHAPE, steps=steps, seed=1, cfg_scale=1.0, sampler="euler")
    assert out.shape == SHAPE
    assert len(calls) == steps
    sigmas = [float(c["sigma"].flatten()[0]) for c in calls]
    targets = [float(c["flow_map_target"].flatten()[0]) for c in calls]
    assert targets[:-1] == pytest.approx(sigmas[1:], abs=1e-5)
    assert targets[-1] == pytest.approx(0.0, abs=1e-5)
    assert sigmas == sorted(sigmas, reverse=True)
    assert all(c["ref_latents"][0].shape == refs[0].shape for c in calls)

def test_checkpoint_without_reference_weights_does_text_to_image_and_rejects_edit():
    gen, calls = _generator(has_image_cond=False)
    out = gen.sample(Conditioning(_cond(), _cond()), SHAPE, steps=2, seed=1, cfg_scale=3.5)
    assert out.shape == SHAPE
    with pytest.raises(ValueError, match="no reference-image weights"):
        refs = [torch.randn(1, IN_CH, 2, 3)]
        gen.sample(Conditioning(_cond(refs), _cond(refs)), SHAPE, steps=2, seed=1, cfg_scale=3.5)

def test_has_image_cond_false_checkpoint_builds_without_image_cond_weights():
    module = Flux3.from_config({**CONFIG, "has_image_cond": False}, pick_operations(torch.float32, torch.float32))
    assert "image_cond" not in module.emb_in
    assert "image_cond" not in module.content_mode_blocks
    assert "image_cond" not in module.early_stream_modulations

def test_real_arch_gives_the_generator_a_sixteen_pixel_grid():
    gen, _ = _generator()
    assert gen.pixel_granularity() == 16
    assert gen.snap_resolution(1000, 600) == (992, 592)
    assert gen.latent_shape_for(992, 592) == (1, IN_CH, 37, 62)


def test_generator_reads_missing_reference_support_from_the_real_arch_config():
    gen, _ = _generator(has_image_cond=False)
    ctx = SimpleNamespace(extra={"generator": gen, "images": [Image.new("RGB", (64, 64))]})
    pipe = GeneratorFlux3Pipe(config=GeneratorFlux3Pipe.get_default_config())
    with pytest.raises(GenerationExecutionError, match="no reference-image weights"):
        pipe._prepare_edit(ctx)
