from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from src.platform.runtime.native.arch.moss_image.model import MossImage, convert_moss_image_state_dict
from src.platform.runtime.native.base import load_into_module
from src.platform.runtime.native.detect.registry import match_model_spec
from src.platform.runtime.native.errors import NativeEngineUnsupportedError
from vendor.gpl.comfyui.ops import pick_operations

_FIXTURE = Path(__file__).parent / "fixtures" / "moss_image_reference_dit.py"

DIM, HEADS, FFN, DEPTH, TEXT_DIM, FREQ, CHANNELS = 64, 2, 96, 3, 48, 32, 16

CONFIG = {
    "image_model": "moss_image", "hidden_size": DIM, "head_dim": DIM // HEADS, "num_heads": HEADS,
    "ffn_dim": FFN, "depth": DEPTH, "context_in_dim": TEXT_DIM, "in_channels": CHANNELS,
    "freq_dim": FREQ, "rope_base": 10000.0,
}


def _load_reference_module():
    spec = importlib.util.spec_from_file_location("moss_image_reference_dit", str(_FIXTURE))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reference():
    module = _load_reference_module()
    torch.manual_seed(0)
    ref = module.MOSSImageTransformer(
        dim=DIM, ffn_dim=FFN, num_heads=HEADS, num_layers=DEPTH,
        text_dim=TEXT_DIM, freq_dim=FREQ, latent_channels=CHANNELS,
    )
    for name, param in ref.named_parameters():
        if name.endswith("norm1.weight") or name.endswith("norm2.weight") or name.endswith("_norm.weight") or "final_norm" in name:
            param.data.uniform_(0.5, 1.5)
    return ref.eval()


@pytest.fixture(scope="module")
def ours(reference):
    model = MossImage.from_config(CONFIG, pick_operations(torch.float32, torch.float32))
    load_into_module(model, convert_moss_image_state_dict(reference.state_dict()), match_model_spec(CONFIG))
    return model.eval()


def _inputs(seed, h, w, lengths):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(len(lengths), CHANNELS, h, w, generator=generator)
    context = torch.randn(len(lengths), max(lengths), TEXT_DIM, generator=generator)
    mask = torch.zeros(len(lengths), max(lengths), dtype=torch.bool)
    for i, n in enumerate(lengths):
        mask[i, :n] = True
    return x, context, mask


def _sigma(value, batch=1):
    return torch.full((batch,), value, dtype=torch.float32)


@pytest.mark.parametrize("h,w,length", [(4, 4, 5), (3, 5, 7), (6, 2, 1), (1, 7, 3)])
def test_matches_the_reference_on_square_and_non_square_latents_and_text_lengths(reference, ours, h, w, length):
    x, context, _ = _inputs(h * 10 + w, h, w, [length])
    sigma = _sigma(0.37)
    with torch.no_grad():
        expected = reference(x, sigma * 1000.0, context)
        actual = ours(x, sigma, context)
    assert actual.shape == (1, CHANNELS, h, w)
    assert expected.abs().mean() > 1e-3
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("skip", [{1}, {0, 2}, frozenset({2})])
def test_skip_layers_match_the_reference_and_change_the_output(reference, ours, skip):
    x, context, _ = _inputs(3, 4, 3, [6])
    sigma = _sigma(0.81)
    with torch.no_grad():
        expected = reference(x, sigma * 1000.0, context, skip_layers=skip)
        actual = ours(x, sigma, context, skip_layers=skip)
        full = ours(x, sigma, context)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert (actual - full).abs().max() > 1e-4


def test_distinct_timesteps_give_distinct_outputs(reference, ours):
    x, context, _ = _inputs(4, 3, 3, [4])
    with torch.no_grad():
        first = ours(x, _sigma(0.2), context)
        second = ours(x, _sigma(0.9), context)
        expected = reference(x, _sigma(0.9) * 1000.0, context)
    assert (first - second).abs().max() > 1e-4
    torch.testing.assert_close(second, expected, atol=2e-5, rtol=2e-5)


def test_padded_batch_with_a_mask_equals_each_unpadded_sample(reference, ours):
    lengths = [3, 8, 5]
    x, context, mask = _inputs(5, 3, 4, lengths)
    context[~mask] = 123.0
    sigma = _sigma(0.55, batch=3)
    with torch.no_grad():
        batched = ours(x, sigma, context, attention_mask=mask)
        expected = reference(x, sigma * 1000.0, context, encoder_attention_mask=mask)
        singles = torch.cat([
            ours(x[i : i + 1], sigma[:1], context[i : i + 1, :n]) for i, n in enumerate(lengths)
        ])
    torch.testing.assert_close(batched, singles, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(batched, expected, atol=2e-5, rtol=2e-5)


def test_padded_context_with_a_mask_equals_the_unpadded_context(ours):
    x, context, mask = _inputs(6, 4, 4, [4])
    padded = torch.cat([context, torch.randn(1, 5, TEXT_DIM)], dim=1)
    padded_mask = torch.cat([mask, torch.zeros(1, 5, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        plain = ours(x, _sigma(0.3), context)
        masked = ours(x, _sigma(0.3), padded, attention_mask=padded_mask)
    torch.testing.assert_close(masked, plain, atol=0, rtol=0)


def test_engine_keyword_arguments_are_ignored(ours):
    x, context, _ = _inputs(7, 3, 3, [4])
    with torch.no_grad():
        plain = ours(x, _sigma(0.4), context)
        extra = ours(x, _sigma(0.4), context, y=None, guidance=None, step_cache=None)
    torch.testing.assert_close(extra, plain, atol=0, rtol=0)


@pytest.mark.parametrize("skip", [{DEPTH}, {-1}, {"1"}])
def test_invalid_skip_layers_are_rejected(ours, skip):
    x, context, _ = _inputs(8, 2, 2, [2])
    with pytest.raises(ValueError, match="skip_layers"):
        ours(x, _sigma(0.5), context, skip_layers=skip)


def test_a_sample_without_valid_text_tokens_is_rejected(ours):
    x, context, _ = _inputs(9, 2, 2, [2])
    with pytest.raises(ValueError, match="text token"):
        ours(x, _sigma(0.5), context, attention_mask=torch.zeros(1, 2, dtype=torch.bool))


def test_conversion_fuses_qkv_in_q_k_v_order_and_drops_rope_bands(reference):
    state = reference.state_dict()
    assert "rope.bands" in state
    converted = convert_moss_image_state_dict(state)
    assert "rope.bands" not in converted
    assert not any(".proj.q." in k or ".proj.k." in k or ".proj.v." in k for k in converted)
    expected = torch.cat([state[f"layers.1.proj.{n}.weight"] for n in "qkv"])
    torch.testing.assert_close(converted["layers.1.proj.qkv.weight"], expected, atol=0, rtol=0)
    assert converted["layers.1.proj.qkv.weight"].shape == (3 * DIM, DIM)


def test_conversion_leaves_an_already_fused_state_dict_untouched(ours):
    state = {k: v for k, v in ours.state_dict().items()}
    assert set(convert_moss_image_state_dict(state)) == set(state)


def test_conversion_drops_attention_hints_and_keeps_everything_else(ours):
    state = dict(ours.state_dict())
    state["layers.0.comfy_attention.config"] = torch.zeros(4, dtype=torch.uint8)
    converted = convert_moss_image_state_dict(state)
    assert "layers.0.comfy_attention.config" not in converted
    assert set(converted) == set(ours.state_dict())


def test_separate_quantised_projections_are_refused(reference):
    state = dict(reference.state_dict())
    state["layers.0.proj.q.weight"] = state["layers.0.proj.q.weight"].to(torch.int8)
    with pytest.raises(NativeEngineUnsupportedError, match="quantised"):
        convert_moss_image_state_dict(state)


def test_built_module_has_exactly_the_converted_reference_keys(reference, ours):
    converted = convert_moss_image_state_dict(reference.state_dict())
    assert {k: tuple(v.shape) for k, v in ours.state_dict().items()} == {k: tuple(v.shape) for k, v in converted.items()}
    assert ours.patch_size == 1
    assert ours.params.in_channels == CHANNELS
