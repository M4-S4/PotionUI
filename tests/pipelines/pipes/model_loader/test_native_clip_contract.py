from __future__ import annotations

import sys
from dataclasses import dataclass

import pytest
import torch

from src.pipelines.pipes._shared.generation import clip_batch as clip_batch_module
from src.pipelines.pipes.model_loader.anima.anima_clip import AnimaClipTextEncoder
from src.pipelines.pipes.model_loader.flux.flux_clip import FluxClipTextEncoder
from src.pipelines.pipes.model_loader.flux3.flux3_clip import Flux3ClipTextEncoder
from src.pipelines.pipes.model_loader.krea2.krea2_clip import Krea2ClipTextEncoder
from src.pipelines.pipes.model_loader.qwen.qwen_clip import QwenClipTextEncoder
from src.pipelines.pipes.model_loader.qwen_image21.qwen_image21_clip import QwenImage21ClipTextEncoder
from src.pipelines.pipes.model_loader.wan22.wan_clip import WanClipTextEncoder
from src.pipelines.pipes.model_loader.z_image.z_image_clip import ZImageClipTextEncoder
from src.platform.runtime.primitives.clip import ClipTextEncoder, ConditioningModel
from tests.pipelines.pipes.model_loader.conftest import CountingLoader, RecordingEncoder

pytestmark = pytest.mark.usefixtures("fresh_embed_cache")


@dataclass(frozen=True)
class ClipFamily:
    id: str
    cls: type
    pooled: bool = False
    empty_negative_encoded: bool = True


FAMILIES = [
    ClipFamily("anima", AnimaClipTextEncoder),
    ClipFamily("flux", FluxClipTextEncoder, pooled=True, empty_negative_encoded=False),
    ClipFamily("flux3", Flux3ClipTextEncoder),
    ClipFamily("krea2", Krea2ClipTextEncoder),
    ClipFamily("qwen", QwenClipTextEncoder),
    ClipFamily("qwen_image21", QwenImage21ClipTextEncoder),
    ClipFamily("wan22", WanClipTextEncoder),
    ClipFamily("z_image", ZImageClipTextEncoder),
]

TRUE_CFG = [f for f in FAMILIES if f.empty_negative_encoded]

IMAGE_FAMILIES = [f for f in FAMILIES if f.id in {"krea2", "qwen", "qwen_image21"}]

families = pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.id)
true_cfg_families = pytest.mark.parametrize("family", TRUE_CFG, ids=lambda f: f.id)
image_families = pytest.mark.parametrize("family", IMAGE_FAMILIES, ids=lambda f: f.id)


def _adapter(family, fingerprint=None):
    fake = RecordingEncoder(pooled=family.pooled)
    return family.cls(fake, device="cpu", model_fingerprint=fingerprint), fake


def _miss_batch(*prompts):
    return [{"prompt": p, "negative_prompt": "", "do_classifier_free_guidance": False} for p in prompts]


@families
def test_is_a_clip_text_encoder(family):
    assert isinstance(_adapter(family)[0], ClipTextEncoder)


@families
def test_packs_prompt_text_and_encoder_roles(family):
    enc, fake = _adapter(family)
    cond = enc.encode_prompt("a cat", "blurry")
    roles = {"context", "attention_mask"} | ({"pooled"} if family.pooled else set())
    assert isinstance(cond, ConditioningModel)
    assert (cond.p_prompt, cond.n_prompt) == ("a cat", "blurry")
    assert set(cond.embeds) == roles
    assert set(cond.n_embeds) == roles
    assert fake.calls == [["a cat"], ["blurry"]]


@families
def test_no_cfg_skips_negative_pass(family):
    enc, fake = _adapter(family)
    cond = enc.encode_prompt("p", "some negative", do_classifier_free_guidance=False)
    assert cond.n_embeds == {}
    assert fake.calls == [["p"]]


@true_cfg_families
def test_empty_negative_still_encoded_for_true_cfg(family):
    enc, fake = _adapter(family)
    cond = enc.encode_prompt("only positive", "")
    assert cond.n_embeds
    assert fake.calls == [["only positive"], [""]]


@families
def test_model_fingerprint_surfaced_for_prompt_encoder_cache(family):
    enc, _ = _adapter(family, fingerprint="fam|te|dit|lora=none")
    assert enc._model_fingerprint == "fam|te|dit|lora=none"


@families
def test_embedding_files_ignored_not_fatal(family):
    enc, _ = _adapter(family)
    cond = enc.encode_prompt("p", "", embedding_files={"tok": "/path/to.pt"})
    assert isinstance(cond, ConditioningModel)


@families
def test_identical_prompt_is_served_from_the_embed_cache(family):
    enc, fake = _adapter(family, fingerprint=f"{family.id}|fp")
    enc.encode_prompt("a cat", "blurry")
    seen = len(fake.calls)
    enc.encode_prompt("a cat", "blurry")
    assert len(fake.calls) == seen


@families
def test_lazy_encoder_never_resolved_on_a_full_embed_cache_hit(family):
    fake = RecordingEncoder(pooled=family.pooled)
    warm = CountingLoader(fake)
    family.cls(device="cpu", model_fingerprint="fp-hit", te_loader=warm).encode_prompt(
        "a fox", "", do_classifier_free_guidance=False
    )
    assert warm.calls == 1
    cold = CountingLoader(fake)
    cond = family.cls(device="cpu", model_fingerprint="fp-hit", te_loader=cold).encode_prompt(
        "a fox", "", do_classifier_free_guidance=False
    )
    assert cold.calls == 0
    assert isinstance(cond, ConditioningModel)


@families
def test_lazy_encoder_resolved_once_for_a_batch_of_misses(family):
    loader = CountingLoader(RecordingEncoder(pooled=family.pooled))
    enc = family.cls(device="cpu", model_fingerprint="fp-batch", te_loader=loader)
    enc.encode_prompts(_miss_batch("a", "b", "c"))
    assert loader.calls == 1


def _spy_on_prompt_embed_key(monkeypatch, family):
    module = sys.modules[family.cls.__module__]
    keys = []
    real_key = module.prompt_embed_key

    def spy(*args, **kwargs):
        key = real_key(*args, **kwargs)
        keys.append(key)
        return key

    monkeypatch.setattr(module, "prompt_embed_key", spy)
    return keys


@image_families
def test_no_images_never_calls_encode_with_an_images_kwarg(family):
    enc, fake = _adapter(family)
    enc.encode_prompt("a cat", "blurry")
    assert fake.image_calls == [[], []]


@image_families
def test_images_condition_both_positive_and_negative_pass(family):
    enc, fake = _adapter(family)
    img = torch.rand(4, 4, 3)
    cond = enc.encode_prompt("edit it", "blurry", images=[img])
    assert isinstance(cond, ConditioningModel)
    assert fake.calls == [["edit it"], ["blurry"]]
    assert fake.image_calls == [[img], [img]]


@image_families
def test_image_cache_key_differs_for_different_images(family, monkeypatch):
    keys = _spy_on_prompt_embed_key(monkeypatch, family)
    enc, _ = _adapter(family, fingerprint="fp")
    enc.encode_prompt("edit it", "", images=[torch.rand(4, 4, 3)], do_classifier_free_guidance=False)
    enc.encode_prompt("edit it", "", images=[torch.rand(4, 4, 3)], do_classifier_free_guidance=False)
    assert keys[0] != keys[1]


@image_families
def test_image_cache_key_same_for_same_image_content(family, monkeypatch):
    keys = _spy_on_prompt_embed_key(monkeypatch, family)
    enc, _ = _adapter(family, fingerprint="fp")
    img = torch.rand(4, 4, 3)
    enc.encode_prompt("edit it", "", images=[img], do_classifier_free_guidance=False)
    enc.encode_prompt("edit it", "", images=[img.clone()], do_classifier_free_guidance=False)
    assert keys[0] == keys[1]


@families
def test_different_prompt_does_not_hit_the_embed_cache(family):
    enc, fake = _adapter(family, fingerprint=f"{family.id}|fp")
    enc.encode_prompt("a cat", "blurry")
    seen = len(fake.calls)
    enc.encode_prompt("a dog", "blurry")
    assert len(fake.calls) > seen


@true_cfg_families
def test_different_negative_prompt_does_not_hit_the_embed_cache(family):
    enc, fake = _adapter(family, fingerprint=f"{family.id}|fp")
    enc.encode_prompt("a cat", "blurry")
    seen = len(fake.calls)
    enc.encode_prompt("a cat", "sharp")
    assert len(fake.calls) > seen


@families
def test_no_fingerprint_means_no_cache_key(family, monkeypatch):
    enc, _ = _adapter(family, fingerprint=None)
    seen = []
    real = clip_batch_module.run_text_encode_batch

    def spy(encoder, device, encode_fns, *, reserve_gb=None, cache_keys=None, encoder_provider=None):
        seen.append(cache_keys)
        return real(
            encoder, device, encode_fns, reserve_gb=reserve_gb, cache_keys=cache_keys,
            encoder_provider=encoder_provider,
        )

    monkeypatch.setattr(clip_batch_module, "run_text_encode_batch", spy)
    enc.encode_prompt("a cat", "blurry")
    assert seen == [[None]]
