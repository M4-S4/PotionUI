from __future__ import annotations

import pytest

from src.pipelines.pipes._shared.generation import clip_batch as clip_batch_module
from src.pipelines.pipes._shared.generation.clip_batch import SequentialWindowClipTextEncoder
from src.platform.runtime.native.text_encoders import prompt_embed_key
from src.platform.runtime.primitives.clip import ClipTextEncoder, ConditioningModel
from tests.pipelines.pipes.model_loader.conftest import CountingLoader, RecordingEncoder

pytestmark = pytest.mark.usefixtures("fresh_embed_cache")


class _Adapter(SequentialWindowClipTextEncoder):
    def _encode_fn_and_key(self, request):
        prompt = request["prompt"]
        negative = request["negative_prompt"]
        cfg = bool(request.get("do_classifier_free_guidance", True))

        def encode():
            pos = self.encoder.encode([prompt])
            neg = self.encoder.encode([negative]) if cfg else {}
            return pos, neg

        return encode, prompt_embed_key(self._model_fingerprint, "dummy", prompt, negative, cfg)


def _adapter(fingerprint=None, **kwargs):
    fake = RecordingEncoder()
    return _Adapter(fake, device="cpu", model_fingerprint=fingerprint, **kwargs), fake


def _spy_batch(monkeypatch):
    seen = []
    real = clip_batch_module.run_text_encode_batch

    def spy(encoder, device, encode_fns, *, reserve_gb=None, cache_keys=None, encoder_provider=None):
        seen.append({"device": device, "cache_keys": cache_keys, "provider": encoder_provider})
        return real(
            encoder, device, encode_fns, reserve_gb=reserve_gb, cache_keys=cache_keys,
            encoder_provider=encoder_provider,
        )

    monkeypatch.setattr(clip_batch_module, "run_text_encode_batch", spy)
    return seen


def test_is_a_clip_text_encoder():
    assert isinstance(_adapter()[0], ClipTextEncoder)


def test_encode_prompt_packs_positive_and_negative():
    enc, fake = _adapter()
    cond = enc.encode_prompt("a cat", "blurry")
    assert isinstance(cond, ConditioningModel)
    assert (cond.p_prompt, cond.n_prompt) == ("a cat", "blurry")
    assert fake.calls == [["a cat"], ["blurry"]]
    assert set(cond.n_embeds) == {"context", "attention_mask"}


def test_encode_prompts_returns_one_conditioning_per_request_in_order():
    enc, _ = _adapter()
    conds = enc.encode_prompts([
        {"prompt": p, "negative_prompt": "", "do_classifier_free_guidance": False} for p in ("a", "bb", "ccc")
    ])
    assert [c.p_prompt for c in conds] == ["a", "bb", "ccc"]
    assert [c.embeds["context"].shape[1] for c in conds] == [1, 2, 3]


def test_routes_through_one_run_text_encode_batch_per_encode_prompts_call(monkeypatch):
    enc, fake = _adapter(fingerprint="fp")
    seen = _spy_batch(monkeypatch)
    enc.encode_prompts([
        {"prompt": p, "negative_prompt": "", "do_classifier_free_guidance": False} for p in ("a", "b")
    ])
    assert len(seen) == 1
    assert seen[0]["device"] == "cpu"
    assert seen[0]["provider"]() is fake
    assert all(k is not None for k in seen[0]["cache_keys"])


def test_no_fingerprint_means_no_cache_key(monkeypatch):
    enc, _ = _adapter(fingerprint=None)
    seen = _spy_batch(monkeypatch)
    enc.encode_prompt("a cat", "blurry")
    assert [s["cache_keys"] for s in seen] == [[None]]


def test_repeated_identical_prompt_hits_the_embed_cache():
    enc, fake = _adapter(fingerprint="fp")
    enc.encode_prompt("a cat", "blurry")
    enc.encode_prompt("a cat", "blurry")
    assert len(fake.calls) == 2


def test_different_prompt_does_not_hit_the_cache():
    enc, fake = _adapter(fingerprint="fp")
    enc.encode_prompt("a cat", "blurry")
    enc.encode_prompt("a dog", "blurry")
    assert len(fake.calls) == 4


def test_lazy_encoder_never_resolved_on_a_full_embed_cache_hit():
    fake = RecordingEncoder()
    warm = CountingLoader(fake)
    _Adapter(device="cpu", model_fingerprint="fp-hit", te_loader=warm).encode_prompt(
        "a fox", "", do_classifier_free_guidance=False
    )
    assert warm.calls == 1
    cold = CountingLoader(fake)
    cond = _Adapter(device="cpu", model_fingerprint="fp-hit", te_loader=cold).encode_prompt(
        "a fox", "", do_classifier_free_guidance=False
    )
    assert cold.calls == 0
    assert isinstance(cond, ConditioningModel)


def test_lazy_encoder_resolved_once_for_a_batch_of_misses():
    loader = CountingLoader(RecordingEncoder())
    enc = _Adapter(device="cpu", model_fingerprint="fp-batch", te_loader=loader)
    enc.encode_prompts([
        {"prompt": p, "negative_prompt": "", "do_classifier_free_guidance": False} for p in ("a", "b", "c")
    ])
    assert loader.calls == 1


def test_lazy_encoder_resolved_once_for_a_mixed_hit_miss_batch():
    fake = RecordingEncoder()
    _Adapter(fake, device="cpu", model_fingerprint="fp-mix").encode_prompt(
        "warm", "", do_classifier_free_guidance=False
    )
    loader = CountingLoader(fake)
    enc = _Adapter(device="cpu", model_fingerprint="fp-mix", te_loader=loader)
    enc.encode_prompts([
        {"prompt": p, "negative_prompt": "", "do_classifier_free_guidance": False} for p in ("warm", "cold1", "cold2")
    ])
    assert loader.calls == 1


def test_lazy_encoder_property_caches_after_first_resolve():
    fake = RecordingEncoder()
    loader = CountingLoader(fake)
    enc = _Adapter(device="cpu", te_loader=loader)
    assert enc.encoder is fake
    assert enc.encoder is fake
    assert loader.calls == 1


def test_eager_encoder_bypasses_te_loader_entirely():
    fake = RecordingEncoder()

    def must_not_run():
        raise AssertionError("te_loader must not run when encoder is already resolved")

    enc = _Adapter(fake, device="cpu", te_loader=must_not_run)
    assert enc.encoder is fake


def test_encoder_setter_replaces_the_encoder():
    enc, _ = _adapter()
    other = RecordingEncoder()
    enc.encoder = other
    assert enc.encoder is other
