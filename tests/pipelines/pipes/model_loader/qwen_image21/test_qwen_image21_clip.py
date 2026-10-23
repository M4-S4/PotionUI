from __future__ import annotations

import torch

from src.platform.runtime.primitives.clip import ConditioningModel
from src.platform.runtime.native.text_encoders import NativeTextEncoder
from src.pipelines.pipes.model_loader.qwen_image21.qwen_image21_clip import QwenImage21ClipTextEncoder


class _FakeEncoder(NativeTextEncoder):
    role = "fake"

    def __init__(self):
        self.calls: list[list[str]] = []
        self.image_calls: list[list] = []

    def encode(self, texts, images=None, grounding_px=768, keep_vision=False):
        self.calls.append(list(texts))
        self.image_calls.append(list(images) if images else [])
        seq = max(1, len(texts[0]))
        out = {
            "context": torch.ones(1, seq, 4),
            "attention_mask": torch.ones(1, seq),
        }
        if images:
            out["image_slots"] = [0]
        return out


def test_forwards_full_image_batch():
    enc = QwenImage21ClipTextEncoder(_FakeEncoder(), device="cpu")
    assert enc.forwards_full_image_batch is True


def test_images_condition_both_positive_and_negative_pass():
    fake = _FakeEncoder()
    enc = QwenImage21ClipTextEncoder(fake, device="cpu")
    img = torch.rand(4, 4, 3)
    cond = enc.encode_prompt("edit it", "blurry", images=[img])
    assert isinstance(cond, ConditioningModel)
    assert fake.calls == [["edit it"], ["blurry"]]
    assert fake.image_calls == [[img], [img]]
    assert cond.embeds["image_slots"] == [0]
    assert cond.n_embeds["image_slots"] == [0]


def test_images_skips_the_negative_pass_when_cfg_off():
    fake = _FakeEncoder()
    enc = QwenImage21ClipTextEncoder(fake, device="cpu")
    img = torch.rand(4, 4, 3)
    cond = enc.encode_prompt("edit it", "neg", images=[img], do_classifier_free_guidance=False)
    assert cond.n_embeds == {}
    assert fake.calls == [["edit it"]]
    assert fake.image_calls == [[img]]


def test_images_converted_from_pil_to_tensor():
    from PIL import Image as PILImage

    fake = _FakeEncoder()
    enc = QwenImage21ClipTextEncoder(fake, device="cpu")
    pil_img = PILImage.new("RGB", (4, 4), color=(255, 0, 0))
    enc.encode_prompt("edit it", "", images=[pil_img], do_classifier_free_guidance=False)
    (seen,) = fake.image_calls[0]
    assert isinstance(seen, torch.Tensor)
    assert seen.shape == (4, 4, 3)
    assert seen.max().item() <= 1.0 and seen.min().item() >= 0.0


def _spy_on_prompt_embed_key(monkeypatch):
    import src.pipelines.pipes.model_loader.qwen_image21.qwen_image21_clip as clip_mod

    calls = []
    real_key = clip_mod.prompt_embed_key

    def spy(*args, **kwargs):
        key = real_key(*args, **kwargs)
        calls.append(key)
        return key

    monkeypatch.setattr(clip_mod, "prompt_embed_key", spy)
    return calls


def test_cache_key_differs_for_different_prompts(monkeypatch):
    calls = _spy_on_prompt_embed_key(monkeypatch)
    fake = _FakeEncoder()
    enc = QwenImage21ClipTextEncoder(fake, device="cpu", model_fingerprint="fp")

    enc.encode_prompt("a cat", "", do_classifier_free_guidance=False)
    enc.encode_prompt("a dog", "", do_classifier_free_guidance=False)

    assert calls[0] != calls[1]


def test_transparent_reference_pixels_reach_the_vision_tower_as_white():
    from PIL import Image

    from src.pipelines.pipes.model_loader.qwen_image21.qwen_image21_clip import _to_image_tensor

    tensor = _to_image_tensor(Image.new("RGBA", (2, 2), (255, 0, 0, 0)))

    assert tensor.shape == (2, 2, 3)
    assert torch.allclose(tensor, torch.ones_like(tensor))
