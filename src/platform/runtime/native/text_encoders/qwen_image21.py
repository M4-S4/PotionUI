from __future__ import annotations

import torch

from ..errors import NativeEngineUnsupportedError
from .base import NativeTextEncoder
from .qwen3 import Qwen3Model
from .qwen3_vl_vision import splice_vision_inputs


class QwenImage21TextEncoder(NativeTextEncoder):
    role = "qwen3vl_8b"

    def __init__(
        self, module: Qwen3Model, tokenizer, variant: str = "qwen3vl_8b",
        device: str | torch.device = "cpu",
    ) -> None:
        self.module = module
        self.tokenizer = tokenizer
        self.role = variant
        self._device = torch.device(device)
        self._has_vision = hasattr(module.model, "visual")
        self._layer = module.cfg.num_hidden_layers - 1

    @torch.inference_mode()
    def encode(
        self, texts: list[str], images: list[torch.Tensor] | None = None,
        grounding_px: int = 768, keep_vision: bool = False,
    ) -> dict[str, torch.Tensor]:
        if images:
            return self._encode_with_images(texts, images, grounding_px, keep_vision)
        ids, mask, prefix_len = self.tokenizer(texts, device=self._device)
        return self._encode_ids(ids, mask, prefix_len)

    def _encode_ids(self, ids, mask, prefix_len) -> dict[str, torch.Tensor]:
        stacked = self.module(ids, attention_mask=mask, layers_to_extract=(self._layer,), capture="output")
        context = stacked.squeeze(1)
        return {"context": context[:, prefix_len:], "attention_mask": mask[:, prefix_len:]}

    def _encode_with_images(
        self, texts: list[str], images: list[torch.Tensor], grounding_px: int, keep_vision: bool,
    ) -> dict[str, torch.Tensor]:
        if not self._has_vision:
            raise NativeEngineUnsupportedError(
                "this qwen-image-2.1 text encoder has no vision tower loaded; "
                "request the vision-enabled variant at load time (load_text_encoder(..., vision=True))"
            )
        if len(texts) != 1:
            raise ValueError("image-conditioned encode() supports exactly one prompt per call")

        ids, mask, prefix_len = self.tokenizer.tokenize_with_images(
            texts[0], num_images=len(images), device=self._device,
        )
        inputs, new_mask, image_spans = splice_vision_inputs(
            self.module, ids, mask, images, device=self._device, grounding_px=grounding_px,
        )

        stacked = self.module(
            None, attention_mask=new_mask, layers_to_extract=(self._layer,), capture="output",
            **inputs,
        )
        context = stacked.squeeze(1)

        keep = torch.ones(context.shape[1], dtype=torch.bool, device=context.device)
        keep[:prefix_len] = False
        slots: list[int] = []
        if not keep_vision:
            for start, size, _grid in image_spans:
                keep[start:start + size] = False
                slots.append(int(keep[:start].sum().item()))

        context = context[:, keep]
        new_mask = new_mask[:, keep.to(new_mask.device)]
        result = {"context": context, "attention_mask": new_mask}
        if slots:
            result["image_slots"] = slots
        return result
