from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from src.pipelines.pipes._shared.generation.clip_batch import SequentialWindowClipTextEncoder
from src.platform.runtime.native.text_encoders import NativeTextEncoder, prompt_embed_key
from src.platform.runtime.primitives.clip import ConditioningModel


class Flux3ClipTextEncoder(SequentialWindowClipTextEncoder):
    cache_tag = "flux3_te"

    def __init__(
        self,
        encoder: Optional[NativeTextEncoder] = None,
        *,
        device: str = "cuda",
        model_fingerprint: Optional[str] = None,
        te_loader: Optional[Callable[[], NativeTextEncoder]] = None,
    ) -> None:
        self._encoder = encoder
        self._te_loader = te_loader
        self.device = device
        self._model_fingerprint = model_fingerprint

    @property
    def encoder(self) -> NativeTextEncoder:
        if self._encoder is None and self._te_loader is not None:
            self._encoder = self._te_loader()
            self._te_loader = None
        return self._encoder

    @encoder.setter
    def encoder(self, value: NativeTextEncoder) -> None:
        self._encoder = value

    def _encode_fn_and_key(self, request: Dict[str, Any]) -> Tuple[Callable[[], Any], Optional[str]]:
        prompt = request["prompt"]
        negative_prompt = request["negative_prompt"]
        do_classifier_free_guidance = bool(request.get("do_classifier_free_guidance", True))

        def _encode():
            pos = self.encoder.encode([prompt])
            neg = self.encoder.encode([negative_prompt]) if do_classifier_free_guidance else {}
            return pos, neg

        cache_key = prompt_embed_key(
            self._model_fingerprint, self.cache_tag, prompt, negative_prompt, do_classifier_free_guidance,
        )
        return _encode, cache_key

    def _pack(self, request: Dict[str, Any], result: Any) -> ConditioningModel:
        embeds, n_embeds = result
        return ConditioningModel(
            p_prompt=request["prompt"],
            n_prompt=request["negative_prompt"],
            embeds=embeds,
            n_embeds=n_embeds,
        )
