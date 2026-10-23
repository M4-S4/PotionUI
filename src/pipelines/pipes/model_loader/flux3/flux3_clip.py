from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from src.pipelines.pipes._shared.generation.clip_batch import SequentialWindowClipTextEncoder
from src.platform.runtime.native.text_encoders import prompt_embed_key


class Flux3ClipTextEncoder(SequentialWindowClipTextEncoder):
    cache_tag = "flux3_te"

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
