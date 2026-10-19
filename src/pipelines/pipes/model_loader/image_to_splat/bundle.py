from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.pipelines.pipes._shared.generation.weak_model_ref import WeakModelRef
from src.platform.runtime.native.engine import NativeModel

logger = logging.getLogger(__name__)


@dataclass
class ImageToSplatModelBundle:
    estimator: NativeModel = field(default=WeakModelRef())
    device: str = "cuda"

    def components(self):
        estimator = self.estimator
        if estimator is None:
            raise ValueError(
                "MoGe-2 was evicted from the model cache before this generation could use it; "
                "re-run to load it again"
            )
        return estimator.module

    def unload(self) -> None:
        unload = getattr(self.estimator, "unload", None)
        if callable(unload):
            try:
                unload()
            except Exception:
                logger.debug("image-to-splat estimator eviction failed", exc_info=True)
