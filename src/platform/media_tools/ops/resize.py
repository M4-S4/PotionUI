from __future__ import annotations

from typing import Optional

from PIL import Image

from src.platform.media_tools.base import MediaTool
from src.platform.media_tools.fastpath import CRF_BY_QUALITY, DEFAULT_CRF
from src.platform.imaging.pixels import MAX_IMAGE_DIMENSION, MAX_VIDEO_DIMENSION, even, resample_ready
from src.platform.media_tools.types import FilterFragment, InvalidToolParams, ToolMedia, ToolSpec

SPEC = dict(
    id="resize",
    label="Resize",
    icon="shrink",
    accepts=("image", "video"),
    produces="same",
    cardinality="one_to_one",
    editor="resize",
    name_suffix="-resize",
    params=[
        {"name": "width", "type": "integer", "label": "Width", "default": None, "min": 1, "max": MAX_IMAGE_DIMENSION, "unit": "px"},
        {"name": "height", "type": "integer", "label": "Height", "default": None, "min": 1, "max": MAX_IMAGE_DIMENSION, "unit": "px"},
        {
            "name": "quality",
            "type": "select",
            "label": "Quality",
            "default": "balanced",
            "options": [
                {"label": "High", "value": "high"},
                {"label": "Balanced", "value": "balanced"},
                {"label": "Small", "value": "small"},
            ],
        },
    ],
)


class Resize(MediaTool):
    def check_params(self, params: dict) -> None:
        if params.get("width") is None and params.get("height") is None:
            raise InvalidToolParams("width", "Give a width, a height, or both")

    @staticmethod
    def _target(
        params: dict, width: Optional[int], height: Optional[int], maximum: int
    ) -> tuple[Optional[int], Optional[int]]:
        target_width = params.get("width")
        target_height = params.get("height")
        if target_width is None and target_height is None:
            raise InvalidToolParams("width", "Give a width, a height, or both")
        target_width = None if target_width is None else int(target_width)
        target_height = None if target_height is None else int(target_height)
        if width and height:
            if target_width is None:
                target_width = max(1, round(width * (target_height / height)))
            elif target_height is None:
                target_height = max(1, round(height * (target_width / width)))
        if target_width is not None and target_width > maximum:
            raise InvalidToolParams("width", f"The target width ({target_width}px) exceeds the {maximum}px limit")
        if target_height is not None and target_height > maximum:
            raise InvalidToolParams("height", f"The target height ({target_height}px) exceeds the {maximum}px limit")
        return target_width, target_height

    def image(self, image: Image.Image, params: dict) -> Image.Image:
        width, height = self._target(params, *image.size, MAX_IMAGE_DIMENSION)
        return resample_ready(image).resize((width, height), Image.Resampling.LANCZOS)

    def video_filter(self, media: ToolMedia, params: dict) -> Optional[FilterFragment]:
        if media.kind != "video":
            return None
        width, height = self._target(params, media.width, media.height, MAX_VIDEO_DIMENSION)
        crf = CRF_BY_QUALITY.get(str(params.get("quality") or "balanced"), DEFAULT_CRF)
        target_width = None if width is None else even(width)
        target_height = None if height is None else even(height)
        if target_width is not None and target_width <= 0:
            raise InvalidToolParams("width", "The target size is too small")
        if target_height is not None and target_height <= 0:
            raise InvalidToolParams("height", "The target size is too small")
        scale = f"scale={target_width if target_width is not None else -2}:{target_height if target_height is not None else -2}"
        return FilterFragment(vf=[scale], width=target_width, height=target_height, crf=crf)


def spec() -> ToolSpec:
    return ToolSpec(**SPEC, impl=Resize())
