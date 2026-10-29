from __future__ import annotations

from typing import Optional

from PIL import Image

from src.platform.media_tools.base import MediaTool
from src.platform.media_tools.ops._imaging import even
from src.platform.media_tools.types import FilterFragment, InvalidToolParams, ToolMedia, ToolSpec

SPEC = dict(
    id="crop",
    label="Crop",
    icon="crop",
    accepts=("image", "video"),
    produces="same",
    cardinality="one_to_one",
    editor="crop",
    name_suffix="-crop",
    params=[
        {"name": "x", "type": "integer", "label": "X", "default": 0, "min": 0, "unit": "px"},
        {"name": "y", "type": "integer", "label": "Y", "default": 0, "min": 0, "unit": "px"},
        {"name": "width", "type": "integer", "label": "Width", "min": 1, "unit": "px"},
        {"name": "height", "type": "integer", "label": "Height", "min": 1, "unit": "px"},
    ],
)


class Crop(MediaTool):
    @staticmethod
    def _box(params: dict, width: Optional[int], height: Optional[int]) -> tuple[int, int, int, int]:
        x, y = int(params.get("x") or 0), int(params.get("y") or 0)
        crop_width, crop_height = int(params["width"]), int(params["height"])
        if width is not None and height is not None:
            if x >= width:
                raise InvalidToolParams("x", f"The crop starts at {x}px, outside the {width}px width")
            if y >= height:
                raise InvalidToolParams("y", f"The crop starts at {y}px, outside the {height}px height")
            if x + crop_width > width:
                raise InvalidToolParams(
                    "width", f"The crop rectangle ({x},{y} {crop_width}x{crop_height}) does not fit inside {width}x{height}"
                )
            if y + crop_height > height:
                raise InvalidToolParams(
                    "height", f"The crop rectangle ({x},{y} {crop_width}x{crop_height}) does not fit inside {width}x{height}"
                )
        return x, y, crop_width, crop_height

    def image(self, image: Image.Image, params: dict) -> Image.Image:
        x, y, width, height = self._box(params, *image.size)
        return image.crop((x, y, x + width, y + height))

    def video_filter(self, media: ToolMedia, params: dict) -> Optional[FilterFragment]:
        if media.kind != "video":
            return None
        x, y, width, height = self._box(params, media.width, media.height)
        crop_width, crop_height = even(width), even(height)
        if crop_width <= 0 or crop_height <= 0:
            raise InvalidToolParams("width" if crop_width <= 0 else "height", "The crop rectangle is too small")
        return FilterFragment(
            vf=[f"crop={crop_width}:{crop_height}:{even(x)}:{even(y)}"],
            width=crop_width,
            height=crop_height,
        )


def spec() -> ToolSpec:
    return ToolSpec(**SPEC, impl=Crop())
