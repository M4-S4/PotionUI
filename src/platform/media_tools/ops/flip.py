from __future__ import annotations

from typing import Optional

from PIL import Image

from src.platform.media_tools.base import MediaTool
from src.platform.media_tools.types import FilterFragment, InvalidToolParams, ToolMedia, ToolSpec

SPEC = dict(
    id="flip",
    label="Flip",
    icon="flip",
    accepts=("image", "video"),
    produces="same",
    cardinality="one_to_one",
    preview=True,
    name_suffix="-flipped",
    params=[
        {
            "name": "axis",
            "type": "select",
            "label": "Axis",
            "default": "horizontal",
            "options": [
                {"label": "Horizontal", "value": "horizontal"},
                {"label": "Vertical", "value": "vertical"},
            ],
        },
    ],
)


class Flip(MediaTool):
    @staticmethod
    def _axis(params: dict) -> str:
        axis = str(params.get("axis") or "horizontal")
        if axis not in ("horizontal", "vertical"):
            raise InvalidToolParams("axis", "Axis must be horizontal or vertical")
        return axis

    def image(self, image: Image.Image, params: dict) -> Image.Image:
        if self._axis(params) == "horizontal":
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    def video_filter(self, media: ToolMedia, params: dict) -> Optional[FilterFragment]:
        if media.kind != "video":
            return None
        return FilterFragment(vf=["hflip" if self._axis(params) == "horizontal" else "vflip"])


def spec() -> ToolSpec:
    return ToolSpec(**SPEC, impl=Flip())
