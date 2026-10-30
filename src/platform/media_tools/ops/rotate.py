from __future__ import annotations

from typing import Optional

from PIL import Image

from src.platform.media_tools.base import MediaTool
from src.platform.media_tools.types import FilterFragment, InvalidToolParams, ToolMedia, ToolSpec

SPEC = dict(
    id="rotate",
    label="Rotate",
    icon="rotate",
    accepts=("image", "video"),
    produces="same",
    cardinality="one_to_one",
    preview=True,
    name_suffix="-rotated",
    params=[
        {
            "name": "degrees",
            "type": "select",
            "label": "Degrees",
            "default": 90,
            "unit": "deg",
            "options": [
                {"label": "90", "value": 90},
                {"label": "180", "value": 180},
                {"label": "270", "value": 270},
            ],
        },
    ],
)


CLOCKWISE_TRANSPOSE = {
    90: Image.Transpose.ROTATE_270,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_90,
}

CLOCKWISE_TRANSPOSE_FILTER = {
    90: ["transpose=1"],
    180: ["transpose=1", "transpose=1"],
    270: ["transpose=2"],
}


class Rotate(MediaTool):
    @staticmethod
    def _degrees(params: dict) -> int:
        degrees = int(params.get("degrees") or 90)
        if degrees not in CLOCKWISE_TRANSPOSE:
            raise InvalidToolParams("degrees", "Degrees must be 90, 180 or 270")
        return degrees

    def image(self, image: Image.Image, params: dict) -> Image.Image:
        return image.transpose(CLOCKWISE_TRANSPOSE[self._degrees(params)])

    def video_filter(self, media: ToolMedia, params: dict) -> Optional[FilterFragment]:
        if media.kind != "video":
            return None
        degrees = self._degrees(params)
        swapped = degrees in (90, 270)
        return FilterFragment(
            vf=list(CLOCKWISE_TRANSPOSE_FILTER[degrees]),
            width=media.height if swapped else None,
            height=media.width if swapped else None,
        )


def spec() -> ToolSpec:
    return ToolSpec(**SPEC, impl=Rotate())
