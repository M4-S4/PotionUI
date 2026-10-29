from __future__ import annotations

from PIL import Image

from src.platform.media_tools.base import MediaTool
from src.platform.imaging.pixels import alpha_bbox
from src.platform.media_tools.types import ToolSpec

SPEC = dict(
    id="crop_subject",
    label="Crop to subject",
    icon="subject",
    accepts=("image",),
    produces="same",
    cardinality="one_to_one",
    requires={"has_alpha": True},
    name_suffix="-subject",
    params=[
        {"name": "margin", "type": "integer", "label": "Margin", "default": 0, "min": 0, "unit": "px"},
        {"name": "alpha_threshold", "type": "integer", "label": "Alpha threshold", "default": 16, "min": 0, "max": 255},
    ],
)


class CropSubject(MediaTool):
    def image(self, image: Image.Image, params: dict) -> Image.Image:
        margin = max(0, int(params.get("margin") or 0))
        threshold = int(params.get("alpha_threshold", 16))
        bbox = alpha_bbox(image, threshold)
        if bbox is None:
            return image
        rgba = image.convert("RGBA")
        width, height = rgba.size
        x, y, box_width, box_height = bbox
        return rgba.crop((
            max(0, x - margin),
            max(0, y - margin),
            min(width, x + box_width + margin),
            min(height, y + box_height + margin),
        ))


def spec() -> ToolSpec:
    return ToolSpec(**SPEC, impl=CropSubject())
