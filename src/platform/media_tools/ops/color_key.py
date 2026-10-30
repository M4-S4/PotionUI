from __future__ import annotations

import numpy as np
from PIL import Image

from src.platform.media_tools.base import MediaTool
from src.platform.imaging.pixels import (
    MAX_CHROMA_DISTANCE,
    border_ring_color,
    despill,
    feather_alpha,
    parse_hex_color,
    rgb_to_cbcr,
    soft_alpha,
)
from src.platform.media_tools.types import InvalidToolParams, ToolSpec

SPEC = dict(
    id="color_key",
    label="Color key",
    icon="keyc",
    accepts=("image",),
    produces="same",
    cardinality="one_to_one",
    preview=True,
    name_suffix="-keyed",
    params=[
        {
            "name": "key_mode",
            "type": "select",
            "label": "Key",
            "default": "auto",
            "options": [
                {"label": "Auto", "value": "auto"},
                {"label": "Color", "value": "color"},
            ],
        },
        {
            "name": "key_color",
            "type": "color",
            "label": "Key color",
            "default": "#00b140",
            "when": {"field": "key_mode", "equals": "color"},
        },
        {"name": "tolerance", "type": "float", "label": "Tolerance", "default": 25, "min": 0, "max": 100, "step": 1},
        {"name": "softness", "type": "float", "label": "Softness", "default": 10, "min": 0, "max": 100, "step": 1},
        {"name": "despill", "type": "checkbox", "label": "Despill", "default": True},
        {"name": "feather", "type": "float", "label": "Feather", "default": 0, "min": 0, "max": 16, "step": 0.5, "unit": "px"},
    ],
)


class ColorKey(MediaTool):
    def check_params(self, params: dict) -> None:
        if params.get("key_mode") == "color" and not params.get("key_color"):
            raise InvalidToolParams("key_color", "Key color is required when keying on a color")

    def image(self, image: Image.Image, params: dict) -> Image.Image:
        key_mode = str(params.get("key_mode") or "auto")
        tolerance = max(0.0, min(100.0, float(params.get("tolerance", 25.0))))
        softness = max(0.0, min(100.0, float(params.get("softness", 10.0))))
        despill_on = bool(params.get("despill", True))
        feather = float(params.get("feather") or 0.0)

        rgba = np.array(image.convert("RGBA"))
        rgb = rgba[..., :3]
        if key_mode == "color":
            key_color = params.get("key_color")
            if not key_color:
                raise InvalidToolParams("key_color", "Key color is required when keying on a color")
            key_rgb = parse_hex_color(key_color)
        else:
            key_rgb = border_ring_color(rgb)

        pixel_cbcr = rgb_to_cbcr(rgb)
        key_cbcr = rgb_to_cbcr(np.array(key_rgb, dtype=np.float64).reshape(1, 1, 3))[0, 0]
        distance = np.sqrt(np.sum((pixel_cbcr - key_cbcr) ** 2, axis=-1))

        threshold = (tolerance / 100.0) * MAX_CHROMA_DISTANCE
        ramp_half = (softness / 100.0) * (MAX_CHROMA_DISTANCE / 2.0)
        keyed = feather_alpha(soft_alpha(distance, threshold, ramp_half), feather)
        alpha = np.round(keyed.astype(np.float64) * rgba[..., 3] / 255.0).astype(np.uint8)

        out_rgb = despill(rgb, key_rgb) if despill_on else rgb
        return Image.fromarray(np.dstack([out_rgb, alpha]).astype(np.uint8))


def spec() -> ToolSpec:
    return ToolSpec(**SPEC, impl=ColorKey())
