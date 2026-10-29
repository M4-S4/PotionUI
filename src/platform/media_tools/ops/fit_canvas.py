from __future__ import annotations

from PIL import Image

from src.platform.media_tools.base import MediaTool
from src.platform.media_tools.ops._imaging import MAX_IMAGE_DIMENSION, parse_hex_color, resample_ready
from src.platform.media_tools.probe import has_alpha_mode
from src.platform.media_tools.types import InvalidToolParams, ToolSpec

SPEC = dict(
    id="fit_canvas",
    label="Fit canvas",
    icon="fit",
    accepts=("image",),
    produces="same",
    cardinality="one_to_one",
    preview=True,
    name_suffix="-canvas",
    params=[
        {"name": "width", "type": "integer", "label": "Width", "default": 1024, "min": 1, "unit": "px"},
        {"name": "height", "type": "integer", "label": "Height", "default": 1024, "min": 1, "unit": "px"},
        {"name": "scale_percent", "type": "float", "label": "Scale", "default": 100, "min": 1, "max": 400, "step": 1, "unit": "%"},
        {"name": "anchor", "type": "anchor", "label": "Anchor", "default": "center"},
        {"name": "fill", "type": "color", "label": "Fill", "default": "transparent", "allow_transparent": True},
    ],
)


TRANSPARENT = "transparent"


def anchor_fractions(anchor: str) -> tuple[float, float]:
    parts = str(anchor or "center").strip().lower().replace("-", "_").split("_")
    horizontal = 0.0 if "left" in parts else 1.0 if "right" in parts else 0.5
    vertical = 0.0 if "top" in parts else 1.0 if "bottom" in parts else 0.5
    return horizontal, vertical


def scaled_size(
    width: int, height: int, canvas_width: int, canvas_height: int, scale_percent: float
) -> tuple[int, int]:
    canvas_short = min(canvas_width, canvas_height)
    factor = canvas_short * (scale_percent / 100.0) / max(width, height)
    return max(1, round(width * factor)), max(1, round(height * factor))


class FitCanvas(MediaTool):
    def image(self, image: Image.Image, params: dict) -> Image.Image:
        width, height = int(params["width"]), int(params["height"])
        if width > MAX_IMAGE_DIMENSION:
            raise InvalidToolParams("width", f"The canvas width ({width}px) exceeds the {MAX_IMAGE_DIMENSION}px limit")
        if height > MAX_IMAGE_DIMENSION:
            raise InvalidToolParams("height", f"The canvas height ({height}px) exceeds the {MAX_IMAGE_DIMENSION}px limit")

        fill = str(params.get("fill") or TRANSPARENT)
        color = None if fill.lower() == TRANSPARENT else parse_hex_color(fill)
        horizontal, vertical = anchor_fractions(params.get("anchor") or "center")
        scale_percent = float(params.get("scale_percent") or 100.0)

        new_width, new_height = scaled_size(*image.size, width, height, scale_percent)
        if max(new_width, new_height) > MAX_IMAGE_DIMENSION:
            raise InvalidToolParams(
                "scale_percent",
                f"The scaled image ({new_width}x{new_height}) would exceed the {MAX_IMAGE_DIMENSION}px limit",
            )
        resized = resample_ready(image).resize((new_width, new_height), Image.Resampling.LANCZOS)
        x = round((width - new_width) * horizontal)
        y = round((height - new_height) * vertical)

        if color is None:
            canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            canvas.paste(resized.convert("RGBA"), (x, y))
            return canvas

        canvas = Image.new("RGB", (width, height), color)
        if has_alpha_mode(resized):
            rgba = resized.convert("RGBA")
            canvas.paste(rgba, (x, y), rgba)
        else:
            canvas.paste(resized.convert("RGB"), (x, y))
        return canvas


def spec() -> ToolSpec:
    return ToolSpec(**SPEC, impl=FitCanvas())
