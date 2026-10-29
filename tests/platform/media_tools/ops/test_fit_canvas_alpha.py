from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.platform.imaging.pixels import alpha_bbox
from src.platform.media_tools import validate_params
from src.platform.media_tools.ops import fit_canvas
from src.platform.media_tools.ops.fit_canvas import anchor_fractions, scaled_size


def fit(image, **params):
    spec = fit_canvas.spec()
    return spec.impl.image(image, validate_params(spec, params))


def rgba(pixels, size=(4, 4)):
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    loaded = image.load()
    for (x, y), value in pixels.items():
        loaded[x, y] = value
    return image


def test_a_transparent_canvas_clips_an_overflowing_axis_and_keeps_alpha_through_resampling():
    source = rgba({(x, y): (30, 200, 10, 90) for x in range(4) for y in range(2)}, size=(4, 2))
    width, height, scale, anchor = 6, 12, 150.0, "left"

    out = fit(source, width=width, height=height, scale_percent=scale, anchor=anchor, fill="transparent")

    new_width, new_height = scaled_size(4, 2, width, height, scale)
    horizontal, vertical = anchor_fractions(anchor)
    x = round((width - new_width) * horizontal)
    y = round((height - new_height) * vertical)
    assert new_width > width
    assert new_height < height
    expected = (
        max(x, 0), max(y, 0),
        min(x + new_width, width) - max(x, 0),
        min(y + new_height, height) - max(y, 0),
    )
    assert alpha_bbox(out, threshold=0) == expected
    inside = np.array(out)[expected[1] + 1, expected[0] + 1]
    assert inside[3] == pytest.approx(90, abs=3)
    assert tuple(inside[:3]) == pytest.approx((30, 200, 10), abs=3)
    assert tuple(np.array(out)[0, 0]) == (0, 0, 0, 0)


def test_a_transparent_canvas_keeps_alpha_when_the_image_is_scaled_down():
    source = rgba({(x, y): (30, 200, 10, 90) for x in range(8) for y in range(8)}, size=(8, 8))

    out = fit(source, width=20, height=20, scale_percent=50.0, anchor="center", fill="transparent")

    new_width, new_height = scaled_size(8, 8, 20, 20, 50.0)
    x0, y0, width, height = alpha_bbox(out, threshold=0)
    assert (width, height) == (new_width, new_height)
    center = np.array(out)[y0 + height // 2, x0 + width // 2]
    assert center[3] == pytest.approx(90, abs=3)
    assert tuple(center[:3]) == pytest.approx((30, 200, 10), abs=3)


def test_a_grayscale_image_with_alpha_lands_on_a_transparent_canvas_with_its_alpha():
    source = Image.new("LA", (4, 4), (150, 100))

    out = fit(source, width=4, height=4, scale_percent=100, anchor="center", fill="transparent")

    assert tuple(np.array(out)[0, 0]) == (150, 150, 150, 100)


def test_a_palette_image_with_a_transparent_index_keeps_that_pixel_transparent():
    palette_image = Image.new("P", (4, 4), 1)
    palette = [0] * (256 * 3)
    palette[3:6] = [200, 100, 50]
    palette_image.putpalette(palette)
    palette_image.info["transparency"] = 0
    palette_image.load()[0, 0] = 0

    out = fit(palette_image, width=4, height=4, scale_percent=100, anchor="center", fill="transparent")

    pixels = np.array(out)
    assert tuple(pixels[0, 0]) == (0, 0, 0, 0)
    assert tuple(pixels[0, 1]) == (200, 100, 50, 255)


def test_semitransparent_pixels_pass_through_exactly_when_nothing_is_resampled():
    source = rgba({
        (0, 0): (10, 20, 30, 0),
        (1, 0): (40, 50, 60, 64),
        (2, 0): (70, 80, 90, 128),
        (3, 0): (100, 110, 120, 255),
    })

    out = fit(source, width=4, height=4, scale_percent=100, anchor="center", fill="transparent")

    assert out.mode == "RGBA"
    assert np.array_equal(np.array(out), np.array(source))


def test_a_hex_fill_still_blends_partial_alpha_over_the_colour():
    source = rgba({(x, y): (255, 0, 0, 128) for x in range(4) for y in range(4)})

    out = fit(source, width=4, height=4, scale_percent=100, anchor="center", fill="#ffffff")

    assert out.mode == "RGB"
    assert tuple(np.array(out)[0, 0]) == (255, 127, 127)
