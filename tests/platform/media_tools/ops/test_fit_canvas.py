from __future__ import annotations

import pytest
from PIL import Image

from src.platform.media_tools import InvalidToolParams
from tests.platform.media_tools.media_factory import make_image
from tests.platform.media_tools.ops.geometry_support import resolved, run_one, tool_of

RED = (200, 0, 0, 255)
CLEAR = (0, 0, 0, 0)


def fit(image: Image.Image, **params) -> Image.Image:
    return tool_of("fit_canvas").image(image, resolved("fit_canvas", **params))


def solid(size=(40, 20), color=(200, 0, 0)) -> Image.Image:
    return Image.new("RGB", size, color)


def test_default_scale_fills_the_canvas_short_side_with_the_long_side():
    out = fit(solid(), width=100, height=60)
    assert out.size == (100, 60)
    assert out.mode == "RGBA"
    assert out.getpixel((20, 15)) == RED
    assert out.getpixel((79, 44)) == RED
    assert out.getpixel((19, 15)) == CLEAR
    assert out.getpixel((80, 15)) == CLEAR
    assert out.getpixel((50, 14)) == CLEAR
    assert out.getpixel((50, 45)) == CLEAR


def test_scale_percent_is_a_share_of_the_canvas_short_side():
    out = fit(solid(), width=100, height=60, scale_percent=50, anchor="top_left")
    assert out.getpixel((29, 14)) == RED
    assert out.getpixel((30, 14)) == CLEAR
    assert out.getpixel((29, 15)) == CLEAR


@pytest.mark.parametrize(
    "anchor, inside, outside",
    [
        ("top_left", (0, 0), (30, 0)),
        ("top_right", (99, 0), (69, 0)),
        ("bottom_left", (0, 59), (0, 44)),
        ("bottom_right", (99, 59), (69, 59)),
        ("top", (50, 0), (34, 0)),
        ("bottom", (50, 59), (50, 44)),
        ("left", (0, 30), (30, 30)),
        ("right", (99, 30), (69, 30)),
    ],
)
def test_anchor_places_the_image_on_the_canvas(anchor, inside, outside):
    out = fit(solid(), width=100, height=60, scale_percent=50, anchor=anchor)
    assert out.getpixel(inside) == RED
    assert out.getpixel(outside) == CLEAR


def test_anchor_accepts_hyphens():
    out = fit(solid(), width=100, height=60, scale_percent=50, anchor="bottom-right")
    assert out.getpixel((99, 59)) == RED


def test_scale_beyond_the_canvas_is_clipped_not_refused():
    out = fit(solid((20, 20)), width=40, height=40, scale_percent=200)
    assert out.size == (40, 40)
    assert out.getpixel((0, 0)) == RED
    assert out.getpixel((39, 39)) == RED


def test_transparent_fill_carries_semitransparent_pixels_through_unchanged():
    source = Image.new("RGBA", (40, 40), (10, 20, 30, 128))
    out = fit(source, width=40, height=40)
    assert out.getpixel((0, 0)) == (10, 20, 30, 128)
    assert out.getpixel((39, 39)) == (10, 20, 30, 128)


def test_hex_fill_gives_an_opaque_rgb_canvas():
    out = fit(solid(), width=100, height=60, fill="#112233")
    assert out.mode == "RGB"
    assert out.getpixel((0, 0)) == (0x11, 0x22, 0x33)
    assert out.getpixel((50, 30)) == (200, 0, 0)


def test_hex_fill_composites_a_semitransparent_source():
    source = Image.new("RGBA", (20, 20), (255, 0, 0, 128))
    out = fit(source, width=20, height=20, fill="#0000ff")
    red, green, blue = out.getpixel((10, 10))
    assert abs(red - 128) <= 2 and green == 0 and abs(blue - 127) <= 2


def test_canvas_over_the_limit_names_the_field():
    with pytest.raises(InvalidToolParams) as caught:
        fit(solid(), width=9000, height=100)
    assert caught.value.field == "width"
    with pytest.raises(InvalidToolParams) as caught:
        fit(solid(), width=100, height=9000)
    assert caught.value.field == "height"


def test_scaled_image_over_the_limit_names_scale_percent():
    with pytest.raises(InvalidToolParams) as caught:
        fit(solid(), width=8000, height=8000, scale_percent=200)
    assert caught.value.field == "scale_percent"


@pytest.mark.parametrize(
    "params, field",
    [
        ({"fill": "banana"}, "fill"),
        ({"anchor": "middle"}, "anchor"),
        ({"scale_percent": 0}, "scale_percent"),
        ({"scale_percent": 401}, "scale_percent"),
        ({"width": 0}, "width"),
    ],
)
def test_invalid_params_name_the_field(params, field):
    with pytest.raises(InvalidToolParams) as caught:
        resolved("fit_canvas", **params)
    assert caught.value.field == field


def test_defaults_are_a_square_centered_transparent_canvas():
    params = resolved("fit_canvas")
    assert (params["width"], params["height"]) == (1024, 1024)
    assert params["anchor"] == "center"
    assert params["fill"] == "transparent"
    assert params["scale_percent"] == 100


def test_a_jpeg_with_transparent_fill_is_saved_as_png(tmp_path, ctx):
    path = tmp_path / "photo.jpg"
    make_image(path, (40, 20))
    out = run_one("fit_canvas", {"width": 100, "height": 60}, path, ctx)
    assert out.path.suffix == ".png"
    assert out.has_alpha is True
    assert (out.width, out.height) == (100, 60)


def test_a_jpeg_with_a_hex_fill_stays_a_jpeg(tmp_path, ctx):
    path = tmp_path / "photo.jpg"
    make_image(path, (40, 20))
    out = run_one("fit_canvas", {"width": 100, "height": 60, "fill": "#ffffff"}, path, ctx)
    assert out.path.suffix == ".jpg"
    assert out.has_alpha is False


def test_palette_source_is_converted_before_scaling():
    source = Image.new("P", (10, 10), 0)
    source.putpalette([255, 0, 0] * 256)
    out = fit(source, width=20, height=20)
    assert out.getpixel((10, 10)) == (255, 0, 0, 255)
