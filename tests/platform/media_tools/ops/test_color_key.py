from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.platform.media_tools import InvalidToolParams
from tests.platform.media_tools.media_factory import make_image
from tests.platform.media_tools.ops.geometry_support import resolved, run_one, tool_of

GREEN = (0, 177, 64)
SHADOW_GREEN = (0, 88, 32)
RED = (200, 30, 30)


def key(image: Image.Image, **params) -> Image.Image:
    return tool_of("color_key").image(image, resolved("color_key", **params))


def green_screen(size=(20, 20), subject=RED, box=(6, 6, 14, 14)) -> Image.Image:
    image = Image.new("RGB", size, GREEN)
    image.paste(subject, box)
    return image


def test_auto_mode_keys_the_border_colour_and_keeps_the_subject():
    out = key(green_screen())
    assert out.mode == "RGBA"
    assert out.size == (20, 20)
    assert out.getpixel((0, 0))[3] == 0
    assert out.getpixel((19, 19))[3] == 0
    assert out.getpixel((10, 10))[3] == 255


def test_explicit_color_mode_keys_the_given_colour():
    source = Image.new("RGB", (20, 20), (255, 0, 255))
    source.paste((255, 255, 255), (6, 6, 14, 14))
    out = key(source, key_mode="color", key_color="#ff00ff")
    assert out.getpixel((0, 0))[3] == 0
    assert out.getpixel((10, 10))[3] == 255


def test_explicit_colour_differs_from_the_border_colour():
    out = key(green_screen(), key_mode="color", key_color="#ff0000")
    assert out.getpixel((0, 0))[3] == 255
    assert out.getpixel((10, 10))[3] == 0


def test_key_color_is_dropped_in_auto_mode():
    assert "key_color" not in resolved("color_key", key_color="#ff0000")
    assert resolved("color_key", key_mode="color")["key_color"] == "#00b140"


def test_key_color_is_required_by_the_tool_when_keying_on_a_colour():
    with pytest.raises(InvalidToolParams) as caught:
        tool_of("color_key").image(green_screen(), {"key_mode": "color"})
    assert caught.value.field == "key_color"


def test_chrominance_keys_an_unevenly_lit_screen_but_keeps_a_dark_subject():
    image = Image.new("RGB", (20, 20), GREEN)
    image.paste(SHADOW_GREEN, (14, 0, 20, 20))
    image.paste((80, 0, 0), (6, 6, 12, 14))
    out = np.array(key(image))
    assert (out[:6, :, 3] == 0).all()
    assert (out[:, 14:, 3] == 0).all()
    assert (out[6:14, 6:12, 3] == 255).all()


def test_despill_pulls_the_key_channel_down_on_kept_pixels():
    subject = (200, 170, 40)
    on = key(green_screen(subject=subject), despill=True)
    off = key(green_screen(subject=subject), despill=False)
    assert on.getpixel((10, 10))[3] == 255
    assert off.getpixel((10, 10))[:3] == subject
    assert on.getpixel((10, 10))[:3] == (200, 120, 40)


def test_hard_edge_has_only_binary_alpha_and_feather_softens_it():
    hard = np.array(key(green_screen(), softness=0, feather=0))[..., 3]
    soft = np.array(key(green_screen(), softness=0, feather=6))[..., 3]
    assert set(np.unique(hard)) <= {0, 255}
    assert ((soft > 0) & (soft < 255)).any()


def test_existing_alpha_of_the_source_is_replaced_by_the_key():
    source = green_screen().convert("RGBA")
    source.putpixel((10, 10), (200, 30, 30, 10))
    out = key(source)
    assert out.getpixel((10, 10))[3] == 255


def test_tolerance_widens_the_keyed_area():
    image = Image.new("RGB", (20, 20), GREEN)
    image.paste((0, 140, 70), (6, 6, 14, 14))
    narrow = np.array(key(image, tolerance=1, softness=0))[..., 3]
    wide = np.array(key(image, tolerance=40, softness=0))[..., 3]
    assert narrow[10, 10] == 255
    assert wide[10, 10] == 0


@pytest.mark.parametrize(
    "params, field",
    [
        ({"key_mode": "chroma"}, "key_mode"),
        ({"tolerance": 101}, "tolerance"),
        ({"softness": -1}, "softness"),
        ({"feather": 17}, "feather"),
        ({"key_mode": "color", "key_color": "banana"}, "key_color"),
        ({"despill": "maybe"}, "despill"),
    ],
)
def test_invalid_params_name_the_field(params, field):
    with pytest.raises(InvalidToolParams) as caught:
        resolved("color_key", **params)
    assert caught.value.field == field


def test_a_jpeg_input_comes_out_as_a_png_with_alpha(tmp_path, ctx):
    path = tmp_path / "screen.jpg"
    green_screen((30, 30), box=(10, 10, 20, 20)).save(path, quality=95)
    out = run_one("color_key", {}, path, ctx)
    assert out.path.suffix == ".png"
    assert out.has_alpha is True
    assert (out.width, out.height) == (30, 30)
    with Image.open(out.path) as saved:
        assert saved.mode == "RGBA"
        assert saved.getpixel((0, 0))[3] == 0
        assert saved.getpixel((15, 15))[3] == 255


def test_a_png_input_stays_a_png(tmp_path, ctx):
    path = tmp_path / "screen.png"
    make_image(path, (20, 20), "RGB", GREEN)
    out = run_one("color_key", {}, path, ctx)
    assert out.path.suffix == ".png"
