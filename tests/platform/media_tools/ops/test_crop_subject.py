from __future__ import annotations

import pytest
from PIL import Image

from src.platform.media_tools import InvalidToolParams, UnsupportedMedia
from tests.platform.media_tools.media_factory import make_image
from tests.platform.media_tools.ops.geometry_support import resolved, run_one, tool_of


def cutout(size=(40, 30), box=(10, 5, 20, 15), alpha=255) -> Image.Image:
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    image.paste((200, 30, 30, alpha), box)
    return image


def crop(image: Image.Image, **params) -> Image.Image:
    return tool_of("crop_subject").image(image, resolved("crop_subject", **params))


def test_crops_to_the_opaque_box():
    out = crop(cutout())
    assert out.size == (10, 10)
    assert out.mode == "RGBA"
    assert out.getpixel((0, 0)) == (200, 30, 30, 255)
    assert out.getpixel((9, 9)) == (200, 30, 30, 255)


def test_margin_pads_on_every_side():
    out = crop(cutout(), margin=3)
    assert out.size == (16, 16)
    assert out.getpixel((0, 0))[3] == 0
    assert out.getpixel((3, 3))[3] == 255


def test_margin_is_clamped_to_the_image():
    out = crop(cutout(box=(2, 1, 12, 11)), margin=5)
    assert out.size == (17, 16)
    assert crop(cutout(), margin=500).size == (40, 30)


def test_alpha_threshold_decides_what_counts_as_subject():
    image = cutout(box=(10, 5, 20, 15))
    image.paste((200, 30, 30, 10), (5, 2, 25, 20))
    image.paste((200, 30, 30, 255), (10, 5, 20, 15))
    assert crop(image, alpha_threshold=16).size == (10, 10)
    assert crop(image, alpha_threshold=5).size == (20, 18)


def test_an_image_with_no_subject_is_returned_unchanged():
    empty = Image.new("RGBA", (40, 30), (0, 0, 0, 0))
    out = crop(empty)
    assert out.size == (40, 30)
    assert out.mode == "RGBA"


def test_an_opaque_rgb_image_is_returned_whole_as_rgba():
    out = crop(Image.new("RGB", (12, 8), (1, 2, 3)))
    assert out.size == (12, 8)
    assert out.mode == "RGBA"


@pytest.mark.parametrize(
    "params, field",
    [({"margin": -1}, "margin"), ({"alpha_threshold": 256}, "alpha_threshold"), ({"alpha_threshold": -1}, "alpha_threshold")],
)
def test_invalid_params_name_the_field(params, field):
    with pytest.raises(InvalidToolParams) as caught:
        resolved("crop_subject", **params)
    assert caught.value.field == field


def test_runs_on_a_png_with_transparency(tmp_path, ctx):
    path = tmp_path / "cutout.png"
    cutout().save(path)
    out = run_one("crop_subject", {"margin": 1}, path, ctx)
    assert (out.width, out.height) == (12, 12)
    assert out.has_alpha is True


def test_refuses_an_image_without_transparency(tmp_path, ctx):
    path = tmp_path / "opaque.png"
    make_image(path, (20, 20))
    with pytest.raises(UnsupportedMedia):
        run_one("crop_subject", {}, path, ctx)
    assert ctx.commands == []
