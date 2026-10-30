"""Tests for the shared alpha helpers (`_shared.imaging.alpha`): matte-strength
smoothstep, flattening and opaque-alpha dropping."""

import numpy as np
import pytest
from PIL import Image

from src.pipelines.pipes._shared.imaging.alpha import (
    apply_matte_strength,
    drop_opaque_alpha,
    flatten_onto,
)


def _rgba(arr):
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


# -- apply_matte_strength -------------------------------------------------

def test_matte_strength_zero_is_identity():
    alpha = np.array([0, 50, 128, 200, 255], dtype=np.uint8)
    assert np.array_equal(apply_matte_strength(alpha, 0), alpha)


def test_matte_strength_hundred_is_hard_threshold_at_128():
    alpha = np.array([0, 100, 127, 128, 129, 200, 255], dtype=np.uint8)
    result = apply_matte_strength(alpha, 100)
    assert list(result) == [0, 0, 0, 255, 255, 255, 255]


def test_matte_strength_commits_mid_grey_pixels():
    """The failure this exists to fix: a matting model that 'removed
    nothing' leaves the whole background at a mid-grey like 140. At
    strength=0 that pixel stays background-colored garbage (140); a
    sufficiently high strength must commit it toward fully opaque since
    140 > 128."""
    mostly_opaque_background = np.full((4, 4), 140, dtype=np.uint8)
    identity = apply_matte_strength(mostly_opaque_background, 0)
    assert np.all(identity == 140)

    committed = apply_matte_strength(mostly_opaque_background, 100)
    assert np.all(committed == 255)


def test_matte_strength_monotonic_in_strength_for_a_fixed_input():
    alpha = np.array([140], dtype=np.uint8)
    values = [int(apply_matte_strength(alpha, s)[0]) for s in (0, 25, 50, 75, 100)]
    assert values == sorted(values)


def test_flatten_onto_turns_transparent_pixels_into_the_background():
    image = Image.new("RGBA", (2, 1), (255, 0, 0, 0))
    image.putpixel((1, 0), (0, 0, 255, 255))

    flat = flatten_onto(image)

    assert flat.mode == "RGB"
    assert flat.getpixel((0, 0)) == (255, 255, 255)
    assert flat.getpixel((1, 0)) == (0, 0, 255)


def test_flatten_onto_blends_partial_alpha_with_the_background():
    flat = flatten_onto(Image.new("RGBA", (1, 1), (0, 0, 0, 128)))

    assert all(126 <= c <= 128 for c in flat.getpixel((0, 0)))


def test_flatten_onto_leaves_opaque_rgb_untouched():
    image = Image.new("RGB", (1, 1), (10, 20, 30))

    assert flatten_onto(image).getpixel((0, 0)) == (10, 20, 30)


@pytest.mark.parametrize("mode", ["RGB", "L", "P", "CMYK", "RGBA", "LA"])
def test_flatten_onto_is_byte_identical_to_an_rgb_convert_for_opaque_images(mode):
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)).convert(mode)

    assert flatten_onto(image).tobytes() == image.convert("RGB").tobytes()


def test_flatten_onto_whitens_a_fully_transparent_image_whatever_rgb_it_hides():
    hidden = np.zeros((24, 32, 4), dtype=np.uint8)
    hidden[..., 0] = 255

    assert np.array_equal(np.array(flatten_onto(_rgba(hidden))), np.full((24, 32, 3), 255, dtype=np.uint8))


def test_flatten_onto_honours_palette_transparency():
    image = Image.new("P", (1, 1), 0)
    image.putpalette([200, 0, 0] + [0, 0, 0] * 255)
    image.info["transparency"] = 0

    assert flatten_onto(image).getpixel((0, 0)) == (255, 255, 255)



def test_drop_opaque_alpha_converts_fully_opaque_rgba_to_rgb():
    image = Image.new("RGBA", (2, 2), (10, 20, 30, 255))

    result = drop_opaque_alpha(image)

    assert result.mode == "RGB"
    assert result.getpixel((0, 0)) == (10, 20, 30)


def test_drop_opaque_alpha_keeps_rgba_with_any_transparent_pixel():
    image = Image.new("RGBA", (2, 2), (10, 20, 30, 255))
    image.putpixel((1, 1), (10, 20, 30, 254))

    result = drop_opaque_alpha(image)

    assert result.mode == "RGBA"
    assert result.getpixel((1, 1)) == (10, 20, 30, 254)


def test_drop_opaque_alpha_leaves_rgb_untouched():
    image = Image.new("RGB", (1, 1), (5, 6, 7))

    assert drop_opaque_alpha(image).getpixel((0, 0)) == (5, 6, 7)

