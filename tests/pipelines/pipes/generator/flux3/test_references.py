from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from src.pipelines.pipes.generator.flux3.references import (
    GRID,
    MAX_SIDE,
    MIN_SIDE,
    prepare_multi_reference,
    prepare_reference,
    reference_canvas,
)


def _gradient(width, height):
    x = np.tile(np.arange(width, dtype=np.uint8) % 251, (height, 1))
    y = np.tile((np.arange(height, dtype=np.uint8) % 251)[:, None], (1, width))
    return Image.fromarray(np.stack([x, y, (x // 2 + y // 2).astype(np.uint8)], axis=-1))


def test_canvas_rounds_up_to_the_grid_without_resizing_small_images():
    fit, canvas = reference_canvas(1000, 1000)
    assert fit == (1000, 1000)
    assert canvas == (1008, 1008)


def test_canvas_never_upscales():
    fit, canvas = reference_canvas(100, 70)
    assert fit == (100, 70)
    assert canvas == (112, 80)


def test_canvas_floors_each_side_at_the_minimum():
    fit, canvas = reference_canvas(20, 20)
    assert fit == (20, 20)
    assert canvas == (MIN_SIDE, MIN_SIDE)


@pytest.mark.parametrize("size", [(4000, 3000), (3000, 4000), (2048, 2048), (9000, 600), (600, 9000)])
def test_canvas_stays_inside_the_pixel_and_side_budget(size):
    fit, canvas = reference_canvas(*size)
    assert canvas[0] * canvas[1] <= 4 * 1024 * 1024
    assert max(canvas) <= MAX_SIDE
    assert all(side % GRID == 0 for side in canvas)
    assert fit[0] <= canvas[0] and fit[1] <= canvas[1]


def test_canvas_keeps_the_aspect_when_fitting():
    fit, _ = reference_canvas(4000, 3000)
    assert abs(fit[0] / fit[1] - 4 / 3) < 0.01


def test_exact_multiple_is_not_padded():
    image, box = prepare_reference(_gradient(1008, 640))
    assert image.size == (1008, 640)
    assert box is None


def test_padded_reference_reports_the_box_of_the_original():
    source = _gradient(1000, 600)
    image, box = prepare_reference(source)
    assert image.size == (1008, 608)
    assert box == (4, 4, 1004, 604)
    assert np.array_equal(np.asarray(image.crop(box)), np.asarray(source))


def test_padding_reflects_without_repeating_the_edge():
    source = _gradient(1000, 600)
    image, box = prepare_reference(source)
    padded, original = np.asarray(image), np.asarray(source)
    assert np.array_equal(padded[box[1]:box[3], box[0] - 1], original[:, 1])
    assert np.array_equal(padded[box[1]:box[3], box[0] + 1000], original[:, 998])


def test_padding_falls_back_to_symmetric_for_a_tiny_side():
    source = _gradient(16, 30)
    image, box = prepare_reference(source)
    assert image.size == (64, 64)
    assert np.array_equal(np.asarray(image.crop(box)), np.asarray(source))
    assert np.array_equal(np.asarray(image)[box[1]:box[3], box[0] - 1], np.asarray(source)[:, 0])


def test_reference_is_converted_to_rgb():
    rgba = Image.new("RGBA", (64, 64), (10, 20, 30, 0))
    image, _ = prepare_reference(rgba)
    assert image.mode == "RGB"


def test_exif_orientation_is_applied():
    buffer = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (160, 96), (1, 2, 3)).save(buffer, format="JPEG", exif=exif)
    buffer.seek(0)
    image, _ = prepare_reference(Image.open(buffer))
    assert image.size == (96, 160)


def test_large_reference_is_fitted_under_the_budget():
    image, box = prepare_reference(Image.new("RGB", (4000, 3000), (9, 9, 9)))
    assert image.width * image.height <= 4 * 1024 * 1024
    assert box is not None


def test_multi_reference_is_fitted_under_one_megapixel_on_the_grid():
    image = prepare_multi_reference(_gradient(2000, 1000))
    assert image.width * image.height <= 1024 * 1024
    assert image.width % GRID == 0 and image.height % GRID == 0
    assert abs(image.width / image.height - 2.0) < 0.05


def test_multi_reference_never_upscales_and_centre_crops():
    source = _gradient(100, 70)
    image = prepare_multi_reference(source)
    assert image.size == (96, 64)
    assert np.array_equal(np.asarray(image), np.asarray(source)[3:67, 2:98])


def test_multi_reference_converts_to_rgb():
    assert prepare_multi_reference(Image.new("RGBA", (64, 64))).mode == "RGB"
