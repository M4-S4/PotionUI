import numpy as np
import pytest

from src.pipelines.pipes._shared.inpainting.region_utils import (
    mask_bbox,
    scaled_crop_box,
    snap_to_multiple,
    working_size,
)


def test_mask_bbox_is_exclusive_and_none_when_empty():
    mask = np.zeros((50, 60), dtype=np.uint8)
    assert mask_bbox(mask) is None
    mask[10:20, 30:45] = 255
    assert mask_bbox(mask) == (10, 20, 30, 45)


def test_context_scales_with_region_size():
    small = scaled_crop_box((4000, 4000), (1000, 1100, 1000, 1100), 0.5, 16)
    large = scaled_crop_box((4000, 4000), (1000, 1800, 1000, 1800), 0.5, 16)
    assert (small[1] - small[0], small[3] - small[2]) == (200, 200)
    assert (large[1] - large[0], large[3] - large[2]) == (1600, 1600)


def test_context_zero_still_pads_one_pixel():
    top, bottom, left, right = scaled_crop_box((500, 500), (100, 200, 100, 200), 0.0, 16)
    assert (top, bottom, left, right) == (99, 201, 99, 201)


def test_min_side_grows_a_tiny_region_symmetrically():
    top, bottom, left, right = scaled_crop_box((1000, 1000), (500, 510, 500, 510), 0.5, 256)
    assert bottom - top == 256 and right - left == 256
    assert top <= 500 and bottom >= 510


def test_crop_box_shifts_inside_the_image_instead_of_shrinking():
    top, bottom, left, right = scaled_crop_box((1000, 1000), (0, 10, 0, 10), 0.5, 256)
    assert (top, left) == (0, 0)
    assert bottom == 256 and right == 256


def test_crop_box_clamps_to_a_small_image():
    box = scaled_crop_box((100, 120), (40, 60, 40, 60), 0.5, 256)
    assert box == (0, 100, 0, 120)


def test_missing_bbox_returns_the_whole_image():
    assert scaled_crop_box((70, 90), None, 0.5, 256) == (0, 70, 0, 90)


def test_region_always_inside_the_box():
    rng = np.random.default_rng(3)
    for _ in range(200):
        h, w = rng.integers(64, 2000, 2)
        a = rng.integers(0, h - 8)
        c = rng.integers(0, w - 8)
        b = rng.integers(a + 1, min(h, a + 600) + 1)
        d = rng.integers(c + 1, min(w, c + 600) + 1)
        top, bottom, left, right = scaled_crop_box((h, w), (a, b, c, d), 0.5, 128)
        assert 0 <= top <= a and b <= bottom <= h
        assert 0 <= left <= c and d <= right <= w


@pytest.mark.parametrize("value,multiple,expected", [(1000, 16, 1008), (1016, 16, 1024), (3, 16, 16), (24, 16, 32), (100, 1, 100)])
def test_snap_to_multiple_rounds_to_nearest_with_a_floor(value, multiple, expected):
    assert snap_to_multiple(value, multiple) == expected


def test_working_size_upscales_a_small_crop_to_the_target_area():
    h, w = working_size(200, 300, 1024 * 1024, 16)
    assert h % 16 == 0 and w % 16 == 0
    assert abs(h * w - 1024 * 1024) / (1024 * 1024) < 0.06
    assert h < w
    assert h > 200 and w > 300


def test_working_size_downscales_a_large_crop():
    h, w = working_size(3000, 2000, 1024 * 1024, 16)
    assert h * w < 1.1 * 1024 * 1024
    assert h > w


def test_working_size_keeps_aspect_within_one_step():
    h, w = working_size(777, 1333, 1024 * 1024, 16)
    assert abs(w / h - 1333 / 777) < 0.05
