import numpy as np
from PIL import Image

from src.platform.imaging.pixels import alpha_bbox, feather_alpha


def _rgba(arr):
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def test_alpha_bbox_finds_opaque_rect():
    arr = np.zeros((20, 30, 4), dtype=np.uint8)
    arr[5:15, 10:20, 3] = 255
    bbox = alpha_bbox(_rgba(arr), threshold=16)
    assert bbox == (10, 5, 10, 10)


def test_alpha_bbox_empty_image_returns_none():
    arr = np.zeros((20, 30, 4), dtype=np.uint8)
    assert alpha_bbox(_rgba(arr), threshold=16) is None


def test_alpha_bbox_threshold_is_exclusive():
    arr = np.zeros((10, 10, 4), dtype=np.uint8)
    arr[..., 3] = 10
    assert alpha_bbox(_rgba(arr), threshold=16) is None

    arr[..., 3] = 17
    assert alpha_bbox(_rgba(arr), threshold=16) == (0, 0, 10, 10)
    assert alpha_bbox(_rgba(arr), threshold=17) is None


def test_feather_zero_is_identity():
    alpha = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    assert np.array_equal(feather_alpha(alpha, 0.0), alpha)


def test_feather_softens_a_hard_edge():
    alpha = np.zeros((20, 20), dtype=np.uint8)
    alpha[:, 10:] = 255
    blurred = feather_alpha(alpha, 4.0)

    edge_strip = blurred[10, 6:14]
    assert any(0 < v < 255 for v in edge_strip)
    assert blurred[10, 0] == 0
    assert blurred[10, 19] == 255
