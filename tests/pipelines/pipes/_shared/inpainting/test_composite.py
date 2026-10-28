import numpy as np

from src.pipelines.pipes._shared.inpainting.mask_utils import (
    composite_region,
    dilate_box,
    feather_alpha,
    match_ring_statistics,
)


def _square(size, lo, hi):
    region = np.zeros((size, size), dtype=bool)
    region[lo:hi, lo:hi] = True
    return region


def test_dilate_box_grows_by_chebyshev_radius():
    region = np.zeros((21, 21), dtype=bool)
    region[10, 10] = True
    grown = dilate_box(region, 3)
    assert grown.sum() == 49
    assert grown[7:14, 7:14].all()
    assert not grown[6, 10] and not grown[10, 14]


def test_dilate_box_zero_radius_is_identity():
    region = _square(20, 5, 9)
    np.testing.assert_array_equal(dilate_box(region, 0), region)


def test_dilate_box_is_clamped_at_the_border():
    region = np.zeros((10, 10), dtype=bool)
    region[0, 0] = True
    grown = dilate_box(region, 2)
    assert grown.sum() == 9


def test_feather_alpha_is_one_inside_zero_outside_and_ramps_between():
    region = _square(200, 80, 120)
    alpha = feather_alpha(region, 12)
    assert (alpha[region] == 1.0).all()
    assert (alpha[~dilate_box(region, 12)] == 0.0).all()
    ring = dilate_box(region, 12) & ~region
    assert 0.0 < alpha[ring].mean() < 1.0
    row = alpha[100, 120:140]
    assert (np.diff(row) <= 1e-6).all()


def _ramp_width(alpha_row):
    return int(((alpha_row > 0) & (alpha_row < 1)).sum())


def test_feather_width_scales_with_the_requested_pixels():
    region = _square(400, 150, 250)
    narrow = _ramp_width(feather_alpha(region, 6)[200])
    wide = _ramp_width(feather_alpha(region, 24)[200])
    assert wide > 3 * narrow


def test_feather_alpha_zero_width_is_the_hard_mask():
    region = _square(50, 10, 20)
    np.testing.assert_array_equal(feather_alpha(region, 0), region.astype(np.float32))


def _scene(size=240):
    rng = np.random.default_rng(7)
    original = rng.integers(60, 180, (size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[90:150, 90:150] = 255
    return original, mask


def test_pixels_outside_the_feathered_region_are_byte_identical():
    original, mask = _scene()
    content = np.full_like(original, 250)
    result = composite_region(original, content, mask, 0.1, False)
    outside = ~dilate_box(mask > 0, 6)
    np.testing.assert_array_equal(result[outside], original[outside])
    assert (result[mask > 0] == 250).all()


def test_outside_identical_with_colour_match_on():
    original, mask = _scene()
    content = np.clip(original.astype(np.int32) + 40, 0, 255).astype(np.uint8)
    result = composite_region(original, content, mask, 0.1, True)
    outside = ~dilate_box(mask > 0, 6)
    np.testing.assert_array_equal(result[outside], original[outside])


def test_feather_pixels_scale_with_region_size():
    original = np.zeros((600, 600, 3), dtype=np.uint8)
    content = np.full_like(original, 255)
    small = np.zeros((600, 600), dtype=np.uint8)
    small[280:320, 280:320] = 255
    large = np.zeros((600, 600), dtype=np.uint8)
    large[100:500, 100:500] = 255
    small_out = composite_region(original, content, small, 0.1, False)[300]
    large_out = composite_region(original, content, large, 0.1, False)[300]

    def partial(row):
        return int(((row[:, 0] > 0) & (row[:, 0] < 255)).sum())

    assert partial(large_out) > 4 * partial(small_out)


def test_colour_match_moves_region_statistics_toward_the_ring():
    original, mask = _scene()
    content = original.copy()
    shifted = np.clip(content.astype(np.int32) + 50, 0, 255).astype(np.uint8)
    region = mask > 0
    plain = composite_region(original, shifted, mask, 0.1, False)
    matched = composite_region(original, shifted, mask, 0.1, True)
    target = original[~dilate_box(region, 6) & dilate_box(region, 40)].mean()
    assert abs(matched[region].mean() - target) < abs(plain[region].mean() - target) / 3


def test_match_ring_statistics_is_identity_when_the_ring_is_tiny():
    content = np.full((10, 10, 3), 200, dtype=np.float32)
    original = np.zeros((10, 10, 3), dtype=np.float32)
    region = np.ones((10, 10), dtype=bool)
    np.testing.assert_array_equal(match_ring_statistics(content, original, region, 2, 6), content)


def test_match_ring_statistics_matches_ring_mean_and_std():
    rng = np.random.default_rng(1)
    original = rng.normal(100, 20, (200, 200, 3)).astype(np.float32)
    content = original * 1.1 + 30
    region = _square(200, 80, 120)
    out = match_ring_statistics(content, original, region, 8, 40)
    ring = dilate_box(region, 40) & ~dilate_box(region, 8)
    np.testing.assert_allclose(out[ring].mean(axis=0), original[ring].mean(axis=0), atol=0.5)


def test_empty_mask_returns_the_original():
    original, _ = _scene()
    content = np.full_like(original, 9)
    result = composite_region(original, content, np.zeros(original.shape[:2], dtype=np.uint8), 0.1, True)
    np.testing.assert_array_equal(result, original)
