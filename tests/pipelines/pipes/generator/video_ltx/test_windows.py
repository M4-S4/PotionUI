from __future__ import annotations

import numpy as np
import pytest

from src.pipelines.pipes.generator.video_ltx.windows import (
    crossfade_windows,
    largest_fitting_frames,
    min_window_frames,
    plan_windows,
)


@pytest.mark.parametrize(("total", "cap", "overlap", "length", "starts"), [
    (136, 121, 24, 81, [0, 55]),
    (125, 121, 24, 81, [0, 44]),
    (40, 17, 8, 17, [0, 8, 15, 23]),
    (50, 17, 0, 17, [0, 16, 33]),
    (1001, 121, 24, 113, [0, 89, 178, 266, 355, 444, 533, 622, 710, 799, 888]),
    (300, 97, 48, 97, [0, 41, 81, 122, 162, 203]),
])
def test_balanced_windows_use_the_fewest_equal_windows_of_the_smallest_valid_length(total, cap, overlap, length, starts):
    assert plan_windows(total, cap, overlap) == (length, starts)


@pytest.mark.parametrize(("total", "cap", "overlap"), [
    (total, cap, overlap)
    for total in (18, 40, 122, 136, 250, 499, 1001)
    for cap in (17, 49, 81, 121)
    for overlap in (0, 8, 16, 24)
    if cap >= min_window_frames(overlap)
])
def test_every_plan_covers_the_clip_with_at_least_the_overlap_using_equal_valid_windows(total, cap, overlap):
    length, starts = plan_windows(total, cap, overlap)
    assert (length - 1) % 8 == 0 and length <= cap
    if total <= cap:
        assert starts == [0] and length <= total < length + 8
        return
    assert starts[0] == 0 and starts[-1] + length == total
    assert all(b - a <= length - overlap for a, b in zip(starts, starts[1:]))
    assert all(b > a for a, b in zip(starts, starts[1:]))
    count = len(starts)
    one_fewer = -(-(total + (count - 2) * overlap) // (count - 1))
    assert count == 2 or -(-(one_fewer - 1) // 8) * 8 + 1 > cap
    assert (length - 8) * count - (count - 1) * overlap < total


def test_a_clip_that_fits_one_window_drops_only_the_tail_that_is_not_a_valid_length():
    assert plan_windows(136, 137, 24) == (129, [0])
    assert plan_windows(129, 1001, 24) == (129, [0])


@pytest.mark.parametrize(("overlap", "expected"), [(0, 9), (4, 9), (8, 17), (16, 33), (24, 49), (48, 97)])
def test_the_smallest_window_advances_by_at_least_the_overlap(overlap, expected):
    assert min_window_frames(overlap) == expected


def test_largest_fitting_frames_walks_down_valid_lengths_to_the_floor():
    seen = []

    def fits(frames):
        seen.append(frames)
        return frames <= 50

    assert largest_fitting_frames(136, fits) == 49
    assert seen == [129, 121, 113, 105, 97, 89, 81, 73, 65, 57, 49]
    assert largest_fitting_frames(136, lambda f: f <= 40, floor=49) is None
    assert largest_fitting_frames(136, lambda f: True, floor=49) == 129


def test_crossfade_ramps_across_the_overlap_and_keeps_the_rest():
    a = np.zeros((5, 1, 1, 3), dtype=np.uint8)
    b = np.full((5, 1, 1, 3), 200, dtype=np.uint8)
    out = crossfade_windows([(0, a), (3, b)], 8)
    assert out.dtype == np.uint8 and out.shape == (8, 1, 1, 3)
    assert out[:3, 0, 0, 0].tolist() == [0, 0, 0]
    assert out[3:5, 0, 0, 0].tolist() == [67, 133]
    assert out[5:, 0, 0, 0].tolist() == [200, 200, 200]


def test_crossfade_without_overlap_is_a_plain_join():
    a = np.full((3, 1, 1, 3), 10, dtype=np.uint8)
    b = np.full((3, 1, 1, 3), 20, dtype=np.uint8)
    assert crossfade_windows([(0, a), (3, b)], 6)[:, 0, 0, 0].tolist() == [10, 10, 10, 20, 20, 20]


@pytest.mark.parametrize(("size", "budget", "grid", "expected"), [
    ((1920, 1080), 0, 64, (1920, 1088)),
    ((960, 540), 2088960, 64, (960, 512)),
    ((3840, 2160), 2088960, 64, (1920, 1024)),
    ((1280, 720), 921600, 32, (1280, 720)),
])
def test_the_pixel_budget_lands_on_the_reference_grid(size, budget, grid, expected):
    from src.pipelines.pipes.generator.video_ltx.main import _fit_pixel_budget

    assert _fit_pixel_budget(*size, budget, grid) == expected
