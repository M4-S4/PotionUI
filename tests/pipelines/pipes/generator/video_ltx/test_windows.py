from __future__ import annotations

import numpy as np
import pytest

from src.pipelines.pipes.generator.video_ltx.windows import crossfade_windows, plan_windows


@pytest.mark.parametrize(("total", "window", "overlap", "expected"), [
    (17, 17, 8, [0]),
    (10, 17, 8, [0]),
    (40, 17, 8, [0, 9, 18, 23]),
    (33, 17, 1, [0, 16]),
    (50, 17, 0, [0, 17, 33]),
])
def test_windows_step_by_window_minus_overlap_and_the_last_one_ends_at_the_clip_end(total, window, overlap, expected):
    starts = plan_windows(total, window, overlap)
    assert starts == expected
    assert starts[-1] + min(window, total) >= total


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
