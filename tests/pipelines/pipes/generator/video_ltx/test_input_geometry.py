from __future__ import annotations

import pytest

from src.pipelines.pipes.generator.video_ltx.input_geometry import fit_to_input


def _fit(width, height, frames, fps=24.0, scale=1.0, max_edge=1920, max_frames=121):
    return fit_to_input(width, height, frames, fps, scale=scale, max_edge=max_edge,
                        max_frames=max_frames, fallback_fps=25.0)


def test_full_hd_input_keeps_its_size_on_the_64px_grid():
    assert _fit(1920, 1080, 97) == (1920, 1088, 97, 24.0)


def test_scale_doubles_a_small_draft():
    assert _fit(480, 272, 49, scale=2.0) == (960, 512, 49, 24.0)


def test_longer_edge_is_capped_with_aspect_kept():
    assert _fit(3840, 2160, 49, max_edge=1920) == (1920, 1088, 49, 24.0)
    assert _fit(1080, 1920, 49, max_edge=1280) == (704, 1280, 49, 24.0)


@pytest.mark.parametrize(("frames", "max_frames", "expected"), [
    (100, 121, 97), (300, 121, 121), (8, 121, 1), (9, 121, 9),
])
def test_frames_snap_down_to_one_plus_eight_k_within_the_cap(frames, max_frames, expected):
    assert _fit(640, 384, frames, max_frames=max_frames)[2] == expected


def test_unknown_fps_falls_back_and_extreme_fps_is_clamped():
    assert _fit(640, 384, 49, fps=0.0)[3] == 25.0
    assert _fit(640, 384, 49, fps=240.0)[3] == 60.0
