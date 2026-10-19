from __future__ import annotations

import math

import numpy as np
import pytest

from src.platform.filesystem.splat_formats import SplatView
from src.platform.runtime.native.arch.image_to_splat import (
    SplatArrays,
    opacity_to_logit,
    preview_size,
    render_splat_preview,
    rgb_to_sh0,
)


def _grid_splats(width: int, height: int, fov_y: float, depth: float, colour_of) -> SplatArrays:
    focal = (height / 2) / math.tan(math.radians(fov_y) / 2)
    cols, rows = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    z = np.full(cols.shape, depth, dtype=np.float32)
    means = np.stack([(cols - width / 2) / focal * z, (rows - height / 2) / focal * z, z], axis=-1).reshape(-1, 3)
    colours = np.array([colour_of(c, r) for r, c in zip(rows.ravel(), cols.ravel())], dtype=np.float32)
    count = means.shape[0]
    return SplatArrays(
        means=means,
        f_dc=rgb_to_sh0(colours),
        opacity=opacity_to_logit(np.full(count, 0.999)),
        log_scales=np.full((count, 3), -5.0),
        rotations=np.tile([1.0, 0.0, 0.0, 0.0], (count, 1)),
    )


def test_a_pixel_aligned_grid_renders_back_its_colours_from_the_source_view():
    view = SplatView(fov_y=50.0, aspect=1.5, depth=4.0, width=24, height=16)
    splats = _grid_splats(24, 16, 50.0, 4.0, lambda c, r: (1.0, 0.0, 0.0) if c < 12 else (0.0, 0.0, 1.0))

    image = np.asarray(render_splat_preview(splats, view, max_size=24))

    assert image.shape == (16, 24, 3)
    assert image[8, 3].tolist() == [255, 0, 0]
    assert image[8, 20].tolist() == [0, 0, 255]


def test_the_nearest_gaussian_wins_a_pixel():
    view = SplatView(fov_y=60.0, aspect=1.0, depth=4.0, width=8, height=8)
    far = _grid_splats(8, 8, 60.0, 8.0, lambda c, r: (0.0, 1.0, 0.0))
    near = _grid_splats(8, 8, 60.0, 2.0, lambda c, r: (1.0, 1.0, 1.0))

    for ordering in ([far, near], [near, far]):
        image = np.asarray(render_splat_preview(SplatArrays.concat(ordering), view, max_size=8))
        assert (image == 255).all()


def test_transparent_and_behind_camera_gaussians_are_ignored_and_holes_are_filled():
    view = SplatView(fov_y=60.0, aspect=1.0, depth=4.0, width=8, height=8)
    solid = _grid_splats(8, 8, 60.0, 4.0, lambda c, r: (0.0, 1.0, 0.0))
    keep = np.ones(solid.count, dtype=bool)
    keep[27] = False
    holed = SplatArrays(solid.means[keep], solid.f_dc[keep], solid.opacity[keep], solid.log_scales[keep], solid.rotations[keep])
    ghost = _grid_splats(8, 8, 60.0, 1.0, lambda c, r: (1.0, 0.0, 0.0))
    ghost.opacity[:] = opacity_to_logit(np.full(ghost.count, 0.05))
    behind = _grid_splats(8, 8, 60.0, 1.0, lambda c, r: (1.0, 0.0, 0.0))
    behind.means[:, 2] *= -1

    image = np.asarray(render_splat_preview(SplatArrays.concat([holed, ghost, behind]), view, max_size=8))

    assert image[3, 3].tolist() == [0, 255, 0]
    assert (image[..., 0] == 0).all()


def test_preview_keeps_the_view_aspect_and_never_upscales():
    assert preview_size(SplatView(50.0, 16 / 9, 4.0, 1920, 1080), 768) == (768, 432)
    assert preview_size(SplatView(50.0, 0.5, 4.0, 400, 800), 768) == (384, 768)
    assert preview_size(SplatView(50.0, 1.5, 4.0, 96, 64), 768) == (96, 64)


def test_an_empty_splat_renders_the_background():
    view = SplatView(fov_y=50.0, aspect=1.0, depth=4.0, width=4, height=4)
    empty = SplatArrays(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3)), np.zeros((0, 4)))

    image = np.asarray(render_splat_preview(empty, view, background=(10, 20, 30)))

    assert image.shape == (4, 4, 3)
    assert (image == [10, 20, 30]).all()


@pytest.mark.parametrize("fov", [30.0, 80.0])
def test_projection_matches_the_opencv_pinhole_of_the_view(fov):
    view = SplatView(fov_y=fov, aspect=1.0, depth=4.0, width=32, height=32)
    focal = 16 / math.tan(math.radians(fov) / 2)
    point = SplatArrays(
        means=[[(20.5 - 16) / focal * 4.0, (5.5 - 16) / focal * 4.0, 4.0]],
        f_dc=rgb_to_sh0([[1.0, 1.0, 1.0]]),
        opacity=opacity_to_logit([0.999]),
        log_scales=[[-5.0, -5.0, -5.0]],
        rotations=[[1.0, 0.0, 0.0, 0.0]],
    )

    image = np.asarray(render_splat_preview(point, view, max_size=32))

    assert image[5, 20].tolist() == [255, 255, 255]
