from __future__ import annotations

import math

import numpy as np
import pytest

from src.platform.filesystem.splat_formats import probe_gaussian_ply
from src.platform.runtime.native.arch.image_to_splat import (
    MEDIAN_DEPTH,
    build_layers,
    lift_to_splat,
    logit_to_opacity,
    sh0_to_rgb,
    write_splat_ply,
)
from src.platform.runtime.native.arch.image_to_splat import config
from src.platform.runtime.native.arch.image_to_splat.lift import (
    depth_jumps,
    foreground_mask,
    hidden_bands,
    inpaint_background,
    normalize_depth,
    pixel_gaussians,
)

H, W = 48, 64
FOCAL = 60.0


def _square_scene(near=2.0, far=6.0):
    depth = np.full((H, W), far, dtype=np.float32)
    depth[16:32, 24:40] = near
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[...] = (200, 30, 30)
    rgb[16:32, 24:40] = (20, 200, 40)
    return rgb, depth, np.ones((H, W), dtype=bool)


def _ramp_step(width=3, near=2.0, far=6.0):
    log_depth = np.full(W, math.log(far))
    log_depth[:30] = math.log(near)
    log_depth[30:30 + width] = np.linspace(math.log(near), math.log(far), width + 2)[1:-1]
    return np.broadcast_to(np.exp(log_depth), (H, W)).astype(np.float32).copy()


def test_depth_is_normalized_to_a_median_of_four():
    depth = np.linspace(1.0, 3.0, H * W, dtype=np.float32).reshape(H, W)
    valid = np.ones((H, W), dtype=bool)
    valid[0] = False
    normalized, usable, scale = normalize_depth(depth, valid)
    assert np.median(normalized[usable]) == pytest.approx(MEDIAN_DEPTH, rel=1e-5)
    assert scale == pytest.approx(MEDIAN_DEPTH / np.median(depth[valid]), rel=1e-6)
    assert not usable[0].any() and (normalized[0] == 0).all()


def test_an_image_without_usable_depth_is_refused():
    with pytest.raises(ValueError, match="usable depth"):
        normalize_depth(np.full((4, 4), np.nan), np.ones((4, 4), dtype=bool))


def test_the_flying_pixels_of_a_blurred_depth_step_snap_to_the_nearer_plateau():
    depth = _ramp_step(width=3)
    cuts = depth_jumps(depth, np.ones((H, W), dtype=bool), 0.35, 5)
    row = cuts.snapped[10]
    assert set(np.round(row, 4).tolist()) == {2.0, 6.0}
    assert row[30] == pytest.approx(2.0) and row[32] == pytest.approx(6.0)
    assert cuts.cuts[10, 30:33].all()
    assert not cuts.cuts[10, :26].any() and not cuts.cuts[10, 37:].any()
    assert cuts.near_side[10, 30] and not cuts.near_side[10, 32]
    assert (cuts.snapped[~cuts.cuts] == depth[~cuts.cuts]).all()


def test_a_steep_but_smooth_slope_is_not_cut():
    cols = np.arange(W, dtype=np.float32)[None, :]
    depth = np.broadcast_to(3.0 * np.exp(0.05 * cols), (H, W)).astype(np.float32)
    cuts = depth_jumps(depth, np.ones((H, W), dtype=bool), 0.35, 5)
    assert not cuts.cuts.any()


def test_the_occluding_square_is_a_foreground_segment():
    _rgb, depth, present = _square_scene()
    cuts = depth_jumps(depth, present, 0.35, 5)
    foreground = foreground_mask(cuts.cuts, cuts.near_side, present, np.zeros_like(present))
    assert foreground[18:30, 26:38].all()
    assert not foreground[:12].any() and not foreground[:, :20].any()


def test_an_occluder_joined_to_its_ground_still_gets_a_hidden_band_behind_its_outline():
    depth = np.full((H, W), 8.0, dtype=np.float32)
    depth[20:, :] = np.linspace(8.0, 4.0, H - 20)[:, None]
    depth[8:, 24:40] = 2.0
    present = np.ones((H, W), dtype=bool)
    sky = np.zeros_like(present)
    cuts = depth_jumps(depth, present, 0.35, 5)
    hidden, fill = hidden_bands(cuts, depth, present, sky, FOCAL)
    assert hidden[10:20, 24:40].any()
    assert hidden[12, 26] and hidden[12, 37]
    assert not hidden[:, :20].any() and not hidden[2:6].any()
    assert (fill | ~hidden).all()
    reach = FOCAL * config.DISOCCLUSION_BASELINE * MEDIAN_DEPTH * (1 / 2.0 - 1 / 8.0)
    assert hidden[12, 24:40].sum() <= 2 * min(reach, config.HIDDEN_MAX_FRACTION * W) + 2


def test_the_background_behind_the_square_is_inpainted_from_its_surroundings():
    rgb, depth, present = _square_scene()
    cuts = depth_jumps(depth, present, 0.35, 5)
    foreground = foreground_mask(cuts.cuts, cuts.near_side, present, np.zeros_like(present))
    background_rgb, background_depth = inpaint_background(rgb, depth, foreground, present)
    centre = background_rgb[22:26, 30:34].reshape(-1, 3)
    assert np.abs(centre.astype(int) - [200, 30, 30]).max() <= 12
    assert background_depth[22:26, 30:34] == pytest.approx(6.0, rel=0.05)
    assert (background_depth >= depth * (1 + config.BACKGROUND_DEPTH_MARGIN) - 1e-5).all()


def test_one_gaussian_per_selected_pixel_on_its_ray_with_colour_and_opacity():
    depth = np.full((4, 6), 3.0, dtype=np.float32)
    rgb = np.full((4, 6, 3), 255, dtype=np.uint8)
    select = np.zeros((4, 6), dtype=bool)
    select[1, 4] = True
    splats = pixel_gaussians(rgb, depth, select, np.full((4, 6), 0.9, dtype=np.float32), 10.0, 10.0)
    assert splats.count == 1
    assert splats.means[0].tolist() == pytest.approx([(4.5 - 3) / 10 * 3, (1.5 - 2) / 10 * 3, 3.0])
    assert sh0_to_rgb(splats.f_dc)[0].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert logit_to_opacity(splats.opacity)[0] == pytest.approx(0.9)


def test_nothing_selected_gives_an_empty_layer():
    splats = pixel_gaussians(np.zeros((3, 3, 3), np.uint8), np.ones((3, 3), np.float32), np.zeros((3, 3), bool),
                             np.ones((3, 3), np.float32), 5.0, 5.0)
    assert splats.count == 0


def test_the_scene_lifts_to_two_layers_with_the_source_view():
    rgb, depth, valid = _square_scene()
    lifted = lift_to_splat(rgb, depth, valid, FOCAL, FOCAL)
    layers = build_layers(rgb, depth, valid, FOCAL, FOCAL)

    assert lifted.front_count == H * W
    assert lifted.back_count == int(layers.hidden.sum())
    assert lifted.splats.count == lifted.front_count + lifted.back_count
    assert lifted.view.fov_y == pytest.approx(math.degrees(2 * math.atan(H / 2 / FOCAL)))
    assert (lifted.view.aspect, lifted.view.width, lifted.view.height) == (W / H, W, H)
    assert lifted.view.depth == MEDIAN_DEPTH

    front = lifted.splats.means[: lifted.front_count].reshape(H, W, 3)
    back = lifted.splats.means[lifted.front_count:]
    assert (back[:, 2] > front[layers.hidden][:, 2]).all()
    assert set(np.round(front[..., 2] / front[..., 2].min(), 3).ravel().tolist()) == {1.0, 3.0}

    alpha = logit_to_opacity(lifted.splats.opacity[: lifted.front_count]).reshape(H, W)
    assert alpha[layers.cuts] == pytest.approx(config.EDGE_OPACITY)
    assert alpha[0, 0] == pytest.approx(config.OPACITY)


def test_dropping_the_cut_pixels_moves_them_to_the_background_layer(monkeypatch):
    from src.platform.runtime.native.arch.image_to_splat import lift

    monkeypatch.setattr(lift, "EDGE_PIXELS", config.EDGE_DROP)
    rgb, depth, valid = _square_scene()
    layers = build_layers(rgb, depth, valid, FOCAL, FOCAL)
    lifted = lift_to_splat(rgb, depth, valid, FOCAL, FOCAL)
    assert layers.cuts.any()
    assert lifted.front_count == H * W - int(layers.cuts.sum())
    assert (layers.hidden | ~layers.cuts).all()


def test_pixels_without_depth_become_a_far_sky():
    rgb, depth, valid = _square_scene()
    valid[:8] = False
    lifted = lift_to_splat(rgb, depth, valid, FOCAL, FOCAL)
    z = lifted.splats.means[: lifted.front_count, 2].reshape(H, W)
    assert z[:8].min() >= config.SKY_DEPTH_MIN
    assert z[:8].min() > z[8:].max()


def test_without_sky_the_masked_pixels_get_no_gaussians(monkeypatch):
    from src.platform.runtime.native.arch.image_to_splat import lift

    monkeypatch.setattr(lift, "KEEP_SKY", False)
    rgb, depth, valid = _square_scene()
    valid[:8] = False
    lifted = lift_to_splat(rgb, depth, valid, FOCAL, FOCAL)
    assert lifted.front_count == H * W - 8 * W


def test_a_flat_wall_has_no_background_layer():
    rgb = np.full((H, W, 3), 128, dtype=np.uint8)
    lifted = lift_to_splat(rgb, np.full((H, W), 5.0, dtype=np.float32), np.ones((H, W), dtype=bool), FOCAL, FOCAL)
    assert lifted.back_count == 0


def test_the_lifted_splat_writes_a_contract_ply(tmp_path):
    rgb, depth, valid = _square_scene()
    lifted = lift_to_splat(rgb, depth, valid, FOCAL, FOCAL)
    path = write_splat_ply(tmp_path / "scene.ply", lifted.splats, lifted.view)
    assert probe_gaussian_ply(str(path)) == lifted.splats.count


def test_mismatched_image_and_depth_are_refused():
    with pytest.raises(ValueError, match="share one height and width"):
        lift_to_splat(np.zeros((4, 4, 3), np.uint8), np.ones((5, 4)), np.ones((5, 4), bool), 1.0, 1.0)
