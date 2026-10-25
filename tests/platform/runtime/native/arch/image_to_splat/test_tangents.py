from __future__ import annotations

import math

import numpy as np
import pytest

from src.platform.runtime.native.arch.image_to_splat import config
from src.platform.runtime.native.arch.image_to_splat import lift
from src.platform.runtime.native.arch.image_to_splat.lift import (
    matrix_quaternions,
    pixel_gaussians,
    surface_tangents,
    tangent_frames,
    unproject,
)
from src.platform.runtime.native.arch.image_to_splat.ply_export import logit_to_opacity

FOCAL = 80.0


def _matrices(quaternions):
    q = np.asarray(quaternions, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=1)


def _covariances(splats):
    m = _matrices(splats.rotations) * np.exp(splats.log_scales.astype(np.float64))[:, None, :]
    return m @ m.transpose(0, 2, 1)


def _transmittance(splats, fov_deg, angle_deg, target_depth, width, height, blur=0.0):
    rot = np.array([
        [math.cos(math.radians(angle_deg)), 0.0, math.sin(math.radians(angle_deg))],
        [0.0, 1.0, 0.0],
        [-math.sin(math.radians(angle_deg)), 0.0, math.cos(math.radians(angle_deg))],
    ])
    target = np.array([0.0, 0.0, target_depth])
    centre = target + rot @ np.array([0.0, 0.0, -target_depth])
    focal = (height / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    p = (splats.means.astype(np.float64) - centre) @ rot
    x, y, z = p.T
    cov = rot.T[None] @ _covariances(splats) @ rot[None]
    j = np.zeros((p.shape[0], 2, 3))
    j[:, 0, 0], j[:, 0, 2] = focal / z, -focal * x / z ** 2
    j[:, 1, 1], j[:, 1, 2] = focal / z, -focal * y / z ** 2
    c2 = j @ cov @ j.transpose(0, 2, 1) + blur * np.eye(2)
    inverse = np.linalg.inv(c2)
    u, v = focal * x / z + width / 2.0, focal * y / z + height / 2.0
    alpha0 = logit_to_opacity(splats.opacity)
    log_keep = np.zeros(width * height)
    radius = 6
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            px, py = np.floor(u).astype(int) + dx, np.floor(v).astype(int) + dy
            d = np.stack([px + 0.5 - u, py + 0.5 - v], axis=-1)
            power = -0.5 * np.einsum("ni,nij,nj->n", d, inverse, d)
            alpha = np.minimum(0.99, alpha0 * np.exp(power))
            inside = (px >= 0) & (px < width) & (py >= 0) & (py < height) & (alpha > 1 / 255)
            np.add.at(log_keep, py[inside] * width + px[inside], np.log1p(-alpha[inside]))
    return np.exp(log_keep).reshape(height, width)


def _inclined_plane(size=48, slope_deg=72.0, depth=4.0):
    cols = np.arange(size) + 0.5 - size / 2.0
    tilt = math.tan(math.radians(slope_deg))
    ray_x = cols / FOCAL
    plane = depth / (1.0 - tilt * ray_x)
    return np.broadcast_to(plane, (size, size)).astype(np.float32).copy()


def test_tangents_of_a_tilted_plane_are_its_central_differences():
    depth = _inclined_plane(slope_deg=60.0)
    points = unproject(depth, FOCAL, FOCAL)
    tangent_u, tangent_v = surface_tangents(points, np.ones(depth.shape, bool), FOCAL, FOCAL, 8.0)
    assert tangent_u[10, 10] == pytest.approx((points[10, 11] - points[10, 9]) / 2)
    assert tangent_v[10, 10] == pytest.approx((points[11, 10] - points[9, 10]) / 2)
    assert np.linalg.norm(tangent_u[10, 10]) > 1.5 * points[10, 10, 2] / FOCAL


def test_a_depth_jump_is_never_spanned_by_a_tangent():
    depth = np.full((8, 8), 6.0, dtype=np.float32)
    depth[:, :4] = 2.0
    points = unproject(depth, FOCAL, FOCAL)
    tangent_u, _ = surface_tangents(points, np.ones(depth.shape, bool), FOCAL, FOCAL, 8.0)
    assert tangent_u[4, 3] == pytest.approx(points[4, 3] - points[4, 2])
    assert tangent_u[4, 4] == pytest.approx(points[4, 5] - points[4, 4])


def test_a_pixel_cut_off_on_both_sides_falls_back_to_its_footprint():
    depth = np.full((5, 5), 6.0, dtype=np.float32)
    depth[2, 2] = 2.0
    points = unproject(depth, FOCAL, FOCAL)
    tangent_u, tangent_v = surface_tangents(points, np.ones(depth.shape, bool), FOCAL, FOCAL, 8.0)
    assert tangent_u[2, 2] == pytest.approx([2.0 / FOCAL, 0.0, 0.0])
    assert tangent_v[2, 2] == pytest.approx([0.0, 2.0 / FOCAL, 0.0])


def test_tangent_length_is_bounded_by_the_footprint_multiple():
    depth = _inclined_plane(slope_deg=60.0)
    points = unproject(depth, FOCAL, FOCAL)
    footprint = points[..., 2] / FOCAL
    for bound in (2.0, 8.0):
        tangent_u, _ = surface_tangents(points, np.ones(depth.shape, bool), FOCAL, FOCAL, bound)
        assert (np.linalg.norm(tangent_u, axis=-1) <= bound * footprint * (1 + 1e-6) + 1e-12).all()


def test_frames_reproduce_the_tangent_covariance_with_a_thin_normal():
    rng = np.random.default_rng(3)
    tangent_u, tangent_v = rng.normal(size=(64, 3)), rng.normal(size=(64, 3))
    thickness = np.full(64, 0.02)
    rotations, log_scales = tangent_frames(tangent_u, tangent_v, thickness, 0.6)
    m = _matrices(rotations) * np.exp(log_scales.astype(np.float64))[:, None, :]
    normal = np.cross(tangent_u, tangent_v)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    expected = 0.36 * (np.einsum("ni,nj->nij", tangent_u, tangent_u) + np.einsum("ni,nj->nij", tangent_v, tangent_v))
    expected += 0.02 ** 2 * np.einsum("ni,nj->nij", normal, normal)
    assert np.abs(m @ m.transpose(0, 2, 1) - expected).max() < 1e-5
    assert np.exp(log_scales[:, 2]) == pytest.approx(thickness, rel=1e-5)


def test_matrix_quaternions_round_trip_every_rotation_including_half_turns():
    rng = np.random.default_rng(5)
    q, _ = np.linalg.qr(rng.normal(size=(200, 3, 3)))
    q *= np.sign(np.linalg.det(q))[:, None, None]
    half_turns = np.array([np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, -1.0]),
                           2 * np.outer([1, -1, 0], [1, -1, 0]) / 2 - np.eye(3)])
    matrices = np.concatenate([q, half_turns])
    quaternions = matrix_quaternions(matrices)
    assert (quaternions[:, 0] >= 0).all()
    assert np.abs(_matrices(quaternions) - matrices).max() < 1e-5


def test_a_fronto_parallel_wall_keeps_pixel_sized_camera_facing_discs():
    depth = np.full((6, 6), 3.0, dtype=np.float32)
    select = np.zeros((6, 6), bool)
    select[2, 3] = True
    splats = pixel_gaussians(np.full((6, 6, 3), 255, np.uint8), depth, select, np.full((6, 6), 0.9, np.float32),
                             FOCAL, FOCAL, present=np.ones((6, 6), bool))
    footprint = 3.0 / FOCAL
    scales = np.sort(np.exp(splats.log_scales[0]))
    assert scales == pytest.approx([config.NORMAL_THICKNESS * footprint] + [config.TANGENT_SCALE * footprint] * 2)
    assert abs(_matrices(splats.rotations)[0, 2, np.argmin(np.exp(splats.log_scales[0]))]) == pytest.approx(1.0)


def _plane_splats(slope_deg):
    depth = _inclined_plane(slope_deg=slope_deg)
    size = depth.shape[0]
    return pixel_gaussians(
        np.full((size, size, 3), 200, np.uint8), depth, np.ones(depth.shape, bool),
        np.full(depth.shape, config.OPACITY, np.float32), FOCAL, FOCAL, present=np.ones(depth.shape, bool),
    )


@pytest.mark.parametrize("angle", [-10.0, 10.0])
def test_an_oblique_plane_stays_fully_covered_ten_degrees_off_the_source_view(angle):
    fov = math.degrees(2 * math.atan(24 / FOCAL))
    transmittance = _transmittance(_plane_splats(72.0), fov, angle, 4.0, 48, 48)
    centre = transmittance[16:32, 16:32]
    assert centre.max() < 0.08


def test_camera_facing_discs_would_leave_the_same_view_see_through(monkeypatch):
    monkeypatch.setattr(lift, "TANGENT_MAX_FOOTPRINTS", 0.0)
    fov = math.degrees(2 * math.atan(24 / FOCAL))
    transmittance = _transmittance(_plane_splats(72.0), fov, -10.0, 4.0, 48, 48)
    assert transmittance[16:32, 16:32].max() > 0.15
