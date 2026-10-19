from __future__ import annotations

import struct

import numpy as np
import pytest

from src.platform.filesystem.splat_formats import (
    GAUSSIAN_PLY_PROPERTIES,
    SplatView,
    inspect_splat,
    probe_gaussian_ply,
    read_ply_header,
)
from src.platform.runtime.native.arch.image_to_splat import (
    SplatArrays,
    logit_to_opacity,
    opacity_to_logit,
    read_splat_ply,
    rgb_to_sh0,
    sh0_to_rgb,
    write_splat_ply,
)
from tests.fixtures.splat_fixtures import SPLAT_COUNT, build_gaussian_ply

VIEW = SplatView(fov_y=42.5, aspect=1.5, depth=4.0, width=96, height=64)


def _splats(count: int = 5, seed: int = 0) -> SplatArrays:
    rng = np.random.default_rng(seed)
    rotations = rng.normal(size=(count, 4))
    return SplatArrays(
        means=rng.normal(size=(count, 3)) + [0, 0, 4],
        f_dc=rng.normal(size=(count, 3)),
        opacity=rng.normal(size=count),
        log_scales=rng.normal(size=(count, 3)) - 4,
        rotations=rotations / np.linalg.norm(rotations, axis=1, keepdims=True),
    )


def test_written_ply_has_the_contract_header_exactly(tmp_path):
    path = write_splat_ply(tmp_path / "scene.ply", _splats(3), VIEW)

    raw = path.read_bytes()
    header = raw[:raw.index(b"end_header\n") + len(b"end_header\n")].decode("ascii")
    assert header.splitlines() == [
        "ply",
        "format binary_little_endian 1.0",
        "comment potionui_view fov_y=42.5 aspect=1.5 depth=4 width=96 height=64 convention=opencv",
        "element vertex 3",
        *[f"property float {name}" for name in GAUSSIAN_PLY_PROPERTIES],
        "end_header",
    ]
    assert len(raw) == len(header) + 3 * 14 * 4


def test_body_is_little_endian_float32_in_property_order(tmp_path):
    splats = _splats(2)
    path = write_splat_ply(tmp_path / "scene.ply", splats, VIEW)

    raw = path.read_bytes()
    body = raw[read_ply_header(path).length:]
    first = struct.unpack("<14f", body[:56])
    expected = (
        *splats.means[0], *splats.f_dc[0], splats.opacity[0], *splats.log_scales[0], *splats.rotations[0],
    )
    assert first == pytest.approx(expected)


def test_round_trip_keeps_every_value_and_the_view(tmp_path):
    splats = _splats(17, seed=3)
    path = write_splat_ply(tmp_path / "scene.ply", splats, VIEW)

    read, view = read_splat_ply(path)

    for name in ("means", "f_dc", "opacity", "log_scales", "rotations"):
        assert np.array_equal(getattr(read, name), getattr(splats, name)), name
    assert view == VIEW


def test_written_ply_passes_the_storage_probe(tmp_path):
    path = write_splat_ply(tmp_path / "scene.ply", _splats(11), VIEW)

    assert probe_gaussian_ply(str(path)) == 11
    info = inspect_splat(path)
    assert (info.format, info.count, info.view) == ("ply", 11, VIEW)


def test_without_a_view_there_is_no_comment(tmp_path):
    path = write_splat_ply(tmp_path / "scene.ply", _splats(2))

    assert b"comment" not in path.read_bytes()[:400]
    assert read_splat_ply(path)[1] is None


def test_reader_accepts_a_trained_ply_with_extra_properties(tmp_path):
    properties = GAUSSIAN_PLY_PROPERTIES[:3] + ("nx", "ny", "nz") + GAUSSIAN_PLY_PROPERTIES[3:] + ("f_rest_0",)
    path = tmp_path / "trained.ply"
    path.write_bytes(build_gaussian_ply(properties=properties))

    splats, view = read_splat_ply(path)

    assert splats.count == SPLAT_COUNT
    assert splats.means[1].tolist() == pytest.approx([0.1, -0.1, 4.1])
    assert splats.rotations[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert view.fov_y == 50.0


def test_mismatched_shapes_are_refused():
    with pytest.raises(ValueError, match="rotations"):
        SplatArrays(
            means=np.zeros((2, 3)), f_dc=np.zeros((2, 3)), opacity=np.zeros(2),
            log_scales=np.zeros((2, 3)), rotations=np.zeros((2, 3)),
        )
    with pytest.raises(ValueError, match="opacity"):
        SplatArrays(
            means=np.zeros((2, 3)), f_dc=np.zeros((2, 3)), opacity=np.zeros(3),
            log_scales=np.zeros((2, 3)), rotations=np.zeros((2, 4)),
        )


def test_concat_stacks_layers_in_order():
    front, back = _splats(2, seed=1), _splats(3, seed=2)
    both = SplatArrays.concat([front, back])
    assert both.count == 5
    assert np.array_equal(both.means[2:], back.means)


def test_colour_and_opacity_transforms_invert_each_other():
    rgb = np.array([[0.0, 0.5, 1.0], [0.2, 0.4, 0.8]], dtype=np.float32)
    assert sh0_to_rgb(rgb_to_sh0(rgb)) == pytest.approx(rgb, abs=1e-6)
    assert rgb_to_sh0([[0.5, 0.5, 0.5]]).tolist() == [[0.0, 0.0, 0.0]]
    alpha = np.array([0.05, 0.5, 0.95], dtype=np.float32)
    assert logit_to_opacity(opacity_to_logit(alpha)) == pytest.approx(alpha, abs=1e-6)
    assert np.isfinite(opacity_to_logit([0.0, 1.0])).all()
