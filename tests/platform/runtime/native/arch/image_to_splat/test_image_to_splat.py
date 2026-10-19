from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from PIL import Image

from src.platform.runtime.native.arch.image_to_splat import (
    STAGE_ESTIMATE,
    STAGE_LIFT,
    detail_long_side,
    prepare_image,
    run_image_to_splat,
)
from src.platform.runtime.native.arch.moge import MoGeGeometry, depth_to_points, focal_to_fx_fy


class FakeEstimator:
    def __init__(self, fov_x_deg: float = 60.0, solvable: bool = True):
        self.fov_x_deg = fov_x_deg
        self.solvable = solvable
        self.calls = []
        self.devices = []

    def to(self, device):
        self.devices.append(str(device))
        return self

    def estimate_geometry(self, pixels, resolution_level, mask_threshold):
        self.calls.append((tuple(pixels.shape), resolution_level, mask_threshold))
        if not self.solvable:
            return [None]
        _, _, height, width = pixels.shape
        aspect = width / height
        fx = 0.5 / math.tan(math.radians(self.fov_x_deg) / 2)
        focal = fx * 2 * aspect / (1 + aspect ** 2) ** 0.5
        fx, fy = focal_to_fx_fy(focal, aspect)
        depth = torch.full((height, width), 7.0)
        depth[height // 3: 2 * height // 3, width // 3: 2 * width // 3] = 3.0
        return [MoGeGeometry(
            points=depth_to_points(depth, fx, fy), depth=depth, mask=torch.ones(height, width, dtype=torch.bool),
            focal=focal, shift=0.0, fx=fx, fy=fy, fov_x_deg=self.fov_x_deg, fov_y_deg=0.0,
        )]


def _image(width, height):
    pixels = np.random.default_rng(0).integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels, "RGB")


def test_detail_sets_the_long_side():
    assert detail_long_side("standard") == 1024
    assert detail_long_side("high") == 1536
    with pytest.raises(ValueError, match="unknown detail"):
        detail_long_side("ultra")


def test_prepare_image_shrinks_to_the_long_side_and_never_enlarges():
    assert prepare_image(_image(2048, 1024), "standard").size == (1024, 512)
    assert prepare_image(_image(1000, 3000), "high").size == (512, 1536)
    assert prepare_image(_image(640, 480), "high").size == (640, 480)
    assert prepare_image(Image.new("RGBA", (40, 30)), "standard").mode == "RGB"


def test_the_run_estimates_on_the_device_then_lifts_on_the_cpu():
    estimator = FakeEstimator(fov_x_deg=60.0)
    stages = []

    lifted = run_image_to_splat(estimator, _image(96, 64), detail="standard", device="cpu", on_stage=stages.append)

    assert stages == [STAGE_ESTIMATE, STAGE_LIFT]
    assert estimator.calls == [((1, 3, 64, 96), 9, 0.5)]
    assert estimator.devices == ["cpu", "cpu"]
    assert lifted.front_count == 96 * 64
    assert lifted.back_count > 0
    fx_px = 48 / math.tan(math.radians(30))
    assert lifted.view.fov_y == pytest.approx(math.degrees(2 * math.atan(32 / fx_px)), rel=1e-5)
    assert lifted.view.width == 96 and lifted.view.height == 64


def test_an_image_without_recoverable_geometry_is_explained():
    with pytest.raises(ValueError, match="MoGe-2 recovered no geometry"):
        run_image_to_splat(FakeEstimator(solvable=False), _image(32, 32), device="cpu")


def test_the_estimator_returns_to_cpu_even_when_estimation_fails():
    estimator = FakeEstimator()

    def _boom(*args, **kwargs):
        raise RuntimeError("out of memory")

    estimator.estimate_geometry = _boom
    with pytest.raises(RuntimeError):
        run_image_to_splat(estimator, _image(32, 32), device="meta")
    assert estimator.devices == ["meta", "cpu"]
