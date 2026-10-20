"""Tests for src.platform.runtime.tensors"""

import numpy as np
import torch
from PIL import Image

from src.platform.runtime.tensors import (
    numpy_to_pil,
    pil_to_numpy_rgb,
    torch_to_numpy,
)


# ---- pil_to_numpy_rgb ----

class TestPilToNumpyRgb:
    def test_rgb_image(self):
        img = Image.new("RGB", (16, 8), color=(10, 20, 30))
        arr = pil_to_numpy_rgb(img)
        assert arr.shape == (8, 16, 3)
        assert arr.dtype == np.uint8
        assert tuple(arr[0, 0]) == (10, 20, 30)

    def test_rgba_image_converted(self):
        img = Image.new("RGBA", (4, 4), color=(100, 150, 200, 255))
        arr = pil_to_numpy_rgb(img)
        assert arr.shape == (4, 4, 3)
        assert tuple(arr[0, 0]) == (100, 150, 200)

    def test_grayscale_image_converted(self):
        img = Image.new("L", (4, 4), color=128)
        arr = pil_to_numpy_rgb(img)
        assert arr.shape == (4, 4, 3)
        assert tuple(arr[0, 0]) == (128, 128, 128)


# ---- numpy_to_pil ----

class TestNumpyToPil:
    def test_rgb_array(self):
        arr = np.zeros((8, 16, 3), dtype=np.uint8)
        arr[0, 0] = [10, 20, 30]
        img = numpy_to_pil(arr)
        assert img.size == (16, 8)
        assert img.mode == "RGB"
        assert img.getpixel((0, 0)) == (10, 20, 30)

    def test_grayscale_array(self):
        arr = np.full((4, 4), 128, dtype=np.uint8)
        img = numpy_to_pil(arr)
        assert img.mode == "L"
        assert img.getpixel((0, 0)) == 128


# ---- torch_to_numpy ----

class TestTorchToNumpy:
    def test_basic_conversion(self):
        t = torch.full((1, 3, 8, 16), 0.5)
        arr = torch_to_numpy(t)
        assert arr.shape == (8, 16, 3)
        assert arr.dtype == np.uint8
        assert arr[0, 0, 0] == 127  # floor(0.5 * 255) = 127

    def test_clamping(self):
        t = torch.tensor([[[[2.0]], [[-1.0]], [[0.5]]]])  # shape (1,3,1,1)
        arr = torch_to_numpy(t)
        assert arr[0, 0, 0] == 255  # clamped from 2.0
        assert arr[0, 0, 1] == 0    # clamped from -1.0
        assert arr[0, 0, 2] == 127  # floor(0.5*255) = 127

    def test_roundtrip_from_uint8(self):
        original = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        t = torch.from_numpy(original).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        recovered = torch_to_numpy(t)
        assert np.allclose(original, recovered, atol=1)
