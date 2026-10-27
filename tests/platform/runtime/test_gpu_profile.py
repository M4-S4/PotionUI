from types import SimpleNamespace

import pytest
import torch

from src.platform.runtime.gpu_generation import GpuGeneration
from src.platform.runtime.gpu_profile import (
    GENERATION_LABELS,
    GpuProfile,
    build_gpu_profile,
    detect_gpu_profile,
    fast_precisions_for,
    generation_for_capability,
    generations_supporting,
)
from tests.fixtures.kernel_capabilities import fake_device


@pytest.mark.parametrize(
    "capability,generation",
    [
        ((8, 0), "ampere"),
        ((8, 6), "ampere"),
        ((8, 7), "ampere"),
        ((8, 9), "ada"),
        ((9, 0), "hopper"),
        ((10, 0), "blackwell"),
        ((12, 0), "blackwell"),
        ((8, 8), "ampere"),
        ((9, 5), "hopper"),
        ((10, 3), "blackwell"),
        ((11, 0), "blackwell"),
        ((12, 1), "blackwell"),
        ((7, 5), "other"),
        ((6, 1), "other"),
        ((0, 0), "other"),
        ((13, 0), "other"),
        (None, "none"),
        ((), "none"),
    ],
)
def test_generation_for_capability(capability, generation):
    assert generation_for_capability(capability) == generation


@pytest.mark.parametrize(
    "capability,snapshot_generation,generation,label",
    [
        ((8, 6), GpuGeneration.AMPERE, "ampere", "RTX 30-series (Ampere)"),
        ((8, 9), GpuGeneration.ADA, "ada", "RTX 40-series (Ada)"),
        ((9, 0), GpuGeneration.HOPPER, "hopper", "Hopper"),
        ((12, 0), GpuGeneration.BLACKWELL, "blackwell", "RTX 50-series (Blackwell)"),
        ((7, 5), GpuGeneration.PRE_AMPERE, "other", "this GPU"),
        ((13, 0), GpuGeneration.UNKNOWN, "other", "this GPU"),
    ],
)
def test_profile_generation_and_label_follow_the_snapshot_generation(capability, snapshot_generation, generation, label):
    device = fake_device(capability)
    profile = build_gpu_profile(device.capability, 24, device.name)

    assert device.generation is snapshot_generation
    assert profile.generation == generation
    assert profile.generation_label == label


@pytest.mark.parametrize("capability", [(7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0), (13, 0)])
def test_nvfp4_support_tracks_the_snapshot_generation(capability):
    device = fake_device(capability)
    profile = build_gpu_profile(device.capability, 24, device.name)

    assert profile.supports("nvfp4") is (device.generation is GpuGeneration.BLACKWELL)


def test_generations_supporting_lists_the_gated_generations():
    assert generations_supporting("nvfp4") == ("blackwell",)
    assert GENERATION_LABELS["blackwell"] == "RTX 50-series (Blackwell)"
    for precision in ("bf16", "fp16", "fp8", "int8"):
        assert generations_supporting(precision) == ()


def test_fast_precisions_per_generation():
    assert set(fast_precisions_for("ampere", (8, 6))) == {"bf16", "fp16"}
    assert set(fast_precisions_for("ada", (8, 9))) == {"bf16", "fp16", "fp8"}
    assert set(fast_precisions_for("hopper", (9, 0))) == {"bf16", "fp16", "fp8"}
    assert set(fast_precisions_for("blackwell", (12, 0))) == {"bf16", "fp16", "fp8", "nvfp4"}
    assert fast_precisions_for("none", None) == ()
    assert not build_gpu_profile((8, 6), 24).is_fast("int8")


def test_other_vendors_get_no_accelerated_quantized_precisions():
    profile = build_gpu_profile((12, 0), 24, vendor_is_nvidia=False)
    assert profile.generation == "other"
    assert profile.fast_precisions == ("bf16", "fp16")


@pytest.mark.parametrize("capability,nvfp4_fast", [((10, 0), False), ((11, 0), False), ((12, 0), True)])
def test_nvfp4_is_fast_only_from_sm120(capability, nvfp4_fast):
    profile = build_gpu_profile(capability, 32)
    assert profile.generation == "blackwell"
    assert profile.is_fast("nvfp4") is nvfp4_fast
    assert profile.is_fast("fp8")


def test_nvfp4_is_supported_only_on_blackwell():
    assert build_gpu_profile((12, 0), 32).supports("nvfp4")
    assert not build_gpu_profile((8, 9), 24).supports("nvfp4")
    assert build_gpu_profile((8, 6), 24).supports("fp8")
    assert not GpuProfile().supports("bf16")


def test_profile_dict_shape():
    profile = build_gpu_profile((8, 9), 23.988, "NVIDIA GeForce RTX 4090")
    assert profile.to_dict() == {
        "generation": "ada",
        "generation_label": "RTX 40-series (Ada)",
        "name": "NVIDIA GeForce RTX 4090",
        "vram_gb": 24.0,
        "compute_capability": "8.9",
        "fast_precisions": ["bf16", "fp16", "fp8"],
    }
    assert profile.vram_class_gb == 24


def test_non_nvidia_device_is_other():
    profile = build_gpu_profile((9, 4), 192, "AMD Instinct", vendor_is_nvidia=False)
    assert profile.generation == "other"
    assert "fp8" not in profile.fast_precisions


def test_no_gpu_profile():
    assert GpuProfile().to_dict() == {
        "generation": "none",
        "generation_label": "no GPU",
        "name": None,
        "vram_gb": 0.0,
        "compute_capability": None,
        "fast_precisions": [],
    }


def test_detect_without_cuda_returns_none(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert detect_gpu_profile().generation == "none"


def test_detect_reads_device_properties(monkeypatch):
    props = SimpleNamespace(major=12, minor=0, total_memory=32 * 1024 ** 3, name="NVIDIA GeForce RTX 5090")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: props)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    profile = detect_gpu_profile()
    assert profile.generation == "blackwell"
    assert profile.vram_gb == 32.0
    assert profile.name == "NVIDIA GeForce RTX 5090"


def test_detect_survives_a_driver_error(monkeypatch):
    def boom(index):
        raise RuntimeError("driver gone")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", boom)
    assert detect_gpu_profile() == GpuProfile()

