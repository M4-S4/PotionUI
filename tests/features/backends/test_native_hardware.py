import sys
from types import SimpleNamespace

import pytest
import torch

from src.features.backends.backend_config import NativeBackendConfig
from src.features.backends.native_hardware import (
    NativeHardwareDefaults,
    detect_native_hardware_defaults,
)
from tests.fixtures.kernel_capabilities import ampere, blackwell, cpu_only, fake_capabilities, fake_device

RESIDENCY = "src.platform.runtime.native.memory.residency"
CPU_DEFAULTS = NativeHardwareDefaults(device="cpu", dtype="float32", gpu_max_vram=0)


@pytest.fixture(autouse=True)
def fresh_cache():
    detect_native_hardware_defaults.cache_clear()
    yield
    detect_native_hardware_defaults.cache_clear()


@pytest.fixture
def vram(monkeypatch):
    calls = SimpleNamespace(total=[], reserve=[])
    state = SimpleNamespace(total_gb=32.0, reserve_gb=1.0)

    def total_vram_gb(device):
        calls.total.append(device)
        return state.total_gb

    def minimum_inference_memory_gb():
        calls.reserve.append(True)
        return state.reserve_gb

    monkeypatch.setattr(f"{RESIDENCY}.total_vram_gb", total_vram_gb)
    monkeypatch.setattr(f"{RESIDENCY}.minimum_inference_memory_gb", minimum_inference_memory_gb)
    return SimpleNamespace(calls=calls, state=state)


@pytest.fixture
def no_torch_probing(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("native hardware defaults must read the kernel snapshot, not torch.cuda")

    for name in ("is_available", "device_count", "get_device_capability", "get_device_name"):
        monkeypatch.setattr(torch.cuda, name, refuse)


def test_no_gpu_defaults_to_cpu_float32(install_capabilities, vram, no_torch_probing):
    install_capabilities(cpu_only())

    assert detect_native_hardware_defaults() == CPU_DEFAULTS
    assert vram.calls.total == []


def test_torch_missing_defaults_to_cpu_float32(monkeypatch, vram):
    monkeypatch.setitem(sys.modules, "src.platform.runtime.native.kernels", None)

    assert detect_native_hardware_defaults() == CPU_DEFAULTS
    assert vram.calls.total == []


@pytest.mark.parametrize("capability", [(8, 0), (8, 6), (8, 9), (9, 0), (12, 0)], ids=["a100", "ampere", "ada", "hopper", "blackwell"])
def test_ampere_or_newer_prefers_bfloat16(install_capabilities, vram, no_torch_probing, capability):
    install_capabilities(fake_capabilities(fake_device(capability)))

    result = detect_native_hardware_defaults()

    assert result.device == "cuda"
    assert result.dtype == "bfloat16"
    assert result.gpu_max_vram == 31


@pytest.mark.parametrize("capability", [(7, 5), (6, 1), (0, 0)], ids=["turing", "pascal", "probe_failed"])
def test_pre_ampere_falls_back_to_float16(install_capabilities, vram, no_torch_probing, capability):
    install_capabilities(fake_capabilities(fake_device(capability)))

    result = detect_native_hardware_defaults()

    assert result.device == "cuda"
    assert result.dtype == "float16"


def test_first_device_decides_on_a_multi_gpu_host(install_capabilities, vram, no_torch_probing):
    install_capabilities(fake_capabilities(fake_device((7, 5)), blackwell()))

    assert detect_native_hardware_defaults().dtype == "float16"


def test_gpu_max_vram_derives_from_total_minus_reserve(install_capabilities, vram, no_torch_probing):
    install_capabilities(fake_capabilities(ampere()))
    vram.state.total_gb = 90.0
    vram.state.reserve_gb = 2.0

    assert detect_native_hardware_defaults().gpu_max_vram == 88
    assert vram.calls.total == ["cuda"]


def test_gpu_max_vram_never_goes_below_one(install_capabilities, vram, no_torch_probing):
    install_capabilities(fake_capabilities(ampere()))
    vram.state.total_gb = 0.5
    vram.state.reserve_gb = 1.0

    assert detect_native_hardware_defaults().gpu_max_vram == 1


def test_result_is_cached_across_calls_and_snapshot_swaps(install_capabilities, vram, no_torch_probing):
    install_capabilities(fake_capabilities(ampere()))

    first = detect_native_hardware_defaults()
    install_capabilities(fake_capabilities(fake_device((7, 5))))
    second = detect_native_hardware_defaults()

    assert first is second
    assert second.dtype == "bfloat16"
    assert len(vram.calls.total) == 1


@pytest.mark.parametrize(
    "capabilities,expected",
    [
        ([], CPU_DEFAULTS),
        ([(7, 5)], NativeHardwareDefaults("cuda", "float16", 31)),
        ([(8, 6)], NativeHardwareDefaults("cuda", "bfloat16", 31)),
        ([(8, 9)], NativeHardwareDefaults("cuda", "bfloat16", 31)),
        ([(12, 0)], NativeHardwareDefaults("cuda", "bfloat16", 31)),
    ],
    ids=["cpu", "turing", "ampere", "ada", "blackwell"],
)
def test_native_backend_config_defaults_follow_the_snapshot(install_capabilities, vram, no_torch_probing, capabilities, expected):
    install_capabilities(fake_capabilities(*(fake_device(capability) for capability in capabilities)))

    config = NativeBackendConfig(id="native", name="Native")

    assert (config.device, config.dtype, config.gpu_max_vram) == (expected.device, expected.dtype, expected.gpu_max_vram)
