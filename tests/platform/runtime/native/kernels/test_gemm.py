from __future__ import annotations

import dataclasses
import logging
import threading
import time
import types

import pytest
import torch
import torch.nn.functional as F

import vendor.gpl.comfyui.ops as comfy_ops
from src.platform.runtime.native.kernels import (
    GemmSupport,
    capabilities,
    gemm,
    gemm_support,
    refresh_capabilities,
)
from src.platform.runtime.native.kernels.hardware import _install_snapshot
from tests.fixtures.kernel_capabilities import (
    ada,
    ampere,
    blackwell,
    cpu_only,
    fake_capabilities,
    fake_device,
    hopper,
)


@pytest.fixture(autouse=True)
def _fresh_verdicts():
    gemm._reset_verdicts()
    yield
    gemm._reset_verdicts()


@pytest.fixture
def current_device(monkeypatch):
    state = types.SimpleNamespace(index=0)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: state.index)
    return state


@pytest.fixture
def trial(monkeypatch):
    state = types.SimpleNamespace(devices=[], error=None)

    def fake_trial(device):
        state.devices.append(device)
        if state.error is not None:
            raise state.error

    monkeypatch.setattr(gemm, "_nvfp4_trial_gemm", fake_trial)
    return state


@pytest.fixture
def probe_counts(monkeypatch):
    counts = types.SimpleNamespace(fp8=0, nvfp4=0)
    real_fp8 = gemm._probe_fp8
    real_nvfp4 = gemm._probe_nvfp4

    def counted_fp8(device):
        counts.fp8 += 1
        return real_fp8(device)

    def counted_nvfp4(device):
        counts.nvfp4 += 1
        return real_nvfp4(device)

    monkeypatch.setattr(gemm, "_probe_fp8", counted_fp8)
    monkeypatch.setattr(gemm, "_probe_nvfp4", counted_nvfp4)
    return counts


def _fail(*args, **kwargs):
    raise AssertionError("must not run")


@pytest.mark.parametrize("make", [ada, hopper, blackwell], ids=["ada", "hopper", "blackwell"])
def test_fp8_is_supported_from_sm89_up(install_capabilities, make):
    install_capabilities(fake_capabilities(make()))
    support = gemm_support(0)
    assert support.fp8_scaled_mm is True
    assert support.fp8_reason is None


def test_fp8_is_unsupported_on_ampere_with_the_sm_in_the_reason(install_capabilities):
    install_capabilities(fake_capabilities(ampere(name="RTX 3090")))
    support = gemm_support(0)
    assert support.fp8_scaled_mm is False
    assert support.fp8_reason == "RTX 3090 is sm86, torch._scaled_mm needs sm89 or newer"


def test_fp8_is_unsupported_when_the_torch_build_lacks_scaled_mm(install_capabilities, monkeypatch):
    install_capabilities(fake_capabilities(hopper()))
    monkeypatch.delattr(torch, "_scaled_mm")
    support = gemm_support(0)
    assert support.fp8_scaled_mm is False
    assert support.fp8_reason == "this torch build has no torch._scaled_mm"


def test_fp8_is_unsupported_without_a_cuda_device(install_capabilities):
    install_capabilities(cpu_only())
    support = gemm_support()
    assert support.fp8_scaled_mm is False
    assert support.fp8_reason == "no CUDA device is available"


def test_fp8_reports_an_unreadable_capability_instead_of_an_sm(install_capabilities):
    install_capabilities(fake_capabilities(fake_device((0, 0), name="Flaky GPU")))
    support = gemm_support(0)
    assert support.fp8_scaled_mm is False
    assert support.fp8_reason == "the compute capability of Flaky GPU could not be read"


def test_nvfp4_is_supported_on_blackwell_when_the_trial_gemm_runs(install_capabilities, trial):
    install_capabilities(fake_capabilities(blackwell()))
    support = gemm_support(0)
    assert support.nvfp4 is True
    assert support.nvfp4_reason is None
    assert trial.devices == [torch.device("cuda", 0)]


def test_nvfp4_is_unsupported_below_sm120_and_never_runs_the_trial(install_capabilities, trial):
    install_capabilities(fake_capabilities(hopper(name="H100")))
    support = gemm_support(0)
    assert support.nvfp4 is False
    assert support.nvfp4_reason == "H100 is sm90, the nvfp4 GEMM needs sm120 or newer"
    assert trial.devices == []


def test_nvfp4_is_unsupported_without_a_cuda_device_and_never_runs_the_trial(install_capabilities, trial):
    install_capabilities(cpu_only())
    support = gemm_support()
    assert support.nvfp4 is False
    assert support.nvfp4_reason == "no CUDA device is available"
    assert trial.devices == []


def test_nvfp4_reports_an_unreadable_capability_and_never_runs_the_trial(install_capabilities, trial):
    install_capabilities(fake_capabilities(fake_device((0, 0), name="Flaky GPU")))
    support = gemm_support(0)
    assert support.nvfp4 is False
    assert support.nvfp4_reason == "the compute capability of Flaky GPU could not be read"
    assert trial.devices == []


@pytest.mark.parametrize(
    "owner,name,described",
    [
        (torch, "float4_e2m1fn_x2", "torch.float4_e2m1fn_x2"),
        (F, "scaled_mm", "torch.nn.functional.scaled_mm"),
        (F, "ScalingType", "torch.nn.functional.ScalingType"),
        (F, "SwizzleType", "torch.nn.functional.SwizzleType"),
    ],
)
def test_nvfp4_is_unsupported_when_the_torch_build_lacks_the_api(
    install_capabilities, trial, monkeypatch, owner, name, described
):
    install_capabilities(fake_capabilities(blackwell()))
    monkeypatch.delattr(owner, name)
    support = gemm_support(0)
    assert support.nvfp4 is False
    assert support.nvfp4_reason == f"this torch build has no {described}"
    assert trial.devices == []


def test_nvfp4_is_unsupported_when_the_trial_gemm_raises(install_capabilities, trial):
    install_capabilities(fake_capabilities(blackwell()))
    trial.error = RuntimeError("kernel not wired")
    support = gemm_support(0)
    assert support.nvfp4 is False
    assert support.nvfp4_reason == "the nvfp4 trial GEMM failed: RuntimeError: kernel not wired"


def test_a_failed_trial_is_not_retried(install_capabilities, trial):
    install_capabilities(fake_capabilities(blackwell()))
    trial.error = RuntimeError("kernel not wired")
    assert gemm_support(0).nvfp4 is False
    assert gemm_support(0).nvfp4 is False
    assert len(trial.devices) == 1


def test_trial_gemm_launches_the_two_level_nvfp4_recipe_and_never_inspects_the_result(monkeypatch):
    calls = []

    def garbage(a, b, scale_a, recipe_a, scale_b, recipe_b, **kwargs):
        calls.append((a, b, scale_a, recipe_a, scale_b, recipe_b, kwargs))
        return torch.full((16, 16), float("nan"))

    monkeypatch.setattr(F, "scaled_mm", garbage)
    gemm._nvfp4_trial_gemm(torch.device("cpu"))

    (a, b, scale_a, recipe_a, scale_b, recipe_b, kwargs), = calls
    assert a.dtype == b.dtype == torch.float4_e2m1fn_x2
    assert tuple(a.shape) == (16, 16)
    assert tuple(b.shape) == (16, 16)
    for scales in (scale_a, scale_b):
        block, global_scale = scales
        assert block.dtype == torch.float8_e4m3fn
        assert tuple(block.shape) == (128, 4)
        assert global_scale.dtype == torch.float32
        assert global_scale.dim() == 0
    assert recipe_a == recipe_b == [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise]
    swizzle = [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE]
    assert kwargs == {"swizzle_a": swizzle, "swizzle_b": swizzle, "output_dtype": torch.bfloat16}


def test_a_trial_gemm_failure_propagates_out_of_the_trial(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("no kernel")

    monkeypatch.setattr(F, "scaled_mm", boom)
    with pytest.raises(RuntimeError, match="no kernel"):
        gemm._nvfp4_trial_gemm(torch.device("cpu"))


def test_asking_only_for_fp8_never_runs_the_nvfp4_trial(install_capabilities, monkeypatch):
    install_capabilities(fake_capabilities(blackwell()))
    monkeypatch.setattr(gemm, "_nvfp4_trial_gemm", _fail)
    support = gemm_support(0)
    assert support.fp8_scaled_mm is True
    assert support.fp8_reason is None
    assert gemm._nvfp4_verdicts == {}


def test_building_the_support_object_probes_nothing(install_capabilities, monkeypatch):
    install_capabilities(fake_capabilities(blackwell()))
    monkeypatch.setattr(gemm, "_probe_fp8", _fail)
    monkeypatch.setattr(gemm, "_probe_nvfp4", _fail)
    assert isinstance(gemm_support(0), GemmSupport)
    assert gemm._fp8_verdicts == {}
    assert gemm._nvfp4_verdicts == {}


def test_nvfp4_trial_runs_at_most_once_per_device(install_capabilities, trial):
    install_capabilities(fake_capabilities(blackwell(), blackwell()))
    for _ in range(3):
        assert gemm_support(0).nvfp4 is True
        assert gemm_support(0).nvfp4_reason is None
    for _ in range(2):
        assert gemm_support(1).nvfp4 is True
    assert trial.devices == [torch.device("cuda", 0), torch.device("cuda", 1)]


def test_each_device_is_probed_once_and_keeps_its_own_fp8_verdict(install_capabilities, probe_counts):
    install_capabilities(fake_capabilities(hopper(), ampere()))
    for _ in range(3):
        assert gemm_support(0).fp8_scaled_mm is True
        assert gemm_support(1).fp8_scaled_mm is False
    assert probe_counts.fp8 == 2
    assert probe_counts.nvfp4 == 0


def test_the_default_device_is_the_current_device_at_call_time(install_capabilities, current_device):
    install_capabilities(fake_capabilities(hopper(), ampere()))
    assert gemm_support().device.index == 0
    assert gemm_support().fp8_scaled_mm is True
    current_device.index = 1
    assert gemm_support().device.index == 1
    assert gemm_support().fp8_scaled_mm is False
    current_device.index = 0
    assert gemm_support().fp8_scaled_mm is True


def test_the_trial_runs_on_the_current_device_by_default(install_capabilities, current_device, trial):
    install_capabilities(fake_capabilities(ampere(), blackwell()))
    current_device.index = 1
    assert gemm_support().nvfp4 is True
    assert trial.devices == [torch.device("cuda", 1)]


@pytest.mark.parametrize("spelling", [1, torch.device("cuda", 1), "cuda:1"], ids=["int", "device", "string"])
def test_every_device_spelling_resolves_to_the_same_device(install_capabilities, spelling):
    install_capabilities(fake_capabilities(hopper(), ampere()))
    support = gemm_support(spelling)
    assert support.device.index == 1
    assert support.fp8_scaled_mm is False


def test_a_cpu_device_resolves_to_the_unsupported_cpu_entry(install_capabilities):
    install_capabilities(fake_capabilities(hopper()))
    support = gemm_support(torch.device("cpu"))
    assert support.device.index is None
    assert support.fp8_scaled_mm is False
    assert support.fp8_reason == "no CUDA device is available"


def test_gemm_support_is_frozen(install_capabilities):
    install_capabilities(fake_capabilities(hopper()))
    support = gemm_support(0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        support.device = support.device


def test_the_capability_snapshot_never_runs_a_probe(monkeypatch):
    monkeypatch.setattr(gemm, "_probe_fp8", _fail)
    monkeypatch.setattr(gemm, "_probe_nvfp4", _fail)
    monkeypatch.setattr(gemm, "_nvfp4_trial_gemm", _fail)
    previous = _install_snapshot(None)
    try:
        capabilities()
        refresh_capabilities()
    finally:
        _install_snapshot(previous)
    assert gemm._fp8_verdicts == {}
    assert gemm._nvfp4_verdicts == {}


def test_the_snapshot_entries_carry_no_gemm_fields(install_capabilities):
    snapshot = install_capabilities(fake_capabilities(blackwell()))
    entry = snapshot.device(0)
    assert not hasattr(entry, "nvfp4")
    assert not hasattr(entry, "fp8_scaled_mm")


def test_a_probe_logs_its_verdict_and_reason_once(install_capabilities, caplog):
    install_capabilities(fake_capabilities(ampere(name="RTX 3090")))
    with caplog.at_level(logging.DEBUG, logger=gemm.logger.name):
        gemm_support(0).fp8_scaled_mm
        gemm_support(0).fp8_reason
    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "fp8 matmul: supported=False device=RTX 3090 "
        "reason=RTX 3090 is sm86, torch._scaled_mm needs sm89 or newer"
    ]


def test_the_nvfp4_probe_logs_its_verdict_once(install_capabilities, trial, caplog):
    install_capabilities(fake_capabilities(blackwell(name="RTX 5090")))
    with caplog.at_level(logging.DEBUG, logger=gemm.logger.name):
        gemm_support(0).nvfp4
        gemm_support(0).nvfp4
    messages = [record.getMessage() for record in caplog.records]
    assert messages == ["nvfp4 matmul: supported=True device=RTX 5090 reason=None"]


def test_concurrent_first_asks_probe_each_family_once(install_capabilities, monkeypatch, probe_counts):
    threads = 8
    install_capabilities(fake_capabilities(blackwell()))
    trial_calls = []

    def slow_trial(device):
        trial_calls.append(device)
        time.sleep(0.05)

    monkeypatch.setattr(gemm, "_nvfp4_trial_gemm", slow_trial)
    barrier = threading.Barrier(threads, timeout=10)
    results = []

    def worker():
        barrier.wait()
        support = gemm_support(0)
        results.append((support.fp8_scaled_mm, support.nvfp4))

    pool = [threading.Thread(target=worker) for _ in range(threads)]
    for thread in pool:
        thread.start()
    for thread in pool:
        thread.join(timeout=10)
    assert results == [(True, True)] * threads
    assert len(trial_calls) == 1
    assert probe_counts.fp8 == 1
    assert probe_counts.nvfp4 == 1


@pytest.fixture
def bound_ops(monkeypatch):
    flags: dict = {}
    monkeypatch.setattr(comfy_ops, "RUNTIME_FLAGS", flags)
    monkeypatch.setattr(comfy_ops, "GEMM_SUPPORT", gemm_support)
    return flags


def test_ops_flags_are_read_live_on_top_of_the_cached_support(
    install_capabilities, current_device, trial, bound_ops
):
    install_capabilities(fake_capabilities(blackwell()))
    assert comfy_ops._fp8_matmul_enabled() is False
    assert comfy_ops._nvfp4_matmul_enabled() is False

    bound_ops["native_fp8_matmul"] = True
    assert comfy_ops._fp8_matmul_enabled() is True
    assert comfy_ops._nvfp4_matmul_enabled() is False
    bound_ops["native_fp8_matmul"] = False
    assert comfy_ops._fp8_matmul_enabled() is False

    bound_ops["native_nvfp4_matmul"] = True
    assert comfy_ops._nvfp4_matmul_enabled() is True
    bound_ops["native_nvfp4_matmul"] = False
    assert comfy_ops._nvfp4_matmul_enabled() is False
    bound_ops["native_nvfp4_matmul"] = True
    assert comfy_ops._nvfp4_matmul_enabled() is True
    assert len(trial.devices) == 1


def test_ops_fp8_flag_never_triggers_the_nvfp4_trial(
    install_capabilities, current_device, monkeypatch, bound_ops
):
    install_capabilities(fake_capabilities(blackwell()))
    monkeypatch.setattr(gemm, "_nvfp4_trial_gemm", _fail)
    bound_ops["native_fp8_matmul"] = True
    assert comfy_ops._fp8_matmul_enabled() is True
    assert gemm._nvfp4_verdicts == {}


def test_ops_flags_stay_off_on_hardware_that_cannot_run_the_kernel(
    install_capabilities, current_device, trial, bound_ops
):
    install_capabilities(fake_capabilities(ampere()))
    bound_ops["native_fp8_matmul"] = True
    bound_ops["native_nvfp4_matmul"] = True
    assert comfy_ops._fp8_matmul_enabled() is False
    assert comfy_ops._nvfp4_matmul_enabled() is False
    assert trial.devices == []


def test_ops_follow_the_current_device_when_it_changes(install_capabilities, current_device, bound_ops):
    install_capabilities(fake_capabilities(hopper(), ampere()))
    bound_ops["native_fp8_matmul"] = True
    assert comfy_ops._fp8_matmul_enabled() is True
    current_device.index = 1
    assert comfy_ops._fp8_matmul_enabled() is False


def test_ops_without_a_bound_support_provider_never_take_the_fast_paths(monkeypatch):
    flags = {"native_fp8_matmul": True, "native_nvfp4_matmul": True}
    monkeypatch.setattr(comfy_ops, "RUNTIME_FLAGS", flags)
    monkeypatch.setattr(comfy_ops, "GEMM_SUPPORT", None)
    assert comfy_ops._fp8_matmul_enabled() is False
    assert comfy_ops._nvfp4_matmul_enabled() is False


def test_nvfp4_trial_gemm_runs_on_real_blackwell_hardware():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")
    if torch.cuda.get_device_capability() < (12, 0):
        pytest.skip("nvfp4 x nvfp4 tensor cores need sm120+ (Blackwell)")
    gemm._nvfp4_trial_gemm(torch.device("cuda", torch.cuda.current_device()))
