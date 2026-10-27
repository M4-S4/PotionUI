from __future__ import annotations

import pytest
import torch

from src.platform.runtime.native.ops.dtype import pick_dtypes
from tests.fixtures.kernel_capabilities import (
    ada,
    ampere,
    blackwell,
    cpu_only,
    fake_capabilities,
    fake_device,
    hopper,
)


@pytest.mark.parametrize(
    "device, expected",
    [
        pytest.param(fake_device((6, 1)), torch.float16, id="pascal"),
        pytest.param(fake_device((7, 5)), torch.float16, id="turing"),
        pytest.param(ampere(), torch.bfloat16, id="ampere"),
        pytest.param(ada(), torch.bfloat16, id="ada"),
        pytest.param(hopper(), torch.bfloat16, id="hopper"),
        pytest.param(blackwell(), torch.bfloat16, id="blackwell"),
    ],
)
def test_compute_dtype_follows_the_snapshot_of_each_generation(install_capabilities, device, expected):
    install_capabilities(fake_capabilities(device))
    storage, compute = pick_dtypes(torch.float32, "cuda:0")
    assert compute == expected
    assert storage == expected


def test_cpu_device_computes_in_bf16_whatever_the_snapshot_holds(install_capabilities):
    install_capabilities(fake_capabilities(fake_device((7, 5))))
    assert pick_dtypes(torch.float32, "cpu")[1] == torch.bfloat16


@pytest.mark.parametrize("spec", ["cuda", "cuda:0", torch.device("cuda")])
def test_cuda_device_without_any_gpu_in_the_snapshot_computes_in_bf16(install_capabilities, spec):
    install_capabilities(cpu_only())
    assert pick_dtypes(torch.float32, spec)[1] == torch.bfloat16


def test_each_device_is_judged_by_its_own_capability(install_capabilities):
    install_capabilities(fake_capabilities(fake_device((7, 5)), ampere()))
    assert pick_dtypes(torch.float32, "cuda:0")[1] == torch.float16
    assert pick_dtypes(torch.float32, "cuda:1")[1] == torch.bfloat16
    assert pick_dtypes(torch.float32, torch.device("cuda", 1))[1] == torch.bfloat16


def test_bare_cuda_follows_the_current_device(monkeypatch, install_capabilities):
    install_capabilities(fake_capabilities(fake_device((7, 5)), ampere()))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert pick_dtypes(torch.float32, "cuda")[1] == torch.float16
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert pick_dtypes(torch.float32, "cuda")[1] == torch.bfloat16


def test_unreadable_capability_computes_in_fp16(install_capabilities):
    install_capabilities(fake_capabilities(fake_device((0, 0))))
    assert pick_dtypes(torch.float32, "cuda:0")[1] == torch.float16


@pytest.mark.parametrize(
    "sd_dtype, storage",
    [
        pytest.param(torch.float8_e4m3fn, torch.float8_e4m3fn, id="fp8-stays-fp8"),
        pytest.param(torch.bfloat16, torch.bfloat16, id="bf16-stays-bf16"),
        pytest.param(torch.float16, torch.float16, id="fp16-stays-fp16"),
        pytest.param(torch.float32, torch.float16, id="fp32-takes-the-compute-dtype"),
        pytest.param(None, torch.float16, id="unknown-takes-the-compute-dtype"),
    ],
)
def test_storage_dtype_rules_on_a_pre_ampere_device(install_capabilities, sd_dtype, storage):
    install_capabilities(fake_capabilities(fake_device((7, 5))))
    result_storage, compute = pick_dtypes(sd_dtype, "cuda:0")
    assert compute == torch.float16
    assert result_storage == storage
