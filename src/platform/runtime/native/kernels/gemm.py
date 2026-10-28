from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from ..io.quant_formats import NVFP4_BLOCK
from .hardware import DeviceCapabilities, capabilities

logger = logging.getLogger(__name__)

_FP8_MIN_CAPABILITY = (8, 9)
_NVFP4_MIN_CAPABILITY = (12, 0)
_NVFP4_TRIAL_SHAPE = (16, 32, 16)
_SCALE_ROW_TILE = 128
_SCALE_COL_TILE = 4
_NVFP4_API = ("scaled_mm", "ScalingType", "SwizzleType")


@dataclass(frozen=True)
class _Verdict:
    supported: bool
    reason: str | None


@dataclass(frozen=True)
class GemmSupport:
    device: DeviceCapabilities

    @property
    def fp8_scaled_mm(self) -> bool:
        return _fp8_verdict(self.device).supported

    @property
    def fp8_reason(self) -> str | None:
        return _fp8_verdict(self.device).reason

    @property
    def nvfp4(self) -> bool:
        return _nvfp4_verdict(self.device).supported

    @property
    def nvfp4_reason(self) -> str | None:
        return _nvfp4_verdict(self.device).reason


_fp8_lock = threading.Lock()
_nvfp4_lock = threading.Lock()
_fp8_verdicts: dict[int | None, _Verdict] = {}
_nvfp4_verdicts: dict[int | None, _Verdict] = {}


def gemm_support(device: int | torch.device | str | None = None) -> GemmSupport:
    return GemmSupport(capabilities().device(device))


def _sm(capability: tuple[int, int]) -> str:
    return f"sm{capability[0]}{capability[1]}"


def _capability_gap(
    device: DeviceCapabilities, minimum: tuple[int, int], feature: str
) -> str | None:
    if device.index is None:
        return "no CUDA device is available"
    if device.capability == (0, 0):
        return f"the compute capability of {device.name} could not be read"
    if not device.at_least(*minimum):
        return f"{device.name} is {_sm(device.capability)}, {feature} needs {_sm(minimum)} or newer"
    return None


def _probe_fp8(device: DeviceCapabilities) -> _Verdict:
    if not hasattr(torch, "_scaled_mm"):
        return _Verdict(False, "this torch build has no torch._scaled_mm")
    gap = _capability_gap(device, _FP8_MIN_CAPABILITY, "torch._scaled_mm")
    return _Verdict(gap is None, gap)


def _nvfp4_api_gap() -> str | None:
    if not hasattr(torch, "float4_e2m1fn_x2"):
        return "torch.float4_e2m1fn_x2"
    for name in _NVFP4_API:
        if not hasattr(F, name):
            return f"torch.nn.functional.{name}"
    return None


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def _trial_block_scale(rows: int, blocks: int, device: torch.device) -> torch.Tensor:
    shape = (_round_up(rows, _SCALE_ROW_TILE), _round_up(blocks, _SCALE_COL_TILE))
    return torch.ones(shape, device=device).to(torch.float8_e4m3fn)


def _nvfp4_trial_gemm(device: torch.device) -> None:
    m, k, n = _NVFP4_TRIAL_SHAPE
    a = torch.zeros(m, k // 2, dtype=torch.uint8, device=device).view(torch.float4_e2m1fn_x2)
    b_row = torch.zeros(n, k // 2, dtype=torch.uint8, device=device).view(torch.float4_e2m1fn_x2)
    blocks = k // NVFP4_BLOCK
    scale_a = _trial_block_scale(m, blocks, device)
    scale_b = _trial_block_scale(n, blocks, device)
    global_a = torch.ones((), dtype=torch.float32, device=device)
    global_b = torch.ones((), dtype=torch.float32, device=device)
    F.scaled_mm(
        a, b_row.t(),
        [scale_a, global_a], [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
        [scale_b, global_b], [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
        swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
        swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
        output_dtype=torch.bfloat16,
    )


def _probe_nvfp4(device: DeviceCapabilities) -> _Verdict:
    missing = _nvfp4_api_gap()
    if missing is not None:
        return _Verdict(False, f"this torch build has no {missing}")
    gap = _capability_gap(device, _NVFP4_MIN_CAPABILITY, "the nvfp4 GEMM")
    if gap is not None:
        return _Verdict(False, gap)
    try:
        _nvfp4_trial_gemm(torch.device("cuda", device.index))
    except Exception as exc:
        return _Verdict(False, f"the nvfp4 trial GEMM failed: {type(exc).__name__}: {exc}")
    return _Verdict(True, None)


def _cached_verdict(
    verdicts: dict[int | None, _Verdict],
    lock: threading.Lock,
    probe: Callable[[DeviceCapabilities], _Verdict],
    label: str,
    device: DeviceCapabilities,
) -> _Verdict:
    verdict = verdicts.get(device.index)
    if verdict is None:
        with lock:
            verdict = verdicts.get(device.index)
            if verdict is None:
                verdict = verdicts[device.index] = probe(device)
                logger.debug(
                    "%s: supported=%s device=%s reason=%s",
                    label, verdict.supported, device.name, verdict.reason,
                )
    return verdict


def _fp8_verdict(device: DeviceCapabilities) -> _Verdict:
    return _cached_verdict(_fp8_verdicts, _fp8_lock, _probe_fp8, "fp8 matmul", device)


def _nvfp4_verdict(device: DeviceCapabilities) -> _Verdict:
    return _cached_verdict(_nvfp4_verdicts, _nvfp4_lock, _probe_nvfp4, "nvfp4 matmul", device)


def _reset_verdicts() -> None:
    with _fp8_lock:
        _fp8_verdicts.clear()
    with _nvfp4_lock:
        _nvfp4_verdicts.clear()
