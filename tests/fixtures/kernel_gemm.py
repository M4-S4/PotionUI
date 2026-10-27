from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Callable, Iterator
from unittest.mock import patch


def gemm_provider(*, fp8: bool = False, nvfp4: bool = False) -> Callable[[], SimpleNamespace]:
    support = SimpleNamespace(
        fp8_scaled_mm=fp8,
        fp8_reason=None if fp8 else "faked unsupported",
        nvfp4=nvfp4,
        nvfp4_reason=None if nvfp4 else "faked unsupported",
    )
    return lambda: support


@contextmanager
def installed_gemm_verdicts(*, fp8: bool = False, nvfp4: bool = False) -> Iterator[None]:
    import vendor.gpl.comfyui.ops as ops
    from src.platform.runtime.native.kernels import gemm, gemm_support

    def verdict(supported: bool):
        return gemm._Verdict(supported, None if supported else "faked unsupported")

    with (
        patch.object(gemm, "_fp8_verdict", lambda device: verdict(fp8)),
        patch.object(gemm, "_nvfp4_verdict", lambda device: verdict(nvfp4)),
        patch.object(ops, "GEMM_SUPPORT", gemm_support),
    ):
        yield
