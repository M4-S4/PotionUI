from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
from unittest.mock import patch

def _verdict(supported: bool):
    from src.platform.runtime.native.kernels import gemm

    return gemm._Verdict(supported, None if supported else "faked unsupported")

@contextmanager
def installed_gemm_verdicts(*, fp8: bool = False, nvfp4: bool = False) -> Iterator[None]:
    import vendor.gpl.comfyui.ops as ops
    from src.platform.runtime.native.kernels import gemm, gemm_support

    with (
        patch.object(gemm, "_fp8_verdict", lambda device: _verdict(fp8)),
        patch.object(gemm, "_nvfp4_verdict", lambda device: _verdict(nvfp4)),
        patch.object(ops, "GEMM_SUPPORT", gemm_support),
    ):
        yield

def install_gemm_verdicts(monkeypatch, *, fp8: bool = False, nvfp4: bool = False) -> None:
    import vendor.gpl.comfyui.ops as ops
    from src.platform.runtime.native.kernels import gemm, gemm_support

    monkeypatch.setattr(gemm, "_fp8_verdict", lambda device: _verdict(fp8))
    monkeypatch.setattr(gemm, "_nvfp4_verdict", lambda device: _verdict(nvfp4))
    monkeypatch.setattr(ops, "GEMM_SUPPORT", gemm_support)
