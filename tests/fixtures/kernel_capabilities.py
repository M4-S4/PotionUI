from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Callable, Iterator, Optional

import pytest

if TYPE_CHECKING:
    from src.platform.runtime.native.kernels import Capabilities, DeviceCapabilities

AMPERE_CAPABILITY = (8, 6)
ADA_CAPABILITY = (8, 9)
HOPPER_CAPABILITY = (9, 0)
BLACKWELL_CAPABILITY = (12, 0)


def fake_device(
    capability: tuple[int, int],
    *,
    index: int = 0,
    name: Optional[str] = None,
    attention_backends: Optional[frozenset[str]] = None,
    fp8_scaled_mm: Optional[bool] = None,
    nvfp4_gemm: Optional[bool] = None,
) -> DeviceCapabilities:
    from src.platform.runtime.native.kernels import SDPA, DeviceCapabilities

    return DeviceCapabilities(
        index=index,
        name=name or f"Fake GPU sm{capability[0]}{capability[1]}",
        capability=capability,
        attention_backends=frozenset({SDPA}) if attention_backends is None else attention_backends,
        fp8_scaled_mm=fp8_scaled_mm,
        nvfp4_gemm=nvfp4_gemm,
    )


def ampere(**overrides) -> DeviceCapabilities:
    return fake_device(AMPERE_CAPABILITY, **overrides)


def ada(**overrides) -> DeviceCapabilities:
    return fake_device(ADA_CAPABILITY, **overrides)


def hopper(**overrides) -> DeviceCapabilities:
    return fake_device(HOPPER_CAPABILITY, **overrides)


def blackwell(**overrides) -> DeviceCapabilities:
    return fake_device(BLACKWELL_CAPABILITY, **overrides)


def fake_capabilities(*devices: DeviceCapabilities) -> Capabilities:
    from src.platform.runtime.native.kernels import SDPA, Capabilities, DeviceCapabilities

    cpu = DeviceCapabilities(
        index=None,
        name="cpu",
        capability=(0, 0),
        attention_backends=frozenset({SDPA}),
    )
    return Capabilities(devices=tuple(devices), cpu=cpu)


def cpu_only() -> Capabilities:
    return fake_capabilities()


@contextmanager
def installed_capabilities(snapshot: Capabilities) -> Iterator[Capabilities]:
    from src.platform.runtime.native.kernels.hardware import _install_snapshot

    previous = _install_snapshot(snapshot)
    try:
        yield snapshot
    finally:
        _install_snapshot(previous)


@pytest.fixture
def install_capabilities() -> Iterator[Callable[[Capabilities], Capabilities]]:
    with ExitStack() as stack:
        yield lambda snapshot: stack.enter_context(installed_capabilities(snapshot))
