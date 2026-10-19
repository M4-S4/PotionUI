from __future__ import annotations

from src.platform.runtime.model_lifecycle.lifecycle import file_size_gb
from src.platform.runtime.native.arch.moge import load as moge_load
from src.platform.runtime.native.engine import NativeModel

GEOMETRY_ESTIMATOR_DTYPE = "float32"


def geometry_estimator_key(path: str) -> str:
    return f"native/moge/{path}"


def geometry_estimator_fingerprint(path: str) -> str:
    return f"{geometry_estimator_key(path)}|{GEOMETRY_ESTIMATOR_DTYPE}"


def geometry_estimator_size_gb(path: str) -> float:
    return 2 * file_size_gb(path)


def load_geometry_estimator(path: str, role: str = "geometry estimator") -> NativeModel:
    try:
        model = moge_load.load_moge2(path)
    except ValueError as exc:
        raise ValueError(f"the {role} could not be loaded: {exc}") from exc
    return NativeModel("text_encoder", model)
