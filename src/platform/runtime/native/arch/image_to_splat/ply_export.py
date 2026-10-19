from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from src.platform.filesystem.splat_formats import (
    GAUSSIAN_PLY_PROPERTIES,
    InvalidSplatError,
    SplatView,
    gaussian_vertex_element,
    parse_view_comment,
    read_ply_header,
)

from .config import SH_C0

__all__ = [
    "SplatArrays",
    "logit_to_opacity",
    "opacity_to_logit",
    "ply_header",
    "read_splat_ply",
    "rgb_to_sh0",
    "sh0_to_rgb",
    "write_splat_ply",
]

PathLike = Union[str, os.PathLike]

_PLY_RECORD = np.dtype([(name, "<f4") for name in GAUSSIAN_PLY_PROPERTIES])
_PLY_NUMPY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}
_OPACITY_EPS = 1e-6


@dataclass
class SplatArrays:
    means: np.ndarray
    f_dc: np.ndarray
    opacity: np.ndarray
    log_scales: np.ndarray
    rotations: np.ndarray

    def __post_init__(self) -> None:
        self.means = _float_rows(self.means, 3, "means")
        count = self.means.shape[0]
        self.f_dc = _float_rows(self.f_dc, 3, "f_dc", count)
        self.log_scales = _float_rows(self.log_scales, 3, "log_scales", count)
        self.rotations = _float_rows(self.rotations, 4, "rotations", count)
        self.opacity = np.ascontiguousarray(np.asarray(self.opacity, dtype=np.float32).reshape(-1))
        if self.opacity.shape[0] != count:
            raise ValueError(f"opacity has {self.opacity.shape[0]} entries for {count} Gaussians")

    @property
    def count(self) -> int:
        return int(self.means.shape[0])

    @classmethod
    def concat(cls, parts: Sequence["SplatArrays"]) -> "SplatArrays":
        return cls(
            means=np.concatenate([part.means for part in parts]),
            f_dc=np.concatenate([part.f_dc for part in parts]),
            opacity=np.concatenate([part.opacity for part in parts]),
            log_scales=np.concatenate([part.log_scales for part in parts]),
            rotations=np.concatenate([part.rotations for part in parts]),
        )


def _float_rows(values, width: int, name: str, count: Optional[int] = None) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must be shaped (N, {width}), got {array.shape}")
    if count is not None and array.shape[0] != count:
        raise ValueError(f"{name} has {array.shape[0]} rows for {count} Gaussians")
    return array


def rgb_to_sh0(rgb) -> np.ndarray:
    return ((np.asarray(rgb, dtype=np.float32) - 0.5) / SH_C0).astype(np.float32)


def sh0_to_rgb(f_dc) -> np.ndarray:
    return np.clip(0.5 + SH_C0 * np.asarray(f_dc, dtype=np.float32), 0.0, 1.0)


def opacity_to_logit(alpha) -> np.ndarray:
    alpha = np.clip(np.asarray(alpha, dtype=np.float64), _OPACITY_EPS, 1.0 - _OPACITY_EPS)
    return np.log(alpha / (1.0 - alpha)).astype(np.float32)


def logit_to_opacity(logit) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.asarray(logit, dtype=np.float64)))).astype(np.float32)


def ply_header(count: int, view: Optional[SplatView] = None) -> bytes:
    lines = ["ply", "format binary_little_endian 1.0"]
    if view is not None:
        lines.append(f"comment {view.to_comment()}")
    lines.append(f"element vertex {int(count)}")
    lines += [f"property float {name}" for name in GAUSSIAN_PLY_PROPERTIES]
    lines.append("end_header")
    return ("\n".join(lines) + "\n").encode("ascii")


def write_splat_ply(path: PathLike, splats: SplatArrays, view: Optional[SplatView] = None) -> Path:
    records = np.empty(splats.count, dtype=_PLY_RECORD)
    columns = (
        ("x", splats.means[:, 0]), ("y", splats.means[:, 1]), ("z", splats.means[:, 2]),
        ("f_dc_0", splats.f_dc[:, 0]), ("f_dc_1", splats.f_dc[:, 1]), ("f_dc_2", splats.f_dc[:, 2]),
        ("opacity", splats.opacity),
        ("scale_0", splats.log_scales[:, 0]), ("scale_1", splats.log_scales[:, 1]),
        ("scale_2", splats.log_scales[:, 2]),
        ("rot_0", splats.rotations[:, 0]), ("rot_1", splats.rotations[:, 1]),
        ("rot_2", splats.rotations[:, 2]), ("rot_3", splats.rotations[:, 3]),
    )
    for name, column in columns:
        records[name] = column
    path = Path(path)
    with open(path, "wb") as handle:
        handle.write(ply_header(splats.count, view))
        handle.write(records.tobytes())
    return path


def read_splat_ply(path: PathLike) -> Tuple[SplatArrays, Optional[SplatView]]:
    header = read_ply_header(path)
    vertex = gaussian_vertex_element(header, path)
    dtype = np.dtype([(prop.name, _PLY_NUMPY_TYPES[prop.type]) for prop in vertex.properties])
    data = np.fromfile(str(path), dtype=dtype, count=vertex.count, offset=header.length)
    if data.shape[0] != vertex.count:
        raise InvalidSplatError(f"PLY splat is truncated ({data.shape[0]} of {vertex.count} Gaussians): {path}")

    def stack(*names: str) -> np.ndarray:
        return np.stack([data[name].astype(np.float32) for name in names], axis=-1)

    splats = SplatArrays(
        means=stack("x", "y", "z"),
        f_dc=stack("f_dc_0", "f_dc_1", "f_dc_2"),
        opacity=data["opacity"].astype(np.float32),
        log_scales=stack("scale_0", "scale_1", "scale_2"),
        rotations=stack("rot_0", "rot_1", "rot_2", "rot_3"),
    )
    view = next((parsed for parsed in map(parse_view_comment, header.comments) if parsed is not None), None)
    return splats, view
