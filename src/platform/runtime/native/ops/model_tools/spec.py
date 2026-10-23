from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import torch

DTYPE_TO_TORCH: dict[str, torch.dtype] = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "BOOL": torch.bool,
}

TORCH_TO_DTYPE: dict[torch.dtype, str] = {v: k for k, v in DTYPE_TO_TORCH.items()}

DTYPE_BYTES: dict[str, int] = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "BOOL": 1,
}

FLOAT_DTYPES = frozenset({"BF16", "F16", "F32"})


@dataclass(frozen=True, slots=True)
class TensorSpec:
    key: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.numel * DTYPE_BYTES[self.dtype]


TensorStream = tuple[list[TensorSpec], Iterator[tuple[str, torch.Tensor]]]
