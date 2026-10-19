from __future__ import annotations

import json

WEIGHT_SUFFIX = ".weight"
SCALE_SUFFIX = ".weight_scale"
DESCRIPTOR_SUFFIX = ".comfy_quant"
QUANT_DESCRIPTOR = {"format": "float8_e4m3fn", "full_precision_matrix_mult": True}
DESCRIPTOR_BYTES = json.dumps(QUANT_DESCRIPTOR).encode("utf-8")
SIDECAR_SUFFIXES = (SCALE_SUFFIX, DESCRIPTOR_SUFFIX, ".scale_weight", ".weight_scale_2", ".input_scale")
FP8_DTYPES = ("F8_E4M3", "F8_E5M2")
