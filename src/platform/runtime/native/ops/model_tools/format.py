from __future__ import annotations

WEIGHT_SUFFIX = ".weight"
SCALE_SUFFIX = ".weight_scale"
SCALE2_SUFFIX = ".weight_scale_2"
DESCRIPTOR_SUFFIX = ".comfy_quant"
SIDECAR_SUFFIXES = (SCALE_SUFFIX, DESCRIPTOR_SUFFIX, ".scale_weight", SCALE2_SUFFIX, ".input_scale")
FP8_DTYPES = ("F8_E4M3", "F8_E5M2")
