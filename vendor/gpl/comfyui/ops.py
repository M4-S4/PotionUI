# Vendored from ComfyUI — https://github.com/comfyanonymous/ComfyUI
# Upstream path: comfy/ops.py @ unknown; vendored ~2025
# License: GPL-3.0 (see LICENSE in this directory). Copyright (c) comfyanonymous and contributors.

from __future__ import annotations

import json
import logging

import torch

logger = logging.getLogger(__name__)

_FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}

# Largest finite magnitude representable in e4m3 (torch.finfo(float8_e4m3fn).max).
_E4M3_MAX = 448.0


_COMFY_QUANT_FP8_FORMATS = frozenset({"float8_e4m3fn", "float8_e5m2"})
_COMFY_QUANT_INT8_TENSORWISE = "int8_tensorwise"
_CONVROT_DEFAULT_GROUPSIZE = 256


def _parse_comfy_quant(blob: torch.Tensor | None) -> dict | None:
    """Decode a ``comfy_quant`` descriptor (UTF-8 JSON bytes stored as a uint8
    tensor) into its dict. A missing OR unparseable blob both return ``None`` --
    the caller then keeps the pre-existing bare-``weight_scale`` dequant
    behavior, matching checkpoints saved before ComfyUI added this descriptor.
    """
    if blob is None:
        return None
    try:
        parsed = json.loads(bytes(blob.tolist()).decode("utf-8"))
    except Exception:  # noqa: BLE001 — any decode/parse failure is "no descriptor"
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_convrot_config(layer_conf: dict, layer_name: str = "") -> tuple[bool, int]:
    """Read ``convrot`` + ``convrot_groupsize`` out of a comfy_quant descriptor.

    Both live at the descriptor's top level in every published checkpoint; the
    ``"params"`` nesting is also read because ComfyUI's own loader accepts it,
    though no released file uses it.

    Published ConvRot descriptors do NOT all state a group size — the compact
    43-byte variant in the wild is exactly ``{"format":...,"convrot":true}`` — so
    falling back to the format default is a real code path, not a guard. It logs,
    because un-rotating at the wrong group size is arithmetically valid and
    silently wrong output.
    """
    params_conf = layer_conf.get("params")
    if not isinstance(params_conf, dict):
        params_conf = {}
    convrot = bool(layer_conf.get("convrot", params_conf.get("convrot", False)))
    for conf in (layer_conf, params_conf):
        if "convrot_groupsize" in conf:
            return convrot, int(conf["convrot_groupsize"])
    if convrot:
        logger.info(
            "ConvRot layer '%s' states no group size (%s); using the format default %d",
            layer_name, sorted(layer_conf), _CONVROT_DEFAULT_GROUPSIZE,
        )
    return convrot, _CONVROT_DEFAULT_GROUPSIZE


def _reject_ambiguous_int8(
    weight: torch.Tensor | None, scale: torch.Tensor | None, layer_name: str,
) -> None:
    """Refuse an integer-coded weight whose rotation state cannot be determined.

    With no descriptor there is nothing to say whether int8 codes are plain or
    ConvRot-rotated. A 0-dim scale means plain tensorwise and dequantises
    correctly, but a per-output-channel scale fits both a plain per-channel
    quantiser and a rotated one — and picking wrong skips the un-rotation and
    produces confident noise. Two of four published int8 checkpoints carry no
    file-level quantisation header at all, so the per-layer descriptor is the only
    thing that ever answers this.
    """
    if weight is None or weight.is_floating_point():
        return
    if scale is None or scale.dim() == 0:
        return
    raise ValueError(
        f"int8 layer '{layer_name}' has a per-output-channel weight_scale but no comfy_quant "
        "descriptor: plain and ConvRot-rotated codes are indistinguishable here"
    )


def _check_quant_format_supported(layer_conf: dict, layer_name: str) -> None:
    """Raise for any ``comfy_quant`` format this module cannot dequantize
    correctly. See the module comment above for what's supported.
    """
    fmt = layer_conf.get("format")
    if fmt in _COMFY_QUANT_FP8_FORMATS or fmt == _COMFY_QUANT_INT8_TENSORWISE:
        return
    raise ValueError(f"quant format {fmt!r} is not supported yet (layer '{layer_name}')")


# ---------------------------------------------------------------------------
# int8_tensorwise ConvRot (QuaRot-style Hadamard rotation) dequantisation
# ---------------------------------------------------------------------------
#
# Written from the published on-disk FORMAT, not ported from an implementation.
# comfy-kitchen's `eager` backend (Apache-2.0, github.com/Comfy-Org/comfy-kitchen)
# is where the format is documented — comfy/quant_ops.py has no pure-torch path
# and dispatches this format to compiled kernels — but no code was taken from it,
# and in particular nothing from its `TensorWiseINT8Layout` (documented as coming
# from dxqb/OneTrainer, whose licence this project has NOT verified) or from the
# parts of its eager backend derived from PyTorch AO. The format facts below are
# all that is needed, and they are facts about the file, not expression:
#
#   * `<layer>.weight`       int8, [out_features, in_features]. One code per byte,
#                            so the stored shape IS the layer's true shape (unlike
#                            nvfp4, which packs two codes per byte).
#   * `<layer>.weight_scale` f32. [out_features, 1] for ConvRot (per output
#                            channel); a 0-dim scalar for plain tensorwise. The
#                            scale's rank is the on-the-wire signal for which.
#   * `<layer>.comfy_quant`  the per-layer descriptor.
#   * `<layer>.bias`         keeps the checkpoint's original float dtype.
#   There is no zero point (codes are symmetric), no `input_scale`, and no
#   `weight_scale_2`.
#
# The rotation: the weight is rotated OFFLINE, grouped along in_features in
# blocks of `convrot_groupsize`, by a normalized regular Hadamard matrix built by
# Kronecker-powering the 4x4 base below and dividing by sqrt(group_size). That
# construction is symmetric and involutory (`H == H.T == H^-1`, verified in the
# tests at 4/16/256/1024), which is what makes dequantisation possible at all:
# undo the scale, then apply the SAME rotation a second time to land back in the
# checkpoint's original, unrotated basis. Group sizes are therefore powers of 4.
#
# The result is an ordinary dequantized weight. It needs no special-casing for
# the plain (unrotated) activation this module always feeds it — activation-side
# rotation only exists on an int8-matmul fast path not implemented here — nor for
# LoRA deltas, which `apply_lora_deltas` adds in that same original basis exactly
# as for every other layer type.
#
# Fidelity note: the reference builds its Hadamard at fp32 for dequantisation,
# while this un-rotates in the activation's compute dtype to avoid a full
# weight-sized fp32 intermediate on every forward (151MB for one 6144x6144
# layer). Measured on a 512x1024 layer at group 256: int8 quantisation alone
# costs 0.786% mean relative error against the original weight, and doing the
# un-rotation in bf16 instead of fp32 takes that to 0.831% — an addition well
# inside the grid error the format already carries.
_CONVROT_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

# Format constant: THE regular 4x4 Hadamard the rotation is built from. Other
# valid H4s exist and are equally orthogonal, but would decode a ConvRot
# checkpoint to noise — this matrix is part of the file format, not a choice.
_CONVROT_H4 = ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1))


def _convrot_group_exponent(group_size: int) -> int:
    """The ``k`` in ``group_size == 4 ** k``, or raise.

    Integer arithmetic rather than ``log(group_size, 4)``: the group size gates a
    reshape of real checkpoint weights, and a float log landing a hair off an
    exact power would reject a valid file.
    """
    remainder = group_size
    exponent = 0
    while remainder > 1 and remainder % 4 == 0:
        remainder //= 4
        exponent += 1
    if group_size < 4 or remainder != 1:
        raise ValueError(f"ConvRot group size must be a power of 4, got {group_size}")
    return exponent


def _build_convrot_hadamard(
    group_size: int, device: "torch.device | str", dtype: torch.dtype,
) -> torch.Tensor:
    """Normalized regular Hadamard matrix for ConvRot, cached per (size, device, dtype)."""
    cache_key = (group_size, str(device), dtype)
    cached = _CONVROT_HADAMARD_CACHE.get(cache_key)
    if cached is not None:
        return cached
    exponent = _convrot_group_exponent(group_size)
    base = torch.tensor(_CONVROT_H4, dtype=dtype, device=device)
    h = base
    for _ in range(exponent - 1):
        h = torch.kron(h, base)
    h = h / (group_size ** 0.5)
    _CONVROT_HADAMARD_CACHE[cache_key] = h
    return h


def _convrot_unrotate_weight(weight: torch.Tensor, hadamard: torch.Tensor, group_size: int) -> torch.Tensor:
    """Undo ConvRot's offline weight rotation (the rotation is symmetric and
    self-inverse, so applying it a second time is the correct un-rotation)."""
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} not divisible by ConvRot group size {group_size}")
    n_groups = in_f // group_size
    grouped = weight.reshape(out_f, n_groups, group_size)
    return torch.matmul(grouped, hadamard).reshape(out_f, in_f)


# ---------------------------------------------------------------------------
# NVFP4 (4-bit) dequantisation
# ---------------------------------------------------------------------------
#
# Format (verified against comfy/quant_ops.py QUANT_ALGOS["nvfp4"] + the pure-
# torch reference in comfy/float.py; the compiled comfy_kitchen kernel is not
# importable here, so float.py is the authority):
#   * weight        uint8  [out, in//2]     two 4-bit e2m1 codes per byte
#   * weight_scale  fp8e4m3 [out, in//16]   per-16-element block scale, stored in
#                                           NVIDIA "to_blocked" swizzled layout
#   * weight_scale_2 f32    scalar          per-tensor scale
#   dequant(v) = e2m1(code) * unblock(weight_scale) * weight_scale_2
#
# Nibble order (comfy/float.py line 95, `packed = (fp4[0::2] << 4) | fp4[1::2]`):
#   even element -> HIGH nibble, odd element -> LOW nibble.
# e2m1 code = (sign<<3)|(exp<<1)|mantissa (comfy/float.py `stochastic_float_to_fp4_e2m1`).

QUANT_NVFP4 = "nvfp4"
_NVFP4_BLOCK = 16
# Largest finite magnitude representable in e2m1 (the fp4 grid {0,.5,1,...,6}).
_F4_MAX = 6.0

# code -> signed value; code = (sign<<3)|(exp<<1)|mantissa. Magnitudes are the
# standard e2m1 grid {0,.5,1,1.5,2,3,4,6}.
_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

# 256-entry LUTs: the packed byte -> its high/low nibble's e2m1 value,
# precomputed once so the per-forward path is a single gather instead of a
# bit-shift + mask + int64 `codes` tensor built via strided [:, 0::2] writes.
# A uint8 index tensor is treated as a boolean MASK by torch's fancy indexing
# (not a gather index), so the byte range is expressed in int32 here -- the
# smallest integer dtype indexing actually accepts (not int64/long).
_BYTE_ARANGE_I32 = torch.arange(256, dtype=torch.int32)
_E2M1_LUT_HIGH = _E2M1_LUT[(_BYTE_ARANGE_I32 >> 4) & 0x0F]
_E2M1_LUT_LOW = _E2M1_LUT[_BYTE_ARANGE_I32 & 0x0F]

# Cache the inverse "to_blocked" permutation per padded (rows, cols) shape.
_INV_BLOCKED_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def _to_blocked(m: torch.Tensor) -> torch.Tensor:
    """NVIDIA cublas d-block scale layout (verbatim from comfy/float.py)."""
    rows, cols = m.shape
    nrb = (rows + 127) // 128
    ncb = (cols + 3) // 4
    padded = m
    if (rows, cols) != (nrb * 128, ncb * 4):
        padded = torch.zeros((nrb * 128, ncb * 4), device=m.device, dtype=m.dtype)
        padded[:rows, :cols] = m
    blocks = padded.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.reshape(nrb * 128, ncb * 4)


def _inv_blocked_index(rows: int, cols: int) -> torch.Tensor:
    """Permutation that un-blocks a stored [rows, cols] scale back to natural order."""
    key = (rows, cols)
    inv = _INV_BLOCKED_CACHE.get(key)
    if inv is None:
        idx = _to_blocked(torch.arange(rows * cols).reshape(rows, cols).float()).flatten().long()
        inv = torch.empty(rows * cols, dtype=torch.long)
        inv[idx] = torch.arange(rows * cols)
        _INV_BLOCKED_CACHE[key] = inv
    return inv


def _unblock_scale(block_stored: torch.Tensor, out_features: int, num_blocks: int) -> torch.Tensor:
    """Recover the natural [out, num_blocks] block scale from the swizzled store."""
    ph, pw = block_stored.shape  # padded to (mult of 128, mult of 4)
    inv = _inv_blocked_index(ph, pw).to(block_stored.device)
    natural = block_stored.to(torch.float32).flatten()[inv].reshape(ph, pw)
    return natural[:out_features, :num_blocks]


def _dequant_values(packed: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """Recover the raw (unscaled) e2m1 magnitudes from packed nvfp4 codes.

    Gathers both nibbles of every byte directly from the 256-entry
    ``_E2M1_LUT_HIGH``/``_E2M1_LUT_LOW`` tables -- no int64 ``codes``
    tensor, no strided ``[:, 0::2]``/``[:, 1::2]`` scatter writes. Interleaving
    high/low via ``stack(..., dim=-1).view(...)`` reproduces the same element
    order the old scatter produced (even columns = high nibble, odd = low).
    """
    idx = packed.to(torch.int32)
    device = packed.device
    high = _E2M1_LUT_HIGH.to(device)[idx]
    low = _E2M1_LUT_LOW.to(device)[idx]
    return torch.stack((high, low), dim=-1).view(out_features, in_features)


def _scale_values(
    values: torch.Tensor, scale: torch.Tensor, out_features: int, in_features: int, num_blocks: int,
) -> torch.Tensor:
    """Apply a ``[out, num_blocks]`` scale to ``[out, in]`` dequantised values.

    Broadcasting a ``[out, blocks, 1]`` view of ``scale`` against a
    ``[out, blocks, 16]`` view of ``values`` performs the identical fp32
    elementwise multiply a ``repeat_interleave``d copy would --
    same two floats multiplied per output element, no allocation to hold the
    16x-duplicated scale.
    """
    scaled = values.view(out_features, num_blocks, _NVFP4_BLOCK) * scale.view(out_features, num_blocks, 1)
    return scaled.view(out_features, in_features)


def precompute_nvfp4_scale(
    block_scale: torch.Tensor, tensor_scale: torch.Tensor, out_features: int, in_features: int,
) -> torch.Tensor:
    num_blocks = in_features // _NVFP4_BLOCK
    block_nat = _unblock_scale(block_scale, out_features, num_blocks)
    return block_nat * tensor_scale.to(torch.float32)


def dequantize_nvfp4(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    tensor_scale: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Dequantise an nvfp4 weight to an fp32 ``[out_features, in_features]`` tensor.
    """
    device = packed.device
    values = _dequant_values(packed, out_features, in_features)
    num_blocks = in_features // _NVFP4_BLOCK
    scale = _unblock_scale(block_scale, out_features, num_blocks).to(device) * tensor_scale.to(torch.float32).to(device)
    return _scale_values(values, scale, out_features, in_features, num_blocks)


def _quantize_nvfp4_dynamic(x2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dynamic per-tensor+per-block nvfp4 activation quantisation for the
    ``_scaled_mm_v2`` fast path.

    ``x2d`` is ``(M, K)`` with ``K % 32 == 0`` (already checked by the
    caller). Returns ``(packed_u8 [M, K // 2], block_scale_swizzled_e4m3,
    global_scale_f32_scalar)`` in the SAME on-disk shape/layout as a
    checkpoint's nvfp4 weight, so :func:`dequantize_nvfp4` (or a
    :func:`_to_blocked` weight scale) can be compared against it directly.

    Algorithm ported from ``comfy/float.py``'s round-to-nearest branch (the
    same authority the weight-format port uses) — its stochastic-
    rounding branch is for SAVING checkpoints and is not used here; a live
    per-forward activation quantiser always rounds to nearest. One
    deliberate addition beyond that reference: an all-zero 16-element block
    (e.g. a padded/masked activation slice — routine at runtime, unlike a
    static trained weight block, which is essentially never exactly zero)
    divides by a zero block scale in the reference formula, i.e. ``0/0 =
    NaN``; the divisor here is floored at a tiny epsilon so that case yields
    an exact ``0`` instead, matching what the (also exactly-zero) stored
    scale would dequantise to anyway.
    """
    m, k = x2d.shape
    x = x2d.detach().reshape(m, -1, _NVFP4_BLOCK).to(torch.float32, copy=True)
    block_amax = x.abs().amax(-1)
    amax = block_amax.amax().clamp(min=1e-12)
    tensor_scale = amax / (_F4_MAX * _E4M3_MAX)
    block = torch.clamp(block_amax / _F4_MAX / tensor_scale, max=_E4M3_MAX).to(torch.float8_e4m3fn)
    safe_block = block.to(torch.float32).clamp(min=torch.finfo(torch.float32).tiny)
    x.div_((tensor_scale * safe_block).unsqueeze(-1))
    sign = torch.signbit(x).to(torch.uint8)
    x.abs_()
    exp = torch.log2(x).add_(1.1925).floor_().clamp_(0, 3)
    normal = exp > 0
    exp_bits = exp.to(torch.uint8)
    mant = torch.pow(2.0, exp.sub_(1.0), out=exp)
    torch.div(x, mant, out=mant).sub_(1.0).mul_(2.0)
    torch.where(normal, mant, x.mul_(2.0), out=mant)
    del x, normal, exp
    mant = mant.round_().clamp_(0, 1).to(torch.uint8)
    codes = ((sign << 3) | (exp_bits << 1) | mant).reshape(m, k)
    packed = (codes[:, 0::2] << 4) | codes[:, 1::2]
    block_sw = _to_blocked(block.to(torch.float32)).to(torch.float8_e4m3fn)
    return packed, block_sw, tensor_scale.reshape(())
