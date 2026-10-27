# Vendored from ComfyUI — https://github.com/comfyanonymous/ComfyUI
# Upstream path: comfy/ops.py @ unknown; vendored ~2025
# License: GPL-3.0 (see LICENSE in this directory). Copyright (c) comfyanonymous and contributors.
# ``linear_with_lora_deltas`` / ``partition_output_branch_deltas`` are PotionUI-original.
# They move a plain LoRA delta off the weight and onto the activation for the DEQUANT
# forwards, reusing the ``_lora_output_branch`` seam the fp8/nvfp4 GEMM fast paths already
# had: ``x @ (up @ down).T == (x @ down.T) @ up.T``, so no ``(out, in)`` delta is
# materialised and no weight is cloned per forward. Deltas that have no such factorisation
# (LoKr) keep the weight-side ``apply_lora_deltas`` path. Upstream ComfyUI bakes patches
# into the weight through ``ModelPatcher`` instead and has no per-forward delta at all.

"""Layer operation namespaces with cast-on-forward and scaled-fp8 support.

Ported and trimmed from ComfyUI's ``comfy/ops.py``. The whole point is the
``operations=`` seam: an arch module is written against ``operations.Linear``,
``operations.Conv2d`` etc. instead of ``torch.nn.*``, and the loader picks a
namespace that matches the checkpoint's storage dtype / quantisation:

  * ``disable_weight_init`` — plain torch layers with init skipped (weights are
    always overwritten by ``load_state_dict``). Used when storage == compute.
  * ``manual_cast`` — casts weights to the activation dtype per forward. Used
    when storage dtype (e.g. fp8 or fp16) differs from compute dtype.

The ops-namespace protocol (what ``operations`` exposes) is:
    Linear, Conv2d, Conv3d, GroupNorm, LayerNorm, RMSNorm, Embedding
each constructible with the same signature as its ``torch.nn`` counterpart.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}

# Largest finite magnitude representable in e4m3 (torch.finfo(float8_e4m3fn).max).
_E4M3_MAX = 448.0

RUNTIME_FLAGS: dict = {}


def cast_to(
    tensor: torch.Tensor | None,
    dtype: torch.dtype,
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.device == device and tensor.dtype == dtype:
        return tensor
    # ``non_blocking`` is only actually async when the source is pinned CPU memory
    # (partial-residency streaming pins the streamed weights); it degrades to a
    # synchronous copy otherwise, so it is always safe to pass through.
    return tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)


def cast_bias_weight(
    module: "CastWeightBiasOp",
    input: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Cast a module's weight/bias to the activation dtype+device.

    A module streamed from pinned CPU RAM (partial residency) sets
    ``stream_non_blocking`` so the per-forward H2D weight copy overlaps compute on
    the current stream; resident modules leave it False (no copy happens anyway).
    """
    if input is not None:
        if dtype is None:
            dtype = input.dtype
        if device is None:
            device = input.device
    nb = getattr(module, "stream_non_blocking", False)
    weight = cast_to(module.weight, dtype, device, non_blocking=nb)
    bias = cast_to(module.bias, dtype, device, non_blocking=nb) if module.bias is not None else None
    return weight, bias


class CastWeightBiasOp:
    """Marker mixin carrying the cast-on-forward flags."""

    comfy_cast_weights = False
    weight_function: list = []
    bias_function: list = []
    # Set True on a module whose weights are STREAMED from pinned CPU RAM under
    # partial residency: the per-forward H2D copy is issued non_blocking so it
    # overlaps compute. Resident modules leave it False. See ``memory/partial.py``.
    stream_non_blocking = False
    # Optional runtime LoRA deltas applied to the (cast/dequant) weight per
    # forward. ``None`` = no LoRA (default). Populated by ``lora.apply`` on
    # cast-mode Linears; standard-mode Linears patch their weight in place
    # instead. See ``apply_lora_deltas``.
    lora_deltas: list | None = None

    @contextmanager
    def prepared_linear(self, sample: torch.Tensor):
        yield self


def apply_lora_deltas(weight: torch.Tensor, deltas: "list | None") -> torch.Tensor:
    """Return ``weight`` with low-rank LoRA deltas added (compute-side).

    Each delta is duck-typed (``lora.key_mapping.LoraDelta``): ``down`` ``(rank,
    in)``, ``up`` ``(out|slice, rank)``, ``alpha``, ``scale`` (strength), and an
    optional ``target_slice`` ``(dim, start, length)`` for partial patches of a
    fused weight (e.g. a diffusers ``to_q`` delta into the native fused ``qkv``).
    Composes with dequant/cast: the caller passes the already-materialised
    compute-dtype weight, so fp8 storage stays untouched. Math is done in fp32
    (``W += (scale * alpha / rank) * up @ down``) then cast back to ``weight``'s
    dtype, matching ComfyUI's intermediate-fp32 convention.
    """
    if not deltas:
        return weight
    out = weight.clone()
    for d in deltas:
        up = d.up.to(device=out.device, dtype=torch.float32)
        down = d.down.to(device=out.device, dtype=torch.float32)
        if getattr(d, "kron", False):
            # LoKr: up/down are the Kronecker factors w1/w2; d.alpha is the
            # pre-divided LyCORIS alpha/dim factor (see LoraDelta docstring).
            delta = torch.kron(up, down).reshape(out.shape) * (float(d.scale) * float(d.alpha))
        else:
            rank = d.down.shape[0]
            delta = (up @ down) * (float(d.scale) * float(d.alpha) / rank)
        delta = delta.to(dtype=out.dtype)
        if d.target_slice is not None:
            _dim, start, length = d.target_slice
            out[start : start + length].add_(delta)
        else:
            out.add_(delta)
    return out


def partition_output_branch_deltas(
    deltas: "list | None", out_features: int,
) -> "tuple[list, list]":
    """Split ``deltas`` into ``(output_side, weight_side)``.

    Per-delta, unlike :func:`_deltas_output_branch_ok`, which the GEMM fast
    paths use as an all-or-nothing gate because they must either take the
    native kernel for the whole layer or not at all. Here a stack mixing one
    LoKr adapter with two plain ones can still put the plain two on the
    activation and only pay a weight-shaped delta for the LoKr one.
    """
    if not deltas:
        return [], []
    output_side, weight_side = [], []
    for d in deltas:
        (output_side if _deltas_output_branch_ok([d], out_features) else weight_side).append(d)
    return output_side, weight_side


def _fuse_output_branch_deltas(
    deltas: "list", dtype: torch.dtype, device: "torch.device",
) -> "tuple[torch.Tensor, torch.Tensor, float]":
    """One ``(down_cat, up_cat, alpha)`` standing in for a whole group of
    deltas that share a ``target_slice``.

    ``Σ_i coeff_i · up_i @ (down_i @ x) == up_cat @ (down_cat @ x)`` with the
    factors concatenated along the rank axis, so a layer carrying N adapters
    costs two matmuls instead of 2N — and, paired with ``addmm_``, ONE
    read-modify-write of the ``(M, out_features)`` output instead of N.

    Equal coefficients ride ``addmm_``'s own ``alpha``. Unequal ones (the
    usual case: same rank, different user strengths) are folded into each
    adapter's own rank rows in fp32 BEFORE the compute-dtype cast — strictly
    better rounding than the pre-fusion path, which applied the coefficient to
    an already-rounded compute-dtype ``(M, out_features)`` term.
    """
    coeffs = [float(d.scale) * float(d.alpha) / d.down.shape[0] for d in deltas]
    if len(deltas) == 1:
        d = deltas[0]
        return (d.down.to(device=device, dtype=dtype),
                d.up.to(device=device, dtype=dtype), coeffs[0])
    up_cat = torch.cat([d.up for d in deltas], 1).to(device=device, dtype=dtype)
    if all(c == coeffs[0] for c in coeffs):
        down_cat = torch.cat([d.down for d in deltas], 0).to(device=device, dtype=dtype)
        return down_cat, up_cat, coeffs[0]
    down_cat = torch.cat(
        [d.down.to(device=device, dtype=torch.float32) * c for d, c in zip(deltas, coeffs)], 0,
    ).to(dtype)
    return down_cat, up_cat, 1.0


NATIVE_LORA_FUSED_ENV = "NATIVE_LORA_FUSED"
_LORA_FUSED_ATTR = "_lora_fused_cache"
_lora_path_layers: "dict[str, set[int]]" = {}


def _lora_fused_enabled() -> bool:
    return bool(RUNTIME_FLAGS.get("native_lora_fused", True))


def _note_lora_path(layer, path: str) -> None:
    _lora_path_layers.setdefault(path, set()).add(id(layer))


def log_lora_path_summary() -> None:
    if not _lora_path_layers:
        return
    parts = ", ".join(f"{path} {len(layers)}" for path, layers in sorted(_lora_path_layers.items()))
    _lora_path_layers.clear()
    logger.info("lora forward paths (layers): %s", parts)


def invalidate_lora_fused_cache(layer) -> None:
    layer.__dict__.pop(_LORA_FUSED_ATTR, None)


def _with_lokr(path: str, deltas: "list") -> str:
    return f"{path} (+lokr)" if any(getattr(d, "kron", False) for d in deltas) else path


def _lora_dequant_path(weight_side: "list", reason: str, output_side: "list" = ()) -> str:
    if weight_side:
        return "weight-side (lokr)" if any(getattr(d, "kron", False) for d in weight_side) else "weight-side"
    return _with_lokr(f"dequant ({reason})", output_side)


def _lokr_output_ok(d, out_features: int) -> bool:
    up, down = d.up, d.down
    return (_lora_fused_enabled() and d.target_slice is None and up.dim() == 2 and down.dim() == 2
            and up.shape[0] * down.shape[0] == out_features)


class _LokrBranch:
    __slots__ = ("w1_first", "chain1", "chain2", "width")

    def __init__(self, w1_first: bool, chain1: list, chain2: list, width: int) -> None:
        self.w1_first = w1_first
        self.chain1 = chain1
        self.chain2 = chain2
        self.width = width


def _lokr_side_chain(full: torch.Tensor, factors: "tuple | None") -> list:
    if factors is not None and sum(t.numel() for t in factors) < full.numel():
        return list(factors)
    return [full]


def _build_lokr_branch(d, dtype: torch.dtype, device: "torch.device") -> "_LokrBranch | None":
    coeff = float(d.scale) * float(d.alpha)
    if coeff == 0.0:
        return None
    factors = getattr(d, "kron_factors", None) or (None, None)
    chain1 = _lokr_side_chain(d.up, factors[0])
    chain2 = _lokr_side_chain(d.down, factors[1])
    chain1 = [(chain1[0].to(device=device, dtype=torch.float32) * coeff).to(dtype)] + [
        m.to(device=device, dtype=dtype) for m in chain1[1:]]
    chain2 = [m.to(device=device, dtype=dtype) for m in chain2]
    o1, i1 = chain1[0].shape[0], chain1[-1].shape[1]
    o2, i2 = chain2[0].shape[0], chain2[-1].shape[1]
    c1 = sum(m.numel() for m in chain1)
    c2 = sum(m.numel() for m in chain2)
    w1_first = i2 * c1 + o1 * c2 <= i1 * c2 + o2 * c1
    if w1_first:
        widths = [m.shape[0] * i2 for m in chain1] + [o1 * m.shape[0] for m in chain2]
    else:
        widths = [i1 * m.shape[0] for m in chain2] + [m.shape[0] * o2 for m in chain1]
    return _LokrBranch(w1_first, chain1, chain2, max(widths))


def _add_lokr_branch(out2d: torch.Tensor, x2d: torch.Tensor, branch: _LokrBranch) -> None:
    chain1, chain2 = branch.chain1, branch.chain2
    o1, i1 = chain1[0].shape[0], chain1[-1].shape[1]
    o2, i2 = chain2[0].shape[0], chain2[-1].shape[1]
    step = max(1, _NVFP4_LORA_BRANCH_CHUNK_BYTES // max(1, branch.width * out2d.element_size()))
    for start in range(0, x2d.shape[0], step):
        xc = x2d[start:start + step]
        oc = out2d[start:start + step]
        n = xc.shape[0]
        if branch.w1_first:
            t = xc.reshape(n, i1, i2)
            for m in reversed(chain1):
                t = torch.matmul(m, t)
            t = t.reshape(n * o1, i2)
            for m in reversed(chain2[1:]):
                t = t @ m.t()
            oc.view(n * o1, o2).addmm_(t, chain2[0].t())
        else:
            t = xc.reshape(n * i1, i2)
            for m in reversed(chain2):
                t = t @ m.t()
            t = t.reshape(n, i1, o2)
            for m in reversed(chain1[1:]):
                t = torch.matmul(m, t)
            oc.view(n, o1, o2).baddbmm_(chain1[0].expand(n, -1, -1), t)


def _lora_fused_key(deltas: "list", dtype: torch.dtype, device: "torch.device") -> tuple:
    return (dtype, device, tuple(
        (id(d), id(d.down), id(d.up), float(d.scale), float(d.alpha), d.target_slice,
         id(getattr(d, "kron_factors", None))) for d in deltas
    ))


def _build_lora_fused(deltas: "list", dtype: torch.dtype, device: "torch.device") -> list:
    groups: "dict" = {}
    for d in deltas:
        if getattr(d, "kron", False):
            continue
        coeff = float(d.scale) * float(d.alpha) / d.down.shape[0]
        if coeff != 0.0:
            groups.setdefault(d.target_slice, []).append((d, coeff))
    fused = []
    for target_slice, group in groups.items():
        first = group[0][0]
        if len(group) == 1 and first.down.dtype == dtype and first.up.dtype == dtype \
                and first.down.device == device and first.up.device == device:
            down_cat, up_cat_t = first.down, first.up.t()
        else:
            down_cat = torch.cat([d.down.to(device=device, dtype=dtype) for d, _c in group], 0)
            up_cat_t = torch.cat([d.up.to(device=device, dtype=dtype).t() for d, _c in group], 0)
            start = 0
            for d, _c in group:
                end = start + d.down.shape[0]
                d.down = down_cat[start:end]
                d.up = up_cat_t[start:end].t()
                start = end
        coeffs = [c for _d, c in group]
        if all(c == coeffs[0] for c in coeffs):
            alpha, scale = coeffs[0], None
        else:
            alpha = 1.0
            scale = torch.cat([
                torch.full((d.down.shape[0],), c, dtype=torch.float32, device=device) for d, c in group
            ])
        fused.append((target_slice, down_cat, up_cat_t, alpha, scale))
    for d in deltas:
        if getattr(d, "kron", False):
            branch = _build_lokr_branch(d, dtype, device)
            if branch is not None:
                fused.append(branch)
    return fused


def _cached_lora_fused(owner, deltas: "list", dtype: torch.dtype, device: "torch.device") -> list:
    entry = owner.__dict__.get(_LORA_FUSED_ATTR)
    if entry is not None and entry[0] == _lora_fused_key(deltas, dtype, device):
        return entry[2]
    fused = _build_lora_fused(deltas, dtype, device)
    owner.__dict__[_LORA_FUSED_ATTR] = (_lora_fused_key(deltas, dtype, device), tuple(deltas), fused)
    return fused


def _add_fused_lora_branch(out2d: torch.Tensor, x2d: torch.Tensor, fused: list) -> None:
    for entry in fused:
        if isinstance(entry, _LokrBranch):
            _add_lokr_branch(out2d, x2d, entry)
            continue
        target_slice, down_cat, up_cat_t, alpha, scale = entry
        hidden = x2d @ down_cat.t()
        if scale is not None:
            hidden.mul_(scale)
        if target_slice is None:
            out2d.addmm_(hidden, up_cat_t, alpha=alpha)
        else:
            _dim, col, length = target_slice
            out2d[:, col:col + length].addmm_(hidden, up_cat_t, alpha=alpha)


def _add_lora_output_branch(
    out: torch.Tensor, x: torch.Tensor, deltas: "list", out_features: int, owner=None,
) -> torch.Tensor:
    """``out`` plus the output-side contribution of ``deltas`` against ``x``.

    ``x`` keeps the caller's leading dims; the branch works on a flattened
    ``(M, in_features)`` view and lands back on ``out``'s shape.

    Under ``no_grad`` the contribution is accumulated straight INTO ``out`` by
    ``addmm_``. ``out`` is the freshly-allocated result of this layer's own
    ``F.linear`` and is aliased by nothing else, and ``addmm_`` fuses the
    ``up`` product with the accumulate, so no ``(M, out_features)`` term is
    ever materialised. That is the whole cost of this function at high token
    counts: a 24576-wide layer at ~8k tokens writes and re-reads ~395 MB per
    materialised term, which is why the pre-fusion form (one term per delta,
    then ``add_``) cost ~3x what this does while looking free at 1k tokens.
    Deltas are fused per ``target_slice`` group first, so a three-adapter
    stack pays that traffic once, not three times.

    Row chunking is gone with the term it existed to bound: the only
    intermediate left is ``(M, Σrank)`` (sub-megabyte at 8k tokens), and
    ``down_cat``/``up_cat`` are rank-sized. :func:`_lora_output_branch`, which
    must RETURN the branch rather than accumulate it, still chunks.

    With grad enabled (never on an inference path here, but the layer is an
    ordinary ``nn.Module`` and someone may) it falls back to
    :func:`_lora_output_branch`, which allocates rather than mutating.
    """
    if not deltas:
        return out
    x2d = x.reshape(-1, x.shape[-1])
    if torch.is_grad_enabled():
        branch = _lora_output_branch(x2d, deltas, out.dtype, out_features)
        return out + branch.reshape(out.shape)

    out2d = out.reshape(-1, out_features)
    x2d = x2d.to(out.dtype)
    if owner is not None and _lora_fused_enabled():
        _add_fused_lora_branch(out2d, x2d, _cached_lora_fused(owner, deltas, out.dtype, x2d.device))
        return out
    groups: "dict" = {}
    for d in deltas:
        if getattr(d, "kron", False):
            branch = _build_lokr_branch(d, out.dtype, x2d.device)
            if branch is not None:
                _add_lokr_branch(out2d, x2d, branch)
            continue
        groups.setdefault(d.target_slice, []).append(d)
    for target_slice, group in groups.items():
        down_cat, up_cat, alpha = _fuse_output_branch_deltas(group, out.dtype, x2d.device)
        hidden = x2d @ down_cat.t()
        if target_slice is None:
            out2d.addmm_(hidden, up_cat.t(), alpha=alpha)
        else:
            # A column slice of a row-major 2D tensor is a non-contiguous view;
            # addmm_ takes it as `self` (cuBLAS carries the leading-dimension
            # stride), so a sliced delta still accumulates in place.
            _dim, col, length = target_slice
            out2d[:, col:col + length].addmm_(hidden, up_cat.t(), alpha=alpha)
    return out


def linear_with_lora_deltas(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: "torch.Tensor | None",
    deltas: "list | None",
    out_features: int,
    owner=None,
) -> torch.Tensor:
    """``F.linear(input, weight, bias)`` with ``deltas`` applied, each on
    whichever side it can be.

    The weight-side form (:func:`apply_lora_deltas`) clones the materialised
    weight and builds an ``(out, in)`` delta on EVERY forward; the output-side
    form touches only ``(M, rank)`` and ``(M, out)`` intermediates. Both are
    the same number: ``x @ (up @ down).T == (x @ down.T) @ up.T``. Only a delta
    with no such factorisation (LoKr) still pays the weight-side cost.
    """
    output_side, weight_side = partition_output_branch_deltas(deltas, out_features)
    if weight_side:
        weight = apply_lora_deltas(weight, weight_side)
    out = F.linear(input, weight, bias)
    return _add_lora_output_branch(out, input, output_side, out_features, owner=owner)


class disable_weight_init:
    """torch layers with parameter init skipped (weights come from load)."""

    class Linear(torch.nn.Linear, CastWeightBiasOp):
        def reset_parameters(self) -> None:
            return None

        def forward_comfy_cast_weights(self, input: torch.Tensor) -> torch.Tensor:
            weight, bias = cast_bias_weight(self, input)
            if self.lora_deltas:
                _note_lora_path(self, "dequant (cast)")
            return linear_with_lora_deltas(input, weight, bias, self.lora_deltas, self.out_features, owner=self)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or self.lora_deltas:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            return super().forward(*args, **kwargs)

    class Conv2d(torch.nn.Conv2d, CastWeightBiasOp):
        def reset_parameters(self) -> None:
            return None

        def forward_comfy_cast_weights(self, input: torch.Tensor) -> torch.Tensor:
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            return super().forward(*args, **kwargs)

    class Conv3d(torch.nn.Conv3d, CastWeightBiasOp):
        def reset_parameters(self) -> None:
            return None

        def forward_comfy_cast_weights(self, input: torch.Tensor) -> torch.Tensor:
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            return super().forward(*args, **kwargs)

    class Conv1d(torch.nn.Conv1d, CastWeightBiasOp):
        def reset_parameters(self) -> None:
            return None

        def forward_comfy_cast_weights(self, input: torch.Tensor) -> torch.Tensor:
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            return super().forward(*args, **kwargs)

    class ConvTranspose1d(torch.nn.ConvTranspose1d, CastWeightBiasOp):
        def reset_parameters(self) -> None:
            return None

        def forward_comfy_cast_weights(self, input: torch.Tensor, output_size=None) -> torch.Tensor:
            weight, bias = cast_bias_weight(self, input)
            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding, self.kernel_size,
                self.dilation,
            )
            return F.conv_transpose1d(
                input, weight, bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation,
            )

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            return super().forward(*args, **kwargs)

    class GroupNorm(torch.nn.GroupNorm, CastWeightBiasOp):
        def reset_parameters(self) -> None:
            return None

        def forward_comfy_cast_weights(self, input: torch.Tensor) -> torch.Tensor:
            weight, bias = cast_bias_weight(self, input)
            return F.group_norm(input, self.num_groups, weight, bias, self.eps)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            return super().forward(*args, **kwargs)

    class LayerNorm(torch.nn.LayerNorm, CastWeightBiasOp):
        def reset_parameters(self) -> None:
            return None

        def forward_comfy_cast_weights(self, input: torch.Tensor) -> torch.Tensor:
            if self.weight is not None:
                weight, bias = cast_bias_weight(self, input)
            else:
                weight, bias = None, None
            return F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            return super().forward(*args, **kwargs)

    class RMSNorm(torch.nn.RMSNorm, CastWeightBiasOp):
        def reset_parameters(self) -> None:
            return None

        def forward_comfy_cast_weights(self, input: torch.Tensor) -> torch.Tensor:
            weight = cast_to(self.weight, input.dtype, input.device) if self.weight is not None else None
            return F.rms_norm(input, self.normalized_shape, weight, self.eps)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            return super().forward(*args, **kwargs)

    class Embedding(torch.nn.Embedding, CastWeightBiasOp):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            # non-persistent: int8_tensorwise dequant state, absent for a plain embedding.
            self.register_buffer("_int8_weight", None, persistent=False)
            self.register_buffer("_int8_weight_scale", None, persistent=False)
            self.register_buffer("_int8_convrot_hadamard", None, persistent=False)
            self._int8_convrot_groupsize: int | None = None

        def reset_parameters(self) -> None:
            self.bias = None
            return None

        def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        ) -> None:
            quant_blob = state_dict.pop(prefix + "comfy_quant", None)
            layer_conf = _parse_comfy_quant(quant_blob)
            if layer_conf is not None:
                fmt = layer_conf.get("format")
                if fmt != _COMFY_QUANT_INT8_TENSORWISE:
                    raise ValueError(
                        f"embedding quant format {fmt!r} is not supported (layer '{prefix.rstrip('.')}')"
                    )
                weight = state_dict.pop(prefix + "weight", None)
                scale = state_dict.pop(prefix + "weight_scale", None)
                if weight is None or scale is None:
                    raise ValueError(
                        f"int8 embedding '{prefix.rstrip('.')}' missing weight/weight_scale"
                    )
                self._int8_weight = weight if weight.dtype == torch.int8 else weight.view(torch.int8)
                self._int8_weight_scale = scale.to(torch.float32)
                convrot, groupsize = _extract_convrot_config(layer_conf, prefix.rstrip("."))
                if convrot:
                    self._int8_convrot_groupsize = groupsize
                    self._int8_convrot_hadamard = _build_convrot_hadamard(groupsize, device="cpu", dtype=torch.float32).clone()
                # Free the empty [num_embeddings, embedding_dim] float weight the
                # constructor allocated; the int8 table is stored on the side.
                self.weight = None
            super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs,
            )
            for name in ("weight", "weight_scale", "comfy_quant"):
                key = prefix + name
                if key in missing_keys:
                    missing_keys.remove(key)

        def forward_comfy_cast_weights(self, input: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
            if self._int8_weight is not None:
                # Optimized path: gather int8 rows first, dequantise only the
                # selected rows (matches the plain/ConvRot int8
                # Linear formula, applied post-gather since both the per-row
                # scale and the ConvRot rotation act within a row only).
                target = out_dtype if out_dtype is not None else torch.bfloat16
                rows = cast_to(self._int8_weight, torch.int8, input.device)[input]
                x = rows.to(torch.float32)
                scale = cast_to(self._int8_weight_scale, torch.float32, input.device)
                if scale.dim() > 0:
                    x = x * scale.reshape(-1)[input].unsqueeze(-1)
                else:
                    x = x * scale
                if self._int8_convrot_hadamard is not None:
                    orig_shape = x.shape
                    flat = x.reshape(-1, orig_shape[-1])
                    hadamard = cast_to(self._int8_convrot_hadamard, flat.dtype, flat.device)
                    flat = _convrot_unrotate_weight(flat, hadamard, self._int8_convrot_groupsize)
                    x = flat.reshape(orig_shape)
                return x.to(target)
            # Embeddings carry no activation to borrow a dtype from; fp8 weights
            # dequantise to out_dtype (bf16 by default).
            target = out_dtype
            if target is None:
                target = self.weight.dtype if self.weight.dtype not in _FP8_DTYPES else torch.bfloat16
            weight = cast_to(self.weight, target, self.weight.device)
            return F.embedding(
                input, weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse,
            )

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or self._int8_weight is not None:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            return super().forward(*args, **kwargs)


class manual_cast(disable_weight_init):
    """Same layers, but always cast weights to the activation dtype per forward."""

    class Linear(disable_weight_init.Linear):
        comfy_cast_weights = True

    class Conv2d(disable_weight_init.Conv2d):
        comfy_cast_weights = True

    class Conv3d(disable_weight_init.Conv3d):
        comfy_cast_weights = True

    class Conv1d(disable_weight_init.Conv1d):
        comfy_cast_weights = True

    class ConvTranspose1d(disable_weight_init.ConvTranspose1d):
        comfy_cast_weights = True

    class GroupNorm(disable_weight_init.GroupNorm):
        comfy_cast_weights = True

    class LayerNorm(disable_weight_init.LayerNorm):
        comfy_cast_weights = True

    class RMSNorm(disable_weight_init.RMSNorm):
        comfy_cast_weights = True

    class Embedding(disable_weight_init.Embedding):
        comfy_cast_weights = True


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


def _deltas_output_branch_ok(deltas: "list | None", out_features: int) -> bool:
    """True iff every delta in ``deltas`` is expressible as an output-side
    low-rank branch: ``((x @ down.T) @ up.T) * (scale * alpha /
    rank)``, algebraically identical to :func:`apply_lora_deltas`'s
    weight-side ``up @ down`` add for that delta.

    A dim-0 ``target_slice`` delta (e.g. a q/k/v-fused weight, where LoRA
    deltas authored per-projection land as a slice of the fused output rows)
    is included: :func:`_lora_output_branch` adds its term into
    ``out[..., start:start+length]``, mirroring the weight-side add's
    ``out[_slice_of(target_slice)]`` exactly. ``out_features`` is required
    here to validate the slice's bounds — the branch function only ever sees
    one delta at a time and can't cross-check them against each other.

    Excluded, conservatively:

      * LoKr (``d.kron``) whose factors don't tile ``out_features`` or that
        targets a slice, and every LoKr when ``NATIVE_LORA_FUSED=off``;
        accepted ones run as the activation-side Kronecker contraction
        (:func:`_add_lokr_branch`).
      * anything that doesn't duck-type as a plain ``LoraDelta`` (missing
        attrs, non-tensor ``up``/``down``, a rank mismatch between them, or a
        ``target_slice`` that isn't a dim-0 slice within ``out_features``).

    Any of the above → ``False``, which routes the caller to today's dequant
    fallback instead.
    """
    if not deltas:
        return True
    for d in deltas:
        try:
            kron = d.kron
            down, up, target_slice = d.down, d.up, d.target_slice
        except AttributeError:
            return False
        if not isinstance(down, torch.Tensor) or not isinstance(up, torch.Tensor):
            return False
        if kron:
            if not _lokr_output_ok(d, out_features):
                return False
            continue
        if down.dim() != 2 or up.dim() != 2 or down.shape[0] != up.shape[1]:
            return False
        if target_slice is not None:
            dim, start, length = target_slice
            if dim != 0 or up.shape[0] != length or start < 0 or start + length > out_features:
                return False
    return True


# Token-dimension chunk budget for _lora_output_branch:
# picked so the largest per-chunk intermediate -- shape (chunk_rows, out_features)
# at out_dtype -- stays in the tens-of-MB range regardless of how many tokens the
# caller passes (stage-2 video upscales can be 4x+ the token count of stage 1).
_NVFP4_LORA_BRANCH_CHUNK_BYTES = 32 * 1024 * 1024


def _lora_output_branch(
    x2d: torch.Tensor, deltas: "list", out_dtype: torch.dtype, out_features: int,
) -> torch.Tensor:
    """Sum of the low-rank output contributions of ``deltas`` (already proven
    eligible by :func:`_deltas_output_branch_ok`) against the flattened
    activation ``x2d`` ``(M, in_features)``. Shared by the nvfp4 and fp8
    ``_scaled_mm``-family fast paths — both add this term to their raw GEMM
    output, before reshaping back to the caller's original leading dims.

    ``x @ (up @ down).T == (x @ down.T) @ up.T`` — computing it in this order
    never materialises the ``(out, in)`` weight-shaped delta, only ``(M,
    rank)``/``(M, length)`` intermediates, where ``length`` is the delta's
    ``target_slice`` width or ``out_features`` for a full-width delta. A
    sliced delta's term is added into ``total[..., start:start+length]``,
    mirroring the weight-side add's ``out[_slice_of(target_slice)]`` in
    :func:`apply_lora_deltas` exactly.

    Dtype: runs in ``out_dtype`` (the GEMM's compute dtype), not fp32. The
    fp32 that :func:`apply_lora_deltas` uses is applied to the ``(out, in)``
    weight-shaped delta -- bounded by weight size regardless of token count --
    and is cast down to compute dtype BEFORE ``F.linear`` ever runs; the real
    GEMM in that fallback path is always compute-dtype, never fp32. Matching
    that here costs no accuracy (LoRA ranks are small -- a handful to a few
    dozen -- so a compute-dtype reduction over the rank axis is extremely
    well-conditioned) while avoiding an ``(M, in_features)``/``(M,
    out_features)`` fp32 allocation that scales with token count instead of
    weight size, which is what OOM'd a 2x-resolution stage-2 video sample.

    Chunking: ``x2d`` is walked in row chunks of
    :data:`_NVFP4_LORA_BRANCH_CHUNK_BYTES` worth of ``(rows, out_features)``
    output so peak memory for this function is independent of ``M`` — large
    token counts (long/high-res video) no longer allocate an
    activation-sized intermediate in one shot.
    """
    m, _k = x2d.shape
    lokr = [b for b in (_build_lokr_branch(d, out_dtype, x2d.device)
                        for d in deltas if getattr(d, "kron", False)) if b is not None]
    prepared = [
        (
            d.down.to(device=x2d.device, dtype=out_dtype),
            d.up.to(device=x2d.device, dtype=out_dtype),
            float(d.scale) * float(d.alpha) / d.down.shape[0],
            d.target_slice,
        )
        for d in deltas if not getattr(d, "kron", False)
    ]
    total = x2d.new_zeros((m, out_features), dtype=out_dtype)
    itemsize = total.element_size()
    chunk_rows = max(1, _NVFP4_LORA_BRANCH_CHUNK_BYTES // max(1, out_features * itemsize))
    for start in range(0, m, chunk_rows):
        end = min(start + chunk_rows, m)
        xc = x2d[start:end].to(out_dtype)
        for down, up, coeff, target_slice in prepared:
            term = (xc @ down.t()) @ up.t()
            term = term * coeff
            if target_slice is not None:
                _dim, s, length = target_slice
                total[start:end, s:s + length] += term
            else:
                total[start:end] += term
        for branch in lokr:
            _add_lokr_branch(total[start:end], xc, branch)
    return total


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
