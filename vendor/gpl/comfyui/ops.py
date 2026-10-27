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

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

RUNTIME_FLAGS: dict = {}


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
