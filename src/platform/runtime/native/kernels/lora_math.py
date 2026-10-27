from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from src.platform.settings.runtime_flags import runtime_flag

logger = logging.getLogger(__name__)

LORA_FUSED_FLAG = "native_lora_fused"

_LORA_FUSED_ATTR = "_lora_fused_cache"
_NVFP4_LORA_BRANCH_CHUNK_BYTES = 32 * 1024 * 1024
_lora_path_layers: "dict[str, set[int]]" = {}


def _lora_fused_enabled() -> bool:
    return bool(runtime_flag(LORA_FUSED_FLAG))


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


def apply_lora_deltas(weight: torch.Tensor, deltas: "list | None") -> torch.Tensor:
    if not deltas:
        return weight
    out = weight.clone()
    for d in deltas:
        up = d.up.to(device=out.device, dtype=torch.float32)
        down = d.down.to(device=out.device, dtype=torch.float32)
        if getattr(d, "kron", False):
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


def _deltas_output_branch_ok(deltas: "list | None", out_features: int) -> bool:
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


def partition_output_branch_deltas(
    deltas: "list | None", out_features: int,
) -> "tuple[list, list]":
    if not deltas:
        return [], []
    output_side, weight_side = [], []
    for d in deltas:
        (output_side if _deltas_output_branch_ok([d], out_features) else weight_side).append(d)
    return output_side, weight_side


def _fuse_output_branch_deltas(
    deltas: "list", dtype: torch.dtype, device: "torch.device",
) -> "tuple[torch.Tensor, torch.Tensor, float]":
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


def _lora_output_branch(
    x2d: torch.Tensor, deltas: "list", out_dtype: torch.dtype, out_features: int,
) -> torch.Tensor:
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


def _add_lora_output_branch(
    out: torch.Tensor, x: torch.Tensor, deltas: "list", out_features: int, owner=None,
) -> torch.Tensor:
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
    output_side, weight_side = partition_output_branch_deltas(deltas, out_features)
    if weight_side:
        weight = apply_lora_deltas(weight, weight_side)
    out = F.linear(input, weight, bias)
    return _add_lora_output_branch(out, input, output_side, out_features, owner=owner)
