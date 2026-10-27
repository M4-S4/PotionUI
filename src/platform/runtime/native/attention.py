"""Attention backend dispatch for native arch modules.

A single ``attention(q, k, v)`` entry point that routes to the fastest attention
kernel available on the current machine, with a uniform tensor contract so every
arch module (Flux, Qwen-Image, Wan, ...) can share it.

Locally written, no upstream lineage: the probe/priority/pin machinery is this
module's own, and each ``_<backend>`` function is a two-to-four-line adapter onto
the kernel package's own documented public entry point
(``torch.nn.functional.scaled_dot_product_attention``,
``flash_attn.flash_attn_func``, ``sageattention.sageattn``,
``sageattn3.sageattn3_blackwell``,
``spas_sage_attn.spas_sage2_attn_meansim_topk_cuda``) — the call signatures are
those packages' public API, not transliterated code. The one piece of
non-mechanical logic, ``_sage``'s conditional V rescale, comes from production
failures observed here and is documented at its definition.

Tensor contract
---------------
Grouped-query attention (fewer K/V heads than query heads) has its own entry
point, :func:`grouped_attention`: backends that consume grouped K/V directly
(:func:`supports_grouped_kv` — torch SDPA's ``enable_gqa``) get them as-is,
every other backend gets them expanded with ``repeat_interleave`` first, which
is what each caller used to do by hand.

``q``, ``k``, ``v`` are **head-split**: shape ``(B, H, L, D)`` (batch, heads,
sequence, head-dim) — exactly the layout ``torch.nn.functional.
scaled_dot_product_attention`` consumes and the layout the Flux DiT already
produces after RoPE. The return is also ``(B, H, L, D)`` (SDPA's output, before
the caller merges heads). ``heads`` is optional metadata (inferred from
``q.shape[1]`` when omitted); ``mask`` is an additive/boolean attention mask
already broadcast to a shape SDPA accepts.

Backends (priority high -> low): ``sage3`` > ``sage2`` > ``sage`` > ``flash`` >
``sdpa``. ``sdpa`` is always available and is the numerical reference. The
accelerated kernels are used only for the plain (no mask, fp16/bf16) path; a
mask or an fp32 input transparently falls back to ``sdpa`` (those kernels
don't support dense masks / fp32), so correctness never depends on the
selected backend.

``sage3`` (SageAttention3, arXiv:2505.11594, thu-ml/SageAttention's
``sageattention3_blackwell`` — a *separate* source-only package, module name
``sageattn3``, entry point ``sageattn3_blackwell(q, k, v, is_causal=False)``,
head-split ``(B, H, L, D)`` fp16/bf16 in — same contract as ``sageattn``, no
``tensor_layout`` kwarg) uses hardware FP4 tensor cores that only exist on
specific Blackwell dies. Its CUDA extension is compiled with the
architecture-specific ``compute_XXXa``/``sm_XXXa`` flags (the trailing ``a``
means "family-specific PTX", not forward/backward compatible the way plain
``sm_90``-style flags are) for exactly three capabilities: ``(10, 0)``
(datacenter Blackwell, B100/B200), ``(12, 0)`` (consumer Blackwell, RTX 50
series, e.g. RTX 5090), and ``(12, 1)`` (Blackwell Ultra/GB20x
variants). So the gate is exact-capability membership in that set, never a
``>=`` range like sage2's sm80+. The upstream build also hard-requires CUDA
runtime >= 12.8 (checked separately, since a stale nvcc/driver combo would
otherwise crash instead of falling back).

Selection: an explicit ``override`` or the ``NATIVE_ATTENTION`` env var wins if
that backend is available, otherwise a warning is logged and the highest-
priority available backend is used.

``sparge`` (SpargeAttention, arXiv:2502.18137, ICML2025; thu-ml/SpargeAttn,
pip/import name ``spas_sage_attn``) is training-free SPARSE attention built on
top of SageAttention2's kernels — a two-stage online filter predicts and skips
low-contribution attention blocks. Unlike every other backend here, it is
**approximate, not numerically near-lossless**: output quality depends on
content and the sparsity/accuracy tradeoff (its ``topk`` parameter). For that
reason ``sparge`` is a **pin-only** backend (:data:`PIN_ONLY_BACKENDS`): it is
never chosen automatically no matter how available it is, only ever selected
by an explicit ``$NATIVE_ATTENTION=sparge``, an explicit ``backend="sparge"``
call arg, or an explicit admin pin. Entry point
``spas_sage2_attn_meansim_topk_cuda`` — same head-split ``(B, H, L, D)``
contract as ``sageattn`` (``tensor_layout="HND"``, its default) and a
documented tune-free default (``topk=0.5``, no per-model calibration
required per the upstream README). Its kernel additionally requires
``seq_len >= 128`` and ``head_dim`` in ``{64, 128}`` — checked per call
(``_sparge_shape_ok``) since no other backend has this constraint; a call
that doesn't qualify falls back to ``sdpa`` like a mask/fp32/CPU call would.
Its compiled kernel only targets Ampere/Ada/Hopper (compute capability major
8 or 9, per ``thu-ml/SpargeAttn``'s ``setup.py`` ``SUPPORTED_ARCHS`` — it does
NOT list Blackwell, so this is unavailable on Blackwell today, unlike sage2/
sage3).
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from src.platform.runtime.native.kernels import (
    FLASH,
    SAGE,
    SAGE2,
    SAGE3,
    SDPA,
    SPARGE,
    capabilities,
    refresh_capabilities,
)
from src.platform.settings.runtime_flags import coerce_runtime_flag, runtime_flag, runtime_flag_values

logger = logging.getLogger(__name__)

Tensor = torch.Tensor

# High -> low priority. sdpa is the always-present floor. PIN_ONLY backends
# (below) are deliberately excluded — never auto-selected regardless of order.
BACKEND_PRIORITY: list[str] = [SAGE3, SAGE2, SAGE, FLASH, SDPA]

# Backends that are valid EXPLICIT pins (name, env var, or admin panel) but are
# never chosen by auto-selection even when available — see the module
# docstring's ``sparge`` section for why (approximate, content-dependent
# quality, unlike the numerically near-lossless backends in BACKEND_PRIORITY).
PIN_ONLY_BACKENDS: frozenset[str] = frozenset({SPARGE})

# spas_sage2_attn_meansim_topk_cuda's own hard requirements (its assertions):
# "seq_len should be not less than 128", "headdim should be in [64, 128]".
_SPARGE_MIN_SEQ_LEN = 128
_SPARGE_VALID_HEAD_DIMS: frozenset[int] = frozenset({64, 128})

_warned_unavailable: set[str] = set()


def set_backend_override(name: str | None) -> None:
    runtime_flag_values()["native_attention_backend"] = coerce_runtime_flag("native_attention_backend", name)


def get_backend_override() -> str | None:
    return runtime_flag("native_attention_backend") or None


def _auto_selectable(available: frozenset[str]) -> list[str]:
    return [b for b in BACKEND_PRIORITY if b in available]


def reset_backend_cache() -> None:
    refresh_capabilities()
    _warned_unavailable.clear()
    _noted_dispatch.clear()
    _grouped_capability.clear()


# Per-backend "can this kernel take K/V with fewer heads than Q, without the
# caller expanding them first?" — computed once per backend name, never per
# call (see `supports_grouped_kv`).
_grouped_capability: dict[str, bool] = {}


def _probe_grouped_sdpa() -> bool:
    """Whether the INSTALLED torch's SDPA takes ``enable_gqa``. Probed by
    calling it rather than by version number: the kwarg landed in torch 2.5 but
    a vendored/patched build is the thing that actually decides."""
    q = torch.zeros(1, 2, 1, 8)
    kv = torch.zeros(1, 1, 1, 8)
    try:
        F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
    except TypeError:
        return False
    return True


def supports_grouped_kv(backend: str) -> bool:
    """Whether ``backend`` consumes grouped K/V (``Hkv < H``) directly.

    ``sdpa`` is probed (:func:`_probe_grouped_sdpa`). Every other backend is
    reported unsupported: flash/sage3/sparge are not installed here to read,
    and sageattention's own GQA handling lives in its compiled ``_qattn_sm*``
    extension — the kernel `sageattn` actually dispatches to on the GPUs this
    runs on — which cannot be inspected or verified from source. Unsupported
    is never a correctness problem, only an allocation one: those backends get
    the expanded layout, exactly as before this seam existed.
    """
    cached = _grouped_capability.get(backend)
    if cached is None:
        cached = _probe_grouped_sdpa() if backend == SDPA else False
        _grouped_capability[backend] = cached
    return cached


def available_backends(device_index: int | None = None) -> list[str]:
    """AUTO-selectable backends on ``device_index`` (``None`` = current device),
    in priority order (high -> low). Always includes sdpa. Excludes
    :data:`PIN_ONLY_BACKENDS` (e.g. ``sparge``) even when available — those
    only ever come back from :func:`get_attention_backend` on an explicit
    request, never as "the best available backend"."""
    return _auto_selectable(capabilities().device(device_index).attention_backends)


def known_backends() -> frozenset[str]:
    """Every backend NAME this dispatcher recognizes as a valid explicit pin —
    :data:`BACKEND_PRIORITY` (auto-selectable) UNION :data:`PIN_ONLY_BACKENDS`
    (e.g. ``sparge``), regardless of whether either is actually installed on
    this machine. This is the single source of truth for "is this a real
    backend name" — callers validating a user-supplied pin (e.g. the admin API)
    should check against this instead of duplicating a name list, or a
    pin-only addition here would silently need a second edit elsewhere to stay
    pinnable end-to-end (see ``api/controllers/backend_controller.py``'s
    ``set_attention_backend``, which does exactly this)."""
    return frozenset(BACKEND_PRIORITY) | PIN_ONLY_BACKENDS


def get_attention_backend(override: str | None = None, device_index: int | None = None) -> str:
    """Resolve the backend to use on ``device_index`` (``None`` = current device).

    Precedence: ``override`` arg, then ``$NATIVE_ATTENTION``, then the in-memory
    pin set via :func:`set_backend_override` (Admin -> Backends -> Optimizations),
    then the highest-priority auto-selectable backend (:func:`available_backends`).
    An EXPLICIT request (any of the first three) is honored if the backend is
    available at all — including a :data:`PIN_ONLY_BACKENDS` member, which is
    otherwise never returned when ``requested`` is ``None``. A requested-but-
    unavailable or unknown choice logs a warning (once per name) and falls back
    to the best auto-selectable backend. No DB read happens here - this runs
    per forward call.
    """
    full_avail = capabilities().device(device_index).attention_backends
    auto_avail = _auto_selectable(full_avail)
    requested = override or runtime_flag("native_attention_backend") or None

    if requested is None:
        return auto_avail[0]

    requested = requested.strip().lower()
    if requested in full_avail:
        return requested

    if requested not in _warned_unavailable:
        _warned_unavailable.add(requested)
        known = list(BACKEND_PRIORITY) + sorted(PIN_ONLY_BACKENDS)
        if requested not in known:
            logger.warning(
                "unknown attention backend %r requested; using %r. Known: %s",
                requested, auto_avail[0], known,
            )
        else:
            logger.warning(
                "attention backend %r requested but unavailable; falling back to %r",
                requested, auto_avail[0],
            )
    return auto_avail[0]


# One-shot dispatch notes: the first call of each kind (kernel actually used /
# fallback + why) logs at INFO so a single generation's log shows definitively
# which attention path ran; repeats stay at DEBUG. Cleared by reset_backend_cache.
_noted_dispatch: set[str] = set()


def _note_dispatch(key: str, fmt: str, *args) -> None:
    if key in _noted_dispatch:
        logger.debug(fmt, *args)
        return
    _noted_dispatch.add(key)
    logger.info(fmt, *args)


_MASK_ALIGN = 16


def _aligned_mask(mask: Tensor | None) -> Tensor | None:
    """Re-home a dense mask in storage whose row stride is a multiple of 16
    elements. torch's memory-efficient SDPA kernel only accepts an ``attn_mask``
    whose last-dim stride is 16-byte aligned; an odd sequence length (text +
    image tokens) fails that check and silently drops to the math kernel, which
    materialises the full ``H×L×L`` score matrix (12 GB at 8k tokens)."""
    if mask is None or mask.ndim < 2:
        return mask
    length = mask.shape[-1]
    pad = (-length) % _MASK_ALIGN
    if pad == 0 and mask.stride(-1) == 1:
        return mask
    buf = torch.empty(*mask.shape[:-1], length + pad, dtype=mask.dtype, device=mask.device)
    buf[..., :length] = mask
    return buf[..., :length]


# Score-matrix elements (B×H×Lq×Lk) above which a masked grouped call expands
# its K/V heads to reach the memory-efficient kernel: 2^26 elements is 128 MB
# of bf16 scores, past which the math kernel's materialisation starts to hurt.
_MASKED_GROUPED_EXPAND_MIN_SCORES = 1 << 26


def _masked_scores(q: Tensor, k: Tensor) -> int:
    return q.shape[0] * q.shape[1] * q.shape[-2] * k.shape[-2]


def _sdpa(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None, grouped: bool = False) -> Tensor:
    # `enable_gqa` is only passed on the grouped path: the kwarg does not exist
    # on every torch this runs against, and `_supports_grouped_kv` is the only
    # thing that has established it does here.
    if mask is not None:
        mask = _aligned_mask(mask)
        if grouped and _masked_scores(q, k) > _MASKED_GROUPED_EXPAND_MIN_SCORES:
            # The memory-efficient kernel does not take grouped K/V together with a
            # dense mask; expand the heads so a masked call can still avoid the
            # math kernel's L×L score matrix. Only worth it once that matrix is
            # big - a short grouped call (an LM prefill, a decode step) keeps its
            # K/V unexpanded and lets the math kernel have its small scores.
            repeat = q.shape[1] // k.shape[1]
            if repeat > 1:
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)
            grouped = False
    if grouped:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False, enable_gqa=True,
        )
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)


def _flash(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    # flash_attn_func wants (B, L, H, D) and fp16/bf16; caller guarantees no mask.
    from flash_attn import flash_attn_func

    out = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=False)
    return out.transpose(1, 2)


# sage v1 internally downcasts V to fp16 (``v = v.to(torch.float16)``) with
# fp16 PV accumulation and no configurable accumulator dtype.
#
# Two competing failure modes, both observed in production:
#
# * Large-V overflow (Qwen-Image, V ~ N(0, 5000^2) at its real joint-attention
#   shape): the kernel's fp16 internals overflow to +/-inf, the softmax-weighted
#   sum turns NaN, and the DiT propagates a black image.  Unconditionally
#   pre-dividing V by 256 (and multiplying the output back) fixed this — the
#   division by a power of two is exact on the input.
# * Small-V quantization noise (LTX 2.3, V magnitudes ~1e2): dividing by 256
#   pushes small V elements toward fp16's subnormal range inside the kernel,
#   and the amplified rounding error (cos=0.999996, max|d|=0.03 per call vs
#   raw V) compounds across ~1500 attention calls (48 blocks x 4 attn ops x
#   8 denoise steps) into a spatially uniform film-grain artifact. ComfyUI
#   passes V raw and shows no grain.
#
# So the rescale is applied CONDITIONALLY: raw pass-through (bit-identical to
# ComfyUI) while ``max|V|`` is comfortably below fp16 territory, and the exact
# production-validated /256 path once V is hot enough to threaten the kernel's
# fp16 internals.  The branch is computed on-GPU via ``torch.where`` (a 0-dim
# tensor scale; ``v / 1.0`` is IEEE-exact), so there is still no per-call host
# sync — an ``isfinite()``-check-and-fallback approach was previously measured
# to erase sage's entire speed advantage (~1.4ms forced sync per call).
# ``nan_to_num`` on the output remains the zero-sync safety net for the
# pathological remainder, so this call can never hand a NaN/Inf back.
_SAGE_V_SCALE = 256.0
# Raw pass-through below this max|V|; above it, the /256 path engages. LTX 2.3
# sits ~1e2, the Qwen-Image overflow repro ~2.6e4 — the threshold splits the
# two regimes with an order of magnitude of margin on each side.
_SAGE_V_SAFE_MAX = 1024.0


# Above this row count, `_sage` trades its zero-sync `torch.where` scale for
# ONE host sync per call: at e.g. the H3 upscale refine's ~78k rows the two
# full-V copies (`v / scale`, `out * scale`) cost ~2.3GB of transient VRAM in
# a regime that is exactly where OOMs happen, while the call count is a few
# hundred per refine — so a ~1ms sync is the cheaper side of the trade there.
# Below it, the production-validated sync-free path is untouched (the sync
# variant was previously measured to erase sage's speed edge on hot paths).
_SAGE_SYNC_ROWS = 32768


def _absmax(t: Tensor) -> Tensor:
    """``t.abs().amax()`` without materializing a full-size ``|t|`` temp --
    two scalar reductions instead. On a 78k-row H3 refine V that temp is
    ~2.8GB, and it was the allocation that OOM'd a real 5090 run inside the
    prescale CHECK itself."""
    return torch.maximum(t.amax(), t.amin().neg())


def _sage(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    # sageattn accepts head-split (B, H, L, D) via tensor_layout="HND".
    from sageattention import sageattn

    if v.shape[2] >= _SAGE_SYNC_ROWS:
        if float(_absmax(v)) > _SAGE_V_SAFE_MAX:
            out = sageattn(q, k, v / _SAGE_V_SCALE, tensor_layout="HND", is_causal=False)
            out = out * _SAGE_V_SCALE
        else:
            # Raw pass-through with NO copy of V -- numerically identical to
            # the scale=1.0 branch below (`v / 1.0` is IEEE-exact), minus the
            # two full-size allocations.
            out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    scale = torch.where(
        _absmax(v) > _SAGE_V_SAFE_MAX,
        v.new_tensor(_SAGE_V_SCALE),
        v.new_tensor(1.0),
    )
    out = sageattn(q, k, v / scale, tensor_layout="HND", is_causal=False)
    out = out * scale
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _sage3(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    # sageattn3_blackwell takes head-split (B, H, L, D) directly — no
    # tensor_layout kwarg (unlike sageattn/sageattn2's "HND" default), per the
    # sageattention3_blackwell README.
    from sageattn3 import sageattn3_blackwell

    return sageattn3_blackwell(q, k, v, is_causal=False)


def _sparge_shape_ok(q: Tensor) -> bool:
    """SpargeAttn's kernel hard-requires ``seq_len >= 128`` and ``head_dim`` in
    ``{64, 128}`` (thu-ml/SpargeAttn's own assertions) — no other backend here
    has this constraint, so it's checked only when ``sparge`` is chosen."""
    return q.shape[2] >= _SPARGE_MIN_SEQ_LEN and q.shape[3] in _SPARGE_VALID_HEAD_DIMS


def _sparge(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    # spas_sage2_attn_meansim_topk_cuda: head-split (B, H, L, D) via
    # tensor_layout="HND" (its default — same contract as sageattn/sageattn2).
    # topk=0.5 is the upstream README's documented tune-free default (no
    # per-model calibration needed); output_dtype pinned to q's own dtype so
    # the kernel's own fp16-only default can't silently narrow a bf16 caller.
    from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda

    return spas_sage2_attn_meansim_topk_cuda(
        q, k, v, is_causal=False, tensor_layout="HND", topk=0.5, output_dtype=q.dtype,
    )


def _resolve_dispatch(q: Tensor, mask: Tensor | None, backend: str | None) -> str:
    """The backend this call will ACTUALLY run on: the selected one, or ``sdpa``
    when the call is mask/dtype/device/shape-constrained past what that kernel
    accepts. Emits the one-shot dispatch note."""
    # Validate availability against the OPERAND's device, not whatever CUDA
    # device happens to be "current" on this thread — a multi-GPU box (e.g. a
    # text encoder spilled to cuda:1 while the DiT runs on cuda:0) can have a
    # Blackwell-only backend available on one card and not the other.
    device_index = q.device.index if q.is_cuda else None
    chosen = get_attention_backend(backend, device_index)

    # Accelerated kernels: CUDA tensors, fp16/bf16 only, and no dense mask.
    # Otherwise sdpa. The device check matters: CPU-side attention calls exist
    # (e.g. LTX's conditioning connectors run on CPU) and sage/flash assert on
    # non-CUDA inputs rather than falling back. ``sparge`` additionally needs
    # seq_len/head_dim within its kernel's hard limits (_sparge_shape_ok) — no
    # other backend has that constraint.
    accelerated = chosen in (SAGE3, SAGE2, SAGE, FLASH, SPARGE)
    if accelerated:
        if mask is not None:
            reason = "mask"
        elif q.dtype not in (torch.float16, torch.bfloat16):
            reason = "dtype"
        elif not q.is_cuda:
            reason = "device"
        elif chosen == SPARGE and not _sparge_shape_ok(q):
            reason = "shape"
        else:
            reason = None
    else:
        reason = None

    if reason is not None:
        _note_dispatch(f"{chosen}->sdpa:{reason}",
                       "attention: %s falls back to sdpa (%s-constrained call, shape %s)",
                       chosen, reason, tuple(q.shape))
        return SDPA
    if accelerated:
        _note_dispatch(chosen, "attention: %s kernels in use (shape %s)", chosen, tuple(q.shape))
    return chosen


def _run(chosen: str, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None, grouped: bool = False) -> Tensor:
    if chosen == FLASH:
        return _flash(q, k, v)
    if chosen == SAGE3:
        return _sage3(q, k, v)
    if chosen == SPARGE:
        return _sparge(q, k, v)
    if chosen in (SAGE2, SAGE):
        return _sage(q, k, v)
    if grouped:
        return _sdpa(q, k, v, mask, grouped=True)
    return _sdpa(q, k, v, mask)


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    heads: int | None = None,
    mask: Tensor | None = None,
    backend: str | None = None,
) -> Tensor:
    """Dispatch scaled-dot-product attention to the selected backend.

    ``q``/``k``/``v``: ``(B, H, L, D)``. Returns ``(B, H, L, D)``. See the module
    docstring for the full contract. A mask or an fp32 input forces the ``sdpa``
    path (the accelerated kernels support neither), so the result is backend-
    independent up to kernel precision. Fewer K/V heads than query heads
    (grouped-query attention) goes through :func:`grouped_attention` instead.
    """
    if heads is not None and q.ndim == 4 and heads != q.shape[1]:
        raise ValueError(f"heads={heads} disagrees with q.shape[1]={q.shape[1]}")

    return _run(_resolve_dispatch(q, mask, backend), q, k, v, mask)


def grouped_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    heads: int | None = None,
    kvheads: int | None = None,
    mask: Tensor | None = None,
    backend: str | None = None,
) -> Tensor:
    """Attention with fewer K/V heads than query heads (grouped-query/MQA).

    ``q`` is ``(B, H, L, D)``, ``k``/``v`` are ``(B, Hkv, S, D)`` with ``H``
    divisible by ``Hkv``; returns ``(B, H, L, D)`` like :func:`attention`.

    Backends that consume grouped K/V directly (:func:`supports_grouped_kv`)
    get them as-is; every other backend gets the K/V heads expanded to ``H``
    with ``repeat_interleave`` first, which is what the caller used to do by
    hand. Both layouts compute the same attention — the expansion is a pure
    materialization of the same values — so the choice is an allocation
    question, not a numerical one.
    """
    if heads is not None and heads != q.shape[1]:
        raise ValueError(f"heads={heads} disagrees with q.shape[1]={q.shape[1]}")
    if kvheads is not None and kvheads != k.shape[1]:
        raise ValueError(f"kvheads={kvheads} disagrees with k.shape[1]={k.shape[1]}")

    q_heads, kv_heads = q.shape[1], k.shape[1]
    if q_heads == kv_heads:
        return attention(q, k, v, mask=mask, backend=backend)
    if kv_heads == 0 or q_heads % kv_heads:
        raise ValueError(f"q heads={q_heads} is not a multiple of kv heads={kv_heads}")

    chosen = _resolve_dispatch(q, mask, backend)
    if supports_grouped_kv(chosen):
        return _run(chosen, q, k, v, mask, grouped=True)

    repeat = q_heads // kv_heads
    return _run(chosen, q, k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1), mask)
