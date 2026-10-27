from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from src.platform.runtime.native.base import release_module_storage
from src.platform.runtime.native.kernels import disable_weight_init, manual_cast
from vendor.gpl.comfyui.ops import _build_convrot_hadamard


def _blob(conf: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def _quantize_int8_tensorwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    row_absmax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = row_absmax / 127.0
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def _convrot_embedding(num_emb: int, dim: int, group: int, seed: int):
    torch.manual_seed(seed)
    w = torch.randn(num_emb, dim) * 0.05
    hadamard = _build_convrot_hadamard(group, device="cpu", dtype=torch.float32)
    n_groups = dim // group
    rotated = torch.matmul(w.reshape(num_emb, n_groups, group), hadamard).reshape(num_emb, dim)
    q, scale = _quantize_int8_tensorwise(rotated)
    sd = {
        "weight": q,
        "weight_scale": scale,
        "comfy_quant": _blob({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group}),
    }
    emb = manual_cast.Embedding(num_emb, dim)
    emb._load_from_state_dict({k: v.clone() for k, v in sd.items()}, "", {}, True, [], [], [])
    return emb, w, q, scale


def test_embedding_consumes_int8_sidecars_no_missing_or_unexpected():
    torch.manual_seed(0)
    num_emb, dim = 20, 16
    w = torch.randn(num_emb, dim) * 0.05
    q, scale = _quantize_int8_tensorwise(w)
    sd = {
        "weight": q,
        "weight_scale": scale,
        "comfy_quant": _blob({"format": "int8_tensorwise"}),
    }
    emb = manual_cast.Embedding(num_emb, dim)
    missing, unexpected = [], []
    emb._load_from_state_dict(dict(sd), "", {}, True, missing, unexpected, [])
    assert missing == [] and unexpected == []
    assert emb.weight is None
    assert emb._int8_weight is not None


def test_embedding_lookup_matches_manual_row_dequant():
    torch.manual_seed(1)
    num_emb, dim = 20, 16
    w = torch.randn(num_emb, dim) * 0.05
    q, scale = _quantize_int8_tensorwise(w)
    sd = {"weight": q, "weight_scale": scale, "comfy_quant": _blob({"format": "int8_tensorwise"})}

    emb = manual_cast.Embedding(num_emb, dim)
    emb._load_from_state_dict(dict(sd), "", {}, True, [], [], [])

    idx = torch.tensor([[1, 5, 19], [0, 2, 7]])
    out = emb(idx)
    expected = (q.to(torch.float32) * scale)[idx].to(out.dtype)
    assert torch.allclose(out, expected)


def test_embedding_lookup_dequant_is_load_bearing():
    torch.manual_seed(2)
    num_emb, dim = 20, 16
    w = torch.randn(num_emb, dim) * 0.05
    q, scale = _quantize_int8_tensorwise(w)
    sd = {"weight": q, "weight_scale": scale, "comfy_quant": _blob({"format": "int8_tensorwise"})}

    emb = manual_cast.Embedding(num_emb, dim)
    emb._load_from_state_dict(dict(sd), "", {}, True, [], [], [])

    idx = torch.tensor([1, 5, 19])
    out = emb(idx)
    undequantized = q.to(out.dtype)[idx]
    assert not torch.allclose(out, undequantized)


def test_embedding_per_tensor_scalar_scale_broadcasts():
    torch.manual_seed(3)
    num_emb, dim = 10, 8
    w = torch.randn(num_emb, dim) * 0.05
    absmax = w.abs().amax().clamp(min=1e-12)
    scale = (absmax / 127.0).reshape(())
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    sd = {"weight": q, "weight_scale": scale, "comfy_quant": _blob({"format": "int8_tensorwise"})}

    emb = manual_cast.Embedding(num_emb, dim)
    emb._load_from_state_dict(dict(sd), "", {}, True, [], [], [])

    idx = torch.tensor([0, 3, 9])
    out = emb(idx)
    expected = (q.to(torch.float32) * scale)[idx].to(out.dtype)
    assert torch.allclose(out, expected)


def test_embedding_convrot_rotated_undoes_the_rotation():
    num_emb, dim, group = 12, 16, 4
    emb, w, q, scale = _convrot_embedding(num_emb, dim, group, seed=4)
    assert emb._int8_convrot_hadamard is not None

    hadamard = _build_convrot_hadamard(group, device="cpu", dtype=torch.float32)
    n_groups = dim // group
    idx = torch.tensor([0, 4, 11])
    out = emb(idx, out_dtype=torch.float32)
    manual_deq = (q.to(torch.float32) * scale)[idx]
    manual_unrot = torch.matmul(manual_deq.reshape(-1, n_groups, group), hadamard).reshape(-1, dim)
    assert torch.allclose(out, manual_unrot, atol=1e-5)
    rel = (out - w[idx]).abs().mean() / w[idx].abs().mean()
    assert rel < 0.2


def test_releasing_one_convrot_embedding_leaves_the_shared_hadamard_intact():
    group = 4
    first, _, _, _ = _convrot_embedding(12, 16, group, seed=5)
    second, _, _, _ = _convrot_embedding(12, 16, group, seed=5)
    idx = torch.tensor([0, 4, 11])
    expected = second(idx, out_dtype=torch.float32)

    release_module_storage(first)

    assert second._int8_convrot_hadamard.numel() == group * group
    assert _build_convrot_hadamard(group, device="cpu", dtype=torch.float32).numel() == group * group
    assert torch.equal(second(idx, out_dtype=torch.float32), expected)
    third, _, _, _ = _convrot_embedding(12, 16, group, seed=5)
    assert torch.equal(third(idx, out_dtype=torch.float32), expected)


def test_unsupported_quant_format_raises():
    num_emb, dim = 10, 8
    sd = {
        "weight": torch.zeros(num_emb, dim, dtype=torch.int8),
        "weight_scale": torch.ones(num_emb, 1),
        "comfy_quant": _blob({"format": "nvfp4"}),
    }
    emb = manual_cast.Embedding(num_emb, dim)
    with pytest.raises(ValueError, match="nvfp4"):
        emb._load_from_state_dict(dict(sd), "", {}, True, [], [], [])


def test_plain_unquantized_embedding_unaffected():
    num_emb, dim = 10, 8
    w = torch.randn(num_emb, dim)
    emb = disable_weight_init.Embedding(num_emb, dim)
    emb._load_from_state_dict({"weight": w}, "", {}, True, [], [], [])
    assert emb._int8_weight is None
    idx = torch.tensor([1, 2, 3])
    assert torch.equal(emb(idx), w[idx])
