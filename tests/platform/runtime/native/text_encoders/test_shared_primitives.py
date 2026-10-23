from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional

from src.platform.runtime.native.ops.embedding import sinusoidal_embedding
from src.platform.runtime.native.ops.padding import pad_to_patch
from src.platform.runtime.native.text_encoders import _functional as fn
from src.platform.runtime.native.text_encoders.base import NativeTextEncoder


def _ref_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _ref_apply_rope(xq, xk, cos, sin):
    org = xq.dtype
    q = (xq * cos) + (_ref_rotate_half(xq) * sin)
    k = (xk * cos) + (_ref_rotate_half(xk) * sin)
    return q.to(org), k.to(org)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rope_helpers_match_the_replaced_copies_bit_for_bit(dtype):
    g = torch.Generator().manual_seed(0)
    xq = torch.randn(2, 4, 5, 8, generator=g).to(dtype)
    xk = torch.randn(2, 4, 5, 8, generator=g).to(dtype)
    cos = torch.randn(1, 1, 5, 8, generator=g)
    sin = torch.randn(1, 1, 5, 8, generator=g)
    assert torch.equal(fn.rotate_half(xq), _ref_rotate_half(xq))
    got = fn.apply_rope_qk(xq, xk, cos, sin)
    want = _ref_apply_rope(xq, xk, cos, sin)
    assert got[0].dtype == dtype
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


def test_rms_norm_module_keeps_the_weight_key_and_math():
    norm = fn.RMSNorm(6, 1e-6)
    assert list(norm.state_dict()) == ["weight"]
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, 6))
    x = torch.randn(3, 6, generator=torch.Generator().manual_seed(1))
    assert torch.equal(norm(x), fn.rms_norm(x, norm.weight, 1e-6))


def _ref_gated_mlp(mlp, x):
    return mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))


def test_gated_mlp_split_layout_keys_and_forward():
    torch.manual_seed(2)
    mlp = fn.GatedMLP(8, 16, nn)
    assert list(mlp.state_dict()) == ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
    x = torch.randn(2, 3, 8)
    assert torch.equal(mlp(x), _ref_gated_mlp(mlp, x))


def test_gated_mlp_merged_layout_matches_the_split_layout():
    torch.manual_seed(3)
    split = fn.GatedMLP(8, 16, nn)
    merged = fn.GatedMLP(8, 16, nn, merged=True)
    assert list(merged.state_dict()) == ["gate_up_proj.weight", "down_proj.weight"]
    with torch.no_grad():
        merged.gate_up_proj.weight.copy_(torch.cat([split.gate_proj.weight, split.up_proj.weight], dim=0))
        merged.down_proj.weight.copy_(split.down_proj.weight)
    x = torch.randn(2, 3, 8)
    assert torch.allclose(merged(x), split(x), atol=1e-6)


def test_module_device_skips_none_and_raises_when_empty():
    assert fn.module_device(nn.Linear(2, 2)) == torch.device("cpu")
    with pytest.raises(RuntimeError, match="no tensors"):
        fn.module_device(nn.Identity())


class _Enc(NativeTextEncoder):
    def __init__(self, module=None):
        if module is not None:
            self.module = module

    def encode(self, texts):
        raise NotImplementedError


def test_base_to_sets_device_and_moves_the_module():
    moved: list[str] = []

    class _Mod(nn.Linear):
        def to(self, device):
            moved.append(str(device))
            return self

    enc = _Enc(_Mod(2, 2))
    assert enc.to("cpu") is enc
    assert enc._device == torch.device("cpu")
    enc.offload()
    assert moved == ["cpu", "cpu"]


def test_base_to_and_unload_tolerate_an_encoder_without_a_module():
    enc = _Enc()
    assert enc.to("cpu") is enc
    enc.unload()


def test_base_unload_releases_module_storage():
    module = nn.Linear(4, 4)
    enc = _Enc(module)
    enc.unload()
    assert all(p.numel() == 0 or p.untyped_storage().size() == 0 for p in module.parameters())


def _ref_ss_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def _ref_h3_embedding(t, dim=256, max_period=10000.0):
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    args = t.to(torch.float32)[:, None] * torch.exp(exponent)[None, :]
    return torch.cat((torch.cos(args), torch.sin(args)), dim=-1)


def _ref_ltx_embedding(timesteps, dim=256, max_period=10000):
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / half
    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    emb = torch.cat([emb[:, half:], emb[:, :half]], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


@pytest.mark.parametrize("dim", [256, 64, 7])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sinusoidal_embedding_matches_every_replaced_copy(dim, dtype):
    t = torch.rand(5, generator=torch.Generator().manual_seed(4)).to(dtype) * 1000
    got = sinusoidal_embedding(t, dim)
    assert got.dtype == torch.float32
    assert torch.equal(got, _ref_ss_embedding(t, dim))
    assert torch.equal(got, _ref_ltx_embedding(t, dim))
    if dim % 2 == 0:
        assert torch.equal(got, _ref_h3_embedding(t, dim, 10000.0))


def _ref_pad_to_patch_size(img, patch_size, padding_mode="circular"):
    pad = ()
    for i in range(img.ndim - 2):
        pad = (0, (patch_size[i] - img.shape[i + 2] % patch_size[i]) % patch_size[i]) + pad
    if all(p == 0 for p in pad):
        return img
    return F.pad(img, pad, mode=padding_mode)


def _ref_z_image_pad(x, p):
    h, w = x.shape[2:]
    pad_h = (p - h % p) % p
    pad_w = (p - w % p) % p
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="circular")
    return x


@pytest.mark.parametrize("shape", [(1, 3, 5, 7, 9), (2, 2, 4, 6, 8), (1, 1, 3, 3, 3)])
def test_pad_to_patch_matches_the_wan_copy(shape):
    x = torch.randn(*shape, generator=torch.Generator().manual_seed(5))
    patch = (1, 2, 2)
    got = pad_to_patch(x, patch)
    assert torch.equal(got, _ref_pad_to_patch_size(x, patch))
    assert all(got.shape[i + 2] % patch[i] == 0 for i in range(3))


@pytest.mark.parametrize("hw", [(7, 9), (8, 8), (5, 8)])
def test_pad_to_patch_matches_the_z_image_inline_pad(hw):
    x = torch.randn(1, 4, *hw, generator=torch.Generator().manual_seed(6))
    assert torch.equal(pad_to_patch(x, (2, 2)), _ref_z_image_pad(x, 2))


def test_pad_to_patch_returns_the_same_tensor_when_aligned():
    x = torch.zeros(1, 1, 4, 4)
    assert pad_to_patch(x, (2, 2)) is x
