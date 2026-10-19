from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import NativeArchModule
from ._functional import GatedMLP, module_device, rotate_half
from .base import NativeTextEncoder

LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"
DELTA_CHUNK_SIZE = 64
LINEAR_KEY_HEAD_DIM = 128
L2_NORM_EPS = 1e-6


@dataclass
class Qwen35Config:
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    vocab_size: int = 248320
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = LINEAR_KEY_HEAD_DIM
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    layer_types: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "Qwen35Config":
        num_layers = int(config["num_layers"])
        layer_types = tuple(config.get("layer_types") or ())
        if not layer_types:
            layer_types = tuple(
                FULL_ATTENTION if (i + 1) % 4 == 0 else LINEAR_ATTENTION for i in range(num_layers)
            )
        if len(layer_types) != num_layers:
            raise ValueError(f"layer_types has {len(layer_types)} entries for {num_layers} layers")
        defaults = cls()
        return cls(
            hidden_size=int(config["hidden_size"]),
            intermediate_size=int(config.get("intermediate_size", defaults.intermediate_size)),
            num_hidden_layers=num_layers,
            vocab_size=int(config["vocab_size"]),
            num_attention_heads=int(config.get("num_attention_heads", defaults.num_attention_heads)),
            num_key_value_heads=int(config.get("num_key_value_heads", defaults.num_key_value_heads)),
            head_dim=int(config.get("head_dim", defaults.head_dim)),
            rms_norm_eps=float(config.get("rms_norm_eps", defaults.rms_norm_eps)),
            rope_theta=float(config.get("rope_theta", defaults.rope_theta)),
            partial_rotary_factor=float(config.get("partial_rotary_factor", defaults.partial_rotary_factor)),
            linear_num_key_heads=int(config.get("linear_num_key_heads", defaults.linear_num_key_heads)),
            linear_num_value_heads=int(config.get("linear_num_value_heads", defaults.linear_num_value_heads)),
            linear_key_head_dim=int(config.get("linear_key_head_dim", defaults.linear_key_head_dim)),
            linear_value_head_dim=int(config.get("linear_value_head_dim", defaults.linear_value_head_dim)),
            linear_conv_kernel_dim=int(config.get("linear_conv_kernel_dim", defaults.linear_conv_kernel_dim)),
            layer_types=layer_types,
        )

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)


class _ZeroCenteredRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, device=None, dtype=None) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.weight.to(device=x.device, dtype=torch.float32)
        return F.rms_norm(x.float(), self.weight.shape, weight=scale, eps=self.eps).type_as(x)


class _GatedRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, device=None, dtype=None) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = self.weight.to(device=x.device, dtype=torch.float32) * x
        return x * F.silu(gate.float())


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + L2_NORM_EPS)


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = DELTA_CHUNK_SIZE,
) -> torch.Tensor:
    query, key, value, beta, g = (t.transpose(1, 2).contiguous().float() for t in (query, key, value, beta, g))
    batch, heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    pad = (-seq_len) % chunk_size
    query, key, value = (F.pad(t, (0, 0, 0, pad)) for t in (query, key, value))
    beta = F.pad(beta, (0, pad))
    g = F.pad(g, (0, pad))
    query = query * k_dim ** -0.5

    k_beta = key * beta.unsqueeze(-1)
    v_beta = value * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = (
        t.reshape(batch, heads, -1, chunk_size, t.shape[-1]) for t in (query, key, value, k_beta, v_beta)
    )
    g = g.reshape(batch, heads, -1, chunk_size).cumsum(dim=-1)
    num_chunks = g.shape[2]

    decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()
    eye = torch.eye(chunk_size, dtype=key.dtype, device=key.device)
    strict_lower = ((k_beta @ key.transpose(-1, -2)) * decay).tril(-1)
    transform = torch.linalg.solve_triangular(
        eye + strict_lower, eye.expand_as(strict_lower), upper=False, unitriangular=True
    )
    value = transform @ v_beta
    k_cumdecay = transform @ (k_beta * g.exp().unsqueeze(-1))

    state = torch.zeros(batch, heads, k_dim, v_dim, dtype=value.dtype, device=value.device)
    out = torch.empty_like(value)
    for i in range(num_chunks):
        q_i, k_i, v_i, g_i = query[:, :, i], key[:, :, i], value[:, :, i], g[:, :, i]
        intra = (q_i @ k_i.transpose(-1, -2)) * decay[:, :, i]
        v_new = v_i - k_cumdecay[:, :, i] @ state
        out[:, :, i] = (q_i * g_i.unsqueeze(-1).exp()) @ state + intra @ v_new
        g_last = g_i[:, :, -1]
        state = (
            state * g_last[..., None, None].exp()
            + (k_i * (g_last.unsqueeze(-1) - g_i).exp().unsqueeze(-1)).transpose(-1, -2) @ v_new
        )
    return out.reshape(batch, heads, -1, v_dim)[:, :, :seq_len].transpose(1, 2)


class _GatedDeltaNet(nn.Module):
    def __init__(self, cfg: Qwen35Config, operations, device=None, dtype=None) -> None:
        super().__init__()
        self.num_k_heads = cfg.linear_num_key_heads
        self.num_v_heads = cfg.linear_num_value_heads
        self.head_k_dim = cfg.linear_key_head_dim
        self.head_v_dim = cfg.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.kernel = cfg.linear_conv_kernel_dim
        conv_dim = 2 * self.key_dim + self.value_dim
        hidden = cfg.hidden_size
        self.in_proj_qkv = operations.Linear(hidden, conv_dim, bias=False, device=device, dtype=dtype)
        self.in_proj_z = operations.Linear(hidden, self.value_dim, bias=False, device=device, dtype=dtype)
        self.in_proj_b = operations.Linear(hidden, self.num_v_heads, bias=False, device=device, dtype=dtype)
        self.in_proj_a = operations.Linear(hidden, self.num_v_heads, bias=False, device=device, dtype=dtype)
        self.conv1d = operations.Conv1d(
            conv_dim, conv_dim, self.kernel, groups=conv_dim, bias=False, padding=0, device=device, dtype=dtype
        )
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads, device=device, dtype=dtype))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads, device=device, dtype=dtype))
        self.norm = _GatedRMSNorm(self.head_v_dim, cfg.rms_norm_eps, device=device, dtype=dtype)
        self.out_proj = operations.Linear(self.value_dim, hidden, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        mixed = self.in_proj_qkv(x).transpose(1, 2)
        mixed = F.silu(self.conv1d(F.pad(mixed, (self.kernel - 1, 0)))).transpose(1, 2)
        query, key, value = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(b, s, self.num_k_heads, self.head_k_dim)
        key = key.reshape(b, s, self.num_k_heads, self.head_k_dim)
        value = value.reshape(b, s, self.num_v_heads, self.head_v_dim)
        z = self.in_proj_z(x).reshape(b, s, self.num_v_heads, self.head_v_dim)

        beta = self.in_proj_b(x).sigmoid()
        decay_rate = self.A_log.to(device=x.device, dtype=torch.float32).exp()
        dt_bias = self.dt_bias.to(device=x.device, dtype=torch.float32)
        g = -decay_rate * F.softplus(self.in_proj_a(x).float() + dt_bias)

        repeat = self.num_v_heads // self.num_k_heads
        if repeat > 1:
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)
        core = chunk_gated_delta_rule(_l2_normalize(query.float()), _l2_normalize(key.float()), value, g, beta)
        core = self.norm(core, z).reshape(b, s, self.value_dim).to(x.dtype)
        return self.out_proj(core)


class _GatedAttention(nn.Module):
    def __init__(self, cfg: Qwen35Config, operations, device=None, dtype=None) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        inner = self.num_heads * self.head_dim
        kv_inner = self.num_kv_heads * self.head_dim
        self.q_proj = operations.Linear(cfg.hidden_size, inner * 2, bias=False, device=device, dtype=dtype)
        self.k_proj = operations.Linear(cfg.hidden_size, kv_inner, bias=False, device=device, dtype=dtype)
        self.v_proj = operations.Linear(cfg.hidden_size, kv_inner, bias=False, device=device, dtype=dtype)
        self.o_proj = operations.Linear(inner, cfg.hidden_size, bias=False, device=device, dtype=dtype)
        self.q_norm = _ZeroCenteredRMSNorm(self.head_dim, cfg.rms_norm_eps, device=device, dtype=dtype)
        self.k_norm = _ZeroCenteredRMSNorm(self.head_dim, cfg.rms_norm_eps, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        query, gate = self.q_proj(x).view(b, s, self.num_heads, self.head_dim * 2).chunk(2, dim=-1)
        gate = gate.reshape(b, s, self.num_heads * self.head_dim)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        value = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        rot = cos.shape[-1]
        query = torch.cat([query[..., :rot] * cos + rotate_half(query[..., :rot]) * sin, query[..., rot:]], dim=-1)
        key = torch.cat([key[..., :rot] * cos + rotate_half(key[..., :rot]) * sin, key[..., rot:]], dim=-1)

        groups = self.num_heads // self.num_kv_heads
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        out = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        out = out.transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)
        return self.o_proj(out * torch.sigmoid(gate))


class _Block(nn.Module):
    def __init__(self, cfg: Qwen35Config, layer_type: str, operations, device=None, dtype=None) -> None:
        super().__init__()
        self.layer_type = layer_type
        if layer_type == LINEAR_ATTENTION:
            self.linear_attn = _GatedDeltaNet(cfg, operations, device=device, dtype=dtype)
        elif layer_type == FULL_ATTENTION:
            self.self_attn = _GatedAttention(cfg, operations, device=device, dtype=dtype)
        else:
            raise ValueError(f"unknown Qwen3.5 layer type: {layer_type!r}")
        self.mlp = GatedMLP(cfg.hidden_size, cfg.intermediate_size, operations, device=device, dtype=dtype)
        self.input_layernorm = _ZeroCenteredRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = _ZeroCenteredRMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        h = self.input_layernorm(x)
        if self.layer_type == LINEAR_ATTENTION:
            x = x + self.linear_attn(h)
        else:
            x = x + self.self_attn(h, cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class _Qwen35Transformer(nn.Module):
    def __init__(self, cfg: Qwen35Config, operations, device=None, dtype=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = operations.Embedding(cfg.vocab_size, cfg.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [_Block(cfg, layer_type, operations, device=device, dtype=dtype) for layer_type in cfg.layer_types]
        )
        self.norm = _ZeroCenteredRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, device=device, dtype=dtype)
        self.register_buffer("inv_freq", torch.empty(cfg.rotary_dim // 2), persistent=False)

    def recompute_inv_freq(self) -> None:
        rot = self.cfg.rotary_dim
        exponent = torch.arange(0, rot, 2, dtype=torch.int64).float() / rot
        self.inv_freq = (1.0 / (self.cfg.rope_theta ** exponent)).to(module_device(self))

    def _rope(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None], emb.sin()[None, None]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids).to(torch.float32)
        cos, sin = self._rope(x.shape[1], x.device)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)


class Qwen35Model(NativeArchModule):
    def __init__(self, cfg: Qwen35Config, operations, device=None, dtype=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = _Qwen35Transformer(cfg, operations, device=device, dtype=dtype)

    @classmethod
    def from_config(cls, config: dict[str, Any], operations: Any) -> "Qwen35Model":
        return cls(Qwen35Config.from_dict(config), operations)

    def post_load(self) -> None:
        self.model.recompute_inv_freq()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)


class MossImageTextEncoder(NativeTextEncoder):
    role = "qwen35_9b"

    def __init__(self, module: Qwen35Model, tokenizer, variant: str = "qwen35_9b",
                 device: str | torch.device = "cpu") -> None:
        self.module = module
        self.tokenizer = tokenizer
        self.role = variant
        self._device = torch.device(device)

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> dict[str, torch.Tensor]:
        if len(texts) != 1:
            raise ValueError(f"the MOSS-Image text encoder takes one prompt per call, got {len(texts)}")
        ids = self.tokenizer(texts, device=self._device)
        return {"context": self.module(ids), "attention_mask": None}
