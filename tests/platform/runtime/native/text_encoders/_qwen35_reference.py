from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-6


def zero_centered_rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + weight.float())


def gated_rms(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x = x.float()
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return weight.float() * normed * F.silu(gate.float())


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + EPS)


def causal_depthwise_conv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    kernel = weight.shape[-1]
    padded = F.pad(x, (kernel - 1, 0))
    out = torch.zeros_like(x)
    for j in range(kernel):
        out = out + weight[:, 0, j][None, :, None] * padded[:, :, j : j + x.shape[-1]]
    return out


def delta_rule(q, k, v, g, beta) -> torch.Tensor:
    b, s, h, dk = q.shape
    dv = v.shape[-1]
    q = q.float() * dk ** -0.5
    k, v, g, beta = k.float(), v.float(), g.float(), beta.float()
    state = torch.zeros(b, h, dk, dv)
    outs = []
    for t in range(s):
        state = state * g[:, t].exp()[:, :, None, None]
        recalled = torch.einsum("bhkv,bhk->bhv", state, k[:, t])
        delta = (v[:, t] - recalled) * beta[:, t][..., None]
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, t], delta)
        outs.append(torch.einsum("bhkv,bhk->bhv", state, q[:, t]))
    return torch.stack(outs, dim=1)


def partial_rope(x: torch.Tensor, rot: int, theta: float) -> torch.Tensor:
    s = x.shape[2]
    inv_freq = theta ** (-torch.arange(0, rot, 2).float() / rot)
    angles = torch.arange(s).float()[:, None] * inv_freq[None, :]
    cos, sin = angles.cos()[None, None], angles.sin()[None, None]
    head, tail = x[..., :rot], x[..., rot:]
    x1, x2 = head[..., : rot // 2], head[..., rot // 2 :]
    rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return torch.cat([rotated, tail], dim=-1)


def linear_attention(sd: dict, p: str, x: torch.Tensor, cfg: dict) -> torch.Tensor:
    b, s, _ = x.shape
    nk, nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    dk, dv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    qkv = (x @ sd[p + "in_proj_qkv.weight"].T).transpose(1, 2)
    qkv = F.silu(causal_depthwise_conv(qkv, sd[p + "conv1d.weight"])).transpose(1, 2)
    q, k, v = qkv[..., : nk * dk], qkv[..., nk * dk : 2 * nk * dk], qkv[..., 2 * nk * dk :]
    q = q.reshape(b, s, nk, dk).repeat_interleave(nv // nk, dim=2)
    k = k.reshape(b, s, nk, dk).repeat_interleave(nv // nk, dim=2)
    v = v.reshape(b, s, nv, dv)
    z = (x @ sd[p + "in_proj_z.weight"].T).reshape(b, s, nv, dv)
    beta = torch.sigmoid(x @ sd[p + "in_proj_b.weight"].T)
    a = x @ sd[p + "in_proj_a.weight"].T
    g = -sd[p + "A_log"].float().exp() * F.softplus(a.float() + sd[p + "dt_bias"].float())
    core = delta_rule(l2_normalize(q), l2_normalize(k), v, g, beta)
    core = gated_rms(core, z, sd[p + "norm.weight"]).reshape(b, s, nv * dv)
    return core @ sd[p + "out_proj.weight"].T


def full_attention(sd: dict, p: str, x: torch.Tensor, cfg: dict) -> torch.Tensor:
    b, s, _ = x.shape
    heads, kv_heads, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    packed = (x @ sd[p + "q_proj.weight"].T).reshape(b, s, heads, 2 * hd)
    q, gate = packed[..., :hd], packed[..., hd:]
    k = (x @ sd[p + "k_proj.weight"].T).reshape(b, s, kv_heads, hd)
    v = (x @ sd[p + "v_proj.weight"].T).reshape(b, s, kv_heads, hd)
    q = zero_centered_rms(q, sd[p + "q_norm.weight"]).transpose(1, 2)
    k = zero_centered_rms(k, sd[p + "k_norm.weight"]).transpose(1, 2)
    v = v.transpose(1, 2)
    rot = int(hd * cfg["partial_rotary_factor"])
    q = partial_rope(q, rot, cfg["rope_theta"])
    k = partial_rope(k, rot, cfg["rope_theta"])
    kv_index = torch.arange(heads) // (heads // kv_heads)
    k, v = k[:, kv_index], v[:, kv_index]
    scores = (q @ k.transpose(-1, -2)) / hd ** 0.5
    causal = torch.ones(s, s, dtype=torch.bool).triu(1)
    probs = scores.masked_fill(causal, float("-inf")).softmax(-1)
    out = (probs @ v).transpose(1, 2).reshape(b, s, heads * hd)
    out = out * torch.sigmoid(gate.reshape(b, s, heads * hd))
    return out @ sd[p + "o_proj.weight"].T


def forward(sd: dict, cfg: dict, input_ids: torch.Tensor) -> torch.Tensor:
    x = sd["model.embed_tokens.weight"].float()[input_ids]
    for i, kind in enumerate(cfg["layer_types"]):
        p = f"model.layers.{i}."
        h = zero_centered_rms(x, sd[p + "input_layernorm.weight"])
        if kind == "linear_attention":
            x = x + linear_attention(sd, p + "linear_attn.", h, cfg)
        else:
            x = x + full_attention(sd, p + "self_attn.", h, cfg)
        h = zero_centered_rms(x, sd[p + "post_attention_layernorm.weight"])
        gated = F.silu(h @ sd[p + "mlp.gate_proj.weight"].T) * (h @ sd[p + "mlp.up_proj.weight"].T)
        x = x + gated @ sd[p + "mlp.down_proj.weight"].T
    return zero_centered_rms(x, sd["model.norm.weight"])
