from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

FAMILY_LAYOUTS: dict[str, tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]] = {
    "flux": (
        {
            "double_blocks.0.img_attn.qkv.weight": (24, 8),
            "double_blocks.0.img_attn.proj.weight": (8, 8),
            "double_blocks.0.img_mlp.0.weight": (32, 8),
            "single_blocks.0.linear1.weight": (40, 8),
        },
        {
            "img_in.weight": (8, 16),
            "txt_in.weight": (8, 12),
            "time_in.in_layer.weight": (8, 4),
            "vector_in.in_layer.weight": (8, 4),
            "guidance_in.in_layer.weight": (8, 4),
            "double_blocks.0.img_mod.lin.weight": (48, 8),
            "double_blocks.0.img_attn.qkv.bias": (24,),
            "double_blocks.0.img_attn.norm.query_norm.scale": (4,),
            "final_layer.linear.weight": (16, 8),
            "final_layer.adaLN_modulation.1.weight": (16, 8),
        },
    ),
    "wan": (
        {
            "blocks.0.self_attn.q.weight": (8, 8),
            "blocks.0.cross_attn.k.weight": (8, 8),
            "blocks.0.ffn.0.weight": (16, 8),
        },
        {
            "patch_embedding.weight": (8, 4, 1, 2, 2),
            "text_embedding.0.weight": (8, 12),
            "time_embedding.0.weight": (8, 4),
            "time_projection.1.weight": (48, 8),
            "blocks.0.modulation": (1, 6, 8),
            "blocks.0.norm3.weight": (8,),
            "blocks.0.self_attn.q.bias": (8,),
            "head.head.weight": (16, 8),
            "head.modulation": (1, 2, 8),
        },
    ),
    "qwen_image": (
        {
            "transformer_blocks.0.attn.to_q.weight": (8, 8),
            "transformer_blocks.0.attn.add_k_proj.weight": (8, 8),
            "transformer_blocks.0.img_mlp.net.0.proj.weight": (16, 8),
        },
        {
            "img_in.weight": (8, 16),
            "txt_in.weight": (8, 12),
            "txt_norm.weight": (12,),
            "time_text_embed.timestep_embedder.linear_1.weight": (8, 4),
            "transformer_blocks.0.img_mod.1.weight": (48, 8),
            "transformer_blocks.0.attn.to_q.bias": (8,),
            "norm_out.linear.weight": (16, 8),
            "proj_out.weight": (16, 8),
        },
    ),
    "z_image": (
        {
            "layers.0.attention.qkv.weight": (24, 8),
            "layers.0.attention.out.weight": (8, 8),
            "layers.0.feed_forward.w1.weight": (16, 8),
        },
        {
            "x_embedder.weight": (8, 16),
            "cap_embedder.1.weight": (8, 12),
            "t_embedder.mlp.0.weight": (8, 4),
            "layers.0.adaLN_modulation.0.weight": (32, 4),
            "layers.0.attention_norm1.weight": (8,),
            "final_layer.linear.weight": (16, 8),
        },
    ),
    "krea2": (
        {
            "blocks.0.attn.q.weight": (8, 8),
            "blocks.0.mlp.fc1.weight": (16, 8),
        },
        {
            "img_in.weight": (8, 16),
            "txt_in.weight": (8, 12),
            "blocks.0.modulation.lin.weight": (48, 8),
            "blocks.0.norm.weight": (8,),
            "final_layer.linear.weight": (16, 8),
        },
    ),
    "moss_image": (
        {
            "layers.0.proj.qkv.weight": (24, 8),
            "layers.0.proj.o.weight": (8, 8),
            "layers.0.proj.ffn_up_gate.weight": (32, 8),
            "layers.0.proj.ffn_down.weight": (8, 16),
        },
        {
            "video_embedder.weight": (8, 16),
            "video_embedder.bias": (8,),
            "text_embedder.proj.weight": (8, 12),
            "text_embedder.blocks.0.qkv.weight": (24, 8),
            "time_embedding.0.weight": (8, 4),
            "time_embedding.2.weight": (8, 8),
            "shared_block_modulation.weight": (48, 8),
            "layers.0.norm1.weight": (8,),
            "layers.0.q_norm.weight": (4,),
            "final_norm_video.weight": (8,),
            "final_linear_video.weight": (16, 8),
            "video_head_modulation.weight": (16, 8),
        },
    ),
    "ltx": (
        {
            "transformer_blocks.0.attn1.to_q.weight": (8, 8),
            "transformer_blocks.0.ff.net.0.proj.weight": (16, 8),
        },
        {
            "patchify_proj.weight": (8, 16),
            "adaln_single.linear.weight": (48, 8),
            "caption_projection.linear_1.weight": (8, 12),
            "scale_shift_table": (2, 8),
            "transformer_blocks.0.scale_shift_table": (6, 8),
            "transformer_blocks.0.attn1.q_norm.weight": (8,),
            "proj_out.weight": (16, 8),
        },
    ),
    "t5xxl": (
        {
            "encoder.block.0.layer.0.SelfAttention.q.weight": (8, 8),
            "encoder.block.0.layer.1.DenseReluDense.wi_0.weight": (16, 8),
            "encoder.block.0.layer.1.DenseReluDense.wo.weight": (8, 16),
        },
        {
            "shared.weight": (32, 8),
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": (4, 2),
            "encoder.block.0.layer.0.layer_norm.weight": (8,),
            "encoder.final_layer_norm.weight": (8,),
        },
    ),
    "qwen3": (
        {
            "model.layers.0.self_attn.q_proj.weight": (8, 8),
            "model.layers.0.mlp.gate_proj.weight": (16, 8),
        },
        {
            "model.embed_tokens.weight": (32, 8),
            "model.layers.0.self_attn.q_norm.weight": (4,),
            "model.layers.0.input_layernorm.weight": (8,),
            "model.norm.weight": (8,),
            "lm_head.weight": (32, 8),
        },
    ),
}


def make_tensors(
    layout: dict[str, tuple[int, ...]], dtype: torch.dtype = torch.bfloat16, seed: int = 0
) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {k: (torch.randn(s, generator=gen) * 0.5).to(dtype) for k, s in layout.items()}


def write_family_fixture(
    path: Path, family: str, dtype: torch.dtype = torch.bfloat16, metadata: dict[str, str] | None = None
) -> tuple[list[str], list[str]]:
    quant, kept = FAMILY_LAYOUTS[family]
    tensors = make_tensors({**quant, **kept}, dtype)
    save_file(tensors, str(path), metadata=metadata or {"format": "pt"})
    return list(quant), list(kept)
