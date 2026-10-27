from __future__ import annotations

import math

import pytest
import torch
from safetensors.torch import save_file

from src.platform.runtime.native.base import load_into_module
from src.platform.runtime.native.text_encoders import loader
from src.platform.runtime.native.text_encoders.loader import (
    _SPECS,
    _make_encoder,
    _normalise_hf_qwen3vl_keys,
    load_text_encoder,
)
from src.platform.runtime.native.text_encoders.qwen3 import (
    FLUX3_LAYERS,
    Flux3TextEncoder,
    Qwen3Model,
    Qwen3TextEncoder,
    Qwen3VLTextEncoder,
)
from src.platform.runtime.native.text_encoders.tokenization import (
    QWEN3_PAD,
    ZIMAGE_TEMPLATE,
    Flux3Tokenizer,
)
from src.platform.runtime.native.kernels import pick_operations

from ._fixtures import tiny_qwen3_state_dict

_CFG = {
    "hidden_size": 16, "num_layers": 36, "vocab_size": 151936,
    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
    "intermediate_size": 32,
}


def _module() -> Qwen3Model:
    torch.manual_seed(0)
    m = Qwen3Model.from_config(_CFG, pick_operations(torch.float32, torch.float32))
    sd = {}
    for k, v in m.state_dict().items():
        if k.endswith("norm.weight"):
            sd[k] = torch.ones_like(v)
        elif v.is_floating_point():
            sd[k] = torch.randn_like(v) * 0.05
        else:
            sd[k] = v.clone()
    load_into_module(m, sd, _SPECS["qwen3"])
    return m.eval()


def _reference_layer_outputs(m: Qwen3Model, ids: torch.Tensor, mask: torch.Tensor, positions: torch.Tensor):
    x = m.model.embed_tokens(ids).float()
    s = x.shape[1]
    freqs = positions.float()[:, None] * m.model.inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[None, None], emb.sin()[None, None]
    low = torch.finfo(x.dtype).min / 4
    causal = torch.empty(s, s).fill_(low).triu_(1)
    keys = (1.0 - mask.float()).reshape(1, 1, 1, s).expand(1, 1, s, s)
    attn = keys.masked_fill(keys.bool(), low) + causal
    outs = []
    for layer in m.model.layers:
        x = layer(x, cos, sin, attn)
        outs.append(x)
    return outs


def _padded_input(real: int = 7, total: int = 12):
    ids = torch.full((1, total), QWEN3_PAD, dtype=torch.long)
    ids[0, :real] = torch.randint(0, 1000, (real,))
    mask = torch.zeros(1, total, dtype=torch.long)
    mask[0, :real] = 1
    return ids, mask


def test_layers_are_hf_hidden_states_four_to_thirty_two():
    assert FLUX3_LAYERS == tuple(k - 1 for k in range(4, 33, 4))


def test_encode_matches_hf_style_reference_for_bfl_contract():
    m = _module()
    ids, mask = _padded_input()
    positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)[0]
    outs = _reference_layer_outputs(m, ids, mask, positions)
    hidden_states = [None] + outs
    expected = torch.stack([hidden_states[k] for k in range(4, 33, 4)], dim=1)
    expected = expected.permute(0, 2, 1, 3).reshape(1, ids.shape[1], -1)

    out = Flux3TextEncoder(m, tokenizer=None)._encode_ids(ids, mask)

    assert out["attention_mask"] is None
    assert out["context"].shape == (1, 12, 8 * 16)
    torch.testing.assert_close(out["context"], expected, atol=1e-5, rtol=1e-5)


def test_stack_is_layer_major():
    m = _module()
    ids, mask = _padded_input()
    ctx = Flux3TextEncoder(m, tokenizer=None)._encode_ids(ids, mask)["context"]
    single = m(ids, attention_mask=mask, layers_to_extract=(FLUX3_LAYERS[2],), capture="output",
               position_ids=((mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)[0])[None].expand(3, -1))
    torch.testing.assert_close(ctx[0, :, 2 * 16:3 * 16], single[0, 0])


def test_padding_rows_use_position_one():
    m = _module()
    ids, mask = _padded_input(real=7, total=12)
    enc = Flux3TextEncoder(m, tokenizer=None)
    ours = enc._encode_ids(ids, mask)["context"]
    naive = m(ids, attention_mask=mask, layers_to_extract=FLUX3_LAYERS, capture="output")
    naive = naive.movedim(1, 2).reshape(1, 12, -1)
    torch.testing.assert_close(ours[:, :7], naive[:, :7], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(ours[:, 7:], naive[:, 7:], atol=1e-4)


def test_forward_stops_after_last_captured_layer():
    m = _module()
    ids, mask = _padded_input()

    def boom(*args, **kwargs):
        raise AssertionError("ran a layer past the last captured one")

    m.model.layers[32].forward = boom
    Flux3TextEncoder(m, tokenizer=None)._encode_ids(ids, mask)


def test_encode_rejects_batches():
    enc = Flux3TextEncoder(_module(), Flux3Tokenizer())
    with pytest.raises(ValueError):
        enc.encode(["a", "b"])


def test_tokenizer_template_and_pad_multiple():
    tok = Flux3Tokenizer()
    ids, mask = tok(["a cat on a mat"])
    expected = tok._tok(ZIMAGE_TEMPLATE.format("a cat on a mat"))["input_ids"]
    real = len(expected)
    assert ids.shape[1] == math.ceil(real / 80) * 80
    assert ids[0, :real].tolist() == expected
    assert int(mask.sum()) == real
    assert (ids[0, real:] == QWEN3_PAD).all()
    assert (mask[0, real:] == 0).all()


def test_tokenizer_caps_at_4096():
    tok = Flux3Tokenizer()
    ids, mask = tok(["word " * 6000])
    assert ids.shape[1] == 4096
    assert int(mask.sum()) == 4096


def test_tokenizer_exact_multiple_gets_no_extra_padding():
    tok = Flux3Tokenizer()
    base = len(tok._tok(ZIMAGE_TEMPLATE.format(""))["input_ids"])
    ids, _ = tok(["x " * (80 - base)])
    real = len(tok._tok(ZIMAGE_TEMPLATE.format("x " * (80 - base)))["input_ids"])
    assert ids.shape[1] == math.ceil(real / 80) * 80


def test_normaliser_maps_hf_language_model_keys_and_drops_lm_head():
    sd = {
        "model.language_model.embed_tokens.weight": 1,
        "model.language_model.layers.0.mlp.up_proj.weight": 2,
        "model.visual.blocks.0.attn.qkv.weight": 3,
        "lm_head.weight": 4,
    }
    assert _normalise_hf_qwen3vl_keys(sd) == {
        "model.embed_tokens.weight": 1,
        "model.layers.0.mlp.up_proj.weight": 2,
        "model.visual.blocks.0.attn.qkv.weight": 3,
    }


def test_normaliser_leaves_comfy_layout_untouched():
    sd = {"model.embed_tokens.weight": 1, "lm_head.weight": 2}
    assert _normalise_hf_qwen3vl_keys(sd) is sd


def test_make_encoder_variant_selects_flux3_and_default_is_unchanged():
    m = _module()
    assert isinstance(_make_encoder("qwen3vl", "qwen3vl_4b", m, "cpu", te_variant="flux3"), Flux3TextEncoder)
    assert isinstance(_make_encoder("qwen3", "qwen3_4b", m, "cpu", te_variant="flux3"), Flux3TextEncoder)
    assert isinstance(_make_encoder("qwen3vl", "qwen3vl_4b", m, "cpu"), Qwen3VLTextEncoder)
    assert isinstance(_make_encoder("qwen3", "qwen3_4b", m, "cpu"), Qwen3TextEncoder)


def test_load_text_encoder_flux3_from_hf_native_file(tmp_path):
    sd = tiny_qwen3_state_dict(num_layers=32, hidden=64, dtype=torch.float32)
    hf = {"model.language_model." + k[len("model."):]: v for k, v in sd.items()}
    hf["model.visual.blocks.0.attn.qkv.weight"] = torch.zeros(4, 4)
    hf["lm_head.weight"] = torch.zeros(8, 64)
    path = tmp_path / "te.safetensors"
    save_file(hf, str(path))

    enc = load_text_encoder(str(path), te_variant="flux3")
    assert isinstance(enc, Flux3TextEncoder)
    out = enc.encode(["a cat"])
    assert out["attention_mask"] is None
    assert out["context"].shape == (1, 80, 8 * 64)
    assert torch.isfinite(out["context"]).all()


def test_engine_load_te_passes_te_variant(monkeypatch):
    from src.platform.runtime.native import engine as engine_mod

    seen = {}

    def fake_load(path, **kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(engine_mod, "load_text_encoder", fake_load)
    monkeypatch.setattr(engine_mod, "_estimate_text_encoder_gb", lambda enc: None)
    engine_mod.NativeEngineLoader()._load_te("x.safetensors", te_variant="flux3")
    assert seen["te_variant"] == "flux3"
