from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from src.platform.runtime.native.base import load_into_module
from src.platform.runtime.native.detect.te_detect import detect_te_config
from src.platform.runtime.native.text_encoders import loader as loader_mod
from src.platform.runtime.native.text_encoders.loader import (
    _SPECS, QWEN35_STRIP_PREFIXES, _build_config, _make_encoder, _normalise_hf_qwen3vl_keys, load_text_encoder,
)
from src.platform.runtime.native.text_encoders.qwen35 import (
    MossImageTextEncoder, Qwen35Config, Qwen35Model, chunk_gated_delta_rule,
)
from vendor.gpl.comfyui.ops import disable_weight_init as plain_ops
from vendor.gpl.comfyui.ops import fp8_ops

from .._quant_layouts import convrot_descriptor, descriptor_blob, reference_quantize_convrot
from . import _qwen35_reference as reference

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "qwen3.5_9b_int8_convrot.keys.txt"

TINY = {
    "hidden_size": 64,
    "num_layers": 4,
    "vocab_size": 300,
    "intermediate_size": 96,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 32,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000000.0,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
}

_DTYPES = {"I8": torch.int8, "U8": torch.uint8, "F32": torch.float32, "BF16": torch.bfloat16}


def _randomize(module: torch.nn.Module, seed: int = 0) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            std = 0.5 if name.endswith("embed_tokens.weight") else 0.15
            p.copy_(torch.randn(p.shape, generator=gen) * std)


def _tiny_module(operations=plain_ops, seed: int = 0, config: dict | None = None) -> Qwen35Model:
    module = Qwen35Model.from_config(config or TINY, operations)
    _randomize(module, seed)
    module.post_load()
    return module


def _fixture_meta_sd() -> dict[str, torch.Tensor]:
    sd = {}
    for line in FIXTURE.read_text().splitlines():
        key, dtype, shape = line.split("\t")
        dims = tuple(int(d) for d in shape.split("x"))
        sd[key] = torch.empty(dims, dtype=_DTYPES[dtype], device="meta")
    return sd


def _stripped_fixture_sd() -> dict[str, torch.Tensor]:
    sd = _normalise_hf_qwen3vl_keys(_fixture_meta_sd())
    return {k: v for k, v in sd.items() if not k.startswith(QWEN35_STRIP_PREFIXES)}


@pytest.mark.parametrize("length", [1, 7, 70])
def test_native_matches_independent_reference(length):
    module = _tiny_module()
    ids = torch.randint(0, TINY["vocab_size"], (1, length), generator=torch.Generator().manual_seed(length))
    with torch.no_grad():
        got = module(ids)
        want = reference.forward(dict(module.state_dict()), TINY, ids)
    assert got.shape == (1, length, TINY["hidden_size"])
    assert torch.allclose(got, want, atol=2e-4, rtol=2e-4)


def test_reference_comparison_can_fail():
    module = _tiny_module()
    ids = torch.randint(0, TINY["vocab_size"], (1, 9))
    sd = dict(module.state_dict())
    sd["model.layers.3.self_attn.q_proj.weight"] = sd["model.layers.3.self_attn.q_proj.weight"].flip(0)
    with torch.no_grad():
        got = module(ids)
        want = reference.forward(sd, TINY, ids)
    assert not torch.allclose(got, want, atol=2e-4, rtol=2e-4)


def test_zero_centered_norm_scales_by_one_plus_weight():
    module = _tiny_module()
    norm = module.model.layers[0].input_layernorm
    with torch.no_grad():
        norm.weight.zero_()
    x = torch.randn(2, 5, TINY["hidden_size"])
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(norm(x), expected, atol=1e-5)


def test_rope_rotates_a_quarter_of_the_head_at_theta_1e7():
    module = _tiny_module()
    cos, sin = module.model._rope(6, torch.device("cpu"))
    assert cos.shape == (1, 1, 6, 8)
    expected = 1e7 ** (-torch.arange(0, 8, 2).float() / 8)
    assert torch.allclose(module.model.inv_freq, expected)


@pytest.mark.parametrize("length", [1, 63, 64, 65, 130])
@pytest.mark.parametrize("decay_scale", [0.3, 3.0])
def test_chunked_delta_rule_matches_recurrent(length, decay_scale):
    gen = torch.Generator().manual_seed(length)
    b, h, dk, dv = 2, 3, 16, 8
    q = reference.l2_normalize(torch.randn(b, length, h, dk, generator=gen))
    k = reference.l2_normalize(torch.randn(b, length, h, dk, generator=gen))
    v = torch.randn(b, length, h, dv, generator=gen)
    g = -torch.rand(b, length, h, generator=gen) * decay_scale
    beta = torch.sigmoid(torch.randn(b, length, h, generator=gen))
    got = chunk_gated_delta_rule(q, k, v, g, beta)
    want = reference.delta_rule(q, k, v, g, beta)
    assert got.shape == (b, length, h, dv)
    assert torch.allclose(got, want, atol=2e-4, rtol=2e-4)


def test_delta_rule_comparison_can_fail():
    gen = torch.Generator().manual_seed(3)
    q = reference.l2_normalize(torch.randn(1, 70, 2, 16, generator=gen))
    k = reference.l2_normalize(torch.randn(1, 70, 2, 16, generator=gen))
    v = torch.randn(1, 70, 2, 8, generator=gen)
    g = -torch.rand(1, 70, 2, generator=gen)
    beta = torch.sigmoid(torch.randn(1, 70, 2, generator=gen))
    got = chunk_gated_delta_rule(q, k, v, g, beta)
    want = reference.delta_rule(q, k, v, g * 0.5, beta)
    assert not torch.allclose(got, want, atol=2e-4, rtol=2e-4)


def test_fixture_header_is_the_real_layout():
    sd = _fixture_meta_sd()
    assert len(sd) == 1409
    assert sd["model.language_model.embed_tokens.weight"].dtype == torch.int8
    assert sd["model.language_model.layers.0.linear_attn.A_log"].dtype == torch.float32
    assert sd["model.language_model.layers.3.self_attn.q_proj.weight"].shape == (8192, 4096)


def test_strip_removes_visual_mtp_and_head_but_keeps_the_language_model():
    raw = _fixture_meta_sd()
    stripped = _stripped_fixture_sd()
    assert any(k.startswith("model.visual.") for k in raw)
    assert any(k.startswith("mtp.") for k in raw)
    assert any(k.startswith("lm_head.") for k in raw)
    assert not any(k.startswith(QWEN35_STRIP_PREFIXES) for k in stripped)
    assert "model.embed_tokens.weight" in stripped
    assert "model.norm.weight" in stripped
    assert len(stripped) == 1409 - 549 - sum(k.startswith(("mtp.", "lm_head.")) for k in raw)


def test_real_header_detects_as_qwen35_9b():
    config = detect_te_config(_stripped_fixture_sd())
    assert config["te_type"] == "qwen35"
    assert config["variant"] == "qwen35_9b"
    assert config["hidden_size"] == 4096
    assert config["vocab_size"] == 248320
    assert config["num_layers"] == 32
    expected = ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(32)]
    assert config["layer_types"] == expected


def test_real_header_dimensions_are_recovered_from_shapes():
    sd = _stripped_fixture_sd()
    config = _build_config(detect_te_config(sd), sd)
    assert config["intermediate_size"] == 12288
    assert config["head_dim"] == 256
    assert config["num_attention_heads"] == 16
    assert config["num_key_value_heads"] == 4
    assert config["linear_num_value_heads"] == 32
    assert config["linear_num_key_heads"] == 16
    assert config["linear_key_head_dim"] == 128
    assert config["linear_value_head_dim"] == 128
    assert config["linear_conv_kernel_dim"] == 4


def test_real_header_config_resolves_rope_and_geometry():
    sd = _stripped_fixture_sd()
    config = Qwen35Config.from_dict(_build_config(detect_te_config(sd), sd))
    assert config.rotary_dim == 64
    assert config.rope_theta == 1e7
    assert config.rms_norm_eps == 1e-6
    assert config.num_hidden_layers == 32
    assert config.layer_types[:4] == ("linear_attention",) * 3 + ("full_attention",)


def test_module_keys_and_shapes_match_the_real_header():
    sd = _stripped_fixture_sd()
    config = _build_config(detect_te_config(sd), sd)
    with torch.device("meta"):
        module = Qwen35Model.from_config(config, plain_ops)
    spec = _SPECS["qwen35"]
    state = module.state_dict()
    assert set(state) <= set(sd)
    extra = [k for k in sd if k not in state and not spec.key_is_expected_unexpected(k)]
    assert extra == []
    for key, tensor in state.items():
        assert tuple(tensor.shape) == tuple(sd[key].shape), key
        quantised = key.removesuffix(".weight") + ".comfy_quant" in sd
        assert (sd[key].dtype == torch.int8) == quantised, key


def test_hidden_size_other_than_the_9b_is_not_detected():
    sd = {
        "model.embed_tokens.weight": torch.empty(10, 2560, device="meta"),
        "model.layers.0.linear_attn.A_log": torch.empty(8, device="meta"),
    }
    assert detect_te_config(sd) is None


def test_qwen3_checkpoints_are_not_detected_as_qwen35():
    sd = {
        "model.embed_tokens.weight": torch.empty(10, 4096, device="meta"),
        "model.layers.0.self_attn.q_norm.weight": torch.empty(128, device="meta"),
        "model.layers.0.self_attn.q_proj.weight": torch.empty(4096, 4096, device="meta"),
    }
    config = detect_te_config(sd)
    assert config is not None and config["te_type"] != "qwen35"


def test_config_rejects_mismatched_layer_types():
    with pytest.raises(ValueError):
        Qwen35Config.from_dict({**TINY, "layer_types": ["full_attention"]})


def test_default_layer_types_put_full_attention_every_fourth_layer():
    config = Qwen35Config.from_dict({k: v for k, v in TINY.items() if k != "layer_types"})
    assert config.layer_types == ("linear_attention",) * 3 + ("full_attention",)


def _convrot_quantise(dense: Qwen35Model, group: int) -> dict[str, torch.Tensor]:
    out = {}
    for key, tensor in dense.state_dict().items():
        stem = key.removesuffix(".weight")
        projected = key.endswith(".weight") and (
            key.endswith("embed_tokens.weight")
            or (stem.endswith("_proj") and not stem.endswith(("in_proj_a", "in_proj_b")))
        )
        if not projected:
            out[key] = tensor
            continue
        codes, scale = reference_quantize_convrot(tensor, group)
        out[key] = codes
        out[stem + ".weight_scale"] = scale
        out[stem + ".comfy_quant"] = descriptor_blob(convrot_descriptor(group))
    return out


def test_int8_convrot_embedding_and_linears_round_trip():
    dense = _tiny_module(seed=5)
    quantised_sd = _convrot_quantise(dense, 16)
    module = Qwen35Model.from_config(TINY, fp8_ops)
    load_into_module(module, quantised_sd, _SPECS["qwen35"])

    assert module.model.embed_tokens._int8_weight is not None
    assert module.model.layers[0].mlp.gate_proj.weight.dtype == torch.int8
    assert module.model.layers[0].linear_attn.in_proj_a.weight.dtype != torch.int8

    ids = torch.randint(0, TINY["vocab_size"], (1, 12), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        got = module(ids)
        want = dense(ids)
    cosine = torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0)
    assert cosine > 0.98
    assert torch.allclose(got, want, atol=0.35)


def test_int8_codes_without_the_rotation_are_visibly_wrong():
    dense = _tiny_module(seed=5)
    quantised_sd = _convrot_quantise(dense, 16)
    for key in list(quantised_sd):
        if key.endswith(".comfy_quant"):
            quantised_sd[key] = descriptor_blob({"format": "int8_tensorwise"})
    module = Qwen35Model.from_config(TINY, fp8_ops)
    load_into_module(module, quantised_sd, _SPECS["qwen35"])
    ids = torch.randint(0, TINY["vocab_size"], (1, 12), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        got = module(ids)
        want = dense(ids)
    cosine = torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0)
    assert cosine < 0.9


class _StubTokenizer:
    def __init__(self, ids: list[int] | None = None) -> None:
        self.ids = ids or [3, 14, 15, 92, 65]

    def __call__(self, texts, device="cpu"):
        if len(texts) != 1:
            raise ValueError("one prompt")
        return torch.tensor([self.ids], dtype=torch.long, device=device)


def test_encoder_returns_last_hidden_state_without_a_mask():
    module = _tiny_module()
    encoder = MossImageTextEncoder(module, _StubTokenizer(), variant="qwen35_9b")
    out = encoder.encode(["a cat"])
    assert set(out) == {"context", "attention_mask"}
    assert out["attention_mask"] is None
    assert out["context"].shape == (1, 5, TINY["hidden_size"])
    ids = torch.tensor([[3, 14, 15, 92, 65]])
    with torch.no_grad():
        assert torch.allclose(out["context"], module(ids))
    assert encoder.role == "qwen35_9b"


def test_encoder_takes_one_prompt_per_call():
    encoder = MossImageTextEncoder(_tiny_module(), _StubTokenizer())
    with pytest.raises(ValueError):
        encoder.encode(["a", "b"])


def test_make_encoder_builds_the_moss_image_encoder():
    module = _tiny_module()
    encoder = _make_encoder("qwen35", "qwen35_9b", module, "cpu")
    assert isinstance(encoder, MossImageTextEncoder)
    assert encoder.role == "qwen35_9b"


def test_loader_end_to_end_strips_extras_and_encodes(tmp_path, monkeypatch):
    config = {
        **TINY,
        "hidden_size": 4096,
        "intermediate_size": 64,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 16,
    }
    dense = _tiny_module(config=config, seed=2)
    sd = {"model.language_model." + k.removeprefix("model."): v.contiguous() for k, v in dense.state_dict().items()}
    sd["lm_head.weight"] = torch.zeros(300, 4096)
    sd["lm_head.weight_scale"] = torch.ones(300, 1)
    sd["mtp.fc.weight"] = torch.zeros(8, 8)
    sd["model.visual.blocks.0.attn.qkv.weight"] = torch.zeros(8, 8)
    path = tmp_path / "qwen35_tiny.safetensors"
    save_file(sd, str(path))

    loaded_keys = []
    real_load = loader_mod.load_into_module

    def recording_load(module, state, spec):
        loaded_keys.extend(state)
        real_load(module, state, spec)

    monkeypatch.setattr(loader_mod, "MossImageTokenizer", _StubTokenizer)
    monkeypatch.setattr(loader_mod, "load_into_module", recording_load)
    encoder = load_text_encoder(path)

    assert loaded_keys
    assert not any(k.startswith(("model.visual.", "mtp.", "lm_head.")) for k in loaded_keys)
    assert "model.embed_tokens.weight" in loaded_keys

    assert isinstance(encoder, MossImageTextEncoder)
    assert encoder.role == "qwen35_9b"
    out = encoder.encode(["a cat"])
    ids = torch.tensor([[3, 14, 15, 92, 65]])
    with torch.no_grad():
        want = dense(ids)
    assert out["context"].shape == (1, 5, 4096)
    assert torch.allclose(out["context"], want, atol=1e-4, rtol=1e-4)


def test_transformers_parity():
    pytest.importorskip("transformers.models.qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    hf_config = Qwen3_5TextConfig(
        vocab_size=TINY["vocab_size"],
        hidden_size=TINY["hidden_size"],
        intermediate_size=TINY["intermediate_size"],
        num_hidden_layers=TINY["num_layers"],
        num_attention_heads=TINY["num_attention_heads"],
        num_key_value_heads=TINY["num_key_value_heads"],
        head_dim=TINY["head_dim"],
        linear_conv_kernel_dim=TINY["linear_conv_kernel_dim"],
        linear_key_head_dim=TINY["linear_key_head_dim"],
        linear_value_head_dim=TINY["linear_value_head_dim"],
        linear_num_key_heads=TINY["linear_num_key_heads"],
        linear_num_value_heads=TINY["linear_num_value_heads"],
        layer_types=TINY["layer_types"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": TINY["rope_theta"],
            "partial_rotary_factor": TINY["partial_rotary_factor"],
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
    )
    hf = Qwen3_5TextModel(hf_config).eval()
    module = _tiny_module(seed=9)
    hf.load_state_dict({k.removeprefix("model."): v for k, v in module.state_dict().items()})
    ids = torch.randint(0, TINY["vocab_size"], (1, 70), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        want = hf(input_ids=ids).last_hidden_state
        got = module(ids)
    assert torch.allclose(got, want, atol=2e-4, rtol=2e-4)


@pytest.fixture(scope="module")
def tokenizer():
    pytest.importorskip("tokenizers")
    from src.platform.runtime.native.text_encoders.tokenization import MossImageTokenizer

    return MossImageTokenizer()


def test_tokenizer_ids_for_a_short_prompt(tokenizer):
    assert tokenizer.encode("A cat") == [248045, 846, 198, 32, 7993, 248046, 198, 248045, 74455, 198, 248068, 198]


def test_tokenizer_ids_for_the_empty_prompt(tokenizer):
    assert tokenizer.encode("") == [248045, 846, 198, 248046, 198, 248045, 74455, 198, 248068, 198]


def test_tokenizer_strips_surrounding_whitespace(tokenizer):
    assert tokenizer.encode("  A cat \n") == tokenizer.encode("A cat")


def test_tokenizer_normalises_to_nfc(tokenizer):
    assert tokenizer.encode("café") == tokenizer.encode("café")


def test_tokenizer_truncates_from_the_end(tokenizer):
    ids = tokenizer.encode("word " * 5000)
    assert len(ids) == 4096
    assert ids[:3] == [248045, 846, 198]


def test_tokenizer_call_returns_a_single_unpadded_row(tokenizer):
    ids = tokenizer(["A cat"], device="cpu")
    assert ids.dtype == torch.long
    assert ids.shape == (1, 12)
    with pytest.raises(ValueError):
        tokenizer(["a", "b"])
