from __future__ import annotations

import hashlib
import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from src.platform.runtime.native.detect.registry import arch_registry
from src.platform.runtime.native.ops.model_tools import families
from src.platform.runtime.native.ops.model_tools.errors import (
    AlreadyQuantizedError,
    StreamMismatchError,
    UnsupportedFamilyError,
    WriteCancelledError,
)
from src.platform.runtime.native.ops.model_tools.formats import get_format
from src.platform.runtime.native.ops.model_tools.quantize import quantize_file, quantize_stream
from src.platform.runtime.native.ops.model_tools.source import read_stream
from src.platform.runtime.native.kernels import detect_quant_format, fp8_ops

from .fixtures import FAMILY_LAYOUTS, make_tensors, write_family_fixture

FAMILIES = sorted(FAMILY_LAYOUTS)
DESCRIPTOR_BYTES = get_format("fp8_e4m3").descriptor_bytes


def load_all(path):
    out = {}
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out, meta


def raw(t):
    return t.contiguous().reshape(-1).view(torch.uint8)


@pytest.mark.parametrize("family", FAMILIES)
def test_keep_list_per_family(tmp_path, family):
    src = tmp_path / "src.safetensors"
    quant_keys, kept_keys = write_family_fixture(src, family)
    plan, tensors, _ = read_stream(src)
    new_plan, _ = quantize_stream(plan, tensors, family)
    by_key = {s.key: s for s in new_plan}
    for key in quant_keys:
        base = key[: -len(".weight")]
        assert by_key[key].dtype == "F8_E4M3"
        assert by_key[base + ".weight_scale"].dtype == "F32" and by_key[base + ".weight_scale"].shape == ()
        assert by_key[base + ".comfy_quant"].dtype == "U8"
    for key in kept_keys:
        assert by_key[key].dtype == "BF16", key
        assert key[: -len(".weight")] + ".weight_scale" not in by_key
    assert len(new_plan) == len(kept_keys) + 3 * len(quant_keys)


@pytest.mark.parametrize("family", FAMILIES)
def test_roundtrip_detect_dequant_and_byte_identical_kept(tmp_path, family):
    src = tmp_path / "src.safetensors"
    dst = tmp_path / "dst.safetensors"
    quant_keys, kept_keys = write_family_fixture(src, family, metadata={"format": "pt", "origin": "fixture"})
    original, _ = load_all(src)
    result = quantize_file(src, dst, family=family)
    assert result.tensor_count == len(kept_keys) + 3 * len(quant_keys)
    out, meta = load_all(dst)
    assert detect_quant_format(meta, out) == "fp8_scaled"
    for key in kept_keys:
        assert out[key].dtype == original[key].dtype
        assert torch.equal(raw(out[key]), raw(original[key]))
    for key in quant_keys:
        base = key[: -len(".weight")]
        w = original[key].float()
        scale = out[base + ".weight_scale"]
        assert scale.dtype == torch.float32 and scale.shape == ()
        assert scale.item() == pytest.approx(w.abs().max().item() / 448.0, rel=1e-6)
        assert out[key].dtype == torch.float8_e4m3fn
        restored = out[key].float() * scale
        assert (restored - w).norm() / w.norm() < 0.08
        assert bytes(out[base + ".comfy_quant"].tolist()) == DESCRIPTOR_BYTES
    assert json.loads(DESCRIPTOR_BYTES) == {"format": "float8_e4m3fn", "full_precision_matrix_mult": True}
    assert "_quantization_metadata" not in meta
    assert not any(k.endswith((".scale_weight", ".input_scale")) or k == "scaled_fp8" for k in out)


def test_metadata_records_source_hash_and_recipe(tmp_path):
    src = tmp_path / "src.safetensors"
    dst = tmp_path / "dst.safetensors"
    quant_keys, _ = write_family_fixture(src, "flux", metadata={"format": "pt", "origin": "fixture"})
    quantize_file(src, dst, family="flux", keep_extra=["single_blocks.*"])
    _, meta = load_all(dst)
    assert meta["origin"] == "fixture"
    assert meta["potionui.quantized_from"] == hashlib.sha256(src.read_bytes()).hexdigest()
    recipe = json.loads(meta["potionui.recipe"])
    assert recipe["family"] == "flux"
    assert recipe["keep_extra"] == ["single_blocks.*"]
    assert recipe["quantized_layers"] == len(quant_keys) - 1


@pytest.mark.parametrize(
    "key, comfy_format, scaling",
    [
        ("fp8_e5m2", "float8_e5m2", "per_tensor_absmax"),
        ("int8", "int8_tensorwise", "per_channel_absmax"),
    ],
)
def test_metadata_recipe_names_the_chosen_format(tmp_path, key, comfy_format, scaling):
    src = tmp_path / "src.safetensors"
    dst = tmp_path / "dst.safetensors"
    quant_keys, _ = write_family_fixture(src, "wan")
    quantize_file(src, dst, family="wan", quant_format=key)
    out, meta = load_all(dst)
    recipe = json.loads(meta["potionui.recipe"])
    assert (recipe["format"], recipe["scaling"], recipe["quantized_layers"]) == (comfy_format, scaling, len(quant_keys))
    assert json.loads(bytes(out[quant_keys[0][: -len(".weight")] + ".comfy_quant"].tolist()))["format"] == comfy_format


def test_keep_extra_keeps_matching_layers(tmp_path):
    src = tmp_path / "src.safetensors"
    write_family_fixture(src, "wan")
    plan, tensors, _ = read_stream(src)
    new_plan, _ = quantize_stream(plan, tensors, "wan", keep_extra=["blocks.0.ffn.*"])
    by_key = {s.key: s for s in new_plan}
    assert by_key["blocks.0.ffn.0.weight"].dtype == "BF16"
    assert by_key["blocks.0.self_attn.q.weight"].dtype == "F8_E4M3"


def test_prefixed_keys_use_same_keep_rules(tmp_path):
    layout_q, layout_k = FAMILY_LAYOUTS["flux"]
    tensors = make_tensors({f"model.diffusion_model.{k}": v for k, v in {**layout_q, **layout_k}.items()})
    src = tmp_path / "p.safetensors"
    save_file(tensors, str(src))
    plan, it, _ = read_stream(src)
    new_plan, _ = quantize_stream(plan, it, "flux")
    by_key = {s.key: s for s in new_plan}
    assert by_key["model.diffusion_model.img_in.weight"].dtype == "BF16"
    assert by_key["model.diffusion_model.double_blocks.0.img_attn.qkv.weight"].dtype == "F8_E4M3"


@pytest.mark.parametrize("family", ["sdxl", "seedvr2", "nonsense"])
def test_refuses_unsupported_families(tmp_path, family):
    src = tmp_path / "src.safetensors"
    write_family_fixture(src, "flux")
    plan, tensors, _ = read_stream(src)
    with pytest.raises(UnsupportedFamilyError):
        quantize_stream(plan, tensors, family)


def test_sdxl_checkpoint_is_refused_with_clear_message(tmp_path):
    src = tmp_path / "sdxl.safetensors"
    save_file(
        make_tensors(
            {
                "model.diffusion_model.input_blocks.0.0.weight": (4, 4, 3, 3),
                "model.diffusion_model.input_blocks.1.1.proj_in.weight": (4, 4),
                "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight": (8, 4),
            }
        ),
        str(src),
    )
    with pytest.raises(UnsupportedFamilyError, match="SDXL"):
        quantize_file(src, tmp_path / "o.safetensors")
    assert not (tmp_path / "o.safetensors").exists()


def test_refuses_already_quantized_source(tmp_path):
    src = tmp_path / "src.safetensors"
    dst = tmp_path / "dst.safetensors"
    write_family_fixture(src, "flux")
    quantize_file(src, dst, family="flux")
    plan, tensors, _ = read_stream(dst)
    with pytest.raises(AlreadyQuantizedError):
        quantize_stream(plan, tensors, "flux")


def test_refuses_destination_equal_to_source(tmp_path):
    src = tmp_path / "src.safetensors"
    write_family_fixture(src, "flux")
    with pytest.raises(StreamMismatchError):
        quantize_file(src, src, family="flux")


def test_cancel_leaves_no_files(tmp_path):
    src = tmp_path / "src.safetensors"
    write_family_fixture(src, "flux")
    with pytest.raises(WriteCancelledError):
        quantize_file(src, tmp_path / "dst.safetensors", family="flux", is_cancelled=lambda: True)
    assert [p.name for p in tmp_path.iterdir()] == ["src.safetensors"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fp16_and_fp32_sources_are_quantized(tmp_path, dtype):
    src = tmp_path / "s.safetensors"
    quant_keys, _ = write_family_fixture(src, "krea2", dtype=dtype)
    plan, tensors, _ = read_stream(src)
    new_plan, _ = quantize_stream(plan, tensors, "krea2")
    assert {s.key: s for s in new_plan}[quant_keys[0]].dtype == "F8_E4M3"


def test_stream_order_matches_plan(tmp_path):
    src = tmp_path / "src.safetensors"
    write_family_fixture(src, "qwen_image")
    plan, tensors, _ = read_stream(src)
    new_plan, new_tensors = quantize_stream(plan, tensors, "qwen_image")
    for spec, (key, tensor) in zip(new_plan, new_tensors, strict=True):
        assert key == spec.key
        assert tuple(tensor.shape) == spec.shape


def test_zero_weight_dequantizes_to_zero(tmp_path):
    src = tmp_path / "z.safetensors"
    save_file({"blocks.0.attn.q.weight": torch.zeros(4, 4, dtype=torch.bfloat16)}, str(src))
    plan, tensors, _ = read_stream(src)
    _, new_tensors = quantize_stream(plan, tensors, "krea2")
    out = dict(new_tensors)
    restored = out["blocks.0.attn.q.weight"].float() * out["blocks.0.attn.q.weight_scale"]
    assert torch.count_nonzero(restored) == 0


def test_loader_consumes_quantized_linear(tmp_path):
    src = tmp_path / "src.safetensors"
    dst = tmp_path / "dst.safetensors"
    write_family_fixture(src, "krea2")
    original, _ = load_all(src)
    quantize_file(src, dst, family="krea2")
    out, _ = load_all(dst)
    key = "blocks.0.attn.q"
    lin = fp8_ops.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    lin.load_state_dict({n: out[f"{key}.{n}"] for n in ("weight", "weight_scale", "comfy_quant")}, strict=False)
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    ref = x.float() @ original[f"{key}.weight"].float().T
    assert (lin(x).float() - ref).norm() / ref.norm() < 0.08


def test_unet_families_exist_in_registry():
    registered = {s.family for s in arch_registry.all()}
    assert set(families.UNET_FAMILY_KEEP) <= registered
