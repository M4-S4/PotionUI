from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from src.platform.runtime.native.arch.moss_image.model import MossImage
from src.platform.runtime.native.lora import apply_loras
from src.platform.runtime.native.ops.model_tools.bake import bake_stream
from src.platform.runtime.native.ops.model_tools.errors import UnsupportedFamilyError
from src.platform.runtime.native.ops.model_tools.errors import BakeError
from src.platform.runtime.native.ops.model_tools.lora_io import LoraInput

from .lora_fixtures import (
    CFG,
    DIM,
    build_moss,
    cpu_only,
    descriptor_bytes,
    drain,
    lokr,
    lora_pair,
    save_lora,
    save_model,
    split_qkv,
)


def _runtime_applied(sd, loras, dtype=torch.float32):
    from src.platform.runtime.native.arch.moss_image.model import convert_moss_image_state_dict
    from src.platform.runtime.native.kernels import pick_operations

    module = MossImage(CFG, operations=pick_operations(torch.float32, torch.float32))
    module = module.to(dtype)
    module.load_state_dict(convert_moss_image_state_dict(dict(sd)), strict=False)
    apply_loras(module, loras)
    return module.state_dict()


def _fuse(out):
    fused = dict(out)
    for key in [k for k in out if k.endswith(".proj.q.weight")]:
        base = key[: -len("q.weight")]
        fused[base + "qkv.weight"] = torch.cat([fused.pop(base + n + ".weight") for n in "qkv"], dim=0)
    return fused


@pytest.fixture
def source(tmp_path):
    module = build_moss()
    sd = {k: v.detach().clone() for k, v in module.state_dict().items()}
    return sd, save_model(tmp_path / "model.safetensors", sd)


def _lora_sd():
    sd = {}
    sd.update(lora_pair("diffusion_model.layers.0.proj.o", DIM, DIM, 3, 1, alpha=6))
    sd.update(lora_pair("diffusion_model.layers.1.proj.ffn_down", DIM, CFG.ffn_dim, 2, 2, alpha=2))
    for part, seed in (("q", 3), ("k", 4), ("v", 5)):
        sd.update(lora_pair(f"diffusion_model.layers.0.proj.{part}", DIM, DIM, 2, seed, alpha=2))
    return sd


def test_baked_lora_equals_runtime_applied_on_fused_source(source, tmp_path, cpu_only):
    sd, path = source
    lora = _lora_sd()
    lora_path = save_lora(tmp_path / "l.safetensors", lora)
    plan, tensors, report = bake_stream(path, [LoraInput(lora_path, 0.7)], "moss")
    baked = drain(plan, tensors)
    expected = _runtime_applied(sd, [(lora, 0.7)])
    for key in sd:
        if key in expected:
            assert torch.equal(baked[key], expected[key]), key
    assert report.family == "moss_image"
    assert report.output_dtype == "F32"
    assert not report.dequantized_fp8
    assert set(report.keys_touched[lora_path]) == {
        "layers.0.proj.o.weight", "layers.1.proj.ffn_down.weight", "layers.0.proj.qkv.weight"
    }


def test_baked_into_split_qkv_source_matches_fused_runtime(source, tmp_path, cpu_only):
    sd, _ = source
    split_path = save_model(tmp_path / "split.safetensors", split_qkv(sd))
    lora = _lora_sd()
    lora_path = save_lora(tmp_path / "l.safetensors", lora)
    plan, tensors, report = bake_stream(split_path, [LoraInput(lora_path, 1.0)], "moss")
    baked = _fuse(drain(plan, tensors))
    expected = _runtime_applied(sd, [(lora, 1.0)])
    for key, value in expected.items():
        if key in baked:
            assert torch.equal(baked[key], value), key
    assert {"layers.0.proj.q.weight", "layers.0.proj.k.weight", "layers.0.proj.v.weight"} <= set(report.keys_touched[lora_path])


def test_baked_lokr_equals_runtime_applied(source, tmp_path, cpu_only):
    sd, path = source
    lora = {}
    lora.update(lokr("diffusion_model.layers.1.proj.o", (4, 2), (4, 8), 7, alpha=2.0, factor_w2=2))
    lora.update(lokr("diffusion_model.layers.0.proj.ffn_down", (2, 3), (8, 8), 8))
    lora_path = save_lora(tmp_path / "k.safetensors", lora)
    plan, tensors, _ = bake_stream(path, [LoraInput(lora_path, 0.5)], "moss")
    baked = drain(plan, tensors)
    expected = _runtime_applied(sd, [(lora, 0.5)])
    changed = [k for k in sd if not torch.equal(sd[k], expected[k])]
    assert set(changed) == {"layers.0.proj.ffn_down.weight", "layers.1.proj.o.weight"}
    for key in changed:
        assert torch.equal(baked[key], expected[key]), key


def test_bf16_source_bakes_within_one_ulp_of_runtime(tmp_path, cpu_only):
    module = build_moss(torch.bfloat16)
    sd = {k: v.detach().clone() for k, v in module.state_dict().items()}
    path = save_model(tmp_path / "bf16.safetensors", sd)
    lora = _lora_sd()
    lora_path = save_lora(tmp_path / "l.safetensors", lora)
    plan, tensors, report = bake_stream(path, [LoraInput(lora_path, 0.8)], "moss")
    baked = drain(plan, tensors)
    expected = _runtime_applied(sd, [(lora, 0.8)], torch.bfloat16)
    assert report.output_dtype == "BF16"
    changed = [k for k in sd if not torch.equal(sd[k], expected[k])]
    assert changed
    for key in changed:
        assert baked[key].dtype == torch.bfloat16
        bound = 2 ** -7 * torch.maximum(sd[key].float().abs(), expected[key].float().abs())
        assert ((baked[key].float() - expected[key].float()).abs() <= bound).all(), key
    for key in sd:
        if key not in changed:
            assert torch.equal(baked[key], sd[key]), key


def test_allow_unmatched_bakes_the_valid_layer_and_reports_the_rest(source, tmp_path, cpu_only):
    sd, path = source
    lora = lora_pair("diffusion_model.layers.0.proj.o", DIM, DIM, 2, 1, alpha=2)
    lora.update(lora_pair("diffusion_model.layers.9.proj.o", DIM, DIM, 2, 2, alpha=2))
    lora_path = save_lora(tmp_path / "l.safetensors", lora)
    plan, tensors, report = bake_stream(path, [LoraInput(lora_path, 1.0)], "moss", allow_unmatched=True)
    baked = drain(plan, tensors)
    expected = _runtime_applied(sd, [(lora, 1.0)])
    assert torch.equal(baked["layers.0.proj.o.weight"], expected["layers.0.proj.o.weight"])
    assert not torch.equal(baked["layers.0.proj.o.weight"], sd["layers.0.proj.o.weight"])
    assert report.unmatched[lora_path] == ["diffusion_model.layers.9.proj.o"]
    assert report.keys_touched[lora_path] == ["layers.0.proj.o.weight"]


def test_untouched_tensors_pass_through_byte_identical(source, tmp_path):
    sd, path = source
    lora_path = save_lora(tmp_path / "l.safetensors", lora_pair("diffusion_model.layers.0.proj.o", DIM, DIM, 2, 1, alpha=2))
    plan, tensors, _ = bake_stream(path, [LoraInput(lora_path, 1.0)], "moss")
    baked = drain(plan, tensors)
    assert {s.key for s in plan} == set(sd)
    for key, value in sd.items():
        if key == "layers.0.proj.o.weight":
            assert not torch.equal(baked[key], value)
        else:
            assert baked[key].view(torch.uint8).tolist() == value.view(torch.uint8).tolist(), key


def _fp8_source(tmp_path, sd):
    out = {}
    scales = {}
    for key, value in sd.items():
        if key.startswith("layers.") and key.endswith((".proj.o.weight", ".proj.ffn_down.weight")):
            scale = value.abs().max() / 400.0
            out[key] = (value / scale).to(torch.float8_e4m3fn)
            stem = key[: -len(".weight")]
            out[stem + ".weight_scale"] = scale.reshape(())
            out[stem + ".comfy_quant"] = descriptor_bytes()
            scales[key] = scale
        else:
            out[key] = value
    return save_model(tmp_path / "fp8.safetensors", out, {"_quantization_metadata": "{}"}), out, scales


def test_fp8_source_is_dequantized_to_bf16_and_baked(source, tmp_path):
    sd, _ = source
    path, packed, scales = _fp8_source(tmp_path, sd)
    lora = lora_pair("diffusion_model.layers.0.proj.o", DIM, DIM, 2, 1, alpha=2)
    lora_path = save_lora(tmp_path / "l.safetensors", lora)
    plan, tensors, report = bake_stream(path, [LoraInput(lora_path, 1.0)], "moss")
    baked = drain(plan, tensors)
    assert report.dequantized_fp8 and report.output_dtype == "BF16"
    assert not any(k.endswith((".weight_scale", ".comfy_quant")) for k in baked)
    assert "_quantization_metadata" not in report.metadata
    key = "layers.0.proj.o.weight"
    delta = lora["diffusion_model.layers.0.proj.o.lora_up.weight"] @ lora["diffusion_model.layers.0.proj.o.lora_down.weight"]
    expected = (packed[key].float() * scales[key] + delta).to(torch.bfloat16)
    assert baked[key].dtype == torch.bfloat16 and torch.equal(baked[key], expected)
    other = "layers.1.proj.ffn_down.weight"
    assert torch.equal(baked[other], (packed[other].float() * scales[other]).to(torch.bfloat16))
    spec = {s.key: s for s in plan}
    assert spec[key].dtype == "BF16" and spec[other].dtype == "BF16"


def test_unknown_layer_is_refused_naming_the_key(source, tmp_path):
    _, path = source
    lora_path = save_lora(tmp_path / "l.safetensors", lora_pair("diffusion_model.layers.9.proj.o", DIM, DIM, 2, 1, alpha=2))
    with pytest.raises(BakeError, match="layers.9.proj.o"):
        bake_stream(path, [LoraInput(lora_path, 1.0)], "moss")


def test_missing_source_key_is_refused(source, tmp_path):
    sd, _ = source
    trimmed = {k: v for k, v in sd.items() if k != "layers.1.proj.o.weight"}
    path = save_model(tmp_path / "trim.safetensors", trimmed)
    lora_path = save_lora(tmp_path / "l.safetensors", lora_pair("diffusion_model.layers.1.proj.o", DIM, DIM, 2, 1, alpha=2))
    with pytest.raises(BakeError, match="missing key.*layers.1.proj.o.weight"):
        bake_stream(path, [LoraInput(lora_path, 1.0)], "moss")


def test_shape_mismatch_is_refused(source, tmp_path):
    _, path = source
    wrong_in = save_lora(tmp_path / "in.safetensors", lora_pair("diffusion_model.layers.0.proj.o", DIM, 99, 2, 1, alpha=2))
    with pytest.raises(BakeError, match="in_features"):
        bake_stream(path, [LoraInput(wrong_in, 1.0)], "moss")
    wrong_out = save_lora(tmp_path / "out.safetensors", lora_pair("diffusion_model.layers.0.proj.o", 5, DIM, 2, 1, alpha=2))
    with pytest.raises(BakeError, match="shape mismatch"):
        bake_stream(path, [LoraInput(wrong_out, 1.0)], "moss")


def test_source_shape_mismatch_is_refused(source, tmp_path):
    sd, _ = source
    broken = dict(sd)
    broken["layers.0.proj.o.weight"] = torch.zeros(DIM, DIM + 1)
    path = save_model(tmp_path / "broken.safetensors", broken)
    lora_path = save_lora(tmp_path / "l.safetensors", lora_pair("diffusion_model.layers.0.proj.o", DIM, DIM, 2, 1, alpha=2))
    with pytest.raises(BakeError, match="shape mismatch"):
        bake_stream(path, [LoraInput(lora_path, 1.0)], "moss")


def test_text_encoder_keys_are_refused_with_a_message(source, tmp_path):
    _, path = source
    sd = lora_pair("diffusion_model.layers.0.proj.o", DIM, DIM, 2, 1, alpha=2)
    sd.update(lora_pair("lora_te_text_model_encoder_layers_0_self_attn_q_proj", 8, 8, 2, 2, alpha=2))
    lora_path = save_lora(tmp_path / "te.safetensors", sd)
    with pytest.raises(BakeError, match="text-encoder"):
        bake_stream(path, [LoraInput(lora_path, 1.0)], "moss")


@pytest.mark.parametrize("extra,match", [
    ({"diffusion_model.layers.0.proj.o.dora_scale": torch.ones(DIM)}, "DoRA"),
    ({"diffusion_model.layers.0.proj.o.hada_w1_a": torch.ones(2, 2)}, "LoHa"),
])
def test_dora_and_loha_are_refused(source, tmp_path, extra, match):
    _, path = source
    sd = lora_pair("diffusion_model.layers.0.proj.o", DIM, DIM, 2, 1, alpha=2)
    sd.update(extra)
    lora_path = save_lora(tmp_path / "x.safetensors", sd)
    with pytest.raises(BakeError, match=match):
        bake_stream(path, [LoraInput(lora_path, 1.0)], "moss")


def test_family_mismatch_and_unsupported_family_are_refused(source, tmp_path):
    _, path = source
    lora_path = save_lora(tmp_path / "l.safetensors", lora_pair("diffusion_model.layers.0.proj.o", DIM, DIM, 2, 1, alpha=2))
    with pytest.raises(BakeError, match="requested"):
        bake_stream(path, [LoraInput(lora_path, 1.0)], "flux")
    with pytest.raises(UnsupportedFamilyError):
        bake_stream(path, [LoraInput(lora_path, 1.0)], "sdxl")


def test_stacked_loras_sum_in_fp32(source, tmp_path):
    sd, path = source
    a = lora_pair("diffusion_model.layers.0.proj.o", DIM, DIM, 2, 1, alpha=2)
    b = lora_pair("diffusion_model.layers.0.proj.o", DIM, DIM, 3, 2, alpha=3)
    pa = save_lora(tmp_path / "a.safetensors", a)
    pb = save_lora(tmp_path / "b.safetensors", b)
    plan, tensors, report = bake_stream(path, [LoraInput(pa, 1.0), LoraInput(pb, -0.5)], "moss")
    baked = drain(plan, tensors)
    stem = "diffusion_model.layers.0.proj.o"
    delta = a[f"{stem}.lora_up.weight"] @ a[f"{stem}.lora_down.weight"] - 0.5 * (b[f"{stem}.lora_up.weight"] @ b[f"{stem}.lora_down.weight"])
    assert torch.allclose(baked["layers.0.proj.o.weight"], sd["layers.0.proj.o.weight"] + delta, atol=1e-6)
    assert set(report.keys_touched) == {pa, pb}
