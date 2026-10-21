from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from src.platform.runtime.native.ops.model_tools.errors import UnsupportedFormatError
from src.platform.runtime.native.ops.model_tools.formats import QUANT_FORMATS, get_format
from src.platform.runtime.native.ops.model_tools.quantize import quantize_file
from src.platform.runtime.native.ops.model_tools.verify import layer_metrics, verify_quantized

from .fixtures import write_family_fixture


def build(tmp_path, family="flux"):
    src = tmp_path / "src.safetensors"
    dst = tmp_path / "dst.safetensors"
    quant_keys, kept_keys = write_family_fixture(src, family)
    quantize_file(src, dst, family=family)
    return src, dst, quant_keys, kept_keys


def load(path):
    with safe_open(str(path), framework="pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def test_report_covers_every_quantized_layer(tmp_path):
    src, dst, quant_keys, kept_keys = build(tmp_path)
    report = verify_quantized(src, dst)
    assert report.ok
    assert sorted(l.key for l in report.layers) == sorted(quant_keys)
    assert report.kept_checked == len(kept_keys)
    assert 0 < report.mean_rel_l2 < 0.08
    assert report.min_cosine > 0.995
    assert report.min_snr_db > 20
    assert report.worst[0].rel_l2 == report.max_rel_l2
    assert len(report.worst) <= 5
    assert report.to_dict()["ok"] is True


def test_sample_limit_bounds_layers(tmp_path):
    src, dst, _, _ = build(tmp_path)
    report = verify_quantized(src, dst, sample_limit=2)
    assert len(report.layers) == 2
    assert report.sampled


def test_altered_kept_tensor_is_flagged(tmp_path):
    src, dst, _, kept_keys = build(tmp_path)
    tensors = load(dst)
    tensors[kept_keys[0]] = tensors[kept_keys[0]] + 1
    bad = tmp_path / "bad.safetensors"
    save_file(tensors, str(bad))
    report = verify_quantized(src, bad)
    assert kept_keys[0] in report.kept_mismatched
    assert not report.ok


def test_missing_tensor_is_flagged(tmp_path):
    src, dst, _, _ = build(tmp_path)
    tensors = {k: v for k, v in load(dst).items() if k != "img_in.weight"}
    bad = tmp_path / "bad.safetensors"
    save_file(tensors, str(bad))
    assert "img_in.weight" in verify_quantized(src, bad).missing


def test_corrupted_scale_degrades_metrics(tmp_path):
    src, dst, quant_keys, _ = build(tmp_path)
    tensors = load(dst)
    target = quant_keys[0]
    scale_key = target[: -len(".weight")] + ".weight_scale"
    tensors[scale_key] = tensors[scale_key] * 3
    bad = tmp_path / "bad.safetensors"
    save_file(tensors, str(bad))
    report = verify_quantized(src, bad)
    assert report.worst[0].key == target
    assert report.worst[0].rel_l2 > 1.0


def test_layer_metrics_identical_and_scaled():
    w = torch.randn(4, 4)
    same = layer_metrics("k", w, w.clone())
    assert same.rel_l2 == 0 and abs(same.cosine - 1.0) < 1e-6 and same.snr_db == 200.0
    half = layer_metrics("k", w, w * 0.5)
    assert abs(half.rel_l2 - 0.5) < 1e-5 and abs(half.cosine - 1.0) < 1e-5


WIDE_LAYERS = {"blocks.0.attn.q.weight": (32, 512), "blocks.0.mlp.fc1.weight": (64, 256)}
FORMAT_ERROR_BOUND = {
    "fp8_e4m3": 0.05, "fp8_e5m2": 0.1, "int8": 0.015, "int8_convrot": 0.015, "nvfp4": 0.13, "int4_convrot": 0.16,
}


def build_wide(tmp_path, key):
    gen = torch.Generator().manual_seed(2)
    src = tmp_path / "src.safetensors"
    dst = tmp_path / f"{key}.safetensors"
    tensors = {k: (torch.randn(shape, generator=gen) * 0.05).to(torch.bfloat16) for k, shape in WIDE_LAYERS.items()}
    tensors["img_in.weight"] = torch.randn(8, 16, generator=gen).to(torch.bfloat16)
    save_file(tensors, str(src))
    quantize_file(src, dst, family="krea2", quant_format=key)
    return src, dst


@pytest.mark.parametrize("key", list(QUANT_FORMATS))
def test_every_format_is_verified_through_its_own_dequantization(tmp_path, key):
    src, dst = build_wide(tmp_path, key)
    report = verify_quantized(src, dst)
    assert report.ok
    assert sorted(l.key for l in report.layers) == sorted(WIDE_LAYERS)
    assert 0 < report.max_rel_l2 < FORMAT_ERROR_BOUND[key]


def test_a_convrot_layer_read_without_its_rotation_fails_the_check(tmp_path):
    src, dst = build_wide(tmp_path, "int8")
    tensors = load(dst)
    convrot = get_format("int8_convrot").descriptor_bytes
    tensors["blocks.0.attn.q.comfy_quant"] = torch.tensor(list(convrot), dtype=torch.uint8)
    bad = tmp_path / "bad.safetensors"
    save_file(tensors, str(bad))
    report = verify_quantized(src, bad)
    assert report.worst[0].key == "blocks.0.attn.q.weight"
    assert report.worst[0].rel_l2 > 0.5


def test_a_descriptor_no_format_writes_is_refused(tmp_path):
    src, dst = build_wide(tmp_path, "int8")
    tensors = load(dst)
    tensors["blocks.0.attn.q.comfy_quant"] = torch.tensor(list(json.dumps({"format": "mxfp8"}).encode()), dtype=torch.uint8)
    bad = tmp_path / "bad.safetensors"
    save_file(tensors, str(bad))
    with pytest.raises(UnsupportedFormatError, match="mxfp8"):
        verify_quantized(src, bad)
