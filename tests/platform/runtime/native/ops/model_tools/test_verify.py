from __future__ import annotations

import torch
from safetensors import safe_open
from safetensors.torch import save_file

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
