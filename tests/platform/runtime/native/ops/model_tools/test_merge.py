from __future__ import annotations

import weakref

import pytest
import torch

from src.platform.runtime.native.lora.key_mapping import LoraDelta
from src.platform.runtime.native.ops.model_tools.delta import compute_delta
from src.platform.runtime.native.ops.model_tools.errors import MergeError
from src.platform.runtime.native.ops.model_tools.lora_io import LoraInput
from src.platform.runtime.native.ops.model_tools.merge import detect_lora_family, kron_to_lora, merge_loras

from .lora_fixtures import CFG, DIM, drain, lokr, lora_pair, save_lora, save_model, build_moss

FLUX_A = "lora_unet_double_blocks_0_img_attn_qkv"
FLUX_B = "lora_unet_single_blocks_1_linear1"


def _recon(out, key):
    up = out[f"{key}.lora_up.weight"].float()
    down = out[f"{key}.lora_down.weight"].float()
    alpha = float(out[f"{key}.alpha"])
    return up @ down * (alpha / down.shape[0])


def _dense(sd, stem, strength):
    up, down = sd[f"{stem}.lora_up.weight"], sd[f"{stem}.lora_down.weight"]
    alpha = float(sd[f"{stem}.alpha"]) if f"{stem}.alpha" in sd else float(down.shape[0])
    return strength * alpha / down.shape[0] * (up @ down)


@pytest.fixture
def two_flux(tmp_path):
    a = {**lora_pair(FLUX_A, 24, 8, 4, 1, alpha=2), **lora_pair(FLUX_B, 12, 8, 3, 2, alpha=3)}
    b = {**lora_pair(FLUX_A, 24, 8, 3, 3, alpha=6)}
    return a, b, save_lora(tmp_path / "a.safetensors", a), save_lora(tmp_path / "b.safetensors", b)


def test_concat_reconstructs_the_summed_delta_exactly(two_flux):
    a, b, pa, pb = two_flux
    plan, tensors, report = merge_loras([LoraInput(pa, 0.7), LoraInput(pb, -0.4)], "concat")
    out = drain(plan, tensors)
    expected = _dense(a, FLUX_A, 0.7) + _dense(b, FLUX_A, -0.4)
    assert torch.allclose(_recon(out, FLUX_A), expected, atol=1e-6)
    assert torch.allclose(_recon(out, FLUX_B), _dense(a, FLUX_B, 0.7), atol=1e-6)
    assert out[f"{FLUX_A}.lora_up.weight"].shape == (24, 7)
    assert float(out[f"{FLUX_A}.alpha"]) == 7.0
    assert report.rank == 7 and report.family == "flux" and report.mode == "concat"
    assert {s.key for s in plan} == set(out)


def test_concat_folds_strength_into_up_not_down(two_flux):
    a, b, pa, pb = two_flux
    plan, tensors, _ = merge_loras([LoraInput(pa, 2.0), LoraInput(pb, 1.0)], "concat")
    out = drain(plan, tensors)
    assert torch.equal(out[f"{FLUX_A}.lora_down.weight"][:4], a[f"{FLUX_A}.lora_down.weight"])
    scaled = a[f"{FLUX_A}.lora_up.weight"] * (2.0 * 2.0 / 4)
    assert torch.allclose(out[f"{FLUX_A}.lora_up.weight"][:, :4], scaled, atol=1e-7)


def test_output_uses_comfy_up_down_alpha_names(two_flux):
    _, _, pa, pb = two_flux
    plan, _, _ = merge_loras([LoraInput(pa, 1.0), LoraInput(pb, 1.0)], "concat")
    assert {s.key for s in plan} == {
        f"{stem}.{suffix}" for stem in (FLUX_A, FLUX_B) for suffix in ("lora_up.weight", "lora_down.weight", "alpha")
    }


def test_reduce_at_full_rank_is_lossless_and_reports_unit_energy(two_flux):
    a, b, pa, pb = two_flux
    plan, tensors, report = merge_loras([LoraInput(pa, 1.0), LoraInput(pb, 1.0)], "reduce", rank=16)
    out = drain(plan, tensors)
    assert torch.allclose(_recon(out, FLUX_A), _dense(a, FLUX_A, 1.0) + _dense(b, FLUX_A, 1.0), atol=1e-5)
    assert report.energy[FLUX_A] == pytest.approx(1.0)


def test_reduce_reports_the_energy_it_actually_kept(two_flux):
    a, b, pa, pb = two_flux
    plan, tensors, report = merge_loras([LoraInput(pa, 1.0), LoraInput(pb, 1.0)], "reduce", rank=2, min_energy=0.0)
    out = drain(plan, tensors)
    full = _dense(a, FLUX_A, 1.0) + _dense(b, FLUX_A, 1.0)
    approx = _recon(out, FLUX_A)
    assert out[f"{FLUX_A}.lora_up.weight"].shape[1] == 2
    s = torch.linalg.svdvals(full)
    assert report.energy[FLUX_A] == pytest.approx(float((s[:2] ** 2).sum() / (s ** 2).sum()), rel=1e-4)
    assert float((approx ** 2).sum() / (full ** 2).sum()) == pytest.approx(report.energy[FLUX_A], rel=1e-4)
    assert report.energy[FLUX_A] < 1.0


def test_reduce_refuses_below_min_energy(two_flux):
    _, _, pa, pb = two_flux
    with pytest.raises(MergeError, match="energy"):
        merge_loras([LoraInput(pa, 1.0), LoraInput(pb, 1.0)], "reduce", rank=1, min_energy=0.999)


def test_reduce_needs_a_rank(two_flux):
    _, _, pa, _ = two_flux
    with pytest.raises(MergeError, match="rank"):
        merge_loras([LoraInput(pa, 1.0)], "reduce")


def test_kron_to_lora_energy_matches_reconstruction():
    gen = torch.Generator().manual_seed(5)
    w1, w2 = torch.randn(3, 4, generator=gen), torch.randn(5, 2, generator=gen)
    full = torch.kron(w1, w2) * 0.5
    up, down, energy = kron_to_lora(w1, w2, 0.5, 4)
    approx = up @ down
    assert energy == pytest.approx(float((approx ** 2).sum() / (full ** 2).sum()), rel=1e-4)
    assert energy < 1.0
    up, down, energy = kron_to_lora(w1, w2, 0.5, 8)
    assert energy == pytest.approx(1.0) and torch.allclose(up @ down, full, atol=1e-5)


def test_kron_order_is_w1_outer_w2_inner():
    w1 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    w2 = torch.tensor([[1.0, -1.0, 0.5, 2.0]])
    up, down, _ = kron_to_lora(w1, w2, 1.0, 2)
    assert torch.allclose(up @ down, torch.kron(w1, w2), atol=1e-5)
    assert not torch.allclose(up @ down, torch.kron(w2, w1), atol=1e-3)


def test_lokr_merges_by_svd_with_default_rank_of_largest_lora(tmp_path, two_flux):
    a, _, pa, _ = two_flux
    k = lokr(FLUX_A, (4, 2), (6, 4), 9, alpha=2.0, factor_w2=2)
    pk = save_lora(tmp_path / "k.safetensors", k)
    plan, tensors, report = merge_loras([LoraInput(pa, 1.0), LoraInput(pk, 1.0)], "concat", min_energy=0.0)
    out = drain(plan, tensors)
    assert any("rank of the largest LoRA" in w for w in report.warnings)
    assert out[f"{FLUX_A}.lora_up.weight"].shape[1] == 4 + 4
    assert report.lokr_energy
    lokr_delta = LoraDelta(down=k[f"{FLUX_A}.lokr_w2_a"] @ k[f"{FLUX_A}.lokr_w2_b"], up=k[f"{FLUX_A}.lokr_w1"], alpha=2.0 / 2, kron=True)
    dense = compute_delta([lokr_delta], (24, 8))
    s = torch.linalg.svdvals(dense)
    got = next(iter(report.lokr_energy.values()))
    assert got <= 1.0 and got >= float((s[:4] ** 2).sum() / (s ** 2).sum()) - 1e-4


def test_lokr_only_needs_explicit_rank_and_honours_it(tmp_path):
    k = lokr(FLUX_A, (4, 2), (6, 4), 9, alpha=2.0, factor_w2=2)
    pk = save_lora(tmp_path / "k.safetensors", k)
    with pytest.raises(MergeError, match="explicit rank"):
        merge_loras([LoraInput(pk, 1.0)], "concat")
    plan, tensors, _ = merge_loras([LoraInput(pk, 1.0)], "concat", rank=8, min_energy=0.99)
    out = drain(plan, tensors)
    lokr_delta = LoraDelta(down=k[f"{FLUX_A}.lokr_w2_a"] @ k[f"{FLUX_A}.lokr_w2_b"], up=k[f"{FLUX_A}.lokr_w1"], alpha=1.0, kron=True)
    assert torch.allclose(_recon(out, FLUX_A), compute_delta([lokr_delta], (24, 8)), atol=1e-4)


def test_mixed_families_are_refused(tmp_path, two_flux):
    _, _, pa, _ = two_flux
    wan = save_lora(tmp_path / "wan.safetensors", lora_pair("diffusion_model.blocks.0.self_attn.q", 8, 8, 2, 1, alpha=2))
    with pytest.raises(MergeError, match="different model families"):
        merge_loras([LoraInput(pa, 1.0), LoraInput(wan, 1.0)], "concat")
    with pytest.raises(MergeError, match="requested"):
        merge_loras([LoraInput(wan, 1.0)], "concat", family="flux")


@pytest.mark.parametrize("stems,family", [
    (["lora_unet_double_blocks_3_img_mlp_0", "diffusion_model.single_blocks.2.linear2", "transformer.single_transformer_blocks.0.attn.to_q"], "flux"),
    (["diffusion_model.blocks.4.attn.wq", "transformer.transformer_blocks.2.attn.to_gate"], "krea2"),
    (["diffusion_model.transformer_blocks.1.img_mlp.net.0.proj", "diffusion_model.transformer_blocks.1.txt_mod.1"], "qwen"),
    (["diffusion_model.blocks.0.self_attn.q", "diffusion_model.blocks.5.cross_attn.o", "diffusion_model.blocks.5.ffn.0"], "wan"),
    (["diffusion_model.layers.0.attention.qkv", "diffusion_model.noise_refiner.1.feed_forward.w1"], "z_image"),
    (["diffusion_model.layers.3.proj.q", "diffusion_model.layers.3.proj.ffn_up_gate"], "moss"),
    (["diffusion_model.transformer_blocks.0.attn1.to_q", "diffusion_model.transformer_blocks.7.audio_attn1.to_k"], "ltx"),
])
def test_family_detection_from_real_key_names(stems, family):
    sd = {}
    for i, stem in enumerate(stems):
        sd.update(lora_pair(stem, 4, 4, 1, i))
    assert detect_lora_family(sd) == family


def test_unknown_layout_is_refused(tmp_path):
    path = save_lora(tmp_path / "u.safetensors", lora_pair("some.weird.layer", 4, 4, 1, 1))
    with pytest.raises(MergeError, match="cannot identify"):
        merge_loras([LoraInput(path, 1.0)], "concat")


@pytest.mark.parametrize("extra,match", [
    ({f"{FLUX_A}.dora_scale": torch.ones(24)}, "DoRA"),
    ({f"{FLUX_A}.hada_w1_a": torch.ones(2, 2)}, "LoHa"),
    ({"lora_unet_conv.lora_up.weight": torch.ones(4, 2, 1, 1), "lora_unet_conv.lora_down.weight": torch.ones(2, 4, 3, 3)}, "convolution"),
    ({"lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight": torch.ones(4, 2), "lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight": torch.ones(2, 4)}, "text-encoder"),
])
def test_unsupported_adapter_types_are_refused_naming_the_file(tmp_path, extra, match):
    sd = {**lora_pair(FLUX_A, 24, 8, 2, 1, alpha=2), **extra}
    path = save_lora(tmp_path / "bad_one.safetensors", sd)
    with pytest.raises(MergeError, match=match) as info:
        merge_loras([LoraInput(path, 1.0)], "concat")
    assert "bad_one.safetensors" in str(info.value)


def test_same_layer_with_two_spellings_needs_a_model(tmp_path):
    a = save_lora(tmp_path / "a.safetensors", lora_pair("lora_unet_double_blocks_0_img_attn_qkv", 24, 8, 2, 1, alpha=2))
    b = save_lora(tmp_path / "b.safetensors", lora_pair("diffusion_model.double_blocks.0.img_attn.qkv", 24, 8, 2, 2, alpha=2))
    with pytest.raises(MergeError, match="model_path"):
        merge_loras([LoraInput(a, 1.0), LoraInput(b, 1.0)], "concat")


def test_shape_conflict_on_one_layer_is_refused(tmp_path):
    a = save_lora(tmp_path / "a.safetensors", lora_pair(FLUX_A, 24, 8, 2, 1, alpha=2))
    b = save_lora(tmp_path / "b.safetensors", lora_pair(FLUX_A, 24, 9, 2, 2, alpha=2))
    with pytest.raises(MergeError, match="shape mismatch"):
        merge_loras([LoraInput(a, 1.0), LoraInput(b, 1.0)], "concat")


def test_model_path_unifies_dialects_and_pads_fused_qkv(tmp_path):
    module = build_moss()
    model = save_model(tmp_path / "m.safetensors", {k: v.detach().clone() for k, v in module.state_dict().items()})
    a = {}
    for part, seed in (("q", 1), ("k", 2)):
        a.update(lora_pair(f"diffusion_model.layers.0.proj.{part}", DIM, DIM, 2, seed, alpha=2))
    b = lora_pair("lora_unet_layers_0_proj_q", DIM, DIM, 3, 7, alpha=3)
    pa = save_lora(tmp_path / "a.safetensors", a)
    pb = save_lora(tmp_path / "b.safetensors", b)
    plan, tensors, report = merge_loras([LoraInput(pa, 1.0), LoraInput(pb, 0.5)], "concat", model_path=model, family="moss")
    out = drain(plan, tensors)
    key = "diffusion_model.layers.0.proj.qkv"
    assert set(out) == {f"{key}.lora_up.weight", f"{key}.lora_down.weight", f"{key}.alpha"}
    expected = torch.zeros(3 * DIM, DIM)
    expected[:DIM] += _dense(a, "diffusion_model.layers.0.proj.q", 1.0) + _dense(b, "lora_unet_layers_0_proj_q", 0.5)
    expected[DIM:2 * DIM] += _dense(a, "diffusion_model.layers.0.proj.k", 1.0)
    assert torch.allclose(_recon(out, key), expected, atol=1e-6)
    assert report.family == "moss"


def test_model_path_refuses_a_lora_for_another_family(tmp_path):
    module = build_moss()
    model = save_model(tmp_path / "m.safetensors", {k: v.detach().clone() for k, v in module.state_dict().items()})
    wan = save_lora(tmp_path / "wan.safetensors", lora_pair("diffusion_model.blocks.0.self_attn.q", 8, 8, 2, 1, alpha=2))
    with pytest.raises(MergeError, match="do not match"):
        merge_loras([LoraInput(wan, 1.0)], "concat", model_path=model)


def test_merge_streams_one_stem_at_a_time(tmp_path):
    stems = [f"lora_unet_double_blocks_{i}_img_attn_qkv" for i in range(10)]
    a, b = {}, {}
    for i, stem in enumerate(stems):
        a.update(lora_pair(stem, 24, 8, 3, i, alpha=3))
        b.update(lora_pair(stem, 24, 8, 2, 100 + i, alpha=2))
    pa, pb = save_lora(tmp_path / "a.safetensors", a), save_lora(tmp_path / "b.safetensors", b)
    for mode, rank in (("concat", None), ("reduce", 4)):
        plan, tensors, _ = merge_loras([LoraInput(pa, 1.0), LoraInput(pb, 1.0)], mode, rank=rank, min_energy=0.0)
        refs, live, seen = [], [], []
        for key, tensor in tensors:
            live.append(sum(1 for r in refs if r() is not None))
            refs.append(weakref.ref(tensor))
            seen.append(key)
        assert seen == [s.key for s in plan]
        assert len(refs) == 30
        assert max(live) <= 2


def test_reduce_energy_failure_happens_before_any_tensor_is_yielded(two_flux):
    _, _, pa, pb = two_flux
    with pytest.raises(MergeError, match="energy"):
        merge_loras([LoraInput(pa, 1.0), LoraInput(pb, 1.0)], "reduce", rank=1, min_energy=0.999)


def test_merge_plan_matches_streamed_shapes_and_dtypes(two_flux):
    _, _, pa, pb = two_flux
    for mode, rank in (("concat", None), ("reduce", 3), ("reduce", 99)):
        plan, tensors, _ = merge_loras([LoraInput(pa, 1.0), LoraInput(pb, 1.0)], mode, rank=rank, min_energy=0.0, dtype="BF16")
        for spec, (key, tensor) in zip(plan, tensors):
            assert spec.key == key and tuple(tensor.shape) == spec.shape
            assert spec.dtype == ("F32" if key.endswith(".alpha") else "BF16")
