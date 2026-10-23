from __future__ import annotations

import json
import os
import shutil
from collections import namedtuple
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from src.pipelines.outputs import Progress, ProgressGenerationOutput
from src.pipelines.pipes.model_tools import placement
from src.pipelines.pipes.model_tools.errors import ModelToolsRefused
from src.pipelines.pipes.model_tools.main import ADMIN_MESSAGE, LOCAL_ONLY_MESSAGE
from src.platform.runtime.native.errors import SamplingCancelled
from src.platform.runtime.native.ops.model_tools.bake import bake_stream
from src.platform.runtime.native.ops.model_tools.lora_io import LoraInput
from tests.platform.runtime.native.ops.model_tools.lora_fixtures import drain

from .conftest import failing_verify, make_depot, model_files, rows, run_pipe

pytestmark = pytest.mark.usefixtures("cpu_only")


def listing(folder: Path) -> list[str]:
    return sorted(os.listdir(folder))


def read(path: Path):
    with safe_open(str(path), framework="pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}, dict(f.metadata() or {})


class TestQuantize:
    def test_writes_next_to_the_source_and_emits_the_card(self, depot, moss):
        sd, source = moss
        outputs = run_pipe(depot, {"mode": "quantize", "model": str(source)})
        card = model_files(outputs)
        assert len(card) == 1
        out = depot.models / "moss-tiny-fp8.safetensors"
        assert out.is_file()
        data = card[0]
        assert data.path == str(out)
        assert data.kind == "model"
        assert data.model_type == "diffusion_model"
        assert data.file_name == out.name
        assert data.size_bytes == out.stat().st_size
        assert data.source_size_bytes == source.stat().st_size
        assert data.format == "fp8_e4m3"
        assert data.summary["layers_quantized"] > 0
        assert data.summary["layers_kept"] > 0
        assert data.summary["worst_layers"]
        assert data.summary["sources"][0]["name"] == source.name
        assert len(data.summary["sources"][0]["sha256"]) == 64
        assert data.summary["licence_note"] == "apache-2.0"
        assert data.summary["verification"]["checked_layers"] > 0
        json.dumps(data.summary)

    def test_output_is_quantized_with_provenance_in_the_header(self, depot, moss):
        sd, source = moss
        run_pipe(depot, {"mode": "quantize", "model": str(source)})
        tensors, meta = read(depot.models / "moss-tiny-fp8.safetensors")
        descriptors = [k for k in tensors if k.endswith(".comfy_quant")]
        assert descriptors
        for key in descriptors:
            weight = tensors[key[: -len(".comfy_quant")] + ".weight"]
            assert weight.dtype == torch.float8_e4m3fn
        assert meta["potionui.tool"] == "model_tools"
        assert meta["potionui.mode"] == "quantize"
        assert json.loads(meta["potionui.source_model"])["name"] == source.name
        assert meta["license"] == "apache-2.0"

    def test_keep_patterns_leave_matching_layers_alone(self, depot, moss):
        sd, source = moss
        run_pipe(depot, {"mode": "quantize", "model": str(source), "keep_patterns": "layers.0.proj.o*\nlayers.1.*"})
        tensors, _ = read(depot.models / "moss-tiny-fp8.safetensors")
        assert tensors["layers.0.proj.o.weight"].dtype == torch.float32
        assert tensors["layers.1.proj.o.weight"].dtype == torch.float32

    def test_custom_name_gets_the_extension(self, depot, moss):
        _, source = moss
        run_pipe(depot, {"mode": "quantize", "model": str(source), "output_name": "my-fp8"})
        assert (depot.models / "my-fp8.safetensors").is_file()

    def test_progress_runs_per_tensor_and_ends_complete(self, depot, moss):
        _, source = moss
        outputs = run_pipe(depot, {"mode": "quantize", "model": str(source)})
        progress = [o for o in outputs if isinstance(o, ProgressGenerationOutput)]
        assert all(isinstance(p.progress, Progress) for p in progress)
        assert any("Writing" in p.state for p in progress)
        assert any("<<MODEL:moss-tiny-fp8.safetensors>>" in p.state for p in progress)
        assert progress[-1].progress.current == progress[-1].progress.max

    def test_already_quantized_source_is_refused(self, depot, moss):
        _, source = moss
        run_pipe(depot, {"mode": "quantize", "model": str(source)})
        with pytest.raises(ModelToolsRefused, match="already carries quantized"):
            run_pipe(
                depot,
                {"mode": "quantize", "model": str(depot.models / "moss-tiny-fp8.safetensors"), "output_name": "twice"},
            )
        assert not (depot.models / "twice.safetensors").exists()


WIDE_SHAPES = {"blocks.0.attn.q.weight": (32, 512), "blocks.0.mlp.fc1.weight": (48, 40), "img_in.weight": (8, 16)}
FORMAT_LABELS = {
    "fp8_e4m3": ("-fp8", torch.float8_e4m3fn),
    "fp8_e5m2": ("-fp8-e5m2", torch.float8_e5m2),
    "int8": ("-int8", torch.int8),
    "int8_convrot": ("-int8-convrot", torch.int8),
    "nvfp4": ("-nvfp4", torch.uint8),
    "int4_convrot": ("-int4-convrot", torch.int8),
}


@pytest.fixture
def wide(depot):
    from safetensors.torch import save_file

    gen = torch.Generator().manual_seed(0)
    path = depot.models / "wide.safetensors"
    save_file({k: (torch.randn(s, generator=gen) * 0.05).to(torch.bfloat16) for k, s in WIDE_SHAPES.items()}, str(path))
    return path


class TestQuantizeFormats:
    @pytest.mark.parametrize("key", list(FORMAT_LABELS))
    def test_each_format_is_written_named_verified_and_labelled(self, depot, wide, key):
        suffix, dtype = FORMAT_LABELS[key]
        outputs = run_pipe(depot, {"mode": "quantize", "model": str(wide), "family": "krea2", "quant_format": key})
        out = depot.models / f"wide{suffix}.safetensors"
        card = model_files(outputs)[0]
        assert card.path == str(out)
        assert card.format == key
        tensors, meta = read(out)
        assert tensors["blocks.0.attn.q.weight"].dtype == dtype
        assert json.loads(meta["potionui.recipe"])["quantized_layers"] == card.summary["layers_quantized"]
        assert card.summary["verification"]["checked_layers"] == card.summary["layers_quantized"]
        assert card.summary["layers_shape_kept"] == (0 if key in ("fp8_e4m3", "fp8_e5m2", "int8") else 1)
        assert card.summary.get("quality_note") == (
            "INT4 output quality is unverified until a GPU check" if key == "int4_convrot" else None
        )

    def test_a_layer_the_format_cannot_hold_keeps_its_source_dtype(self, depot, wide):
        outputs = run_pipe(depot, {"mode": "quantize", "model": str(wide), "family": "krea2", "quant_format": "nvfp4"})
        tensors, _ = read(depot.models / "wide-nvfp4.safetensors")
        assert tensors["blocks.0.mlp.fc1.weight"].dtype == torch.bfloat16
        assert "blocks.0.mlp.fc1.comfy_quant" not in tensors
        summary = model_files(outputs)[0].summary
        assert (summary["layers_quantized"], summary["layers_shape_kept"]) == (1, 1)

    def test_an_unknown_format_is_refused_before_anything_is_written(self, depot, wide):
        with pytest.raises(ModelToolsRefused, match="Unknown quantization format"):
            run_pipe(depot, {"mode": "quantize", "model": str(wide), "quant_format": "int3"})
        assert listing(depot.models) == ["wide.safetensors"]


class TestQuantizeTextEncoder:
    def test_a_text_encoder_is_quantized_into_the_text_encoder_folder(self, depot):
        from tests.platform.runtime.native.ops.model_tools.fixtures import write_family_fixture

        source = depot.dirs["text_encoder"] / "t5.safetensors"
        quant_keys, kept_keys = write_family_fixture(source, "t5xxl")
        outputs = run_pipe(depot, {"mode": "quantize", "model": str(source)})
        out = depot.dirs["text_encoder"] / "t5-fp8.safetensors"
        assert out.is_file()
        card = model_files(outputs)[0]
        assert card.model_type == "text_encoder"
        assert card.kind == "model"
        assert card.summary["family"] == "t5xxl"
        assert card.summary["layers_quantized"] == len(quant_keys)
        assert listing(depot.models) == []

    @pytest.mark.parametrize("key", ["int8", "int8_convrot", "int4_convrot"])
    def test_integer_formats_are_refused_for_text_encoders(self, depot, key):
        from tests.platform.runtime.native.ops.model_tools.fixtures import write_family_fixture

        source = depot.dirs["text_encoder"] / "t5.safetensors"
        write_family_fixture(source, "t5xxl")
        with pytest.raises(ModelToolsRefused, match="text encoders can be quantized to fp8_e4m3, fp8_e5m2, nvfp4 only"):
            run_pipe(depot, {"mode": "quantize", "model": str(source), "quant_format": key})
        assert listing(depot.dirs["text_encoder"]) == ["t5.safetensors"]

    def test_sdxl_style_checkpoints_are_refused_with_the_reason(self, depot):
        from safetensors.torch import save_file

        path = depot.models / "sdxl.safetensors"
        save_file(
            {
                "model.diffusion_model.input_blocks.0.0.weight": torch.zeros(4, 4),
                "model.diffusion_model.input_blocks.1.0.weight": torch.zeros(4, 4),
            },
            str(path),
        )
        with pytest.raises(ModelToolsRefused, match="SDXL"):
            run_pipe(depot, {"mode": "quantize", "model": str(path)})
        assert listing(depot.models) == ["sdxl.safetensors"]


class TestMergeIntoModel:
    def test_matches_the_shared_bake_and_names_the_output_from_the_loras(self, depot, moss, lora_files):
        sd, source = moss
        a, b = lora_files
        outputs = run_pipe(
            depot, {"mode": "merge_into_model", "model": str(source), "loras": rows((a, 0.7), (b, -0.4))}
        )
        out = depot.models / "moss-tiny+style-a+style-b.safetensors"
        assert out.is_file()
        plan, tensors, _ = bake_stream(source, [LoraInput(str(a), 0.7), LoraInput(str(b), -0.4)], "moss")
        expected = drain(plan, tensors)
        written, meta = read(out)
        assert set(written) == set(expected)
        for key, value in expected.items():
            assert torch.equal(written[key], value), key
        assert not torch.equal(written["layers.0.proj.o.weight"], sd["layers.0.proj.o.weight"])
        card = model_files(outputs)[0]
        assert card.kind == "model"
        assert card.summary["layers_merged"] >= 2
        assert [s["strength"] for s in card.summary["sources"]] == [None, 0.7, -0.4]
        loras = json.loads(meta["potionui.loras"])
        assert [l["name"] for l in loras] == ["style-a.safetensors", "style-b.safetensors"]
        assert all(len(l["sha256"]) == 64 for l in loras)

    @pytest.mark.parametrize("strength", [0, "x", float("nan")])
    def test_unusable_strengths_are_refused(self, depot, moss, lora_files, strength):
        _, source = moss
        with pytest.raises(ModelToolsRefused, match="strength"):
            run_pipe(
                depot,
                {"mode": "merge_into_model", "model": str(source), "loras": [{"model": str(lora_files[0]), "strength": strength}]},
            )

    def test_needs_a_lora(self, depot, moss):
        _, source = moss
        with pytest.raises(ModelToolsRefused, match="at least one LoRA"):
            run_pipe(depot, {"mode": "merge_into_model", "model": str(source), "loras": []})

    def test_a_lora_for_another_layout_is_refused_and_nothing_is_left_behind(self, depot, moss):
        _, source = moss
        from tests.platform.runtime.native.ops.model_tools.lora_fixtures import lora_pair, save_lora

        bad = save_lora(depot.loras / "bad.safetensors", lora_pair("diffusion_model.nope.layer", 4, 4, 2, 1))
        before = listing(depot.models)
        with pytest.raises(ModelToolsRefused):
            run_pipe(depot, {"mode": "merge_into_model", "model": str(source), "loras": rows((bad, 1.0))})
        assert listing(depot.models) == before

    def test_unsupported_adapter_kinds_are_refused(self, depot, moss):
        _, source = moss
        from tests.platform.runtime.native.ops.model_tools.lora_fixtures import lora_pair, save_lora

        sd = lora_pair("diffusion_model.layers.0.proj.o", 16, 16, 2, 1)
        sd["diffusion_model.layers.0.proj.o.dora_scale"] = torch.ones(16)
        dora = save_lora(depot.loras / "dora.safetensors", sd)
        with pytest.raises(ModelToolsRefused, match="DoRA"):
            run_pipe(depot, {"mode": "merge_into_model", "model": str(source), "loras": rows((dora, 1.0))})


class TestMergeAndQuantize:
    def test_quantizes_the_merged_weights_and_reports_against_the_merge(self, depot, moss, lora_files):
        sd, source = moss
        a, _ = lora_files
        outputs = run_pipe(
            depot, {"mode": "merge_and_quantize", "model": str(source), "loras": rows((a, 1.0))}
        )
        out = depot.models / "moss-tiny+style-a-fp8.safetensors"
        assert out.is_file()
        tensors, meta = read(out)
        assert any(k.endswith(".comfy_quant") for k in tensors)
        card = model_files(outputs)[0]
        assert card.format == "fp8_e4m3"
        assert card.summary["layers_quantized"] > 0
        assert card.summary["verification"]["reference"] == "merged fp32"
        assert card.summary["worst_layers"]
        assert meta["potionui.mode"] == "merge_and_quantize"
        assert "potionui.recipe" in meta and "potionui.loras" in meta

        weight = tensors["layers.0.proj.o.weight"].to(torch.float32)
        scale = tensors["layers.0.proj.o.weight_scale"].to(torch.float32)
        plan, baked, _ = bake_stream(source, [LoraInput(str(a), 1.0)], "moss")
        merged = drain(plan, baked)["layers.0.proj.o.weight"]
        base_err = torch.linalg.vector_norm(weight * scale - sd["layers.0.proj.o.weight"]).item()
        merged_err = torch.linalg.vector_norm(weight * scale - merged).item()
        assert merged_err < base_err


class TestMergeAndQuantizeFormats:
    def test_nvfp4_is_checked_against_the_merge_and_counts_the_layers_it_cannot_hold(self, depot, moss, lora_files):
        _, source = moss
        a, _ = lora_files
        outputs = run_pipe(
            depot,
            {"mode": "merge_and_quantize", "model": str(source), "loras": rows((a, 1.0)), "quant_format": "nvfp4"},
        )
        out = depot.models / "moss-tiny+style-a-nvfp4.safetensors"
        tensors, _ = read(out)
        card = model_files(outputs)[0]
        assert card.format == "nvfp4"
        assert card.summary["layers_quantized"] > 0
        assert card.summary["layers_shape_kept"] > 0
        assert card.summary["verification"]["checked_layers"] == card.summary["layers_quantized"]
        assert 0 < card.summary["max_rel_l2"] < 0.2
        quantized = [k[: -len(".comfy_quant")] for k in tensors if k.endswith(".comfy_quant")]
        assert all(tensors[k + ".weight"].dtype == torch.uint8 for k in quantized)


class TestMergeLoras:
    @pytest.mark.parametrize("method", ["concat", "reduce"])
    def test_lands_in_the_lora_folder_as_a_lora_card(self, depot, lora_files, method):
        a, b = lora_files
        outputs = run_pipe(
            depot,
            {"mode": "merge_loras", "loras": rows((a, 0.5), (b, 1.0)), "merge_method": method, "target_rank": 4, "min_energy": 0.0},
        )
        out = depot.loras / "style-a+style-b.safetensors"
        assert out.is_file()
        card = model_files(outputs)[0]
        assert card.kind == "lora"
        assert card.model_type == "lora"
        assert card.path == str(out)
        assert card.summary["rank"] >= 1
        assert card.summary["layers_merged"] == 2
        assert card.source_size_bytes == a.stat().st_size + b.stat().st_size
        tensors, meta = read(out)
        assert any(k.endswith(".lora_up.weight") for k in tensors)
        assert meta["potionui.mode"] == "merge_loras"
        assert card.summary["rank"] == (6 if method == "concat" else 4)

    def test_reduce_below_min_energy_is_refused_without_a_file(self, depot, lora_files):
        a, b = lora_files
        before = listing(depot.loras)
        with pytest.raises(ModelToolsRefused, match="energy"):
            run_pipe(
                depot,
                {"mode": "merge_loras", "loras": rows((a, 1.0), (b, 1.0)), "merge_method": "reduce", "target_rank": 1, "min_energy": 0.999},
            )
        assert listing(depot.loras) == before

    def test_needs_two_loras(self, depot, lora_files):
        with pytest.raises(ModelToolsRefused, match="two LoRAs"):
            run_pipe(depot, {"mode": "merge_loras", "loras": rows((lora_files[0], 1.0))})

    def test_a_non_lora_file_in_the_list_is_refused(self, depot, moss, lora_files):
        _, source = moss
        with pytest.raises(ModelToolsRefused, match="expected one of: lora"):
            run_pipe(depot, {"mode": "merge_loras", "loras": rows((lora_files[0], 1.0), (source, 1.0))})


class TestCancel:
    def test_cancel_mid_write_raises_and_leaves_no_files(self, depot, moss):
        _, source = moss
        before = listing(depot.models)
        calls = {"n": 0}

        def cancelled():
            calls["n"] += 1
            return calls["n"] > 6

        with pytest.raises(SamplingCancelled):
            run_pipe(depot, {"mode": "quantize", "model": str(source)}, cancelled=cancelled)
        assert calls["n"] > 6
        assert listing(depot.models) == before

    def test_cancel_before_start_writes_nothing(self, depot, moss):
        _, source = moss
        before = listing(depot.models)
        with pytest.raises(SamplingCancelled):
            run_pipe(depot, {"mode": "quantize", "model": str(source)}, cancelled=lambda: True)
        assert listing(depot.models) == before

    def test_cancel_after_the_write_discards_the_staged_file(self, depot, moss, wrap_verify):
        _, source = moss
        before = listing(depot.models)
        state = {"verified": False}

        def wrapper(inner):
            def verify(staged):
                result = inner(staged)
                state["verified"] = True
                return result

            return verify

        wrap_verify(wrapper)
        with pytest.raises(SamplingCancelled):
            run_pipe(depot, {"mode": "quantize", "model": str(source)}, cancelled=lambda: state["verified"])
        assert state["verified"]
        assert listing(depot.models) == before


class TestVerification:
    @pytest.mark.parametrize("mode", ["quantize", "merge_into_model"])
    def test_a_failed_check_publishes_nothing(self, depot, moss, lora_files, wrap_verify, mode):
        _, source = moss
        wrap_verify(failing_verify)
        before = listing(depot.models)
        config = {"mode": mode, "model": str(source), "loras": rows((lora_files[0], 1.0))}
        with pytest.raises(ModelToolsRefused, match="forced"):
            run_pipe(depot, config)
        assert listing(depot.models) == before

    def test_a_failed_check_on_overwrite_keeps_the_old_file(self, depot, moss, wrap_verify):
        _, source = moss
        wrap_verify(failing_verify)
        existing = depot.models / "moss-tiny-fp8.safetensors"
        existing.write_bytes(b"old")
        with pytest.raises(ModelToolsRefused):
            run_pipe(depot, {"mode": "quantize", "model": str(source), "overwrite": True})
        assert existing.read_bytes() == b"old"
        assert listing(depot.models) == ["moss-tiny-fp8.safetensors", "moss-tiny.safetensors"]


class TestFileAppearsMidRun:
    def appear(self, target):
        def wrapper(inner):
            def verify(staged):
                result = inner(staged)
                target.write_bytes(b"someone else")
                return result

            return verify

        return wrapper

    def test_without_overwrite_the_new_file_is_never_clobbered(self, depot, moss, wrap_verify):
        _, source = moss
        target = depot.models / "moss-tiny-fp8.safetensors"
        wrap_verify(self.appear(target))
        with pytest.raises(ModelToolsRefused, match="appeared while"):
            run_pipe(depot, {"mode": "quantize", "model": str(source)})
        assert target.read_bytes() == b"someone else"
        assert listing(depot.models) == ["moss-tiny-fp8.safetensors", "moss-tiny.safetensors"]

    def test_the_no_clobber_fallback_also_refuses_without_hard_links(self, depot, moss, wrap_verify, monkeypatch):
        _, source = moss
        target = depot.models / "moss-tiny-fp8.safetensors"
        wrap_verify(self.appear(target))

        def no_links(*_args, **_kwargs):
            raise OSError("hard links unsupported")

        monkeypatch.setattr(os, "link", no_links)
        with pytest.raises(ModelToolsRefused, match="appeared while"):
            run_pipe(depot, {"mode": "quantize", "model": str(source)})
        assert target.read_bytes() == b"someone else"

    def test_the_fallback_still_publishes_when_nothing_is_in_the_way(self, depot, moss, monkeypatch):
        _, source = moss

        def no_links(*_args, **_kwargs):
            raise OSError("hard links unsupported")

        monkeypatch.setattr(os, "link", no_links)
        run_pipe(depot, {"mode": "quantize", "model": str(source)})
        assert listing(depot.models) == ["moss-tiny-fp8.safetensors", "moss-tiny.safetensors"]

    def test_with_overwrite_the_new_file_is_replaced(self, depot, moss, wrap_verify):
        _, source = moss
        target = depot.models / "moss-tiny-fp8.safetensors"
        wrap_verify(self.appear(target))
        run_pipe(depot, {"mode": "quantize", "model": str(source), "overwrite": True})
        assert target.read_bytes() != b"someone else"


class TestRefusals:
    def test_non_admin_is_refused_before_anything_is_touched(self, depot, moss):
        _, source = moss
        before = listing(depot.models)
        with pytest.raises(ModelToolsRefused, match=ADMIN_MESSAGE):
            run_pipe(depot, {"mode": "quantize", "model": str(source)}, is_admin=False)
        assert listing(depot.models) == before

    @pytest.mark.parametrize("flag", [None, "true", 1, "yes"])
    def test_only_a_real_true_counts_as_admin(self, depot, moss, flag):
        _, source = moss
        with pytest.raises(ModelToolsRefused, match=ADMIN_MESSAGE):
            run_pipe(depot, {"mode": "quantize", "model": str(source)}, is_admin=flag)

    @pytest.mark.parametrize("service", [None, "none-roots"])
    def test_a_host_without_model_roots_is_refused_as_not_local(self, depot, moss, service):
        from types import SimpleNamespace

        _, source = moss
        models = None if service is None else SimpleNamespace(roots=None)
        with pytest.raises(ModelToolsRefused) as err:
            run_pipe(depot, {"mode": "quantize", "model": str(source)}, models=models)
        assert str(err.value) == LOCAL_ONLY_MESSAGE
        assert "remote" in LOCAL_ONLY_MESSAGE and "ComfyUI" in LOCAL_ONLY_MESSAGE

    def test_not_enough_free_space_names_both_sizes(self, depot, moss, monkeypatch):
        _, source = moss
        usage = namedtuple("usage", "total used free")
        monkeypatch.setattr(placement.shutil, "disk_usage", lambda _p: usage(10**9, 10**9 - 1000, 1000))
        before = listing(depot.models)
        with pytest.raises(ModelToolsRefused, match=r"needs about .*MiB and the drive must keep .* free .* only .*MiB is available"):
            run_pipe(depot, {"mode": "quantize", "model": str(source)})
        assert listing(depot.models) == before

    @pytest.mark.parametrize(
        "name",
        ["a/b", "..\\x", "x:y", "CON", "nul.txt", ".hidden", "a..b", "bad|name", "x" * 200, "q?", "trail.", "tab\tname"],
    )
    def test_bad_output_names_are_refused(self, depot, moss, name):
        _, source = moss
        before = listing(depot.models)
        with pytest.raises(ModelToolsRefused):
            run_pipe(depot, {"mode": "quantize", "model": str(source), "output_name": name})
        assert listing(depot.models) == before
        assert listing(depot.base) == sorted(["diffusion_models", "loras", "text_encoders"])

    def test_existing_output_needs_the_overwrite_box(self, depot, moss):
        _, source = moss
        existing = depot.models / "moss-tiny-fp8.safetensors"
        existing.write_bytes(b"keep me")
        with pytest.raises(ModelToolsRefused, match="Overwrite"):
            run_pipe(depot, {"mode": "quantize", "model": str(source)})
        assert existing.read_bytes() == b"keep me"
        assert listing(depot.models) == ["moss-tiny-fp8.safetensors", "moss-tiny.safetensors"]

    def test_overwrite_replaces_and_leaves_no_staging_files(self, depot, moss):
        _, source = moss
        existing = depot.models / "moss-tiny-fp8.safetensors"
        existing.write_bytes(b"old")
        run_pipe(depot, {"mode": "quantize", "model": str(source), "overwrite": True})
        assert existing.stat().st_size > 3
        assert listing(depot.models) == ["moss-tiny-fp8.safetensors", "moss-tiny.safetensors"]

    def test_a_failed_overwrite_keeps_the_old_file(self, depot, moss):
        _, source = moss
        existing = depot.models / "moss-tiny-fp8.safetensors"
        existing.write_bytes(b"old")
        calls = {"n": 0}

        def cancelled():
            calls["n"] += 1
            return calls["n"] > 4

        with pytest.raises(SamplingCancelled):
            run_pipe(depot, {"mode": "quantize", "model": str(source), "overwrite": True}, cancelled=cancelled)
        assert existing.read_bytes() == b"old"

    def test_the_output_can_never_replace_an_input(self, depot, moss):
        _, source = moss
        with pytest.raises(ModelToolsRefused, match="overwrite its own input"):
            run_pipe(
                depot,
                {"mode": "quantize", "model": str(source), "output_name": "moss-tiny", "overwrite": True},
            )

    def test_a_source_outside_the_model_folders_is_refused(self, depot, moss, tmp_path):
        _, source = moss
        outside = tmp_path / "elsewhere.safetensors"
        shutil.copy(source, outside)
        with pytest.raises(ModelToolsRefused, match="not inside a configured model folder"):
            run_pipe(depot, {"mode": "quantize", "model": str(outside)})
        assert not (tmp_path / "elsewhere-fp8.safetensors").exists()

    def test_a_lora_given_as_the_model_is_refused(self, depot, lora_files):
        with pytest.raises(ModelToolsRefused, match="expected one of"):
            run_pipe(depot, {"mode": "quantize", "model": str(lora_files[0])})

    @pytest.mark.parametrize("model", ["", None, "missing.safetensors", "x.ckpt"])
    def test_missing_or_unsupported_model_is_refused(self, depot, model):
        with pytest.raises(ModelToolsRefused):
            run_pipe(depot, {"mode": "quantize", "model": model})

    def test_a_read_only_model_folder_is_refused(self, tmp_path):
        depot = make_depot(tmp_path, read_only=True)
        from tests.platform.runtime.native.ops.model_tools.lora_fixtures import build_moss, save_model

        sd = {k: v.detach().clone() for k, v in build_moss().state_dict().items()}
        source = save_model(depot.models / "m.safetensors", sd)
        with pytest.raises(ModelToolsRefused, match="read-only"):
            run_pipe(depot, {"mode": "quantize", "model": source})

    def test_unknown_mode_is_refused(self, depot, moss):
        with pytest.raises(ModelToolsRefused, match="Unknown mode"):
            run_pipe(depot, {"mode": "explode", "model": str(moss[1])})
