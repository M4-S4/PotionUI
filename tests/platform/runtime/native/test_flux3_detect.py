from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from src.platform.runtime.model_headers.signatures import TRANSFORMER_EXTRACTABLE_FAMILIES, family_from_keys
from src.platform.runtime.native.arch.flux3.model import Flux3
from src.platform.runtime.native.detect.registry import match_model_spec
from src.platform.runtime.native.detect.unet_detect import detect_unet_config
from src.platform.runtime.native.errors import NativeEngineUnsupportedError
from vendor.gpl.comfyui.ops import pick_operations

_FIXTURES = Path(__file__).parent / "arch" / "fixtures"


@pytest.fixture(scope="module")
def reference():
    path = _FIXTURES / "flux3_reference_dit.py"
    spec = importlib.util.spec_from_file_location("flux3_reference_dit_detect", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _meta_state_dict(reference, flow_map):
    with torch.device("meta"):
        dit = reference.DiT(flow_map=flow_map)
    return {k: torch.empty(tuple(v.shape), device="meta") for k, v in dit.state_dict().items()}


def test_checkpoint_key_list_is_the_357_reference_keys(reference):
    keys = set((_FIXTURES / "flux3_image_klein_bf16.keys.txt").read_text().split())
    assert len(keys) == 357
    assert keys == set(_meta_state_dict(reference, False))


def test_base_checkpoint_is_detected_with_real_shapes(reference):
    config = detect_unet_config(_meta_state_dict(reference, False))
    assert config["image_model"] == "flux3"
    assert config["hidden_size"] == 3072
    assert config["num_heads"] == 24
    assert config["head_dim"] == 128
    assert config["depth"] == 5
    assert config["depth_single_blocks"] == 28
    assert config["context_in_dim"] == 20480
    assert config["in_channels"] == 128
    assert config["mlp_hidden"] == 9216
    assert config["axes_dim"] == [32, 32, 32, 32]
    assert config["flow_map"] is False
    assert config["has_image_cond"] is True
    spec = match_model_spec(config)
    assert (spec.family, spec.variant) == ("flux3", "flux3_klein_base")
    assert spec.sampling_settings["guidance"] == "cfg"
    assert spec.sampling_settings["shift"] == 6.93


def test_flow_map_checkpoint_selects_the_distilled_spec(reference):
    config = detect_unet_config(_meta_state_dict(reference, True))
    assert config["flow_map"] is True
    spec = match_model_spec(config)
    assert (spec.family, spec.variant) == ("flux3", "flux3_klein")
    assert spec.sampling_settings["guidance"] == "flow_map"
    assert spec.sampling_settings["default_steps"] == 16


@pytest.mark.parametrize("flow_map,expected_keys", [(False, 357), (True, 359)])
def test_detected_config_builds_a_module_with_exactly_the_checkpoint_keys(reference, flow_map, expected_keys):
    sd = _meta_state_dict(reference, flow_map)
    config = detect_unet_config(sd)
    with torch.device("meta"):
        model = Flux3.from_config(config, pick_operations(torch.bfloat16, torch.bfloat16))
    built = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    assert len(built) == expected_keys
    assert built == {k: tuple(v.shape) for k, v in sd.items()}


@pytest.mark.parametrize("flow_map,variant", [(False, "flux3_klein_base"), (True, "flux3_klein")])
def test_torch_free_dispatch_agrees_with_native_detection(reference, flow_map, variant):
    sd = _meta_state_dict(reference, flow_map)
    match = family_from_keys(set(sd), lambda key: tuple(sd[key].shape))
    assert (match.family, match.variant) == ("flux3", variant)
    assert match_model_spec(detect_unet_config(sd)).variant == variant
    assert "flux3" in TRANSFORMER_EXTRACTABLE_FAMILIES


def test_nvfp4_checkpoint_raises_a_clear_unsupported_error(reference):
    sd = _meta_state_dict(reference, False)
    sd["single_blocks.0.q_proj.weight_scale_2"] = torch.empty((), device="meta")
    with pytest.raises(NativeEngineUnsupportedError, match="NVFP4"):
        detect_unet_config(sd)


def test_flux3_module_exposes_no_hidden_size_so_loras_use_the_name_for_name_map(reference):
    config = detect_unet_config(_meta_state_dict(reference, False))
    with torch.device("meta"):
        model = Flux3.from_config(config, pick_operations(torch.bfloat16, torch.bfloat16))
    assert not hasattr(model.params, "hidden_size")
    assert model.params.in_channels == 128
    assert model.patch_size == 1
