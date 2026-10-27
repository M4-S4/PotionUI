from __future__ import annotations

import json

import pytest
import torch

from src.platform.runtime.native.io.quant_formats import (
    COMFY_QUANT_INT8_TENSORWISE,
    check_quant_format_supported,
    extract_convrot_config,
    parse_comfy_quant,
    reject_ambiguous_int8,
)

from .._quant_layouts import (
    CONFIRMED_DESCRIPTORS,
    CONVROT_DEFAULT_GROUPSIZE,
    CONVROT_WILD_FORMS,
    INT8_PLAIN_DESCRIPTOR,
    convrot_descriptor,
    convrot_wild_blob,
    descriptor_blob,
)

_STATED_GROUPSIZE = 64


def _blob(conf: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def _text_blob(text: str) -> torch.Tensor:
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.uint8)


def test_parse_comfy_quant_none_for_missing_blob():
    assert parse_comfy_quant(None) is None


def test_parse_comfy_quant_none_for_corrupt_blob():
    garbage = torch.tensor([0xFF, 0xFE, 0x00, 0x01], dtype=torch.uint8)
    assert parse_comfy_quant(garbage) is None


def test_parse_comfy_quant_decodes_json():
    assert parse_comfy_quant(_blob({"format": "float8_e4m3fn"})) == {"format": "float8_e4m3fn"}


@pytest.mark.parametrize("text", ["[1, 2]", "3", '"format"', "null"])
def test_a_blob_that_is_json_but_not_an_object_is_no_descriptor(text):
    assert parse_comfy_quant(_text_blob(text)) is None


@pytest.mark.parametrize("conf", CONFIRMED_DESCRIPTORS)
def test_published_descriptors_round_trip(conf):
    blob = descriptor_blob(conf)
    assert blob.dtype == torch.uint8
    assert blob.numel() == len(json.dumps(conf).encode("utf-8"))
    assert parse_comfy_quant(blob) == conf


@pytest.mark.parametrize("nested", [False, True])
def test_stated_convrot_groupsize_is_read_not_defaulted(nested):
    conf = convrot_descriptor(_STATED_GROUPSIZE, nested=nested)
    convrot, groupsize = extract_convrot_config(conf, "proj")
    assert convrot is True
    assert groupsize == _STATED_GROUPSIZE


def test_convrot_without_stated_groupsize_uses_the_format_default(caplog):
    conf = convrot_descriptor(_STATED_GROUPSIZE, drop_groupsize=True)
    with caplog.at_level("INFO"):
        convrot, groupsize = extract_convrot_config(conf, "proj")
    assert (convrot, groupsize) == (True, CONVROT_DEFAULT_GROUPSIZE)
    assert "no group size" in caplog.text or "states no group size" in caplog.text


@pytest.mark.parametrize("form", CONVROT_WILD_FORMS, ids=lambda f: f"{f[0]}b")
def test_wild_descriptor_variants_parse_at_their_published_byte_length(form):
    expected_bytes, _, _, extras = form
    blob = convrot_wild_blob(CONVROT_DEFAULT_GROUPSIZE, form)
    assert blob.numel() == expected_bytes
    conf = parse_comfy_quant(blob)
    assert conf is not None
    convrot, groupsize = extract_convrot_config(conf, "proj")
    assert convrot is True
    assert groupsize == CONVROT_DEFAULT_GROUPSIZE
    if extras.get("per_row"):
        assert conf["per_row"] is True


def test_plain_int8_descriptor_reports_no_convrot():
    assert extract_convrot_config(dict(INT8_PLAIN_DESCRIPTOR), "proj")[0] is False


def test_a_descriptor_without_convrot_reports_the_default_groupsize_and_stays_quiet(caplog):
    with caplog.at_level("INFO"):
        convrot, groupsize = extract_convrot_config({"format": "float8_e4m3fn"}, "proj")
    assert (convrot, groupsize) == (False, CONVROT_DEFAULT_GROUPSIZE)
    assert caplog.text == ""


def test_the_default_groupsize_log_names_the_layer(caplog):
    with caplog.at_level("INFO"):
        extract_convrot_config({"format": COMFY_QUANT_INT8_TENSORWISE, "convrot": True}, "blocks.3.proj")
    assert "blocks.3.proj" in caplog.text


@pytest.mark.parametrize("fmt", ["float8_e4m3fn", "float8_e5m2", COMFY_QUANT_INT8_TENSORWISE])
def test_the_formats_the_layers_dequantise_are_accepted(fmt):
    check_quant_format_supported({"format": fmt}, "proj")


@pytest.mark.parametrize("conf, shown", [
    ({"format": "mxfp8"}, "'mxfp8'"),
    ({"format": "nvfp4"}, "'nvfp4'"),
    ({}, "None"),
])
def test_any_other_format_is_refused_naming_the_format_and_the_layer(conf, shown):
    with pytest.raises(ValueError, match=f"quant format {shown} is not supported yet \\(layer 'blocks.0.proj'\\)"):
        check_quant_format_supported(conf, "blocks.0.proj")


@pytest.mark.parametrize("weight, scale", [
    (torch.zeros(4, 8), torch.ones(4, 1)),
    (None, torch.ones(4, 1)),
    (torch.zeros(4, 8, dtype=torch.int8), None),
    (torch.zeros(4, 8, dtype=torch.int8), torch.tensor(0.5)),
], ids=["float-weight", "no-weight", "no-scale", "scalar-scale"])
def test_a_layer_whose_rotation_state_is_decidable_is_not_rejected(weight, scale):
    reject_ambiguous_int8(weight, scale, "proj")


def test_int8_codes_with_a_per_channel_scale_and_no_descriptor_are_rejected():
    weight = torch.zeros(4, 8, dtype=torch.int8)
    with pytest.raises(ValueError, match="int8 layer 'blocks.1.proj' has a per-output-channel weight_scale but no comfy_quant"):
        reject_ambiguous_int8(weight, torch.ones(4, 1), "blocks.1.proj")
