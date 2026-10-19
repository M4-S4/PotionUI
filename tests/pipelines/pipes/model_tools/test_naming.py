from __future__ import annotations

import pytest

from src.pipelines.pipes.model_tools import naming
from src.pipelines.pipes.model_tools.errors import ModelToolsRefused


@pytest.mark.parametrize(
    "mode, source, loras, expected",
    [
        ("quantize", "flux-dev.safetensors", [], "flux-dev-fp8"),
        ("merge_into_model", "flux-dev.safetensors", ["style.safetensors"], "flux-dev+style"),
        ("merge_into_model", "flux-dev.safetensors", ["a.safetensors", "b.sft"], "flux-dev+a+b"),
        ("merge_and_quantize", "flux-dev.safetensors", ["style.safetensors"], "flux-dev+style-fp8"),
        ("merge_loras", None, ["a.safetensors", "b.safetensors"], "a+b"),
    ],
)
def test_default_names_derive_from_the_inputs(mode, source, loras, expected):
    assert naming.resolve_stem("", mode, source, loras) == expected
    assert naming.resolve_stem(None, mode, source, loras) == expected
    assert naming.resolve_stem("   ", mode, source, loras) == expected


def test_default_names_are_sanitized_not_refused():
    stem = naming.resolve_stem("", "quantize", "My Model: v2 (final)?.safetensors", [])
    assert stem == "My_Model_v2_(final)-fp8"
    naming.validate_stem(stem)


def test_long_default_names_are_shortened_with_a_stable_digest():
    loras = [f"a-very-long-lora-name-number-{i}.safetensors" for i in range(12)]
    first = naming.resolve_stem("", "merge_into_model", "base.safetensors", loras)
    second = naming.resolve_stem("", "merge_into_model", "base.safetensors", loras)
    assert first == second
    assert len(first) <= naming.MAX_STEM_LENGTH
    other = naming.resolve_stem("", "merge_into_model", "base.safetensors", loras[:-1] + ["different.safetensors"])
    assert other != first


@pytest.mark.parametrize(
    "given, expected",
    [("mine", "mine"), ("mine.safetensors", "mine"), ("Mine.SAFETENSORS", "Mine"), (" x.sft ", "x"), ("v1.5-fp8", "v1.5-fp8")],
)
def test_user_names_lose_the_extension_and_surrounding_space(given, expected):
    assert naming.resolve_stem(given, "quantize", "m.safetensors", []) == expected


@pytest.mark.parametrize(
    "given",
    [
        "a/b", "a\\b", "../x", "..", "x:y", "a*b", "a?b", 'a"b', "a<b", "a>b", "a|b", ".x", "x.",
        "a..b", "COM1", "lpt9.old", "x\x00y", "x" * 151,
    ],
)
def test_user_names_with_separators_or_reserved_parts_are_refused(given):
    with pytest.raises(ModelToolsRefused):
        naming.resolve_stem(given, "quantize", "m.safetensors", [])


def test_the_extension_is_added_exactly_once():
    assert naming.file_name("a") == "a.safetensors"
