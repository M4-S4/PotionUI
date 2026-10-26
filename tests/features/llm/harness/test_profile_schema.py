import pydantic
import pytest

from src.features.llm.dto import ProfileOverridesRequest
from src.features.llm.harness.profile import LAYERED_FIELDS
from src.features.llm.harness.profile_schema import (
    PROFILE_FIELDS,
    clean_overrides,
    merge_overrides,
    schema_json,
    validate_overrides,
)


def field(name: str) -> dict:
    return next(item for item in schema_json()["fields"] if item["name"] == name)


def test_the_schema_describes_every_profile_field_exactly_once():
    names = [spec.name for spec in PROFILE_FIELDS]

    assert sorted(names) == sorted(LAYERED_FIELDS)
    assert len(names) == len(set(names))


def test_every_schema_field_has_a_label_and_a_group():
    for item in schema_json()["fields"]:
        assert item["label"] and item["group"] and item["kind"]


def test_the_schema_matches_the_documented_shapes():
    assert field("replay") == {
        "name": "replay",
        "kind": "enum",
        "label": "Reasoning replay",
        "group": "reasoning",
        "options": ["send_all", "drop_earlier", "preserve", "never"],
    }
    assert field("tool_choice")["kind"] == "set"
    assert field("tool_choice")["options"] == ["auto", "none", "required", "named"]
    assert field("context_window")["kind"] == "int"
    assert field("context_window")["min"] == 1024


def test_only_verified_is_read_only():
    assert [item["name"] for item in schema_json()["fields"] if item.get("readonly")] == ["verified"]


def test_valid_overrides_are_normalised_to_plain_json():
    result = validate_overrides(
        {
            "replay": "preserve",
            "context_window": 65536,
            "tool_choice": ["named", "auto"],
            "reasoning_levels": ["low", "high"],
            "vision": True,
            "prices": {"input": 2, "output": 8.5},
            "reasoning_toggle": "think",
        }
    )

    assert result == {
        "replay": "preserve",
        "context_window": 65536,
        "tool_choice": ["auto", "named"],
        "reasoning_levels": ["low", "high"],
        "vision": True,
        "prices": {"input": 2.0, "output": 8.5, "cached_input": None, "cache_write": None, "reasoning": None},
        "reasoning_toggle": "think",
    }


def test_null_marks_a_field_for_clearing():
    assert validate_overrides({"replay": None, "context_window": None}) == {"replay": None, "context_window": None}


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"bogus": 1}, "bogus"),
        ({"bogus": None}, "bogus"),
        ({"verified": {"suite": "x", "passed_at": "y"}}, "verified"),
        ({"verified": None}, "verified"),
        ({"adapter": "x"}, "adapter"),
        ({"replay": "sometimes"}, "replay"),
        ({"context_window": 512}, "at least 1024"),
        ({"context_window": "big"}, "context_window"),
        ({"context_window": 4096.5}, "context_window"),
        ({"context_window": True}, "context_window"),
        ({"max_output_tokens": 0}, "at least 1"),
        ({"max_images": -1}, "at least 0"),
        ({"cache_min_tokens": -5}, "at least 0"),
        ({"vision": "yes"}, "vision"),
        ({"vision": 1}, "vision"),
        ({"tool_choice": ["auto", "sometimes"]}, "unknown values"),
        ({"tool_choice": "auto"}, "tool_choice"),
        ({"sampling_rejected": ["temperature", "seed"]}, "unknown values"),
        ({"reasoning_levels": ["low", " "]}, "blank"),
        ({"reasoning_levels": "low"}, "reasoning_levels"),
        ({"reasoning_field": "  "}, "blank"),
        ({"reasoning": "sometimes"}, "reasoning"),
        ({"cache": "forever"}, "cache"),
        ({"tool_id_rule": "random"}, "tool_id_rule"),
        ({"structured_output": "yaml"}, "structured_output"),
        ({"prices": {"input": -1}}, "negative"),
        ({"prices": {"free": 1}}, "prices"),
    ],
)
def test_invalid_overrides_are_rejected_with_the_field_named(patch, message):
    with pytest.raises(ValueError, match=message):
        validate_overrides(patch)


def test_a_non_object_patch_is_rejected():
    with pytest.raises(ValueError, match="object"):
        validate_overrides(["replay"])


def test_the_smallest_context_window_is_accepted():
    assert validate_overrides({"context_window": 1024}) == {"context_window": 1024}


def test_merge_sets_keeps_and_clears():
    merged = merge_overrides(
        {"replay": "preserve", "vision": True, "context_window": 4096},
        {"context_window": None, "cache": "explicit", "vision": False},
    )

    assert merged == {"replay": "preserve", "vision": False, "cache": "explicit"}


def test_merge_into_nothing():
    assert merge_overrides(None, {"replay": "never"}) == {"replay": "never"}
    assert merge_overrides({}, {"replay": None}) == {}


def test_merge_rejects_an_invalid_patch_without_touching_the_existing_overrides():
    existing = {"replay": "never"}

    with pytest.raises(ValueError):
        merge_overrides(existing, {"vision": "yes"})

    assert existing == {"replay": "never"}


def test_cleaning_stored_overrides_keeps_valid_values_and_reports_the_rest():
    cleaned, problems = clean_overrides(
        {"replay": "never", "vision": "yes", "bogus": 1, "context_window": None, "cache": "explicit"}
    )

    assert cleaned == {"replay": "never", "cache": "explicit"}
    assert len(problems) == 2
    assert any("vision" in problem for problem in problems) and any("bogus" in problem for problem in problems)


def test_cleaning_nothing_is_empty():
    assert clean_overrides(None) == ({}, [])
    assert clean_overrides({}) == ({}, [])


def test_the_request_dto_validates_and_normalises_the_overrides():
    request = ProfileOverridesRequest(overrides={"replay": "preserve", "context_window": None})

    assert request.overrides == {"replay": "preserve", "context_window": None}


def test_the_request_dto_rejects_invalid_overrides_with_a_validation_error():
    with pytest.raises(pydantic.ValidationError, match="replay"):
        ProfileOverridesRequest(overrides={"replay": "sometimes"})
