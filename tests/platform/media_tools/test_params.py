from __future__ import annotations

import pytest

from src.platform.media_tools import InvalidToolParams, validate_params
from src.platform.media_tools.conditions import when_matches
from tests.platform.media_tools.fake_tools import make_spec

from src.platform.media_tools import MediaTool


def spec_with(*params, impl=None):
    return make_spec("probe", impl, params=list(params))


def integer(**kw):
    return {"name": "n", "type": "integer", "label": "Count", **kw}


def number(**kw):
    return {"name": "f", "type": "float", "label": "Amount", **kw}


def test_defaults_are_applied_and_optional_none_is_kept():
    spec = spec_with(
        {"name": "a", "type": "integer", "default": 3},
        {"name": "b", "type": "integer", "default": None},
        {"name": "c", "type": "checkbox", "default": True},
    )

    assert validate_params(spec, {}) == {"a": 3, "b": None, "c": True}


def test_param_without_a_default_is_required():
    spec = spec_with(integer(min=1))

    with pytest.raises(InvalidToolParams) as raised:
        validate_params(spec, {})

    assert raised.value.field == "n"
    assert "required" in raised.value.message


def test_none_and_blank_count_as_absent():
    spec = spec_with(integer(default=4))

    assert validate_params(spec, {"n": None}) == {"n": 4}
    assert validate_params(spec, {"n": "  "}) == {"n": 4}


def test_unknown_parameter_is_rejected_by_name():
    spec = spec_with(integer(default=1))

    with pytest.raises(InvalidToolParams) as raised:
        validate_params(spec, {"n": 1, "typo": 2})

    assert raised.value.field == "typo"


def test_params_none_is_the_same_as_empty():
    assert validate_params(spec_with(integer(default=1)), None) == {"n": 1}


@pytest.mark.parametrize("raw,expected", [(5, 5), ("5", 5), (5.0, 5), (" 7 ", 7)])
def test_integer_coercion(raw, expected):
    assert validate_params(spec_with(integer()), {"n": raw}) == {"n": expected}


@pytest.mark.parametrize("raw", [5.5, "abc", True, [], float("nan"), float("inf"), "2.5"])
def test_integer_rejects_non_whole_values(raw):
    with pytest.raises(InvalidToolParams) as raised:
        validate_params(spec_with(integer()), {"n": raw})

    assert raised.value.field == "n"


@pytest.mark.parametrize("raw", [0, 11])
def test_integer_range_is_inclusive(raw):
    spec = spec_with(integer(min=1, max=10))

    with pytest.raises(InvalidToolParams):
        validate_params(spec, {"n": raw})
    assert validate_params(spec, {"n": 1}) == {"n": 1}
    assert validate_params(spec, {"n": 10}) == {"n": 10}


def test_range_message_names_the_bound_and_unit():
    spec = spec_with(number(min=0.5, unit="s"))

    with pytest.raises(InvalidToolParams) as raised:
        validate_params(spec, {"f": 0.1})

    assert "0.5 s" in raised.value.message


@pytest.mark.parametrize("raw,expected", [(1, 1.0), ("2.5", 2.5), (0.25, 0.25), (" 3 ", 3.0)])
def test_float_coercion(raw, expected):
    assert validate_params(spec_with(number()), {"f": raw}) == {"f": expected}


@pytest.mark.parametrize("raw", ["x", True, float("nan"), float("-inf"), {}])
def test_float_rejects_non_numbers(raw):
    with pytest.raises(InvalidToolParams):
        validate_params(spec_with(number()), {"f": raw})


def test_float_range():
    spec = spec_with(number(min=0.25, max=4))

    with pytest.raises(InvalidToolParams):
        validate_params(spec, {"f": 4.01})
    assert validate_params(spec, {"f": 4}) == {"f": 4.0}


@pytest.mark.parametrize(
    "raw,expected",
    [(True, True), (False, False), ("true", True), ("False", False), ("yes", True), ("0", False), (1, True), (0, False)],
)
def test_checkbox_coercion(raw, expected):
    spec = spec_with({"name": "k", "type": "checkbox", "default": False})

    assert validate_params(spec, {"k": raw}) == {"k": expected}


@pytest.mark.parametrize("raw", ["maybe", 2, [], 0.5])
def test_checkbox_rejects_other_values(raw):
    with pytest.raises(InvalidToolParams):
        validate_params(spec_with({"name": "k", "type": "checkbox", "default": False}), {"k": raw})


def test_select_matches_label_value_options_and_plain_options():
    labelled = spec_with(
        {"name": "s", "type": "select", "options": [{"label": "High", "value": "high"}, {"label": "Low", "value": "low"}]}
    )
    plain = spec_with({"name": "s", "type": "select", "options": ["a", "b"]})

    assert validate_params(labelled, {"s": "low"}) == {"s": "low"}
    assert validate_params(plain, {"s": "b"}) == {"s": "b"}


def test_select_returns_the_declared_option_type():
    spec = spec_with(
        {"name": "degrees", "type": "select", "options": [{"label": "90", "value": 90}, {"label": "180", "value": 180}]}
    )

    assert validate_params(spec, {"degrees": "180"}) == {"degrees": 180}
    assert validate_params(spec, {"degrees": 90}) == {"degrees": 90}


def test_select_rejects_a_value_outside_the_options_and_lists_them():
    spec = spec_with({"name": "s", "type": "select", "label": "Axis", "options": ["h", "v"]})

    with pytest.raises(InvalidToolParams) as raised:
        validate_params(spec, {"s": "x"})

    assert raised.value.field == "s"
    assert "h, v" in raised.value.message


def test_select_does_not_confuse_booleans_with_numbers():
    spec = spec_with({"name": "s", "type": "select", "options": [0, 1]})

    with pytest.raises(InvalidToolParams):
        validate_params(spec, {"s": True})


@pytest.mark.parametrize("raw,expected", [("#FF8800", "#ff8800"), ("ff8800", "#ff8800"), ("#abc", "#aabbcc"), (" #00B140 ", "#00b140")])
def test_color_normalises_to_lowercase_hex(raw, expected):
    assert validate_params(spec_with({"name": "c", "type": "color"}), {"c": raw}) == {"c": expected}


@pytest.mark.parametrize("raw", ["red", "#12", "#1234567", "#gggggg", 5])
def test_color_rejects_non_hex(raw):
    with pytest.raises(InvalidToolParams):
        validate_params(spec_with({"name": "c", "type": "color"}), {"c": raw})


def test_color_transparent_only_when_allowed():
    allowed = spec_with({"name": "c", "type": "color", "allow_transparent": True, "default": "transparent"})
    refused = spec_with({"name": "c", "type": "color"})

    assert validate_params(allowed, {}) == {"c": "transparent"}
    assert validate_params(allowed, {"c": "Transparent"}) == {"c": "transparent"}
    with pytest.raises(InvalidToolParams):
        validate_params(refused, {"c": "transparent"})


def test_string_accepts_text_numbers_and_keeps_blank_text():
    spec = spec_with({"name": "t", "type": "string", "default": "x"})

    assert validate_params(spec, {"t": "hello"}) == {"t": "hello"}
    assert validate_params(spec, {"t": ""}) == {"t": ""}
    assert validate_params(spec, {"t": 12}) == {"t": "12"}


def test_string_max_is_a_length_limit_and_rejects_non_text():
    spec = spec_with({"name": "t", "type": "string", "max": 3})

    with pytest.raises(InvalidToolParams):
        validate_params(spec, {"t": "abcd"})
    with pytest.raises(InvalidToolParams):
        validate_params(spec, {"t": ["a"]})
    assert validate_params(spec, {"t": "abc"}) == {"t": "abc"}


@pytest.mark.parametrize(
    "raw,expected",
    [("center", "center"), ("TOP-LEFT", "top_left"), ("bottom_right", "bottom_right"), (" left ", "left")],
)
def test_anchor_accepts_the_nine_positions_in_either_spelling(raw, expected):
    assert validate_params(spec_with({"name": "a", "type": "anchor"}), {"a": raw}) == {"a": expected}


@pytest.mark.parametrize("raw", ["middle", "top-top", "", 3])
def test_anchor_rejects_anything_else(raw):
    with pytest.raises(InvalidToolParams):
        validate_params(spec_with({"name": "a", "type": "anchor", "default": "center"}), {"a": raw or "nope"})


def test_unknown_field_types_pass_through_untouched():
    spec = make_spec("probe", None, params=[{"name": "p", "type": "mystery", "default": {"a": 1}}], component="c")

    assert validate_params(spec, {}) == {"p": {"a": 1}}


KEY_PARAMS = [
    {"name": "key_mode", "type": "select", "default": "auto", "options": ["auto", "color"]},
    {"name": "key_color", "type": "color", "default": "#00b140", "when": {"field": "key_mode", "equals": "color"}},
]


def test_when_false_drops_the_param_and_skips_validation():
    spec = spec_with(*KEY_PARAMS)

    assert validate_params(spec, {"key_mode": "auto", "key_color": "not a colour"}) == {"key_mode": "auto"}


def test_when_true_validates_and_keeps_the_param():
    spec = spec_with(*KEY_PARAMS)

    assert validate_params(spec, {"key_mode": "color", "key_color": "#112233"}) == {
        "key_mode": "color",
        "key_color": "#112233",
    }
    with pytest.raises(InvalidToolParams) as raised:
        validate_params(spec, {"key_mode": "color", "key_color": "nope"})
    assert raised.value.field == "key_color"


def test_when_reads_defaults_of_other_params():
    spec = spec_with(
        {"name": "key_mode", "type": "select", "default": "color", "options": ["auto", "color"]},
        KEY_PARAMS[1],
    )

    assert validate_params(spec, {}) == {"key_mode": "color", "key_color": "#00b140"}


def test_required_param_behind_a_false_when_is_not_required():
    spec = spec_with(
        {"name": "key_mode", "type": "select", "default": "auto", "options": ["auto", "color"]},
        {"name": "key_color", "type": "color", "when": {"field": "key_mode", "equals": "color"}},
    )

    assert validate_params(spec, {}) == {"key_mode": "auto"}
    with pytest.raises(InvalidToolParams):
        validate_params(spec, {"key_mode": "color"})


def test_check_params_hook_runs_on_the_resolved_values():
    class Ordered(MediaTool):
        def check_params(self, params):
            if params["end"] <= params["start"]:
                raise InvalidToolParams("end", "End must be after start")

    spec = make_spec(
        "probe",
        Ordered(),
        params=[
            {"name": "start", "type": "float", "default": 0},
            {"name": "end", "type": "float"},
        ],
    )

    assert validate_params(spec, {"end": 2}) == {"start": 0.0, "end": 2.0}
    with pytest.raises(InvalidToolParams) as raised:
        validate_params(spec, {"start": 3, "end": 2})
    assert raised.value.field == "end"


def test_validation_is_idempotent():
    spec = spec_with(*KEY_PARAMS, number(default=1.5, min=0, max=3))
    first = validate_params(spec, {"key_mode": "color", "key_color": "ABC", "f": "2"})

    assert validate_params(spec, first) == first


@pytest.mark.parametrize(
    "when,values,expected",
    [
        (None, {}, True),
        ({"field": "a", "equals": 1}, {"a": 1}, True),
        ({"field": "a", "equals": 1}, {"a": 2}, False),
        ({"field": "a", "equals": 1}, {"a": True}, False),
        ({"field": "a", "operator": "not_equals", "value": 1}, {"a": 2}, True),
        ({"field": "a", "in": [1, 2]}, {"a": 2}, True),
        ({"field": "a", "not_in": [1, 2]}, {"a": 2}, False),
        ({"field": "a", "greater_than": 3}, {"a": "4"}, True),
        ({"field": "a", "less_than_or_equals": 3}, {"a": 3}, True),
        ({"field": "a", "contains": "ell"}, {"a": "hello"}, True),
        ({"field": "a", "is_empty": True}, {"a": ""}, True),
        ({"field": "a", "is_not_empty": True}, {"a": "x"}, True),
        ([{"field": "a", "equals": 1}, {"field": "b", "equals": 2}], {"a": 1, "b": 2}, True),
        ([{"field": "a", "equals": 1}, {"field": "b", "equals": 2}], {"a": 1, "b": 3}, False),
        ({"logic": "OR", "conditions": [{"field": "a", "equals": 1}, {"field": "b", "equals": 2}]}, {"b": 2}, True),
        ({"logic": "AND", "conditions": [{"field": "a", "equals": 1}, {"field": "b", "equals": 2}]}, {"a": 1}, False),
        ({"field": "a", "bogus": 1}, {"a": 1}, False),
        ({"equals": 1}, {"a": 1}, False),
        ("a=1", {"a": 1}, False),
    ],
)
def test_when_grammar(when, values, expected):
    assert when_matches(when, values) is expected
