from decimal import Decimal

import pytest

from tests.fixtures.flat_variable_ops import apply_variable_operations

from src.platform.resources.prompt_variables import (
    MAX_NUMBER_GRID,
    build_variable_wire_value,
    build_variables_for_submit,
    number_grid,
    render_prompt_variable_lines,
    validate_variables_map,
)


def num(**kw):
    base = {"type": "number", "mode": "shuffle", "min": 18, "max": 80, "step": 1}
    base.update(kw)
    return base


class TestGrid:
    def test_whole_number_grid(self):
        grid = number_grid(num())
        assert grid.count == 63
        assert grid.format_index(0) == "18"
        assert grid.format_index(62) == "80"

    def test_decimal_places_follow_the_step(self):
        assert number_grid(num(min=0, max=1, step=0.1)).values() == [
            "0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0",
        ]
        grid = number_grid(num(min=0, max=0.2, step=0.05))
        assert grid.values() == ["0.00", "0.05", "0.10", "0.15", "0.20"]

    def test_values_come_from_the_index_not_float_accumulation(self):
        grid = number_grid(num(min=0, max=3, step=0.1))
        assert grid.format_index(3) == "0.3"
        assert grid.format_index(29) == "2.9"

    def test_max_not_on_the_grid_is_unreachable(self):
        assert number_grid(num(min=0, max=10, step=3)).values() == ["0", "3", "6", "9"]

    def test_single_value_grid(self):
        grid = number_grid(num(min=5, max=5))
        assert grid.values() == ["5"]

    def test_negative_ranges(self):
        assert number_grid(num(min=-2, max=1, step=1)).values() == ["-2", "-1", "0", "1"]

    def test_step_defaults_to_one(self):
        assert number_grid({"type": "number", "min": 1, "max": 3}).values() == ["1", "2", "3"]

    def test_string_numbers_are_accepted(self):
        assert number_grid(num(min="1", max="2", step="0.5")).values() == ["1.0", "1.5", "2.0"]

    def test_invalid_definition_has_no_grid(self):
        assert number_grid(num(min="x")) is None
        assert number_grid(num(step=0)) is None


class TestValidation:
    def test_accepts_all_three_modes(self):
        for mode in ("shuffle", "pin", "per-image"):
            assert validate_variables_map({"age": num(mode=mode, value=30)}) == []

    def test_mode_defaults_to_shuffle(self):
        assert validate_variables_map({"age": {"type": "number", "min": 1, "max": 2}}) == []

    def test_grid_cap_boundary(self):
        assert validate_variables_map({"a": num(min=0, max=MAX_NUMBER_GRID - 1)}) == []
        errors = validate_variables_map({"a": num(min=0, max=MAX_NUMBER_GRID)})
        assert len(errors) == 1 and "201 values" in errors[0]

    def test_grid_error_names_the_step_to_use(self):
        (error,) = validate_variables_map({"a": num(min=0, max=1000)})
        assert "Raise the step to at least 6" in error

    def test_decimal_step_hint_keeps_the_step_places(self):
        (error,) = validate_variables_map({"a": num(min=0, max=100, step=0.1)})
        assert "at least 0.6" in error

    @pytest.mark.parametrize("definition, fragment", [
        (num(min=None), "numeric min"),
        (num(max="abc"), "numeric max"),
        (num(min=True), "numeric min"),
        (num(step=0), "step must be a number greater than 0"),
        (num(step=-1), "step must be a number greater than 0"),
        (num(step="x"), "step must be a number greater than 0"),
        (num(min=10, max=5), "greater than max"),
        (num(step=0.0000001), "decimal places"),
        (num(min=0.5, step=1), "more decimal places than step"),
        (num(value="abc"), "value must be a number"),
        (num(mode="sometimes"), "invalid mode"),
        (num(min=float("inf")), "numeric min"),
    ])
    def test_teaching_errors(self, definition, fragment):
        errors = validate_variables_map({"age": definition})
        assert len(errors) == 1
        assert fragment in errors[0] and "'age'" in errors[0]

    def test_unknown_type_message_lists_number(self):
        (error,) = validate_variables_map({"a": {"type": "date"}})
        assert "Use text, choice or number" in error

    def test_choice_when_cannot_read_a_number_variable(self):
        errors = validate_variables_map({
            "age": num(),
            "hair": {"type": "choice", "options": [{"text": "grey", "when": {"var": "age", "values": ["70"]}}]},
        })
        assert len(errors) == 1 and "not a choice variable" in errors[0]


class TestWire:
    def test_pin_sends_the_clamped_formatted_value(self):
        assert build_variable_wire_value(num(mode="pin", value=30)) == "30"
        assert build_variable_wire_value(num(mode="pin", value=500)) == "80"
        assert build_variable_wire_value(num(mode="pin", value=-5)) == "18"

    def test_pin_rounds_to_the_step_places(self):
        assert build_variable_wire_value(num(mode="pin", min=0, max=1, step=0.1, value=0.35)) == "0.4"

    def test_pin_without_a_value_uses_min(self):
        assert build_variable_wire_value(num(mode="pin")) == "18"

    def test_per_image_sends_the_enumerated_grid(self):
        wire = build_variable_wire_value(num(mode="per-image", min=1, max=4))
        assert wire == "{1|2|3|4}"

    def test_single_value_grid_is_not_wrapped(self):
        assert build_variable_wire_value(num(mode="per-image", min=7, max=7)) == "7"

    def test_invalid_definition_sends_nothing(self):
        assert build_variable_wire_value(num(min="x")) == ""


class TestSubmitTwin:
    def test_shuffle_rolls_one_grid_value(self):
        result = build_variables_for_submit([dict(num(), name="age")], random_fn=lambda: 0.0)
        assert result == {"age": "18"}
        result = build_variables_for_submit([dict(num(), name="age")], random_fn=lambda: 0.999999)
        assert result == {"age": "80"}

    def test_shuffle_never_indexes_past_the_grid(self):
        result = build_variables_for_submit([dict(num(), name="age")], random_fn=lambda: 1.0)
        assert result == {"age": "80"}

    def test_pin_and_per_image(self):
        result = build_variables_for_submit([
            dict(num(mode="pin", value=40), name="a"),
            dict(num(mode="per-image", min=1, max=3), name="b"),
        ])
        assert result == {"a": "40", "b": "{1|2|3}"}

    def test_invalid_number_is_skipped(self):
        assert build_variables_for_submit([dict(num(min="x"), name="age")]) == {}

    def test_chat_roll_feeds_a_condition(self):
        from src.features.prompt.expander import expand_prompts

        variables = build_variables_for_submit([dict(num(mode="pin", value=70), name="age")])
        got = expand_prompts('<<if $age > 65>>grey<<else>>young<<end>>', "", count=1, base_seed=0, variables=variables)
        assert got[0].positive == "grey"


class TestRenderAndOperations:
    def test_render_line_shuffle(self):
        (line,) = render_prompt_variable_lines([dict(num(), name="age", lastRoll="42")])
        assert line == "age: number from 18 to 80 in steps of 1 — shuffles each generation; last roll: 42"

    def test_render_line_pin_and_per_image(self):
        lines = render_prompt_variable_lines([
            dict(num(mode="pin", value=30), name="a"),
            dict(num(mode="per-image"), name="b"),
        ])
        assert lines[0].endswith("pinned to 30")
        assert lines[1].endswith("re-rolls independently per image")

    def test_render_skips_an_invalid_number(self):
        assert render_prompt_variable_lines([dict(num(min="x"), name="age")]) == []

    def test_set_operation_builds_a_number_entry(self):
        result = apply_variable_operations([], [
            {"op": "set", "name": "age", "type": "number", "min": 18, "max": 80, "step": 1, "mode": "pin", "value": 30},
        ])
        assert result == [{"name": "age", "type": "number", "mode": "pin", "min": 18, "max": 80, "step": 1, "value": 30}]
        assert validate_variables_map({"age": result[0]}) == []


HOSTILE = [
    {"min": 0, "max": "1e100"},
    {"min": "1e30", "max": "1e30"},
    {"min": 0, "max": "1e999999999"},
    {"min": "-1e999999999", "max": 0},
    {"min": 0, "max": 10, "step": "1e-30"},
    {"min": 0, "max": 10, "step": "1e30"},
    {"min": 0, "max": 10, "step": 0},
    {"min": 0, "max": 10, "step": -1},
    {"min": 0, "max": 10, "step": "-1e-30"},
    {"min": 0, "max": 10, "value": "1e999999999"},
    {"min": "1" * 5000, "max": "1" * 5000},
    {"min": 0, "max": 10 ** 5000},
    {"min": 0, "max": 1.5e300},
    {"min": "NaN", "max": 1},
    {"min": 0, "max": "Infinity"},
    {"min": 0.1234567890123, "max": 1, "step": "0.0000001"},
    {"min": "999999999999.123456", "max": "999999999999.123456", "step": "0.000001"},
]


class TestHostileNumbers:
    @pytest.mark.parametrize("fields", HOSTILE, ids=[str(i) for i in range(len(HOSTILE))])
    def test_validate_returns_a_teaching_error(self, fields):
        errors = validate_variables_map({"age": {"type": "number", **fields}})
        assert len(errors) == 1 and errors[0].startswith("'age':")

    @pytest.mark.parametrize("fields", HOSTILE, ids=[str(i) for i in range(len(HOSTILE))])
    def test_grid_wire_and_render_never_raise(self, fields):
        var = {"name": "age", "type": "number", "mode": "pin", **fields}
        assert number_grid(var) is None
        assert build_variable_wire_value(var) == ""
        assert build_variables_for_submit([var]) == {}
        assert render_prompt_variable_lines([var]) == []

    @pytest.mark.parametrize("fields", HOSTILE, ids=[str(i) for i in range(len(HOSTILE))])
    def test_set_operation_then_validate_never_raises(self, fields):
        (entry,) = apply_variable_operations([], [{"op": "set", "name": "age", "type": "number", **fields}])
        assert len(validate_variables_map({"age": entry})) == 1

    def test_limits_boundary_is_accepted(self):
        assert validate_variables_map({"a": num(min=-999999999999, max=-999999999900, step=1)}) == []
        assert validate_variables_map({"a": num(min=0, max=0.000001, step=0.000001)}) == []
        assert validate_variables_map({"a": num(min=0, max=10 ** 12)}) != []
