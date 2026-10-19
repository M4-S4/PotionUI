import json
from pathlib import Path

import pytest

from src.platform.resources.prompt_variables import (
    apply_variable_operations,
    build_variable_wire_value,
    number_grid,
    render_prompt_variable_lines,
)

CORPUS = json.loads(
    (Path(__file__).resolve().parents[2] / "fixtures" / "prompt_conditions" / "number_variables.json").read_text(
        encoding="utf-8"
    )
)
CASES = CORPUS["cases"]


def test_corpus_header_and_unique_names():
    assert CORPUS["format"] == "prompt-conditions-corpus"
    assert CORPUS["kind"] == "number-variables"
    names = [c["name"] for c in CASES]
    assert len(names) == len(set(names))
    assert len(CASES) >= 30


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_python_matches_corpus(case):
    var = {"type": "number", **case["def"]}
    grid = number_grid(var)
    expected = case["expected"]
    assert (grid is not None) is expected["valid"]
    assert build_variable_wire_value(var) == expected["wire"]
    if grid is None:
        assert expected["count"] == 0
    else:
        assert grid.count == expected["count"]
        assert grid.format_index(0) == expected["first"]
        assert grid.format_index(grid.count - 1) == expected["last"]


def test_chat_snapshot_shaped_number_operation_round_trips():
    snapshot_entry = {
        "name": "age",
        "type": "number",
        "mode": "pin",
        "min": 18,
        "max": 80,
        "step": 1,
        "value": "99",
    }
    applied = apply_variable_operations([], [{"op": "set", **snapshot_entry}])
    assert applied == [snapshot_entry]
    assert build_variable_wire_value(applied[0]) == "80"
    assert render_prompt_variable_lines(applied) == [
        "age: number from 18 to 80 in steps of 1 — pinned to 80"
    ]


def test_chat_snapshot_without_value_pins_to_min():
    applied = apply_variable_operations(
        [], [{"op": "set", "name": "w", "type": "number", "mode": "shuffle", "min": 0.8, "max": 1.4, "step": 0.1}]
    )
    assert applied[0]["value"] is None
    assert build_variable_wire_value({**applied[0], "mode": "pin"}) == "0.8"
