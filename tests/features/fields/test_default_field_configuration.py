from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import yaml

from src.features.fields.field_factory import DefaultField
from src.features.forms.binding import _pattern_violation

YUE2_TAB = Path("content/presets/marketplace/YuE2/modes/song/tabs/advanced.yml")

ABC = "X:1\nT:Song\nM:4/4\nL:1/8\nK:C\n\"C\"E2 G2 c2 G2 | \"F\"A2 c2 \"G\"B4 |]\n"


def yue2_abc_configuration():
    def walk(node):
        if isinstance(node, dict):
            if node.get("name") == "abc":
                return node["configuration"]
            for value in node.values():
                found = walk(value)
                if found:
                    return found
        elif isinstance(node, list):
            for value in node:
                found = walk(value)
                if found:
                    return found
        return None

    return walk(yaml.safe_load(YUE2_TAB.read_text()))


def test_configuration_reaches_the_schema_of_a_default_typed_field():
    field = DefaultField(Mock())
    schema = field.output(
        {"type": "string", "name": "n", "configuration": {"placeholder": "p", "pattern": "^a$"}}
    )
    assert schema["configuration"] == {"placeholder": "p", "pattern": "^a$"}


def test_a_field_without_configuration_keeps_the_old_schema():
    schema = DefaultField(Mock()).output({"type": "number", "name": "n", "label": "N", "default": 3})
    assert "configuration" not in schema
    assert "min" not in schema and "max" not in schema


def test_number_bounds_are_not_moved_by_the_configuration_block():
    schema = DefaultField(Mock()).output(
        {"type": "number", "name": "seed", "default": "-1", "configuration": {"min": -1}}
    )
    assert schema["default"] == "-1"
    assert "min" not in schema


def test_the_yue2_abc_pattern_accepts_real_multiline_abc_and_rejects_the_rest():
    config = yue2_abc_configuration()
    assert _pattern_violation(ABC, config) is None
    assert _pattern_violation("X:1\nCDEF|", config) == config["pattern_message"]
    assert _pattern_violation("T:X: inline\nK:C", config) == config["pattern_message"]
