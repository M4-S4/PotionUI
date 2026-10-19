import json

import pytest

from src.features.llm.tools.base import ToolContext
from src.features.llm.tools.builtin.prompt_variables_tool import ManagePromptVariablesTool
from src.platform.resources.prompt_variables import apply_variable_operations


def _ctx(variables=None):
    return ToolContext(user_id="user-1", session_metadata={"form_state": {"variables": variables or []}})


def _age(**extra):
    return {"name": "age", "type": "number", "min": 18, "max": 90, "step": 1, "mode": "shuffle", "value": 18, **extra}


async def _preview(ops, variables=None):
    return await ManagePromptVariablesTool().execute(_ctx(variables), operations=ops)


async def _confirmed(ops, variables=None):
    result = await ManagePromptVariablesTool().execute_confirmed(_ctx(variables), operations=ops)
    assert result.success, result.error
    return json.loads(result.data)


@pytest.mark.asyncio
async def test_create_number_previews_range_step_and_mode():
    result = await _preview([{"op": "set", "name": "age", "type": "number", "min": 18, "max": 90, "mode": "per-image"}])
    assert result.success, result.error
    row = json.loads(result.data)["proposed_changes"][0]
    assert row["field_name"] == "${age}"
    assert row["new_value"] == "number from 18 to 90 in steps of 1 — re-rolls independently per image"


@pytest.mark.asyncio
async def test_confirmed_wire_op_carries_numbers_the_frontend_can_apply():
    payload = await _confirmed([{"op": "set", "name": "weight", "type": "number", "min": 0.5, "max": 1.5, "step": 0.25}])
    assert payload["operations"] == [
        {"op": "set", "name": "weight", "type": "number", "mode": "shuffle", "min": 0.5, "max": 1.5, "step": 0.25, "value": 0.5}
    ]


@pytest.mark.asyncio
async def test_pin_uses_value_and_clamps_into_range():
    payload = await _confirmed([{"op": "set", "name": "age", "type": "number", "min": 18, "max": 90, "mode": "pin", "value": 120}])
    assert payload["operations"][0]["value"] == 90
    assert payload["operations"][0]["mode"] == "pin"


@pytest.mark.asyncio
async def test_update_keeps_omitted_fields_of_an_existing_number():
    payload = await _confirmed([{"op": "set", "name": "age", "mode": "pin", "value": 70}], [_age()])
    op = payload["operations"][0]
    assert (op["type"], op["min"], op["max"], op["step"], op["mode"], op["value"]) == ("number", 18, 90, 1, "pin", 70)


@pytest.mark.asyncio
async def test_wire_ops_round_trip_through_apply_variable_operations():
    payload = await _confirmed([{"op": "set", "name": "age", "type": "number", "min": 18, "max": 90, "mode": "pin", "value": 66}])
    applied = apply_variable_operations([], payload["operations"])
    assert applied == [{"name": "age", "type": "number", "mode": "pin", "min": 18, "max": 90, "step": 1, "value": 66}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op,fragment",
    [
        ({"name": "age", "type": "number", "max": 90}, "needs a numeric min"),
        ({"name": "age", "type": "number", "min": 90, "max": 18}, "greater than max"),
        ({"name": "age", "type": "number", "min": 0, "max": 1000}, "more than the limit of 200"),
        ({"name": "age", "type": "number", "min": 1, "max": 5, "step": 0}, "step must be a number greater than 0"),
        ({"name": "age", "type": "number", "min": 1, "max": 5, "mode": "random"}, "invalid mode"),
        ({"name": "age", "type": "number", "min": "abc", "max": 5}, "needs a numeric min"),
    ],
)
async def test_number_limits_reuse_the_shared_validator(op, fragment):
    result = await _preview([{"op": "set", **op}])
    assert not result.success
    assert fragment in result.error


@pytest.mark.asyncio
async def test_remove_describes_a_number_variable():
    result = await _preview([{"op": "remove", "name": "age"}], [_age()])
    row = json.loads(result.data)["proposed_changes"][0]
    assert row["old_value"] == "number from 18 to 90 in steps of 1 — shuffles each generation"


def test_schema_offers_number_fields():
    props = ManagePromptVariablesTool().parameters["properties"]["operations"]["items"]["properties"]
    assert "number" in props["type"]["enum"]
    assert {"min", "max", "step"} <= set(props)
