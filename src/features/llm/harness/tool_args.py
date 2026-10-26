import json
from typing import Any, Dict, Optional

from src.features.llm.harness.assembler import parse_args
from src.features.llm.harness.messages import ToolCallPart


def arguments_text(call: ToolCallPart) -> str:
    if call.args is None:
        return "{}"
    raw = call.args_raw
    if raw is not None and raw.strip() and parse_args(raw) == call.args:
        return raw
    return json.dumps(call.args)


def arguments_object(call: ToolCallPart) -> Dict[str, Any]:
    return call.args if call.args is not None else {}


def object_from_wire(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return parse_args(value)
    return None
