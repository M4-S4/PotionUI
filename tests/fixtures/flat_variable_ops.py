from typing import Any, Dict, List

from src.features.tab_drafts.validation import flat_definition


def apply_variable_operations(variables: Any, operations: Any) -> List[Dict[str, Any]]:
    by_name = {v["name"]: dict(v) for v in variables or []}
    for op in operations or []:
        if op.get("op") == "remove":
            by_name.pop(op["name"], None)
        elif op.get("op") == "set":
            by_name[op["name"]] = {**flat_definition(op), "name": op["name"]}
    return list(by_name.values())
