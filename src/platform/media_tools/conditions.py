from __future__ import annotations

import math
import re
from typing import Any, Callable

OPERATORS = frozenset({
    "equals", "not_equals", "in", "not_in", "greater_than", "less_than",
    "greater_than_or_equals", "less_than_or_equals", "contains", "not_contains",
    "is_empty", "is_not_empty",
})

_RESERVED = {"field", "operator", "value"}
_NUMBER_PREFIX = re.compile(r"^\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _strict_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    return a == b


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    found = _NUMBER_PREFIX.match(str(value)) if value is not None else None
    return float(found.group(0)) if found else math.nan


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


_COMPARATORS: dict[str, Callable[[Any, Any], bool]] = {
    "equals": _strict_equal,
    "not_equals": lambda a, b: not _strict_equal(a, b),
    "in": lambda a, b: isinstance(b, list) and any(_strict_equal(a, item) for item in b),
    "not_in": lambda a, b: isinstance(b, list) and not any(_strict_equal(a, item) for item in b),
    "greater_than": lambda a, b: _number(a) > _number(b),
    "less_than": lambda a, b: _number(a) < _number(b),
    "greater_than_or_equals": lambda a, b: _number(a) >= _number(b),
    "less_than_or_equals": lambda a, b: _number(a) <= _number(b),
    "contains": lambda a, b: _text(b) in _text(a),
    "not_contains": lambda a, b: _text(b) not in _text(a),
    "is_empty": lambda a, b: _empty(a),
    "is_not_empty": lambda a, b: not _empty(a),
}


def _operator_and_value(when: dict) -> tuple[str | None, Any]:
    operator = when.get("operator")
    if operator is not None:
        return operator, when.get("value")
    for key in when:
        if key in OPERATORS and key not in _RESERVED:
            return key, when[key]
    return None, None


def when_matches(when: Any, values: dict[str, Any]) -> bool:
    if when is None:
        return True
    if isinstance(when, list):
        return all(when_matches(item, values) for item in when)
    if not isinstance(when, dict):
        return False
    if "logic" in when and "conditions" in when:
        conditions = when["conditions"] if isinstance(when["conditions"], list) else []
        if not conditions:
            return True
        results = (when_matches(item, values) for item in conditions)
        if when["logic"] == "AND":
            return all(results)
        if when["logic"] == "OR":
            return any(results)
        return False
    field = when.get("field")
    operator, expected = _operator_and_value(when)
    comparator = _COMPARATORS.get(operator or "")
    if field is None or comparator is None:
        return False
    try:
        return bool(comparator(values.get(field), expected))
    except Exception:
        return False
