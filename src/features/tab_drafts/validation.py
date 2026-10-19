import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from src.features.tab_drafts.errors import TabDraftInvalid, TabDraftTooLarge
from src.platform.resources.prompt_variables import (
    MAX_CHOICE_OPTIONS,
    MAX_OPTION_CHARS,
    MAX_TEXT_VALUE_CHARS,
    name_error,
    validate_variables_map,
)

MAX_DOC_BYTES = 256 * 1024
MAX_OPERATIONS = 500
MAX_DRAFTS_PER_USER = 200
MAX_OPTIONS = MAX_CHOICE_OPTIONS
MAX_TEXT_CHARS = MAX_TEXT_VALUE_CHARS
MAX_TAB_ID_CHARS = 64
TAB_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
DOC_KEYS = {"variables"}


def check_tab_id(tab_id: Any) -> str:
    if not isinstance(tab_id, str) or not tab_id or len(tab_id) > MAX_TAB_ID_CHARS or not TAB_ID_RE.match(tab_id):
        raise TabDraftInvalid(
            f"tab_id must be 1-{MAX_TAB_ID_CHARS} letters, digits, dashes or underscores."
        )
    return tab_id


def doc_size(doc: Dict[str, Any]) -> int:
    return len(json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def check_doc_size(doc: Dict[str, Any]) -> None:
    size = doc_size(doc)
    if size > MAX_DOC_BYTES:
        raise TabDraftTooLarge(size, MAX_DOC_BYTES)


def _cap_errors(variables: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for name, definition in variables.items():
        if not isinstance(definition, dict):
            continue
        kind = definition.get("type")
        if kind == "text":
            value = definition.get("value")
            if isinstance(value, str) and len(value) > MAX_TEXT_CHARS:
                errors.append(f"'{name}': a text value is limited to {MAX_TEXT_CHARS} characters.")
        elif kind == "choice":
            options = definition.get("options")
            if not isinstance(options, (list, tuple)):
                continue
            if len(options) > MAX_OPTIONS:
                errors.append(f"'{name}': a choice variable is limited to {MAX_OPTIONS} options.")
            for option in options:
                text = option.get("text") if isinstance(option, dict) else option
                if isinstance(text, str) and len(text) > MAX_OPTION_CHARS:
                    errors.append(f"'{name}': an option is limited to {MAX_OPTION_CHARS} characters.")
                    break
    return errors


def variable_errors(variables: Dict[str, Any]) -> List[str]:
    return validate_variables_map(variables) + _cap_errors(variables)


def new_variable_errors(before: Dict[str, Any], after: Dict[str, Any]) -> List[str]:
    known = Counter(variable_errors(before))
    fresh: List[str] = []
    for error in variable_errors(after):
        if known[error] > 0:
            known[error] -= 1
        else:
            fresh.append(error)
    return fresh


def normalize_doc(doc: Any) -> Dict[str, Any]:
    if not isinstance(doc, dict):
        raise TabDraftInvalid("doc must be an object.")
    unknown = sorted(str(key) for key in doc if key not in DOC_KEYS)
    if unknown:
        raise TabDraftInvalid(f"Unknown doc fields: {', '.join(unknown)}.")
    raw = doc.get("variables")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TabDraftInvalid("variables must be an object mapping name to definition.")
    variables: Dict[str, Any] = {}
    for key, definition in raw.items():
        name = key.strip() if isinstance(key, str) else ""
        if name in variables:
            raise TabDraftInvalid(f"Duplicate variable name '{name}'.")
        variables[name] = definition
    errors = variable_errors(variables)
    if errors:
        raise TabDraftInvalid("; ".join(errors))
    normalized = {"variables": variables}
    check_doc_size(normalized)
    return normalized


def flat_definition(op: Dict[str, Any]) -> Dict[str, Any]:
    definition = _flat_definition(op)
    if definition["type"] != "text" and op.get("linked") is True:
        definition["linked"] = True
    return definition


def _flat_definition(op: Dict[str, Any]) -> Dict[str, Any]:
    if op.get("type") == "choice":
        mode = op.get("mode") or "shuffle"
        return {
            "type": "choice",
            "options": op.get("options") or ["", ""],
            "mode": mode,
            "pinnedIndex": op.get("pinned_index") if mode == "pin" else None,
        }
    if op.get("type") == "number":
        return {
            "type": "number",
            "mode": op.get("mode") or "shuffle",
            "min": op.get("min"),
            "max": op.get("max"),
            "step": op.get("step", 1),
            "value": op.get("value"),
        }
    return {"type": "text", "value": op.get("value") or ""}


def parse_operations(operations: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not isinstance(operations, list) or not operations:
        raise TabDraftInvalid("ops must be a non-empty list.")
    if len(operations) > MAX_OPERATIONS:
        raise TabDraftInvalid(f"A batch is limited to {MAX_OPERATIONS} operations.")
    parsed: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for index, op in enumerate(operations):
        problems: List[str] = []
        kind = op.get("op") if isinstance(op, dict) else None
        raw_name = op.get("name") if isinstance(op, dict) else None
        name = raw_name.strip() if isinstance(raw_name, str) else ""
        if not isinstance(op, dict):
            problems.append("Operation must be an object.")
        elif kind not in ("set", "remove"):
            problems.append(f"Unknown operation '{kind}'. Use set or remove.")
        elif not name:
            problems.append("Empty variable name.")
        else:
            problem = name_error(name)
            if problem:
                problems.append(problem)
            elif kind == "set" and op.get("type") not in (None, "text", "choice", "number"):
                problems.append(f"'{name}': invalid type '{op.get('type')}'. Use text, choice or number.")
        if problems:
            failures.append({"index": index, "name": name or None, "op": kind, "errors": problems})
            continue
        parsed.append(
            {"index": index, "op": kind, "name": name, "def": flat_definition(op) if kind == "set" else None}
        )
    return parsed, failures


def apply_operations(
    current: Dict[str, Any], operations: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    before = dict(current.get("variables") or {})
    after = dict(before)
    for op in operations:
        if op["op"] == "remove":
            after.pop(op["name"], None)
        else:
            after[op["name"]] = op["def"]
    failures: List[Dict[str, Any]] = []
    by_name: Dict[str, Optional[Dict[str, Any]]] = {}
    for op in operations:
        by_name[op["name"]] = op
    for error in new_variable_errors(before, after):
        owner = next((op for name, op in by_name.items() if error.startswith(f"'{name}'")), None)
        failures.append(
            {
                "index": owner["index"] if owner else None,
                "name": owner["name"] if owner else None,
                "op": owner["op"] if owner else None,
                "errors": [error],
            }
        )
    return {**current, "variables": after}, failures
