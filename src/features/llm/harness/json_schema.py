import json
import re
from typing import Any, Callable, Dict, List, Mapping

MAX_REF_HOPS = 32
DEFINITION_KEYS = ("$defs", "definitions")
SCALAR_TYPES = frozenset({"string", "number", "integer", "boolean"})
COMBINATORS = ("anyOf", "oneOf", "allOf")

TYPE_CHECKS: Dict[str, Callable[[Any], bool]] = {
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "array": lambda value: isinstance(value, list),
    "object": lambda value: isinstance(value, dict),
    "null": lambda value: value is None,
}


def validate(value: Any, schema: Any, path: str = "$", root: Any = None) -> List[str]:
    root = schema if root is None else root
    schema = _resolve(schema, root)
    if not isinstance(schema, Mapping):
        return []
    types = schema.get("type")
    if types is not None:
        names = [types] if isinstance(types, str) else list(types)
        if not any(TYPE_CHECKS.get(name, lambda _: True)(value) for name in names):
            return [f"{path}: expected {' or '.join(names)}, got {_kind(value)}"]
    errors: List[str] = []
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: must equal {json.dumps(schema['const'])}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: must be one of {json.dumps(schema['enum'])}")
    for part in schema.get("allOf", ()):
        errors += validate(value, part, path, root)
    if "anyOf" in schema and not any(not validate(value, part, path, root) for part in schema["anyOf"]):
        errors.append(f"{path}: matches none of the allowed shapes")
    if "oneOf" in schema and sum(not validate(value, part, path, root) for part in schema["oneOf"]) != 1:
        errors.append(f"{path}: must match exactly one of the allowed shapes")
    if isinstance(value, str):
        errors += _check_string(value, schema, path)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        errors += _check_number(value, schema, path)
    elif isinstance(value, list):
        errors += _check_array(value, schema, path, root)
    elif isinstance(value, dict):
        errors += _check_object(value, schema, path, root)
    return errors


def _kind(value: Any) -> str:
    for name in ("boolean", "integer", "number", "string", "array", "object", "null"):
        if TYPE_CHECKS[name](value):
            return name
    return type(value).__name__


def _resolve(schema: Any, root: Any) -> Any:
    for _ in range(MAX_REF_HOPS):
        if not isinstance(schema, Mapping) or "$ref" not in schema:
            return schema
        ref = schema["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#"):
            raise ValueError(f"unsupported schema reference {ref!r}")
        node = root
        for token in (item for item in ref[1:].split("/") if item):
            node = node[token.replace("~1", "/").replace("~0", "~")]
        schema = node
    raise ValueError("schema references loop")


def _check_string(value: str, schema: Mapping[str, Any], path: str) -> List[str]:
    errors: List[str] = []
    if "minLength" in schema and len(value) < schema["minLength"]:
        errors.append(f"{path}: must have at least {schema['minLength']} characters")
    if "maxLength" in schema and len(value) > schema["maxLength"]:
        errors.append(f"{path}: must have at most {schema['maxLength']} characters")
    if "pattern" in schema and not re.search(schema["pattern"], value):
        errors.append(f"{path}: must match the pattern {schema['pattern']}")
    return errors


def _check_number(value: float, schema: Mapping[str, Any], path: str) -> List[str]:
    errors: List[str] = []
    if "minimum" in schema and value < schema["minimum"]:
        errors.append(f"{path}: must be at least {schema['minimum']}")
    if "maximum" in schema and value > schema["maximum"]:
        errors.append(f"{path}: must be at most {schema['maximum']}")
    return errors


def _check_array(value: list, schema: Mapping[str, Any], path: str, root: Any) -> List[str]:
    errors: List[str] = []
    if "minItems" in schema and len(value) < schema["minItems"]:
        errors.append(f"{path}: must have at least {schema['minItems']} items")
    if "maxItems" in schema and len(value) > schema["maxItems"]:
        errors.append(f"{path}: must have at most {schema['maxItems']} items")
    if isinstance(schema.get("items"), Mapping):
        for index, item in enumerate(value):
            errors += validate(item, schema["items"], f"{path}[{index}]", root)
    return errors


def _check_object(value: dict, schema: Mapping[str, Any], path: str, root: Any) -> List[str]:
    errors: List[str] = []
    properties = schema.get("properties") or {}
    for name in schema.get("required", ()):
        if name not in value:
            errors.append(f"{path}: missing required property {name!r}")
    for name, item in value.items():
        if name in properties:
            errors += validate(item, properties[name], f"{path}.{name}", root)
        elif schema.get("additionalProperties") is False:
            errors.append(f"{path}: unexpected property {name!r}")
        elif isinstance(schema.get("additionalProperties"), Mapping):
            errors += validate(item, schema["additionalProperties"], f"{path}.{name}", root)
    return errors


def strict_schema(schema: Mapping[str, Any]) -> Dict[str, Any]:
    return _strict(schema, schema)


def _strict(schema: Mapping[str, Any], root: Any) -> Dict[str, Any]:
    result = dict(schema)
    for key in COMBINATORS:
        if key in result:
            result[key] = [_strict(branch, root) if isinstance(branch, Mapping) else branch for branch in result[key]]
    for key in DEFINITION_KEYS:
        if key in result:
            result[key] = {name: _strict(item, root) for name, item in result[key].items()}
    if isinstance(result.get("items"), Mapping):
        result["items"] = _strict(result["items"], root)
    if isinstance(result.get("properties"), Mapping):
        required = set(result.get("required", ()))
        result["properties"] = {
            name: _strict(prop, root) if name in required else _nullable(_strict(prop, root), root)
            for name, prop in result["properties"].items()
        }
        result["required"] = list(result["properties"])
        if not isinstance(result.get("additionalProperties"), Mapping):
            result["additionalProperties"] = False
    return result


def prune_nulls(value: Any, schema: Any, root: Any = None) -> Any:
    root = schema if root is None else root
    schema = _resolve(schema, root)
    if not isinstance(schema, Mapping):
        return value
    for key in COMBINATORS:
        for branch in schema.get(key, ()):
            value = prune_nulls(value, branch, root)
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        required = set(schema.get("required", ()))
        pruned = {}
        for name, item in value.items():
            prop = properties.get(name)
            if prop is None:
                pruned[name] = item
            elif item is not None or name in required or _allows_null(prop, root):
                pruned[name] = prune_nulls(item, prop, root)
        return pruned
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        return [prune_nulls(item, schema["items"], root) for item in value]
    return value


def _allows_null(schema: Any, root: Any) -> bool:
    schema = _resolve(schema, root)
    if not isinstance(schema, Mapping):
        return True
    types = schema.get("type")
    if types == "null" or (isinstance(types, list) and "null" in types):
        return True
    if ("enum" in schema and None in schema["enum"]) or ("const" in schema and schema["const"] is None):
        return True
    return any(_allows_null(branch, root) for key in ("anyOf", "oneOf") for branch in schema.get(key, ()))


def _nullable(schema: Dict[str, Any], root: Any) -> Dict[str, Any]:
    if _allows_null(schema, root):
        return schema
    types = schema.get("type")
    names = [types] if isinstance(types, str) else types
    plain = isinstance(names, list) and set(names) <= SCALAR_TYPES and not any(key in schema for key in ("$ref", *COMBINATORS))
    if not plain:
        return {"anyOf": [schema, {"type": "null"}]}
    nullable = {**schema, "type": [*names, "null"]}
    if "enum" in schema:
        nullable["enum"] = [*schema["enum"], None]
    return nullable
