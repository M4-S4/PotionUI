import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Set

import yaml

_FULL_VAR_RE = re.compile(r"^\s*\{\{\s*vars\.(\w+)\s*\}\}\s*$")
_EMBEDDED_VAR_RE = re.compile(r"\{\{\s*vars\.(\w+)\s*\}\}")
_SCALARS = (str, int, float, bool)


class ChildrenVarsError(ValueError):
    pass


@dataclass
class ChildrenRef:
    file: str
    vars: Dict[str, Any] = field(default_factory=dict)


def parse_children_ref(children: Any) -> Optional[ChildrenRef]:
    if isinstance(children, str):
        return ChildrenRef(children)
    if not isinstance(children, dict):
        return None
    unknown = set(children) - {"file", "vars"}
    if unknown:
        raise ChildrenVarsError(f"children include: unknown keys {sorted(unknown)}")
    file = children.get("file")
    if not isinstance(file, str) or not file:
        raise ChildrenVarsError("children include: 'file' must be a non-empty string")
    values = children.get("vars") or {}
    if not isinstance(values, dict):
        raise ChildrenVarsError(f"children include {file}: 'vars' must be a mapping")
    for key, value in values.items():
        if not isinstance(value, _SCALARS):
            raise ChildrenVarsError(
                f"children include {file}: var '{key}' must be a scalar, got {type(value).__name__}"
            )
    return ChildrenRef(file, dict(values))


def _contains_include(node: Any) -> bool:
    if isinstance(node, dict):
        if isinstance(node.get("children"), (str, dict)):
            return True
        return any(_contains_include(v) for v in node.values())
    if isinstance(node, list):
        return any(_contains_include(v) for v in node)
    return False


def _substitute(node: Any, values: Dict[str, Any], used: Set[str], missing: Set[str]) -> Any:
    if isinstance(node, dict):
        return {k: _substitute(v, values, used, missing) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute(v, values, used, missing) for v in node]
    if not isinstance(node, str):
        return node

    def lookup(name: str) -> Any:
        if name not in values:
            missing.add(name)
            return None
        used.add(name)
        return values[name]

    full = _FULL_VAR_RE.match(node)
    if full:
        return lookup(full.group(1))
    return _EMBEDDED_VAR_RE.sub(lambda m: str(lookup(m.group(1))), node)


def _apply_children_vars(data: Any, ref: ChildrenRef, source: Path) -> Any:
    if not ref.vars:
        return data
    if _contains_include(data):
        raise ChildrenVarsError(f"{source}: a fragment included with vars cannot include another fragment")
    used: Set[str] = set()
    missing: Set[str] = set()
    result = _substitute(data, ref.vars, used, missing)
    if missing:
        raise ChildrenVarsError(f"{source}: missing vars {sorted(missing)}")
    unknown = set(ref.vars) - used
    if unknown:
        raise ChildrenVarsError(f"{source}: unknown vars {sorted(unknown)}")
    return result


def read_children_fragment(path: Path, ref: ChildrenRef) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return _apply_children_vars(data, ref, path)
