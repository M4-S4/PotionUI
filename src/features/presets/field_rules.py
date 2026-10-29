from typing import Any, Dict, List

from src.platform.media_tools.params import ANCHORS, TRANSPARENT, coerce_value
from src.platform.media_tools.types import InvalidToolParams


def configuration_error(field_type: str, configuration: Any, default: Any) -> str:
    config = configuration if isinstance(configuration, dict) else {}
    if field_type == "segmented":
        return _segmented_error(config, default)
    if field_type == "anchor":
        return _anchor_error(default)
    if field_type == "color":
        return _color_error(config, default)
    return ""


def _segmented_error(config: Dict[str, Any], default: Any) -> str:
    raw = config.get("options")
    if not isinstance(raw, list) or not raw:
        return "a 'segmented' field needs a non-empty `configuration.options` list"
    values: List[Any] = []
    for entry in raw:
        if isinstance(entry, dict):
            if "value" not in entry:
                return f"segmented option {entry!r} needs a `value`"
            value = entry["value"]
        else:
            value = entry
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return f"segmented option value {value!r} must be a string or a number"
        values.append(value)
    if len({str(value) for value in values}) != len(values):
        return "segmented option values must be unique"
    if default is not None and str(default) not in {str(value) for value in values}:
        return f"default {default!r} is not one of the declared options"
    return ""


def _anchor_error(default: Any) -> str:
    if default is not None and default not in ANCHORS:
        return f"default for an 'anchor' field must be one of {', '.join(ANCHORS)}, got {default!r}"
    return ""


def _color_error(config: Dict[str, Any], default: Any) -> str:
    allow = config.get("allow_transparent", False)
    if not isinstance(allow, bool):
        return f"`allow_transparent` must be a boolean, got {allow!r}"
    if default is None:
        return ""
    if not isinstance(default, str):
        return f"default for a 'color' field must be a #RRGGBB string, got {type(default).__name__}"
    if default.strip().lower() == TRANSPARENT and not allow:
        return "default 'transparent' needs `configuration.allow_transparent: true`"
    try:
        coerce_value({"name": "default", "label": "default", "type": "color", "allow_transparent": allow}, default)
    except InvalidToolParams as exc:
        return exc.message
    return ""
