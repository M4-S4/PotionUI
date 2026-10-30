from typing import Any, Dict, List

from src.platform.media_tools.params import coerce_value
from src.platform.media_tools.types import InvalidToolParams

VALUE_LABEL = "Value"


def coerce_field_value(param: Dict[str, Any], value: Any) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        return value
    try:
        return coerce_value({"name": "value", "label": VALUE_LABEL, **param}, value)
    except InvalidToolParams as exc:
        raise ValueError(exc.message) from exc


def normalize_options(raw: Any) -> List[Dict[str, Any]]:
    options: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return options
    for entry in raw:
        if isinstance(entry, dict):
            if "value" not in entry or isinstance(entry["value"], (list, dict)):
                continue
            option = {"label": entry.get("label") or str(entry["value"]), "value": entry["value"]}
            if entry.get("disabled") is True:
                option["disabled"] = True
            options.append(option)
        elif isinstance(entry, (str, int, float)) and not isinstance(entry, bool):
            options.append({"label": str(entry), "value": entry})
    return options
