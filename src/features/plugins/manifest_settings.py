import logging
from typing import Any, Dict, List, Optional

from src.features.plugins.repository import PluginRepository
from src.platform.plugins.manifest import SettingSpec
from src.platform.plugins.registry import PluginRegistry

logger = logging.getLogger(__name__)

TRUE_WORDS = ("1", "true", "yes", "on")
VALUE_TYPES = ("string", "number", "boolean", "select")


def _specs(manifest: Any) -> List[SettingSpec]:
    return [
        raw if isinstance(raw, SettingSpec) else SettingSpec.model_validate(raw)
        for raw in manifest.settings or []
    ]


def secret_setting_keys(registry: PluginRegistry, plugin_id: str) -> Optional[List[str]]:
    manifest = registry.get_plugin(plugin_id)
    if manifest is None:
        return None
    return [spec.name for spec in _specs(manifest) if spec.is_secret]


def _bound(spec: SettingSpec, name: str) -> Optional[float]:
    value = (spec.model_extra or {}).get(name)
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _option_values(spec: SettingSpec) -> List[str]:
    options = (spec.model_extra or {}).get("options") or []
    return [
        str(option["value"] if isinstance(option, dict) else option)
        for option in options
        if not isinstance(option, dict) or "value" in option
    ]


def _coerce_number(plugin_id: str, spec: SettingSpec, raw: str) -> Any:
    as_int = isinstance(spec.default, int) and not isinstance(spec.default, bool)
    try:
        number = int(raw) if as_int else float(raw)
    except ValueError:
        return spec.default
    low, high = _bound(spec, "min"), _bound(spec, "max")
    clamped = number
    if low is not None and clamped < low:
        clamped = low
    if high is not None and clamped > high:
        clamped = high
    if clamped != number:
        clamped = int(clamped) if as_int else float(clamped)
        logger.warning(
            "Plugin %s setting %s=%s is outside %s..%s; using %s",
            plugin_id, spec.name, raw, low, high, clamped,
        )
    return clamped


def _coerce(plugin_id: str, spec: SettingSpec, raw: Optional[str]) -> Any:
    if raw is None:
        return spec.default
    if spec.type == "string":
        keeps_empty = spec.default is None or (spec.model_extra or {}).get("allow_empty")
        return raw if raw or keeps_empty else spec.default
    text = raw.strip()
    if not text:
        return spec.default
    if spec.type == "boolean":
        return text.lower() in TRUE_WORDS
    if spec.type == "select":
        wanted = text.lower()
        return next((value for value in _option_values(spec) if value.lower() == wanted), spec.default)
    return _coerce_number(plugin_id, spec, text)


def resolve_plugin_settings(
    repo: PluginRepository, registry: PluginRegistry, plugin_id: str
) -> Dict[str, Any]:
    manifest = registry.get_plugin(plugin_id)
    if manifest is None:
        return {}
    stored = {row.setting_key: row.setting_value for row in repo.get_plugin_settings(plugin_id)}
    return {
        spec.name: _coerce(plugin_id, spec, stored.get(spec.name))
        for spec in _specs(manifest)
        if spec.type in VALUE_TYPES
    }


def settings_json_schema(registry: PluginRegistry, plugin_id: str) -> Optional[Dict[str, Any]]:
    manifest = registry.get_plugin(plugin_id)
    if manifest is None:
        return None
    properties: Dict[str, Any] = {}
    specs = _specs(manifest)
    for spec in specs:
        if spec.type not in VALUE_TYPES:
            continue
        prop: Dict[str, Any] = {"type": "string" if spec.type == "select" else spec.type}
        if spec.label:
            prop["title"] = spec.label
        if spec.description:
            prop["description"] = spec.description
        if spec.default is not None:
            prop["default"] = spec.default
        if spec.is_secret:
            prop["format"] = "password"
        if spec.type == "select":
            prop["enum"] = _option_values(spec)
        if spec.type == "number":
            for name, key in (("min", "minimum"), ("max", "maximum")):
                if _bound(spec, name) is not None:
                    prop[key] = _bound(spec, name)
        properties[spec.name] = prop
    return {
        "type": "object",
        "properties": properties,
        "required": [spec.name for spec in specs if spec.required],
    }
