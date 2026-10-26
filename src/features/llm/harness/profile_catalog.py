import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

from src.features.llm.harness.profile_schema import PROFILE_FIELDS
from src.features.llm.harness.profile_seed import SeedTable

MODELS_DEV_URL = "https://models.dev/api.json"

PROVIDER_ADAPTERS = {
    "anthropic": "anthropic-messages",
    "openai": "openai-responses",
    "google": "gemini-interactions",
    "deepseek": "openai-chat",
}

_CATALOG_HEADER = re.compile(r"^catalog:.*$", re.MULTILINE)
_PRICE_FIELDS = {
    "input": "input",
    "output": "output",
    "cache_read": "cached_input",
    "cache_write": "cache_write",
    "reasoning": "reasoning",
}
_FIELD_ORDER = {spec.name: index for index, spec in enumerate(PROFILE_FIELDS)}


class _Flow(dict):
    pass


class _CatalogDumper(yaml.SafeDumper):
    pass


def _represent_flow(dumper: yaml.SafeDumper, data: _Flow) -> Any:
    return dumper.represent_mapping("tag:yaml.org,2002:map", dict(data), flow_style=True)


def _represent_list(dumper: yaml.SafeDumper, data: list) -> Any:
    flat = all(item is None or isinstance(item, (str, int, float, bool)) for item in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flat)


_CatalogDumper.add_representer(_Flow, _represent_flow)
_CatalogDumper.add_representer(list, _represent_list)


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if value >= 1 else None


def _prices(cost: Any) -> Optional[_Flow]:
    if not isinstance(cost, dict):
        return None
    prices = _Flow()
    for source, target in _PRICE_FIELDS.items():
        value = cost.get(source)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            prices[target] = value
    return prices if "input" in prices and "output" in prices else None


def _reasoning(model: Mapping[str, Any]) -> Dict[str, Any]:
    if model.get("reasoning") is False:
        return {"reasoning": "none"}
    options = {
        option.get("type"): option for option in model.get("reasoning_options") or [] if isinstance(option, dict)
    }
    if "effort" in options:
        values = options["effort"].get("values") or []
        return {"reasoning": "effort", "reasoning_levels": [str(value) for value in values]}
    if "budget_tokens" in options:
        return {"reasoning": "budget"}
    if "toggle" in options:
        return {"reasoning": "toggle"}
    return {}


def _interleaved(model: Mapping[str, Any]) -> Dict[str, Any]:
    interleaved = model.get("interleaved")
    if isinstance(interleaved, dict):
        fields: Dict[str, Any] = {"interleaved_reasoning": True}
        if isinstance(interleaved.get("field"), str):
            fields["reasoning_field"] = interleaved["field"]
        return fields
    return {"interleaved_reasoning": True} if interleaved is True else {}


def _fields(model: Mapping[str, Any]) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    limit = model.get("limit") or {}
    if _positive_int(limit.get("context")):
        fields["context_window"] = _positive_int(limit["context"])
    if _positive_int(limit.get("output")):
        fields["max_output_tokens"] = _positive_int(limit["output"])
    if isinstance(model.get("tool_call"), bool):
        fields["tools"] = model["tool_call"]
    fields["vision"] = "image" in ((model.get("modalities") or {}).get("input") or [])
    fields.update(_reasoning(model))
    fields.update(_interleaved(model))
    if model.get("temperature") is False:
        fields["sampling_rejected"] = ["temperature"]
    prices = _prices(model.get("cost"))
    if prices is not None:
        fields["prices"] = prices
    return dict(sorted(fields.items(), key=lambda item: _FIELD_ORDER[item[0]]))


def _is_chat_model(model: Mapping[str, Any]) -> bool:
    modalities = model.get("modalities") or {}
    return (
        model.get("type") is None
        and "text" in (modalities.get("input") or [])
        and "text" in (modalities.get("output") or [])
    )


def build_catalog(models_dev: Mapping[str, Any]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for provider, adapter in PROVIDER_ADAPTERS.items():
        models = (models_dev.get(provider) or {}).get("models") or {}
        for model_id in sorted(models):
            model = models[model_id]
            if isinstance(model, dict) and _is_chat_model(model):
                entries.append({"adapter": adapter, "model": model_id, "fields": _fields(model)})
    return entries


def _flowed(entry: Mapping[str, Any]) -> Dict[str, Any]:
    fields = dict(entry["fields"])
    if isinstance(fields.get("prices"), dict):
        fields["prices"] = _Flow(fields["prices"])
    return {**entry, "fields": fields}


def render_seed(seed_text: str, catalog: Sequence[Mapping[str, Any]]) -> str:
    matches = _CATALOG_HEADER.findall(seed_text)
    if len(matches) != 1:
        raise ValueError("the seed table must have exactly one top-level catalog key")
    head = seed_text[: _CATALOG_HEADER.search(seed_text).start()]
    if not catalog:
        return f"{head}catalog: []\n"
    body = yaml.dump([_flowed(entry) for entry in catalog], Dumper=_CatalogDumper, sort_keys=False, default_flow_style=False, width=1000)
    indented = "".join(f"  {line}\n" if line else "\n" for line in body.splitlines())
    return f"{head}catalog:\n{indented}"


def refresh_seed(seed_text: str, models_dev: Mapping[str, Any]) -> str:
    rendered = render_seed(seed_text, build_catalog(models_dev))
    SeedTable.from_mapping(yaml.safe_load(rendered))
    return rendered
