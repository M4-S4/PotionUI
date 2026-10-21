import copy
from typing import Any, Dict, List, Optional

from src.features.cloud.redaction import is_sensitive_name
from src.platform.security.redaction import SECRET_MASK
from src.platform.security.secrets import ENVELOPE_PREFIX, get_secret_cipher

HTTP_REQUEST_TYPE = "action.http_request"


def _is_sealed(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(ENVELOPE_PREFIX)


def _http_nodes(graph: Any) -> List[Dict[str, Any]]:
    if not isinstance(graph, dict):
        return []
    return [n for n in graph.get("nodes") or [] if isinstance(n, dict) and n.get("type") == HTTP_REQUEST_TYPE]


def _header_rows(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    config = node.get("config")
    rows = config.get("headers") if isinstance(config, dict) else None
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _name_key(row: Dict[str, Any]) -> str:
    return str(row.get("name") or "").strip().lower()


def _wants_secret(row: Dict[str, Any]) -> bool:
    return bool(row.get("secret")) or is_sensitive_name(_name_key(row))


def protect_graph(graph: Dict[str, Any], existing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    protected = copy.deepcopy(graph)
    stored: Dict[str, Dict[str, Any]] = {}
    for node in _http_nodes(existing):
        stored[str(node.get("id"))] = {_name_key(row): row.get("value") for row in _header_rows(node)}
    cipher = get_secret_cipher()
    for node in _http_nodes(protected):
        previous = stored.get(str(node.get("id")), {})
        for row in _header_rows(node):
            value = row.get("value")
            if value == SECRET_MASK:
                kept = previous.get(_name_key(row))
                row["value"] = kept if _is_sealed(kept) else ""
            elif isinstance(value, str) and value and _wants_secret(row) and not _is_sealed(value):
                row["value"] = cipher.encrypt(value)
    return protected


def mask_config(config: Any) -> Any:
    if not isinstance(config, dict) or not isinstance(config.get("headers"), list):
        return config
    masked = dict(config)
    masked["headers"] = [
        {**row, "value": SECRET_MASK} if isinstance(row, dict) and _is_sealed(row.get("value")) else row
        for row in config["headers"]
    ]
    return masked


def mask_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    if not _http_nodes(graph):
        return graph
    masked = dict(graph)
    masked["nodes"] = [
        {**node, "config": mask_config(node.get("config"))} if node.get("type") == HTTP_REQUEST_TYPE else node
        for node in graph["nodes"]
    ]
    return masked


def reveal_value(value: Any, *, context: str) -> Any:
    if _is_sealed(value):
        return get_secret_cipher().decrypt(value, context=context)
    return value
