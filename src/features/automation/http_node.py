from typing import Any, Dict, List, Mapping, Optional, Tuple

from src.features.automation.expr import render_template
from src.features.automation.header_secrets import reveal_value
from src.features.cloud.redaction import is_sensitive_name
from src.features.automation.http_request import (
    DEFAULT_TIMEOUT_S,
    HttpResult,
    RequestSpec,
    Transport,
    aiohttp_transport,
    execute,
)


def _rows(raw: Any) -> List[Mapping[str, Any]]:
    return [row for row in raw if isinstance(row, Mapping)] if isinstance(raw, list) else []


def _render(value: Any, context: Dict[str, Any]) -> str:
    return render_template(str(value), context) if value else ""


def _pairs(raw: Any, context: Dict[str, Any], *, sealed: bool = False) -> Tuple[Tuple[str, str], ...]:
    pairs = []
    for row in _rows(raw):
        name = _render(row.get("name"), context).strip()
        if not name:
            continue
        value = row.get("value")
        if sealed:
            value = reveal_value(value, context=f"automation http header {name}")
        pairs.append((name, _render(value, context)))
    return tuple(pairs)


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _secret_values(raw: Any, context: Dict[str, Any]) -> Tuple[str, ...]:
    values = []
    for row in _rows(raw):
        name = _render(row.get("name"), context).strip()
        if name and row.get("secret") and not is_sensitive_name(name):
            revealed = reveal_value(row.get("value"), context=f"automation http header {name}")
            values.append(_render(revealed, context))
    return tuple(v for v in values if v)


def build_spec(config: Mapping[str, Any], context: Dict[str, Any]) -> RequestSpec:
    mode = str(config.get("body_mode") or "none")
    if mode == "form":
        body: Any = _pairs(config.get("body_form"), context)
    elif mode in ("json", "text"):
        body = _render(config.get("body"), context)
    else:
        body = None
    return RequestSpec(
        method=str(config.get("method") or "GET").upper(),
        url=_render(config.get("url"), context),
        query=_pairs(config.get("query"), context),
        headers=_pairs(config.get("headers"), context, sealed=True),
        body_mode=mode,
        body=body,
        timeout_s=_number(config.get("timeout_s"), DEFAULT_TIMEOUT_S),
        expected_status=str(config.get("expected_status") or "2xx"),
        retries=int(_number(config.get("retries"), 0)),
        follow_redirects=bool(config.get("follow_redirects")),
        allow_private_targets=bool(config.get("allow_private_targets")),
        secret_values=_secret_values(config.get("headers"), context),
    )


def build_output(result: HttpResult, pick_template: Optional[str], context: Dict[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "status": result.status,
        "ok": result.ok,
        "headers": result.headers,
        "body": result.body,
        "truncated": result.truncated,
        "url": result.url,
        "attempts": result.attempts,
        "pick": None,
    }
    if pick_template:
        output["pick"] = render_template(pick_template, {**context, "response": output})
    return output


async def perform(config: Mapping[str, Any], context: Dict[str, Any], transport: Optional[Transport] = None) -> Dict[str, Any]:
    result = await execute(build_spec(config, context), transport or aiohttp_transport)
    return build_output(result, config.get("response_pick"), context)
