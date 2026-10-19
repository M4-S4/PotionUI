from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel

DISCOVERY_TIMEOUT_SECONDS = 8.0
CONTEXT_WINDOW_PROBE_TIMEOUT_SECONDS = 3.0


class DiscoveredModel(BaseModel):
    id: str
    label: Optional[str] = None
    size: Optional[int] = None
    modified_at: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class DiscoveredWindow(BaseModel):
    tokens: int
    kind: Literal["runtime", "trained"]


class ModelListingError(Exception):
    pass


def normalize_base_url(base_url: str) -> str:
    url = (base_url or "").strip().rstrip("/")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ModelListingError("Base URL must be an http:// or https:// address.")
    return url


async def fetch_json(
    provider: str,
    base_url: str,
    path: str,
    headers: Optional[Dict[str, str]] = None,
    *,
    body: Optional[Dict[str, Any]] = None,
    timeout: float = DISCOVERY_TIMEOUT_SECONDS,
) -> Any:
    url = normalize_base_url(base_url)
    target = f"{provider} at {url}"
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            if body is None:
                response = await client.get(f"{url}{path}", headers=headers or {})
            else:
                response = await client.post(f"{url}{path}", headers=headers or {}, json=body)
    except httpx.ConnectError:
        raise ModelListingError(f"Couldn't reach {target}: connection refused.")
    except httpx.TimeoutException:
        raise ModelListingError(
            f"Couldn't reach {target}: timed out after {timeout:g}s."
        )
    except httpx.HTTPError as exc:
        raise ModelListingError(f"Couldn't reach {target}: {type(exc).__name__}.")
    if response.status_code in (401, 403):
        raise ModelListingError(
            f"{provider} at {url} rejected the credentials (HTTP {response.status_code}). Check the API key."
        )
    if response.status_code == 404:
        raise ModelListingError(f"{provider} at {url} has no model list at {path} (HTTP 404). Check the base URL.")
    if response.status_code >= 400:
        raise ModelListingError(f"{provider} at {url} answered HTTP {response.status_code}.")
    try:
        return response.json()
    except ValueError:
        raise ModelListingError(f"{provider} at {url} returned a response that isn't JSON.")


def sorted_models(models: List[DiscoveredModel]) -> List[DiscoveredModel]:
    return sorted(models, key=lambda m: m.id.lower())


def positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if value >= 1 else None
