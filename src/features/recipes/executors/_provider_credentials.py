from __future__ import annotations

from typing import Any, Dict, Optional

from src.features.recipes.executors._async_bridge import run_sync


def resolve_provider_registry(factory=None):
    """Best-effort provider registry resolution - `None` on any failure (no
    provider plugin installed, discovery not wired, ...) so a caller can
    degrade gracefully rather than crash a recipe step over an optional
    lookup. `factory` is the injectable seam tests use; defaults to the real
    module-level registry's async discovery."""
    try:
        if factory is not None:
            return run_sync(factory())
        from src.features.providers.registry import ensure_providers_discovered

        return run_sync(ensure_providers_discovered())
    except Exception:
        return None


def credential_prompt_for_provider(registry: Any, provider_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if registry is None or not provider_id:
        return None
    metadata = registry.get_provider_metadata(provider_id)
    if metadata is None:
        return None
    secret_keys = registry.get_provider_secret_keys(provider_id)
    if not secret_keys:
        return None
    field_name = secret_keys[0]
    current = registry.get_provider_current_settings(provider_id) or {}
    return {
        "id": provider_id,
        "name": metadata.name,
        "website": metadata.website,
        "field_name": field_name,
        "configured": bool(current.get(field_name)),
    }
