from typing import Any, Optional
from urllib.parse import quote

PUBLIC_BASE_URL_SETTING = "public_base_url"
REQUEST_ORIGIN_KEY = "request_origin"


def generation_details_path(generation_id: str) -> str:
    return f"/history?generation={quote(str(generation_id), safe='')}"


def resolve_public_base_url(settings: Any, request_origin: Optional[str] = None) -> str:
    configured = ""
    if settings is not None:
        try:
            configured = str(settings.get_setting(PUBLIC_BASE_URL_SETTING, "") or "").strip()
        except Exception:
            configured = ""
    return (configured or (request_origin or "").strip()).rstrip("/")


def generation_details_url(generation_id: str, settings: Any = None, request_origin: Optional[str] = None) -> str:
    return f"{resolve_public_base_url(settings, request_origin)}{generation_details_path(generation_id)}"
