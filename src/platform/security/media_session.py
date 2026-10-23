import hashlib
import ipaddress
import time
from pathlib import Path
from typing import Optional

from fastapi import Request, Response
from jose import JWTError, jwt

MEDIA_COOKIE = "potionui_media"
MEDIA_COOKIE_PATH = "/api"
PARTITIONED_SUFFIX = "_p"


def media_cookie_name() -> str:
    from src.platform.security.current_user import get_auth

    try:
        secret = get_auth().tokens._config.secret_key
    except Exception:
        return MEDIA_COOKIE
    if not isinstance(secret, str) or not secret:
        return MEDIA_COOKIE
    identity = f"{secret}\0{_instance_location()}"
    return f"{MEDIA_COOKIE}_{hashlib.sha256(identity.encode()).hexdigest()[:12]}"


def partitioned_media_cookie_name() -> str:
    return f"{media_cookie_name()}{PARTITIONED_SUFFIX}"


def _instance_location() -> str:
    from src.platform.database.database import db

    try:
        return str(Path(db.db_path).resolve())
    except Exception:
        return ""


def _is_secure(request: Request) -> bool:
    forwarded = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    return request.url.scheme == "https" or forwarded == "https"


def token_lifetime_seconds(token: str, now: Optional[float] = None) -> int:
    try:
        expires = jwt.get_unverified_claims(token).get("exp")
    except JWTError:
        return 0
    if not isinstance(expires, (int, float)):
        return 0
    return max(0, int(expires - (time.time() if now is None else now)))


def _is_loopback_host(request: Request) -> bool:
    host = request.url.hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _accepts_partitioned(request: Request) -> bool:
    return _is_secure(request) or _is_loopback_host(request)


def _cookie_header(name: str, value: str, partitioned: bool, **attributes) -> str:
    probe = Response()
    probe.set_cookie(name, value, path=MEDIA_COOKIE_PATH, httponly=True, **attributes)
    header = probe.headers["set-cookie"]
    return f"{header}; Partitioned" if partitioned else header


def _cookie_headers(request: Request, value: str, **attributes) -> list[str]:
    headers = [
        _cookie_header(
            media_cookie_name(), value, False,
            samesite="lax", secure=_is_secure(request), **attributes,
        )
    ]
    if _accepts_partitioned(request):
        headers.append(
            _cookie_header(
                partitioned_media_cookie_name(), value, True,
                samesite="none", secure=True, **attributes,
            )
        )
    return headers


def media_cookie_headers(request: Request, token: str) -> list[str]:
    lifetime = token_lifetime_seconds(token)
    if lifetime <= 0:
        return []
    return _cookie_headers(request, token, max_age=lifetime)


def set_media_cookie(response: Response, request: Request, token: Optional[str]) -> None:
    headers = media_cookie_headers(request, token) if token else []
    if not headers:
        clear_media_cookie(response, request)
        return
    for header in headers:
        response.headers.append("set-cookie", header)


def clear_media_cookie(response: Response, request: Request) -> None:
    for header in _cookie_headers(request, "", max_age=0, expires=0):
        response.headers.append("set-cookie", header)


def media_cookie_token(request: Request) -> Optional[str]:
    return (
        request.cookies.get(media_cookie_name())
        or request.cookies.get(partitioned_media_cookie_name())
        or None
    )
