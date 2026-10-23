from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from src.plugin_api import PluginRepository, plugin_settings

PLUGIN_ID = "oidc-auth"

LOOPBACK_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1"})


def _normalize_issuer(raw: Optional[str]) -> Optional[str]:
    value = (raw or "").strip().rstrip("/")
    return value or None


@dataclass(frozen=True)
class OIDCSettings:
    issuer_url: Optional[str]
    client_id: Optional[str]
    client_secret: Optional[str]
    scopes: str
    label: str
    redirect_uri_override: Optional[str]
    auto_redirect: bool = False

    def is_configured(self) -> bool:
        if not (self.issuer_url and self.client_id and self.client_secret):
            return False
        parsed = urlparse(self.issuer_url)
        if parsed.scheme == "https":
            return True
        return parsed.scheme == "http" and parsed.hostname in LOOPBACK_HOSTNAMES


def load_settings(repository: Optional[PluginRepository] = None) -> OIDCSettings:
    values = plugin_settings(PLUGIN_ID, repository)

    return OIDCSettings(
        issuer_url=_normalize_issuer(values["issuer_url"]),
        client_id=(values["client_id"] or "").strip() or None,
        client_secret=(values["client_secret"] or "").strip() or None,
        scopes=values["scopes"],
        label=values["label"],
        redirect_uri_override=(values["redirect_uri"] or "").strip() or None,
        auto_redirect=bool(values["auto_redirect"]),
    )
