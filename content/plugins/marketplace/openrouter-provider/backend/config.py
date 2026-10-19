from typing import ClassVar, List, Optional
from urllib.parse import urlsplit

from pydantic import Field, field_validator

from src.plugin_api.cloud import CloudBackendConfig

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterConfig(CloudBackendConfig):
    driver: str = Field(default="cloud.openrouter")
    base_url: str = Field(
        default=DEFAULT_BASE_URL,
        title="API address",
        description="Change only to reach OpenRouter through a proxy",
    )
    app_title: str = Field(
        default="PotionUI",
        title="App name",
        description="Shown to OpenRouter as the name of this app in its usage statistics",
    )
    app_url: str = Field(
        default="https://potionui.com",
        title="App address",
        description="Shown to OpenRouter as the address of this app, so its usage appears on OpenRouter's app rankings. Leave empty to send none.",
    )
    upstream_providers: str = Field(
        default="",
        title="Allowed upstream providers",
        description="Comma separated. When set, OpenRouter may only route requests to these providers. Leave empty to allow all.",
    )

    @field_validator("app_url")
    @classmethod
    def _app_url_is_a_web_address(cls, value: str) -> str:
        text = value.strip()
        if not text:
            return ""
        parts = urlsplit(text)
        if not text.isascii() or not text.isprintable() or " " in text or parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("Enter the app address as a web address such as https://example.com, or leave it empty.")
        return text

    engine_label: ClassVar[Optional[str]] = "OpenRouter"
    accepts_user_hash: ClassVar[bool] = True
    api_key_source: ClassVar[str] = "Created at openrouter.ai under Keys."

    def upstream_list(self) -> List[str]:
        return [name.strip() for name in self.upstream_providers.split(",") if name.strip()]
