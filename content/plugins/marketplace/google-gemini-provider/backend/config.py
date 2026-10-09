from typing import ClassVar, Optional

from pydantic import Field

from src.plugin_api.cloud import CloudBackendConfig

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GoogleConfig(CloudBackendConfig):
    driver: str = Field(default="cloud.google")
    base_url: str = Field(
        default=DEFAULT_BASE_URL,
        title="API address",
        description="Change only to reach the Gemini API through a proxy",
    )

    engine_label: ClassVar[Optional[str]] = "Google Gemini"
    api_key_source: ClassVar[str] = "Created in Google AI Studio (aistudio.google.com) under Get API key."
