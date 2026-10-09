from typing import ClassVar, Optional

from pydantic import Field, field_validator

from src.plugin_api.cloud import CloudBackendConfig

REGIONS = {
    "global": "https://api.bfl.ai/v1",
    "eu": "https://api.eu.bfl.ai/v1",
    "us": "https://api.us.bfl.ai/v1",
}
DEFAULT_REGION = "global"


class BflConfig(CloudBackendConfig):
    driver: str = Field(default="cloud.bfl")
    region: str = Field(
        default=DEFAULT_REGION,
        title="Region",
        description="global lets BFL pick the nearest cluster and fail over. eu keeps every request in the EU, us in the US.",
        json_schema_extra={"options": list(REGIONS)},
    )
    base_url: str = Field(
        default="",
        title="API address",
        description="Leave empty to use the region above. Set it only to reach BFL through a proxy, for example https://proxy.example/v1.",
    )
    timeout_seconds: int = Field(
        default=600,
        ge=30,
        le=3600,
        title="Timeout (seconds)",
        description="Longest a single picture may take before PotionUI stops waiting. BFL keeps a finished picture for 10 minutes.",
    )
    send_user_hash: bool = Field(
        default=False,
        title="Send an anonymous user id",
        description="Send an anonymous id per user so BFL can tell users apart without knowing who they are. It cannot be traced back to a person without this server's secret key.",
    )

    engine_label: ClassVar[Optional[str]] = "Black Forest Labs"
    api_key_source: ClassVar[str] = "Created at api.bfl.ai under API Keys."

    @field_validator("region")
    @classmethod
    def _known_region(cls, value: str) -> str:
        text = (value or "").strip().lower() or DEFAULT_REGION
        if text not in REGIONS:
            raise ValueError(f"Choose one of the regions: {', '.join(REGIONS)}.")
        return text

    def api_address(self) -> str:
        return self.base_url or REGIONS[self.region]
