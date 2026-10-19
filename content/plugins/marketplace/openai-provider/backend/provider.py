import asyncio
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Mapping, Optional

from src.plugin_api.cloud import (
    CloudBackendConfig,
    CloudError,
    CloudHealth,
    CloudJob,
    CloudModelSpec,
    CloudProvider,
    CloudRequest,
    load_catalog_file,
    valid_specs,
)

from .catalog import CATALOG_FILE, catalog_specs
from .config import DEFAULT_BASE_URL, OpenAIConfig
from .errors import CREDITS_HINT, lacks_listing_scope, map_error
from .mapping import edit_form, parse_result, request_fields

logger = logging.getLogger(__name__)

DATA_NOTICE = (
    "Your prompts, any pictures you add and the settings you choose are sent to OpenAI (openai.com), which runs "
    "the model, so OpenAI's API data policy applies. PotionUI keeps each finished picture on this server and does "
    "not rely on OpenAI to keep it. No user name or email is sent. If an administrator turns on the anonymous user "
    "id, an anonymous id is sent that cannot be traced back to a person without this server's secret key."
)


class OpenAIProvider(CloudProvider):
    key: ClassVar[str] = "openai"
    label: ClassVar[str] = "OpenAI"
    config_class: ClassVar[type[CloudBackendConfig]] = OpenAIConfig
    data_notice: ClassVar[str] = DATA_NOTICE
    credits_hint: ClassVar[str] = CREDITS_HINT
    default_base_url: ClassVar[str] = DEFAULT_BASE_URL
    catalog_file: ClassVar[Path] = CATALOG_FILE

    @classmethod
    def auth_headers(cls, config: CloudBackendConfig) -> Mapping[str, str]:
        headers = dict(super().auth_headers(config))
        if config.organization:
            headers["OpenAI-Organization"] = config.organization
        if config.project:
            headers["OpenAI-Project"] = config.project
        return headers

    def map_error(self, status: int, headers: Mapping[str, str], body: Any) -> Optional[CloudError]:
        return map_error(status, headers, body)

    async def _listed_models(self) -> Optional[Dict[str, Any]]:
        try:
            payload = await self.http.request_json("GET", "/models")
        except CloudError as error:
            if not lacks_listing_scope(error):
                raise
            logger.info("[OPENAI] this key may not list models; offering the whole catalog")
            return None
        items = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return None
        return {item["id"]: item for item in items if isinstance(item, dict) and isinstance(item.get("id"), str)}

    async def discover(self) -> List[CloudModelSpec]:
        listed = await self._listed_models()
        specs: List[CloudModelSpec] = []
        for spec in catalog_specs(load_catalog_file(self.catalog_file)):
            if listed is not None and spec.provider_model_id not in listed:
                logger.info(f"[OPENAI] {spec.provider_model_id} is not offered to this key")
                continue
            shutdown = (listed or {}).get(spec.provider_model_id, {}).get("shutdown_date")
            if isinstance(shutdown, str) and shutdown and not spec.deprecated_at:
                spec = replace(spec, deprecated_at=shutdown)
            specs.append(spec)
        return valid_specs(specs, "OPENAI")

    def _user(self, request: CloudRequest) -> Optional[str]:
        return request.user_ref or None

    async def submit(self, request: CloudRequest) -> CloudJob:
        moderation, user = self.config.moderation, self._user(request)
        timeout = float(self.config.timeout_seconds)
        if request.task == "img_edit":
            form = await asyncio.to_thread(edit_form, request, moderation, user)
            response = await self.http.request("POST", "/images/edits", data=form, timeout_s=timeout)
        else:
            body = request_fields(request, moderation, user)
            response = await self.http.request("POST", "/images/generations", json=body, timeout_s=timeout)
        request_id = response.headers.get("x-request-id")
        result = parse_result(response.json(), request, request_id)
        return CloudJob(job_id=request_id or f"openai-{request.client_reference or 'image'}", result=result)

    async def probe(self) -> CloudHealth:
        try:
            await self.http.request_json("GET", "/models")
        except CloudError as error:
            if lacks_listing_scope(error):
                return CloudHealth(ok=True, message="The key works, but it may not list models. The whole catalog is offered.")
            raise
        return CloudHealth(ok=True)
