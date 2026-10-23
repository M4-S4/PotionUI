import asyncio
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Mapping, Optional

from src.plugin_api.cloud import (
    CloudBackendConfig,
    CloudError,
    CloudHealth,
    CloudHttp,
    CloudJob,
    CloudModelSpec,
    CloudProvider,
    CloudRequest,
    CloudStatus,
    load_catalog_file,
    parse_retry_after,
    valid_specs,
)

from .config import DEFAULT_BASE_URL, OpenRouterConfig
from .mapping import build_body, endpoint_list, model_spec, parse_result
from .video import (
    FIRST_POLL_SECONDS,
    build_video_body,
    is_video_spec,
    merge_director,
    parse_status,
    polling_target,
    video_spec,
)

logger = logging.getLogger(__name__)

SUGGESTIONS_FILE = Path(__file__).resolve().parents[1] / "cloud_models.yml"
ENDPOINT_LOOKUPS_AT_ONCE = 4
ROUTE_NOT_OFFERED = ("HTTP 404", "HTTP 405")

DATA_NOTICE = (
    "Your prompts, any pictures you add and the settings you choose are sent to OpenRouter (openrouter.ai). "
    "OpenRouter passes them on to the company that runs the chosen model, and which company that is depends on "
    "the model and on the routing settings of this backend, so the data policy of that company applies as well. "
    "PotionUI downloads each finished picture straight away and does not rely on OpenRouter to keep it. "
    "No user name or email is sent. If an administrator turns on the anonymous user id, an anonymous id is sent "
    "that cannot be traced back to a person without this server's secret key."
)


class OpenRouterProvider(CloudProvider):
    key: ClassVar[str] = "openrouter"
    label: ClassVar[str] = "OpenRouter"
    config_class: ClassVar[type[CloudBackendConfig]] = OpenRouterConfig
    data_notice: ClassVar[str] = DATA_NOTICE
    credits_hint: ClassVar[str] = "An administrator can add credits at openrouter.ai."
    default_base_url: ClassVar[str] = DEFAULT_BASE_URL
    catalog_file: ClassVar[Path] = SUGGESTIONS_FILE

    def attribution_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.config.app_url:
            headers["HTTP-Referer"] = self.config.app_url
        if self.config.app_title:
            headers["X-OpenRouter-Title"] = self.config.app_title
        return headers

    def director_overrides(self) -> Dict[str, Dict[str, Any]]:
        entries = load_catalog_file(self.catalog_file).get("director") or {}
        if not isinstance(entries, dict):
            return {}
        return {str(model_id): value for model_id, value in entries.items() if isinstance(value, dict)}

    def map_error(self, status: int, headers: Mapping[str, str], body: Any) -> Optional[CloudError]:
        error = body.get("error") if isinstance(body, dict) else None
        error = error if isinstance(error, dict) else {}
        metadata = error.get("metadata") if isinstance(error.get("metadata"), dict) else {}
        message = str(error.get("message") or "")[:300]
        retry_after = parse_retry_after(headers.get("retry-after"))
        reasons = metadata.get("reasons")
        if status == 403 and reasons:
            listed = ", ".join(str(reason) for reason in reasons) if isinstance(reasons, list) else str(reasons)
            facts = [f"reasons: {listed}"]
            for label, field in (("provider", "provider_name"), ("model", "model_slug")):
                if metadata.get(field):
                    facts.append(f"{label}: {metadata[field]}")
            return CloudError(
                "refused", "The model's content filter refused this request. Try a different prompt or picture.",
                detail="moderation refused (" + "; ".join(facts) + ")",
            )
        if status == 502:
            return CloudError(
                "unavailable", "The model is down at the moment. Try again shortly.",
                detail=f"HTTP 502: {message}", retry_after_s=retry_after,
            )
        if status == 503:
            return CloudError(
                "unavailable", "No provider could serve this request with the current routing settings.",
                detail=f"HTTP 503: {message}", retry_after_s=retry_after,
            )
        return super().map_error(status, headers, body)

    async def _get(self, target: str) -> Any:
        return await self.http.request_json("GET", target, headers=self.attribution_headers())

    async def _endpoints(self, model_id: str, gate: asyncio.Semaphore) -> List[Any]:
        async with gate:
            try:
                return endpoint_list(await self._get(f"/images/models/{model_id}/endpoints"))
            except CloudError as error:
                if error.kind in ("auth", "credits"):
                    raise
                logger.warning(f"[OPENROUTER] no endpoint details for {model_id}: {error.kind}")
                return []

    async def discover(self) -> List[CloudModelSpec]:
        return [*await self._discover_images(), *await self._discover_videos()]

    async def _discover_videos(self) -> List[CloudModelSpec]:
        try:
            payload = await self._get("/videos/models")
        except CloudError as error:
            if not error.detail.startswith(ROUTE_NOT_OFFERED):
                raise
            logger.info("[OPENROUTER] this account's API does not offer video models")
            return []
        items = payload.get("data", []) if isinstance(payload, dict) else payload
        overrides = self.director_overrides()
        specs = valid_specs((video_spec(item) for item in items if isinstance(items, list)), "OPENROUTER")
        return [
            replace(spec, director=merge_director(spec.director, overrides[spec.provider_model_id]))
            if spec.provider_model_id in overrides else spec
            for spec in specs
        ]

    async def _discover_images(self) -> List[CloudModelSpec]:
        payload = await self._get("/images/models")
        items = payload.get("data", []) if isinstance(payload, dict) else payload
        items = [item for item in items if isinstance(item, dict) and isinstance(item.get("id"), str)] if isinstance(items, list) else []
        gate = asyncio.Semaphore(ENDPOINT_LOOKUPS_AT_ONCE)
        endpoints = await asyncio.gather(*(self._endpoints(item["id"], gate) for item in items))
        return valid_specs((model_spec(item, details) for item, details in zip(items, endpoints)), "OPENROUTER")

    async def submit_video(self, request: CloudRequest) -> CloudJob:
        body = await asyncio.to_thread(
            build_video_body, request, self.config.upstream_list(),
            request.user_ref or None,
        )
        payload = await self.http.request_json(
            "POST", "/videos", json=body, headers=self.attribution_headers(),
            timeout_s=float(self.config.timeout_seconds),
        )
        job_id = payload.get("id") if isinstance(payload, dict) else None
        if not isinstance(job_id, str) or not job_id:
            raise CloudError("failed", "OpenRouter did not accept the video job.", detail="no job id in the answer")
        return CloudJob(
            job_id=job_id,
            handle={"target": polling_target(self.http, payload, job_id)},
            poll_after_s=FIRST_POLL_SECONDS,
        )

    async def poll(self, job: CloudJob) -> CloudStatus:
        payload = await self.http.request_json("GET", job.handle["target"], headers=self.attribution_headers())
        return parse_status(self.http, payload, job.job_id)

    async def submit(self, request: CloudRequest) -> CloudJob:
        if is_video_spec(request.model):
            return await self.submit_video(request)
        body = await asyncio.to_thread(
            build_body, request, self.config.upstream_list(), request.user_ref or None
        )
        payload = await self.http.request_json(
            "POST", "/images", json=body, headers=self.attribution_headers(),
            timeout_s=float(self.config.timeout_seconds),
        )
        result = parse_result(payload, request)
        job_id = result.provider_job_id or f"openrouter-{request.client_reference or 'image'}"
        return CloudJob(job_id=job_id, result=result)

    async def probe(self) -> CloudHealth:
        await self._get("/images/models")
        return CloudHealth(ok=True)
