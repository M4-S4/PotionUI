import asyncio
import logging
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Set, Tuple

from src.plugin_api.cloud import (
    CloudArtifact,
    CloudBackendConfig,
    CloudError,
    CloudHealth,
    CloudHttp,
    CloudJob,
    CloudModelSpec,
    CloudProvider,
    CloudRequest,
    CloudResult,
    CloudStatus,
    load_catalog_file,
    parse_retry_after,
    valid_specs,
)

from .catalog import CATALOG_FILE, catalog_specs
from .config import BflConfig
from .mapping import (
    CONTENT_MODERATED,
    ERROR,
    NOT_FOUND,
    QUEUED,
    READY,
    REQUEST_MODERATED,
    body_message,
    build_body,
    credits_cost,
    error_reason,
    media_type_for,
    moderation_error,
    polling_target,
    progress_of,
)

logger = logging.getLogger(__name__)

FIRST_POLL_SECONDS = 1.0
REGIONAL_TIMEOUT_SECONDS = 30.0
EXPIRED_STATUSES = ("HTTP 401", "HTTP 403", "HTTP 404", "HTTP 410")
EXPIRED_MESSAGE = "The finished picture expired at BFL before it could be downloaded. Generate it again."
TIMEOUT_MESSAGE = "BFL did not finish in time. It may still finish this job and bill it."
STOPPED_MESSAGE = "Stopped waiting. BFL may still finish this job and bill it."

DATA_NOTICE = (
    "Your prompts, any pictures you add and the settings you choose are sent to Black Forest Labs (bfl.ai), "
    "which runs the FLUX models. The backend's region decides where: global lets BFL pick a cluster, eu keeps "
    "requests in the EU and us in the US. FLUX 3 Image may also look things up on the web while it plans a "
    "picture unless Web grounding is turned off. BFL keeps each finished picture for 10 minutes; PotionUI "
    "downloads it straight away and does not rely on BFL to keep it. No user name or email is sent. If an "
    "administrator turns on the anonymous user id, an anonymous id is sent that cannot be traced back to a "
    "person without this server's secret key."
)


class BflProvider(CloudProvider):
    key: ClassVar[str] = "bfl"
    label: ClassVar[str] = "Black Forest Labs"
    config_class: ClassVar[type[CloudBackendConfig]] = BflConfig
    data_notice: ClassVar[str] = DATA_NOTICE
    supports_cancel: ClassVar[bool] = False
    idempotent_submit: ClassVar[bool] = False
    auth_header: ClassVar[Tuple[str, str]] = ("x-key", "")
    catalog_file: ClassVar[Path] = CATALOG_FILE
    max_poll_s: ClassVar[float] = 8.0
    timeout_message: ClassVar[str] = TIMEOUT_MESSAGE

    def __init__(self, config: CloudBackendConfig, http: CloudHttp) -> None:
        super().__init__(config, http)
        self._abandoned: Set[str] = set()
        self._regional: Dict[str, CloudHttp] = {}

    @classmethod
    def api_base_url(cls, config: CloudBackendConfig) -> str:
        return config.api_address()

    def map_error(self, status: int, headers: Mapping[str, str], body: Any) -> Optional[CloudError]:
        message = body_message(body)[:300]
        retry_after = parse_retry_after(headers.get("retry-after"))
        detail = f"HTTP {status}: {message}"
        if status == 401:
            return CloudError("auth", "BFL rejected the API key. Check it in Administration, Backends.", detail=detail)
        if status == 402:
            return CloudError(
                "credits", "The BFL account is out of credits. An administrator can add credits at api.bfl.ai.",
                detail=detail, retry_after_s=retry_after,
            )
        if status == 403:
            return CloudError("refused", "BFL refused this request. The key may not be allowed to use this model.", detail=detail)
        if status == 429:
            return CloudError(
                "rate_limited", "BFL is busy with this account's other jobs. Try again shortly.",
                detail=detail, retry_after_s=retry_after,
            )
        if status in (404, 405):
            return CloudError("invalid_request", "BFL does not offer this model at the configured address.", detail=detail)
        if status in (400, 422):
            return CloudError("invalid_request", "BFL rejected the request settings.", detail=detail)
        if status >= 500:
            return CloudError("unavailable", "BFL is having trouble at the moment. Try again shortly.", detail=detail, retry_after_s=retry_after)
        return None

    async def discover(self) -> List[CloudModelSpec]:
        return valid_specs(catalog_specs(load_catalog_file(self.catalog_file)), "BFL")

    async def submit(self, request: CloudRequest) -> CloudJob:
        user_ref = request.user_ref if self.config.send_user_hash else None
        body = await asyncio.to_thread(build_body, request, user_ref)
        payload = await self.http.request_json(
            "POST", f"/{request.model.provider_model_id}", json=body, timeout_s=float(self.config.timeout_seconds),
        )
        job_id = payload.get("id") if isinstance(payload, dict) else None
        if not isinstance(job_id, str) or not job_id:
            raise CloudError("failed", "BFL did not accept the job.", detail="no task id in the answer")
        submit_cost = credits_cost(payload.get("cost"), payload, "submit")
        handle: Dict[str, Any] = {
            **polling_target(self.http, payload, job_id),
            "output_format": body.get("output_format"),
            "submit_cost": submit_cost,
            "running": False,
        }
        return CloudJob(job_id=job_id, handle=handle, poll_after_s=FIRST_POLL_SECONDS)

    def regional_http(self, origin: str) -> CloudHttp:
        if origin not in self._regional:
            self._regional[origin] = CloudHttp(
                origin, auth_headers=type(self).auth_headers(self.config),
                timeout_s=REGIONAL_TIMEOUT_SECONDS, error_mapper=self.map_error,
            )
        return self._regional[origin]

    async def close(self) -> None:
        regional, self._regional = self._regional, {}
        for http in regional.values():
            await http.close()

    async def _poll_payload(self, job: CloudJob) -> Any:
        handle = job.handle
        http = self.regional_http(handle["origin"]) if handle.get("origin") else self.http
        return await http.request_json("GET", handle["target"], params=handle.get("params"))

    def _forget(self, job_id: str) -> None:
        self._abandoned.discard(job_id)

    async def poll(self, job: CloudJob) -> CloudStatus:
        if job.job_id in self._abandoned:
            self._forget(job.job_id)
            return CloudStatus(state="cancelled", message=STOPPED_MESSAGE)
        payload = await self._poll_payload(job)
        if not isinstance(payload, dict):
            raise CloudError("failed", "BFL sent an answer that could not be read.", detail="poll response is not an object")
        status = str(payload.get("status") or "").strip().lower()
        if status == READY:
            self._forget(job.job_id)
            return CloudStatus(state="succeeded", progress=1.0, result=self._result(job, payload))
        if status in (REQUEST_MODERATED, CONTENT_MODERATED):
            self._forget(job.job_id)
            raise moderation_error(status, payload, job.job_id)
        if status == ERROR:
            self._forget(job.job_id)
            return CloudStatus(state="failed", message=f"BFL reported Error: {error_reason(payload)}")
        if status == NOT_FOUND:
            self._forget(job.job_id)
            return CloudStatus(state="expired", message=f"BFL no longer knows task {job.job_id}")
        if status in QUEUED and not job.handle.get("running"):
            state = "queued"
        else:
            state = "running"
            if isinstance(job.handle, dict):
                job.handle["running"] = True
        return CloudStatus(state=state, progress=progress_of(payload.get("progress")))

    def _result(self, job: CloudJob, payload: Dict[str, Any]) -> CloudResult:
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        sample = result.get("sample")
        if isinstance(sample, list):
            sample = next((item for item in sample if isinstance(item, str) and item), None)
        if not isinstance(sample, str) or not sample:
            raise CloudError("failed", "BFL finished without a picture.", detail=f"task {job.job_id} is Ready without result.sample")
        cost = credits_cost(payload.get("cost"), payload, "settled") or job.handle.get("submit_cost")
        seed = result.get("seed")
        return CloudResult(
            artifacts=(CloudArtifact(modality="image", index=0, url=sample, media_type=media_type_for(job.handle.get("output_format"), sample)),),
            cost=cost,
            seed_used=seed if isinstance(seed, int) and not isinstance(seed, bool) else None,
            provider_job_id=job.job_id,
        )

    async def fetch(self, artifact: CloudArtifact, dest: Path, *, max_bytes: Optional[int] = None) -> Path:
        try:
            return await super().fetch(artifact, dest, max_bytes=max_bytes)
        except CloudError as error:
            if error.detail.startswith(EXPIRED_STATUSES):
                raise CloudError("expired", EXPIRED_MESSAGE, detail=f"download refused, the signed address has likely expired ({error.detail})") from None
            raise

    async def cancel(self, job: CloudJob) -> bool:
        self._abandoned.add(job.job_id)
        return False

    async def check(self) -> CloudHealth:
        try:
            payload = await self.http.request_json("GET", "/credits")
        except CloudError as error:
            if error.kind in ("auth", "credits", "refused"):
                return CloudHealth(ok=False, message=error.user_message)
            raise
        credits = payload.get("credits") if isinstance(payload, dict) else None
        try:
            balance = Decimal(str(credits)) / Decimal(100) if credits is not None and not isinstance(credits, bool) else None
        except (InvalidOperation, ValueError):
            balance = None
        return CloudHealth(ok=True, credits_usd=balance if balance is not None and balance.is_finite() else None)
