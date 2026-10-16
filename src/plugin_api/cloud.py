from typing import Any, Dict, Mapping, Optional, Sequence

from src.features.cloud.clock import Clock, MonotonicClock
from src.features.cloud.contracts import (
    CANONICAL_PARAMS,
    CLOUD_ENGINE,
    EXTRA_PARAM_PREFIX,
    MEDIA_ROLES,
    PARAM_ASPECT_RATIO,
    PARAM_BACKGROUND,
    PARAM_COUNT,
    PARAM_DURATION_S,
    PARAM_ENHANCE_PROMPT,
    PARAM_FPS,
    PARAM_GENERATE_AUDIO,
    PARAM_GUIDANCE,
    PARAM_INSTRUMENTAL,
    PARAM_LYRICS,
    PARAM_NEGATIVE_PROMPT,
    PARAM_OUTPUT_FORMAT,
    PARAM_PROMPT,
    PARAM_QUALITY,
    PARAM_RESOLUTION,
    PARAM_SEED,
    PARAM_SIZE,
    PARAM_STEPS,
    PARAM_STRENGTH,
    PARAM_VOICE,
    ROLE_FIRST_FRAME,
    ROLE_LAST_FRAME,
    ROLE_MASK,
    ROLE_REFERENCE,
    ROLE_SOURCE_AUDIO,
    ROLE_SOURCE_IMAGE,
    ROLE_SOURCE_VIDEO,
    TASK_KINDS,
    TERMINAL_STATES,
    CloudArtifact,
    CloudBackendConfig,
    CloudCost,
    CloudError,
    CloudHealth,
    CloudJob,
    CloudModelSpec,
    CloudProvider,
    CloudRequest,
    CloudResult,
    CloudStatus,
    JobState,
    LocalMedia,
    MediaInputSpec,
    ParamSpec,
    PriceLine,
    TaskKind,
    is_canonical_param,
    load_catalog_file,
    spec_problems,
    valid_specs,
)
from src.features.cloud.director import DirectorEstimateError
from src.features.cloud.http import CloudHttp, HttpResponse, default_error_mapper, parse_retry_after
from src.features.cloud.inline import decode_inline, encode_media
from src.features.cloud.rate_limit import TokenBucket
from src.features.cloud.redaction import redact_headers, redact_url, scrub_text
from src.features.cloud.registration import register_cloud_provider
from src.platform.plugins.runtime_registries import get_container
from src.platform.security.user import AccountType

__all__ = [
    "CLOUD_BLOCKS",
    "CANONICAL_PARAMS",
    "CLOUD_ENGINE",
    "CloudArtifact",
    "CloudBackendConfig",
    "CloudCost",
    "CloudError",
    "CloudHealth",
    "CloudHttp",
    "CloudJob",
    "CloudModelSpec",
    "CloudProvider",
    "CloudRequest",
    "CloudResult",
    "CloudStatus",
    "Clock",
    "DirectorEstimateError",
    "EXTRA_PARAM_PREFIX",
    "HttpResponse",
    "JobState",
    "LocalMedia",
    "MEDIA_ROLES",
    "MediaInputSpec",
    "MonotonicClock",
    "PARAM_ASPECT_RATIO",
    "PARAM_BACKGROUND",
    "PARAM_COUNT",
    "PARAM_DURATION_S",
    "PARAM_ENHANCE_PROMPT",
    "PARAM_FPS",
    "PARAM_GENERATE_AUDIO",
    "PARAM_GUIDANCE",
    "PARAM_INSTRUMENTAL",
    "PARAM_LYRICS",
    "PARAM_NEGATIVE_PROMPT",
    "PARAM_OUTPUT_FORMAT",
    "PARAM_PROMPT",
    "PARAM_QUALITY",
    "PARAM_RESOLUTION",
    "PARAM_SEED",
    "PARAM_SIZE",
    "PARAM_STEPS",
    "PARAM_STRENGTH",
    "PARAM_VOICE",
    "ParamSpec",
    "PriceLine",
    "ROLE_FIRST_FRAME",
    "ROLE_LAST_FRAME",
    "ROLE_MASK",
    "ROLE_REFERENCE",
    "ROLE_SOURCE_AUDIO",
    "ROLE_SOURCE_IMAGE",
    "ROLE_SOURCE_VIDEO",
    "TASK_KINDS",
    "TERMINAL_STATES",
    "TaskKind",
    "TokenBucket",
    "decode_inline",
    "default_error_mapper",
    "encode_media",
    "estimate_cloud_cost",
    "is_canonical_param",
    "load_catalog_file",
    "parse_retry_after",
    "redact_headers",
    "redact_url",
    "register_cloud_provider",
    "scrub_text",
    "spec_problems",
    "valid_specs",
]

CLOUD_BLOCKS = (
    "tabs/image.yml",
    "tabs/image_edit.yml",
    "tabs/video.yml",
    "tabs/img2video.yml",
    "tabs/references_edit.yml",
    "tabs/references_img2video.yml",
    "tabs/provider_options.yml",
    "tabs/_image_params.yml",
    "tabs/_video_params.yml",
    "models/txt2img.yml",
    "models/img_edit.yml",
    "models/inpaint.yml",
    "models/txt2video.yml",
    "models/img2video.yml",
    "models/ref2video.yml",
    "models/video2video.yml",
    "models/txt2audio.yml",
    "models/txt2speech.yml",
    "models/upscale_image.yml",
    "models/upscale_video.yml",
)


def estimate_cloud_cost(
    user, model_id: str, shots: Sequence[Mapping[str, Any]], driver: Optional[str] = None
) -> Dict[str, Any]:
    """What a run of shots on an enabled cloud model would cost, as
    `POST /api/cloud/models/{model_id}/estimate` answers it. Costs are shown
    to administrators only: anyone else, or no user, gets `PermissionError`.
    Each shot is `{"task", "params"}`, as in the route's body. Shots the route
    would refuse raise `ValueError` (`DirectorEstimateError` when the
    estimator refuses them), and a model that is not an enabled cloud model
    raises `ModelNotFoundException`."""
    if getattr(user, "account_type", None) != AccountType.ADMIN:
        raise PermissionError("Only an administrator can see what a cloud run costs")
    from src.features.cloud.dto import DirectorEstimateRequest
    from src.features.cloud.routes import estimate_cost

    request = DirectorEstimateRequest.model_validate({"shots": list(shots), "driver": driver})
    return estimate_cost(get_container().cloud_capabilities, model_id, request)
