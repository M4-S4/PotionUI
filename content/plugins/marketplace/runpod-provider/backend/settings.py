"""Reading this plugin's own admin-configured settings.

`PluginRepository` decrypts `is_secret` settings transparently (see
`src.plugin_api.storage`'s docstring) - `load_settings()` re-reads on every
call rather than caching, so an admin rotating the API key in Admin -> Plugins
takes effect on the next request without a process restart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from src.plugin_api import PluginRepository, plugin_settings

PLUGIN_ID = "runpod-provider"


@dataclass(frozen=True)
class RunPodSettings:
    api_key: Optional[str]
    region: Optional[str]
    gpu_type_id: str
    volume_size_gb: int
    worker_image: Optional[str]
    container_registry_auth_id: Optional[str]
    allowed_cuda_versions: Tuple[str, ...]


def _parse_cuda_versions(raw: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())

def load_settings(repository: Optional[PluginRepository] = None) -> RunPodSettings:
    values = plugin_settings(PLUGIN_ID, repository)

    return RunPodSettings(
        api_key=values["api_key"],
        region=values["region"],
        gpu_type_id=values["gpu_type_id"],
        volume_size_gb=values["volume_size_gb"],
        worker_image=values["worker_image"],
        container_registry_auth_id=values["container_registry_auth_id"],
        allowed_cuda_versions=_parse_cuda_versions(values["allowed_cuda_versions"]),
    )
