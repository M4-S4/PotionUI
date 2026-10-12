"""Queueing downloads and following them.

`queue_model_download` puts one model file on the application's download queue
in the name of `created_by`; the worker fetches it, verifies `checksum_sha256`
when given, indexes it into the catalog and fires `download.after_complete`.
`get_download` reads a download's current state. Both answer `Download.to_dict()`
dicts (`id`, `status`, `progress`, `filename`, `destination_path`,
`error_message` ...). `DownloadQueueException` is the queue's refusal, and its
`UnsafeFilenameException` subtype a file name it will not write.
"""

from typing import Any, Dict, List, Optional

from src.features.downloads.exceptions import (
    DownloadNotFoundException,
    DownloadQueueException,
    UnsafeFilenameException,
)
from src.platform.plugins.runtime_registries import get_container

__all__ = [
    "DownloadNotFoundException",
    "DownloadQueueException",
    "UnsafeFilenameException",
    "get_download",
    "queue_model_download",
]


async def queue_model_download(
    url: str,
    *,
    filename: Optional[str] = None,
    model_type: Optional[str] = None,
    subdir: Optional[str] = None,
    tags: Optional[List[str]] = None,
    checksum_sha256: Optional[str] = None,
    provider_id: Optional[str] = None,
    created_by: Optional[str] = None,
    destination_backend_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Queue one model file; returns the queued download.

    `model_type` picks the type's write folder and `subdir` a folder under it;
    `provider_id` names the marketplace provider whose credentials the worker
    attaches; `destination_backend_id` sends the file to a `native.remote`
    worker's depot. Raises `DownloadQueueException` when the queue refuses.
    """
    download = await get_container().download_queue.queue_model_download(
        url,
        filename=filename,
        tags=list(tags) if tags is not None else None,
        checksum_sha256=checksum_sha256,
        provider_id=provider_id,
        created_by=created_by,
        model_type=model_type,
        subdir=subdir,
        destination_backend_id=destination_backend_id,
    )
    return download.to_dict()


def get_download(download_id: str) -> Optional[Dict[str, Any]]:
    """A download's current state, or None once the application no longer has it."""
    if not download_id:
        return None
    try:
        download = get_container().download_queue.get_download(download_id)
    except DownloadNotFoundException:
        return None
    return download.to_dict()
