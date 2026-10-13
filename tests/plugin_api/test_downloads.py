"""Tests for the `src.plugin_api.downloads` surface."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.features.downloads.exceptions import DownloadNotFoundException
from src.plugin_api.downloads import get_download, queue_model_download


def _download(**fields):
    defaults = dict(id="dl-1", status="pending", filename="m.safetensors")
    defaults.update(fields)
    return SimpleNamespace(to_dict=lambda: dict(defaults))


@pytest.mark.asyncio
async def test_queue_model_download_queues_on_the_application_queue_and_answers_the_record():
    queue = Mock(queue_model_download=AsyncMock(return_value=_download()))
    container = SimpleNamespace(download_queue=queue)

    with patch("src.plugin_api.downloads.get_container", return_value=container):
        result = await queue_model_download(
            "https://example.org/m.safetensors",
            filename="m.safetensors", model_type="lora", subdir="my-plugin", tags=("a", "b"),
            checksum_sha256="a" * 64, provider_id="civitai", created_by="user-1",
        )

    assert result == {"id": "dl-1", "status": "pending", "filename": "m.safetensors"}
    queue.queue_model_download.assert_awaited_once_with(
        "https://example.org/m.safetensors",
        filename="m.safetensors", tags=["a", "b"], checksum_sha256="a" * 64, provider_id="civitai",
        created_by="user-1", model_type="lora", subdir="my-plugin", destination_backend_id=None,
    )


@pytest.mark.asyncio
async def test_queue_model_download_lets_the_queues_refusal_through():
    from src.features.downloads.exceptions import DownloadQueueException

    queue = Mock(queue_model_download=AsyncMock(side_effect=DownloadQueueException("blocked by a hook")))
    container = SimpleNamespace(download_queue=queue)

    with patch("src.plugin_api.downloads.get_container", return_value=container):
        with pytest.raises(DownloadQueueException):
            await queue_model_download("https://example.org/m.safetensors")


def test_get_download_answers_the_current_state():
    queue = Mock(get_download=Mock(return_value=_download(status="completed")))
    container = SimpleNamespace(download_queue=queue)

    with patch("src.plugin_api.downloads.get_container", return_value=container):
        result = get_download("dl-1")

    assert result["status"] == "completed"
    queue.get_download.assert_called_once_with("dl-1")


def test_get_download_answers_none_for_a_download_the_application_no_longer_has():
    queue = Mock(get_download=Mock(side_effect=DownloadNotFoundException("gone")))
    container = SimpleNamespace(download_queue=queue)

    with patch("src.plugin_api.downloads.get_container", return_value=container):
        assert get_download("dl-1") is None
        assert get_download("") is None
