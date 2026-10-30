"""Tests for the download.after_complete hook fired by the download worker."""
import threading
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.features.downloads import worker as worker_module
from src.features.downloads.hooks import DOWNLOAD_HOOKS
from src.features.downloads.models import Download, DownloadSettings, DownloadStatus, DownloadType
from src.features.downloads.worker import DownloadWorker
from src.features.models.indexer import DuplicateOf


def _download(**fields):
    defaults = dict(
        id='dl-1', url='https://example.org/m.safetensors', destination_path='/depot/loras/m.safetensors',
        filename='m.safetensors', type=DownloadType.MODEL, status=DownloadStatus.PENDING,
        checksum_sha256='a' * 64, created_by='user-1',
    )
    defaults.update(fields)
    return Download(**defaults)


@pytest.fixture
def repository():
    repo = Mock()
    repo.update_status.return_value = None
    return repo


@pytest.fixture
def registry():
    registry = Mock()
    context = Mock()
    context.data = {}
    registry.execute_hook.return_value = (context, [])
    return registry


def _worker(repository, registry, indexer=None):
    settings = DownloadSettings()
    settings.verify_checksum = False
    return DownloadWorker(
        settings=settings, repo=repository, connection_hub=AsyncMock(),
        provider_registry_factory=lambda: None, backend_registry=Mock(),
        local_model_indexer=indexer, plugin_registry=registry,
    )


async def _complete(worker, download):
    worker.repo.get_by_id.return_value = download

    async def fake_download_file(self, dl):
        return True

    with patch.object(DownloadWorker, "_download_file", fake_download_file), \
         patch("src.features.models.native_availability_reconciler.native_availability_projector") as rec:
        rec.reconcile = AsyncMock()
        await worker._process_download(download.id)


def _after_complete_payload(registry):
    calls = [c for c in registry.execute_hook.call_args_list if c.args[0] == DOWNLOAD_HOOKS.after_complete]
    assert len(calls) == 1, calls
    return calls[0].kwargs["initial_data"]


@pytest.mark.asyncio
async def test_indexed_model_download_fires_with_the_model_id(repository, registry):
    indexer = Mock(return_value={"indexed": True, "model_id": "m1", "duplicate_of": None, "warning": None})
    download = _download()

    await _complete(_worker(repository, registry, indexer), download)

    payload = _after_complete_payload(registry)
    assert payload == {
        "download_id": "dl-1",
        "url": "https://example.org/m.safetensors",
        "type": "model",
        "filename": "m.safetensors",
        "destination_path": "/depot/loras/m.safetensors",
        "checksum_sha256": "a" * 64,
        "created_by": "user-1",
        "destination_backend_id": None,
        "model_id": "m1",
        "duplicate_of": None,
        "warning": None,
    }


@pytest.mark.asyncio
async def test_duplicate_file_fires_with_the_model_it_duplicates(repository, registry):
    duplicate = DuplicateOf(model_id="m-existing", model_type="lora", root="/depot", path="/depot/loras/other.safetensors")
    indexer = Mock(return_value={"indexed": False, "model_id": None, "duplicate_of": duplicate.to_dict(), "warning": None})

    await _complete(_worker(repository, registry, indexer), _download())

    payload = _after_complete_payload(registry)
    assert payload["model_id"] is None
    assert payload["duplicate_of"] == "m-existing"


@pytest.mark.asyncio
async def test_fires_after_the_indexer_even_when_indexing_fails(repository, registry):
    indexer = Mock(side_effect=RuntimeError("outside every model root"))

    await _complete(_worker(repository, registry, indexer), _download())

    payload = _after_complete_payload(registry)
    assert payload["model_id"] is None
    assert payload["duplicate_of"] is None
    repository.update_status.assert_any_call('dl-1', DownloadStatus.COMPLETED, None)


@pytest.mark.asyncio
async def test_media_download_fires_without_a_model(repository, registry):
    indexer = Mock()
    download = _download(type=DownloadType.MEDIA, destination_path='/outputs/x.png', filename='x.png')

    await _complete(_worker(repository, registry, indexer), download)

    payload = _after_complete_payload(registry)
    assert payload["type"] == "media"
    assert payload["model_id"] is None
    indexer.assert_not_called()


@pytest.mark.asyncio
async def test_hf_repo_child_fires_without_a_model(repository, registry):
    download = _download(group_id='parent-1')
    repository.refresh_group.return_value = (None, False)

    await _complete(_worker(repository, registry, Mock()), download)

    assert _after_complete_payload(registry)["model_id"] is None


@pytest.mark.asyncio
async def test_a_failing_hook_never_fails_the_download(repository, registry):
    registry.execute_hook.side_effect = RuntimeError("plugin blew up")
    indexer = Mock(return_value={"indexed": True, "model_id": "m1", "duplicate_of": None, "warning": None})

    await _complete(_worker(repository, registry, indexer), _download())

    repository.update_status.assert_any_call('dl-1', DownloadStatus.COMPLETED, None)
    assert not any(c.args[1] == DownloadStatus.FAILED for c in repository.update_status.call_args_list)


@pytest.mark.asyncio
async def test_a_failing_download_record_read_never_fails_the_download(repository, registry):
    download = _download()
    statuses = []
    repository.update_status.side_effect = lambda download_id, status, *rest: statuses.append(status)

    def get_by_id(download_id):
        if DownloadStatus.COMPLETED in statuses:
            raise RuntimeError("database is locked")
        return download

    repository.get_by_id.side_effect = get_by_id

    await _complete(_worker(repository, registry), download)

    assert statuses[-1] == DownloadStatus.COMPLETED
    assert DownloadStatus.FAILED not in statuses
    repository.increment_retry.assert_not_called()
    assert not any(c.args[0] == DOWNLOAD_HOOKS.after_complete for c in registry.execute_hook.call_args_list)


@pytest.mark.asyncio
async def test_handlers_run_off_the_worker_loop(repository, registry):
    threads = []

    def record_thread(hook_name, context=None, initial_data=None):
        threads.append(threading.current_thread())
        return Mock(data={}), []

    registry.execute_hook.side_effect = record_thread

    await _complete(_worker(repository, registry), _download())

    assert len(threads) == 1
    assert threads[0] is not threading.current_thread()


@pytest.mark.asyncio
async def test_a_slow_handler_does_not_hold_the_worker(repository, registry, monkeypatch):
    monkeypatch.setattr(worker_module, "AFTER_COMPLETE_HOOK_TIMEOUT_S", 0.05)
    finished = threading.Event()

    def slow_handler(hook_name, context=None, initial_data=None):
        time.sleep(0.5)
        finished.set()
        return Mock(data={}), []

    registry.execute_hook.side_effect = slow_handler

    await _complete(_worker(repository, registry), _download())

    assert not finished.is_set()
    repository.update_status.assert_any_call('dl-1', DownloadStatus.COMPLETED, None)
    assert finished.wait(5)


@pytest.mark.asyncio
async def test_without_a_registry_nothing_is_fired(repository):
    indexer = Mock(return_value={"indexed": True, "model_id": "m1", "duplicate_of": None, "warning": None})

    await _complete(_worker(repository, None, indexer), _download())

    repository.update_status.assert_any_call('dl-1', DownloadStatus.COMPLETED, None)


@pytest.mark.asyncio
async def test_failed_download_fires_nothing(repository, registry):
    worker = _worker(repository, registry, Mock())
    worker.settings.auto_retry_failed = False
    download = _download()
    repository.get_by_id.return_value = download

    async def failing_download_file(self, dl):
        raise RuntimeError("connection reset")

    with patch.object(DownloadWorker, "_download_file", failing_download_file):
        await worker._process_download(download.id)

    assert not any(c.args[0] == DOWNLOAD_HOOKS.after_complete for c in registry.execute_hook.call_args_list)
