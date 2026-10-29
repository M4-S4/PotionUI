from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.features.downloads.models import Download, DownloadSettings, DownloadStatus, DownloadType
from src.features.downloads.worker import DownloadWorker

REMOTE_WARNING_PREFIX = "Downloaded, but the catalog could not be refreshed automatically"


def _download(**fields) -> Download:
    defaults = dict(
        id='dl-1', url='https://example.org/m.safetensors', destination_path='/models/loras/m.safetensors',
        filename='m.safetensors', type=DownloadType.MODEL, status=DownloadStatus.PENDING,
    )
    defaults.update(fields)
    return Download(**defaults)


class _Row:
    def __init__(self, download: Download):
        self.events = []
        self.status = DownloadStatus.PENDING
        self.error_message = None
        self.status_writes = []
        self.repo = Mock()
        self.repo.get_by_id.return_value = download
        self.repo.increment_retry.return_value = 1
        self.repo.update_status.side_effect = self._update_status
        self.hub = AsyncMock()
        self.hub.send_download_status.side_effect = self._send_status
        self.hub.send_notification.side_effect = self._send_notification

    def _update_status(self, download_id, status, error_message=None):
        self.events.append(f"status:{status.value}")
        self.status_writes.append((status, error_message))
        self.status = status
        self.error_message = error_message
        return True

    async def _send_status(self, download_id, status, filename, **kwargs):
        self.events.append(f"ws:{status}")

    async def _send_notification(self, *args, **kwargs):
        self.events.append("notification")

    def hook_registry(self):
        registry = Mock()

        def execute_hook(hook, initial_data=None):
            self.events.append("after_complete")
            return Mock(data={}), []

        registry.execute_hook.side_effect = execute_hook
        return registry

    def completed_writes(self):
        return [write for write in self.status_writes if write[0] == DownloadStatus.COMPLETED]


def _worker(row: _Row, **kwargs) -> DownloadWorker:
    settings = DownloadSettings()
    settings.verify_checksum = False
    return DownloadWorker(
        settings=settings, repo=row.repo, connection_hub=row.hub,
        provider_registry_factory=lambda: None, **kwargs,
    )


async def _run_local(worker: DownloadWorker, download: Download) -> None:
    async def fake_download_file(self, dl):
        return True

    with patch.object(DownloadWorker, "_download_file", fake_download_file):
        await worker._process_download(download.id)


async def _run_remote(worker: DownloadWorker, download: Download) -> None:
    async def fake_fetch_remote(self, dl):
        return True

    with patch.object(DownloadWorker, "_fetch_remote", fake_fetch_remote):
        await worker._process_download(download.id)


def _remote_worker(row: _Row, index_backend) -> DownloadWorker:
    backend_registry = Mock()
    backend_registry.get_backend.return_value = Mock()
    backend_model_indexer = Mock()
    backend_model_indexer.index_backend = index_backend
    return _worker(row, backend_registry=backend_registry, backend_model_indexer=backend_model_indexer)


@pytest.mark.asyncio
async def test_a_local_model_is_indexed_before_the_download_reports_completed():
    download = _download()
    row = _Row(download)

    def indexer(path):
        row.events.append("index")
        return {"indexed": True, "model_id": "m1", "duplicate_of": None, "warning": None}

    await _run_local(_worker(row, local_model_indexer=indexer, plugin_registry=row.hook_registry()), download)

    assert row.events == [
        "status:downloading", "ws:started",
        "index",
        "status:completed", "ws:completed", "notification",
        "after_complete",
    ]


@pytest.mark.asyncio
async def test_a_remote_download_reindexes_the_backend_before_it_reports_completed():
    download = _download(destination_backend_id='remote-1')
    row = _Row(download)

    async def index_backend(backend):
        row.events.append("index")

    await _run_remote(_remote_worker(row, index_backend), download)

    assert row.events == [
        "status:downloading", "ws:started",
        "index",
        "status:completed", "ws:completed", "notification",
    ]


@pytest.mark.asyncio
async def test_an_indexer_warning_lands_on_the_completed_row_exactly_once():
    download = _download()
    row = _Row(download)
    indexer = Mock(return_value={"indexed": True, "model_id": "m1", "duplicate_of": None, "warning": "header unreadable"})

    await _run_local(_worker(row, local_model_indexer=indexer), download)

    assert row.completed_writes() == [(DownloadStatus.COMPLETED, "header unreadable")]
    assert (row.status, row.error_message) == (DownloadStatus.COMPLETED, "header unreadable")


@pytest.mark.asyncio
async def test_a_remote_reindex_failure_lands_on_the_completed_row_exactly_once():
    download = _download(destination_backend_id='remote-1')
    row = _Row(download)

    async def index_backend(backend):
        raise RuntimeError("worker unreachable")

    await _run_remote(_remote_worker(row, index_backend), download)

    writes = row.completed_writes()
    assert len(writes) == 1
    assert writes[0][1].startswith(REMOTE_WARNING_PREFIX)
    assert "worker unreachable" in writes[0][1]
    assert row.status == DownloadStatus.COMPLETED


@pytest.mark.asyncio
async def test_an_indexer_exception_is_attempted_before_completion_and_still_completes():
    download = _download()
    row = _Row(download)

    def indexer(path):
        row.events.append("index")
        raise RuntimeError("outside every model root")

    await _run_local(_worker(row, local_model_indexer=indexer, plugin_registry=row.hook_registry()), download)

    assert row.events == [
        "status:downloading", "ws:started",
        "index",
        "status:completed", "ws:completed", "notification",
        "after_complete",
    ]
    assert row.completed_writes() == [(DownloadStatus.COMPLETED, None)]
    assert not any(write[0] == DownloadStatus.FAILED for write in row.status_writes)
    row.repo.increment_retry.assert_not_called()


@pytest.mark.asyncio
async def test_a_failing_row_read_while_indexing_still_completes_the_download():
    download = _download()
    row = _Row(download)
    reads = []

    def get_by_id(download_id):
        reads.append(download_id)
        if len(reads) > 1:
            raise RuntimeError("database is locked")
        return download

    row.repo.get_by_id.side_effect = get_by_id
    indexer = Mock()

    await _run_local(_worker(row, local_model_indexer=indexer), download)

    indexer.assert_not_called()
    assert row.completed_writes() == [(DownloadStatus.COMPLETED, None)]
    assert not any(write[0] == DownloadStatus.FAILED for write in row.status_writes)
    row.repo.increment_retry.assert_not_called()
