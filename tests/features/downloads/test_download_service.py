import tempfile
from pathlib import Path

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.features.downloads.models import Download, DownloadType, DownloadStatus, DownloadSettings
from src.features.downloads.queue import DownloadQueue
from tests.fixtures.model_roots import make_roots

_ROOTS_TMP_DIR = Path(tempfile.mkdtemp(prefix="potionui-download-service-test-"))


def _build_queue(**kwargs):
    kwargs.setdefault('settings', Mock(get_setting=Mock(return_value=None)))
    kwargs.setdefault('connection_hub', AsyncMock())
    kwargs.setdefault('resolver', make_roots(_ROOTS_TMP_DIR))
    return DownloadQueue(**kwargs)


@pytest.fixture
def mock_download_repo():
    """Mock DownloadRepository"""
    repo = Mock()
    repo.create = Mock()
    repo.get_by_id = Mock()
    repo.get_pending = Mock(return_value=[])
    repo.get_active = Mock(return_value=[])
    repo.get_all = Mock(return_value=[])
    repo.count_by_status = Mock(return_value={})
    repo.count_total = Mock(return_value=0)
    repo.update_status = Mock(return_value=True)
    repo.update_progress = Mock(return_value=True)
    repo.increment_retry = Mock(return_value=True)
    repo.get_settings = Mock(return_value=DownloadSettings())
    return repo


@pytest.fixture
def mock_plugin_registry():
    """Mock PluginRegistry"""
    registry = Mock()
    mock_context = Mock()
    mock_context.data = {}
    registry.execute_hook.return_value = (mock_context, [])
    return registry


@pytest.fixture
def sample_download():
    """Sample download for testing"""
    return Download(
        id="test-download-1",
        type=DownloadType.MODEL,
        url="https://example.com/model.safetensors",
        destination_path="/models/model.safetensors",
        filename="model.safetensors",
        status=DownloadStatus.PENDING,
        progress=0.0,
        total_bytes=1000000,
        downloaded_bytes=0,
        created_at=datetime(2024, 1, 1, 12, 0, 0)
    )


class TestQueueDownload:
    """Tests for queueing downloads"""

    @pytest.mark.asyncio
    async def test_queue_model_download_with_tags(self, mock_download_repo, mock_plugin_registry, sample_download):
        """Test queueing a model download with tags"""
        mock_download_repo.create.return_value = sample_download

        manager = _build_queue(
            download_repository=mock_download_repo,
            plugin_registry=mock_plugin_registry,
        )

        mock_worker = AsyncMock()
        mock_worker.get_queue_position.return_value = 0
        manager.worker = mock_worker

        with patch.object(manager, 'conn', AsyncMock()) as mock_conn:
            mock_conn.send_download_queued = AsyncMock()

            result = await manager.queue_model_download(
                url="https://example.com/model.safetensors",
                tags=["sdxl", "checkpoint"]
            )

        created_download = mock_download_repo.create.call_args[0][0]
        assert created_download.tags == ["sdxl", "checkpoint"]


class TestRetryDownload:
    """Tests for retry functionality"""

    @pytest.mark.asyncio
    async def test_retry_failed_download(self, mock_download_repo, mock_plugin_registry, sample_download):
        """Test retrying a failed download"""

        sample_download.status = DownloadStatus.FAILED
        mock_download_repo.get_by_id.return_value = sample_download

        manager = _build_queue(
            download_repository=mock_download_repo,
            plugin_registry=mock_plugin_registry,
        )

        mock_worker = AsyncMock()
        mock_worker.retry.return_value = True
        manager.worker = mock_worker

        result = await manager.retry_download("test-download-1")

        mock_worker.retry.assert_called_once_with("test-download-1")

    @pytest.mark.asyncio
    async def test_retry_cancelled_download(self, mock_download_repo, mock_plugin_registry, sample_download):
        """A cancelled download can be retried - "download again"."""
        sample_download.status = DownloadStatus.CANCELLED
        mock_download_repo.get_by_id.return_value = sample_download

        manager = _build_queue(
            download_repository=mock_download_repo,
            plugin_registry=mock_plugin_registry,
        )

        mock_worker = AsyncMock()
        mock_worker.retry.return_value = True
        manager.worker = mock_worker

        result = await manager.retry_download("test-download-1")

        mock_worker.retry.assert_called_once_with("test-download-1")
        assert result is sample_download

    @pytest.mark.asyncio
    async def test_retry_completed_download(self, mock_download_repo, mock_plugin_registry, sample_download):
        """A completed download can be retried - "download again"."""
        sample_download.status = DownloadStatus.COMPLETED
        mock_download_repo.get_by_id.return_value = sample_download

        manager = _build_queue(
            download_repository=mock_download_repo,
            plugin_registry=mock_plugin_registry,
        )

        mock_worker = AsyncMock()
        mock_worker.retry.return_value = True
        manager.worker = mock_worker

        result = await manager.retry_download("test-download-1")

        mock_worker.retry.assert_called_once_with("test-download-1")
        assert result is sample_download

    @pytest.mark.asyncio
    async def test_retry_downloading_raises_naming_the_state(self, mock_download_repo, mock_plugin_registry, sample_download):
        """Retrying an in-flight download is refused; the error names the actual status."""
        from src.features.downloads.exceptions import DownloadOperationException

        sample_download.status = DownloadStatus.DOWNLOADING
        mock_download_repo.get_by_id.return_value = sample_download

        manager = _build_queue(
            download_repository=mock_download_repo,
            plugin_registry=mock_plugin_registry,
        )

        mock_worker = AsyncMock()
        mock_worker.retry.return_value = False
        manager.worker = mock_worker

        with pytest.raises(DownloadOperationException, match="downloading"):
            await manager.retry_download("test-download-1")

    @pytest.mark.asyncio
    async def test_retry_group_retries_every_retryable_child(self, mock_download_repo, mock_plugin_registry):
        """A grouped retry retries every child in a retryable state, not just failed ones."""
        parent = Download(
            id="parent-1", type=DownloadType.HF_REPO,
            status=DownloadStatus.FAILED, filename="repo",
        )
        completed_child = Download(
            id="child-completed", type=DownloadType.MODEL,
            status=DownloadStatus.COMPLETED, filename="a.safetensors", group_id="parent-1",
        )
        failed_child = Download(
            id="child-failed", type=DownloadType.MODEL,
            status=DownloadStatus.FAILED, filename="b.safetensors", group_id="parent-1",
        )
        pending_child = Download(
            id="child-pending", type=DownloadType.MODEL,
            status=DownloadStatus.PENDING, filename="c.safetensors", group_id="parent-1",
        )

        mock_download_repo.get_by_id.return_value = parent
        mock_download_repo.get_children.return_value = [completed_child, failed_child, pending_child]

        manager = _build_queue(
            download_repository=mock_download_repo,
            plugin_registry=mock_plugin_registry,
        )

        mock_worker = AsyncMock()
        mock_worker.retry.side_effect = lambda child_id: child_id in ("child-completed", "child-failed")
        manager.worker = mock_worker

        await manager.retry_download("parent-1")

        assert mock_worker.retry.await_count == 3
        mock_worker.retry.assert_any_await("child-completed")
        mock_worker.retry.assert_any_await("child-failed")
        mock_worker.retry.assert_any_await("child-pending")
        mock_download_repo.refresh_group.assert_called_once_with("parent-1")

    @pytest.mark.asyncio
    async def test_retry_group_raises_only_when_no_child_retryable(self, mock_download_repo, mock_plugin_registry):
        """A grouped retry raises only when every child is non-retryable."""
        from src.features.downloads.exceptions import DownloadOperationException

        parent = Download(
            id="parent-1", type=DownloadType.HF_REPO,
            status=DownloadStatus.FAILED, filename="repo",
        )
        pending_child = Download(
            id="child-pending", type=DownloadType.MODEL,
            status=DownloadStatus.PENDING, filename="c.safetensors", group_id="parent-1",
        )

        mock_download_repo.get_by_id.return_value = parent
        mock_download_repo.get_children.return_value = [pending_child]

        manager = _build_queue(
            download_repository=mock_download_repo,
            plugin_registry=mock_plugin_registry,
        )

        mock_worker = AsyncMock()
        mock_worker.retry.return_value = False
        manager.worker = mock_worker

        with pytest.raises(DownloadOperationException):
            await manager.retry_download("parent-1")


class TestFilenameOverride:
    """Tests for filename override in download queueing"""

    @pytest.mark.asyncio
    async def test_queue_with_custom_filename(self, mock_download_repo, mock_plugin_registry, sample_download):
        """Test that filename override takes precedence"""

        mock_download_repo.create.return_value = sample_download

        manager = _build_queue(
            download_repository=mock_download_repo,
            plugin_registry=mock_plugin_registry,
        )

        mock_worker = AsyncMock()
        mock_worker.get_queue_position.return_value = 0
        manager.worker = mock_worker

        with patch.object(manager, 'conn', AsyncMock()) as mock_conn:
            mock_conn.send_download_queued = AsyncMock()

            result = await manager.queue_model_download(
                url="https://example.com/model.safetensors",
                filename="custom_name.safetensors"
            )

        create_call = mock_download_repo.create.call_args
        created_download = create_call[0][0]
        assert created_download.filename == "custom_name.safetensors"


class TestChecksumVerification:
    """Tests for checksum verification"""

    @pytest.mark.asyncio
    async def test_checksum_stored_with_download(self, mock_download_repo, mock_plugin_registry, sample_download):
        """Test that checksum is stored with the download"""

        mock_download_repo.create.return_value = sample_download

        manager = _build_queue(
            download_repository=mock_download_repo,
            plugin_registry=mock_plugin_registry,
        )

        mock_worker = AsyncMock()
        mock_worker.get_queue_position.return_value = 0
        manager.worker = mock_worker

        with patch.object(manager, 'conn', AsyncMock()) as mock_conn:
            mock_conn.send_download_queued = AsyncMock()

            result = await manager.queue_model_download(
                url="https://example.com/model.safetensors",
                checksum_sha256="abc123def456"
            )

        create_call = mock_download_repo.create.call_args
        created_download = create_call[0][0]
        assert created_download.checksum_sha256 == "abc123def456"


class TestQueueLifecycle:
    """Tests for queue start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_queue_start_creates_worker(self, mock_download_repo, mock_plugin_registry):
        """Test that starting the queue creates a worker."""

        manager = _build_queue(
            download_repository=mock_download_repo,
            plugin_registry=mock_plugin_registry,
        )

        with patch('src.features.downloads.queue.DownloadWorker') as MockWorker:
            mock_worker_instance = AsyncMock()
            MockWorker.return_value = mock_worker_instance

            await manager.start()

            MockWorker.assert_called_once()
            mock_worker_instance.start.assert_called_once()
            assert manager.worker is not None

            await manager.stop()

    @pytest.mark.asyncio
    async def test_queue_stop_stops_worker(self, mock_download_repo, mock_plugin_registry):
        """Test that stopping the manager stops the worker"""

        manager = _build_queue(
            download_repository=mock_download_repo,
            plugin_registry=mock_plugin_registry,
        )

        with patch('src.features.downloads.queue.DownloadWorker') as MockWorker:
            mock_worker_instance = AsyncMock()
            MockWorker.return_value = mock_worker_instance

            await manager.start()
            await manager.stop()

            mock_worker_instance.stop.assert_called_once()
            assert manager.worker is None
