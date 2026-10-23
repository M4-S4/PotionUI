import hashlib
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.features.downloads.models import Download, DownloadSettings
from src.features.downloads.worker import DownloadWorker
from src.features.models import file_digest as file_digest_module
from src.features.models.hash_cache_repository import model_hash_cache_repo
from src.features.models.indexer import ModelScanner
from tests.features.downloads.fakes import FakeResponse, FakeSession

CHUNKS = (b"abcd", b"efgh", b"ijkl")
CONTENT = b"".join(CHUNKS)


def _worker():
    repo = Mock()
    repo.is_destination_claimed.return_value = False
    return DownloadWorker(
        settings=DownloadSettings(chunk_size_kb=1),
        repo=repo,
        connection_hub=AsyncMock(),
        provider_registry_factory=lambda: None,
    )


def _download(tmp_path, checksum=None):
    path = tmp_path / "m.safetensors"
    return Download(
        id="d1", url="https://example.invalid/m", filename=path.name, destination_path=str(path),
        filename_supplied=True, checksum_sha256=checksum,
    )


async def _fetch(worker, download, *responses):
    worker.session = FakeSession(*responses)
    assert await worker._download_file(download) is True


@pytest.mark.asyncio
async def test_a_streamed_download_seeds_the_hash_cache_with_the_digest_of_what_it_wrote(tmp_path, mock_db):
    worker, download = _worker(), _download(tmp_path)

    await _fetch(worker, download, FakeResponse(chunks=CHUNKS))

    stat = (tmp_path / "m.safetensors").stat()
    cached = model_hash_cache_repo.get(str(tmp_path / "m.safetensors"))
    assert (cached.sha256, cached.size, cached.mtime_ns) == (
        hashlib.sha256(CONTENT).hexdigest(), stat.st_size, stat.st_mtime_ns,
    )


@pytest.mark.asyncio
async def test_the_indexer_takes_the_streamed_digest_without_reading_the_file(tmp_path, mock_db):
    worker, download = _worker(), _download(tmp_path)
    await _fetch(worker, download, FakeResponse(chunks=CHUNKS))
    stat = (tmp_path / "m.safetensors").stat()

    with patch.object(ModelScanner, "calculate_sha256", side_effect=AssertionError("read twice")):
        found = ModelScanner(resolver=None)._digest_for(str(tmp_path / "m.safetensors"), stat.st_size, stat.st_mtime_ns)

    assert found == hashlib.sha256(CONTENT).hexdigest()


@pytest.mark.asyncio
async def test_a_resumed_download_hashes_the_part_it_already_had(tmp_path, mock_db):
    worker, download = _worker(), _download(tmp_path)
    (tmp_path / "m.safetensors.part").write_bytes(b"abcd")

    await _fetch(worker, download, FakeResponse(status=206, chunks=(b"efgh", b"ijkl"), headers={"Content-Length": "8"}))

    assert (tmp_path / "m.safetensors").read_bytes() == CONTENT
    assert model_hash_cache_repo.get(str(tmp_path / "m.safetensors")).sha256 == hashlib.sha256(CONTENT).hexdigest()


@pytest.mark.asyncio
async def test_the_checksum_is_verified_from_the_stream_without_reading_the_file_again(tmp_path, mock_db, monkeypatch):
    worker = _worker()
    download = _download(tmp_path, hashlib.sha256(CONTENT).hexdigest())
    await _fetch(worker, download, FakeResponse(chunks=CHUNKS))
    monkeypatch.setattr(file_digest_module, "read_chunks", lambda *a, **k: pytest.fail("read twice"))

    assert await worker._verify_checksum(download) is True
    assert worker._streamed_digests == {}


@pytest.mark.asyncio
async def test_a_streamed_digest_that_differs_from_the_checksum_fails_verification(tmp_path, mock_db):
    worker = _worker()
    download = _download(tmp_path, "0" * 64)
    await _fetch(worker, download, FakeResponse(chunks=CHUNKS))

    assert await worker._verify_checksum(download) is False


@pytest.mark.asyncio
async def test_a_digest_that_does_not_cover_the_whole_file_is_never_recorded(tmp_path, mock_db):
    worker, download = _worker(), _download(tmp_path)
    path = tmp_path / "m.safetensors"
    path.write_bytes(CONTENT)

    worker._record_streamed_digest(download, path, "f" * 64, len(CONTENT) - 1)

    assert model_hash_cache_repo.get(str(path)) is None


@pytest.mark.asyncio
async def test_a_server_that_ignores_the_range_request_restarts_the_file_instead_of_appending(tmp_path, mock_db):
    worker, download = _worker(), _download(tmp_path)
    (tmp_path / "m.safetensors.part").write_bytes(b"abcd")

    await _fetch(worker, download, FakeResponse(status=200, chunks=CHUNKS))

    assert (tmp_path / "m.safetensors").read_bytes() == CONTENT
    assert model_hash_cache_repo.get(str(tmp_path / "m.safetensors")).sha256 == hashlib.sha256(CONTENT).hexdigest()


@pytest.mark.asyncio
async def test_a_failed_download_leaves_no_streamed_digest_behind(tmp_path, mock_db):
    worker = _worker()
    worker.settings = DownloadSettings(chunk_size_kb=1, auto_retry_failed=False, max_retries=0)
    download = _download(tmp_path, hashlib.sha256(CONTENT).hexdigest())
    worker.repo.get_by_id.return_value = download
    worker._streamed_digests[download.id] = "stale"

    async def failing(self_, dl):
        raise RuntimeError("boom")

    with patch.object(DownloadWorker, "_download_file", failing):
        await worker._process_download(download.id)

    assert worker._streamed_digests == {}
