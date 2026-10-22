import asyncio
import hashlib
import os
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.features.downloads.models import Download, DownloadSettings
from src.features.downloads.worker import DownloadWorker
from src.features.models import file_digest as file_digest_module
from src.features.models.indexing_governor import IndexingGovernor
from src.features.models.read_throttle import ReadThrottle
from src.features.models.hash_cache_repository import model_hash_cache_repo
from src.features.models.indexer import ModelScanner


@pytest.fixture
def worker():
    return DownloadWorker(
        settings=DownloadSettings(max_concurrent_downloads=1, auto_retry_failed=False, max_retries=0, chunk_size_kb=1024),
        repo=Mock(),
        connection_hub=AsyncMock(),
        provider_registry_factory=lambda: None,
    )


def _download(path, checksum):
    return Download(id="d1", url="https://example.invalid/m", destination_path=str(path), filename=path.name, checksum_sha256=checksum)


@pytest.mark.asyncio
async def test_a_verified_download_seeds_the_hash_cache_under_its_path_size_and_mtime(worker, tmp_path, mock_db):
    content = b"downloaded weights"
    path = tmp_path / "model.safetensors"
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()

    assert await worker._verify_checksum(_download(path, digest)) is True

    cached = model_hash_cache_repo.get(str(path))
    stat = path.stat()
    assert (cached.sha256, cached.size, cached.mtime_ns) == (digest, stat.st_size, stat.st_mtime_ns)


@pytest.mark.asyncio
async def test_the_indexer_reuses_the_seeded_digest_without_reading_the_file_again(worker, tmp_path, mock_db):
    content = b"downloaded weights"
    path = tmp_path / "model.safetensors"
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    await worker._verify_checksum(_download(path, digest))
    stat = path.stat()

    with patch.object(ModelScanner, "calculate_sha256", side_effect=AssertionError("must not read the file again")):
        found = ModelScanner(resolver=None)._digest_for(str(path), stat.st_size, stat.st_mtime_ns)

    assert found == digest


@pytest.mark.asyncio
async def test_the_cache_records_the_file_as_it_was_before_hashing_so_a_later_change_is_not_trusted(
    worker, tmp_path, mock_db, monkeypatch
):
    content = b"downloaded weights"
    path = tmp_path / "model.safetensors"
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    before = path.stat()
    real = file_digest_module.read_chunks

    def changing_underfoot(chunk_path, *args, **kwargs):
        yield from real(chunk_path, *args, **kwargs)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 5_000_000_000))

    monkeypatch.setattr(file_digest_module, "read_chunks", changing_underfoot)

    assert await worker._verify_checksum(_download(path, digest)) is True

    cached = model_hash_cache_repo.get(str(path))
    assert cached.mtime_ns == before.st_mtime_ns
    assert cached.mtime_ns != path.stat().st_mtime_ns


@pytest.mark.asyncio
async def test_a_checksum_mismatch_seeds_nothing(worker, tmp_path, mock_db):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"downloaded weights")

    assert await worker._verify_checksum(_download(path, "0" * 64)) is False

    assert model_hash_cache_repo.get(str(path)) is None


@pytest.mark.asyncio
async def test_a_cache_failure_does_not_fail_the_verification(worker, tmp_path, mock_db):
    content = b"downloaded weights"
    path = tmp_path / "model.safetensors"
    path.write_bytes(content)

    with patch.object(model_hash_cache_repo, "put", side_effect=RuntimeError("database is locked")):
        assert await worker._verify_checksum(_download(path, hashlib.sha256(content).hexdigest())) is True


@pytest.mark.asyncio
async def test_a_slow_verification_never_blocks_the_event_loop(worker, tmp_path, mock_db, monkeypatch):
    content = b"downloaded weights"
    path = tmp_path / "model.safetensors"
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    real = file_digest_module.read_chunks

    def slow(chunk_path, *args, **kwargs):
        for chunk in real(chunk_path, *args, **kwargs):
            time.sleep(0.25)
            yield chunk

    monkeypatch.setattr(file_digest_module, "read_chunks", slow)
    ticks = []

    async def heartbeat():
        while True:
            ticks.append(time.monotonic())
            await asyncio.sleep(0.01)

    task = asyncio.create_task(heartbeat())
    started = time.monotonic()

    verified = await worker._verify_checksum(_download(path, digest))

    task.cancel()
    elapsed = time.monotonic() - started
    assert verified is True and elapsed >= 0.25
    assert len(ticks) >= 10
    gaps = [later - earlier for earlier, later in zip(ticks, ticks[1:])]
    assert max(gaps) < 0.15


@pytest.mark.asyncio
async def test_the_verification_read_is_paced_by_the_throttle_it_is_given(worker, tmp_path, mock_db):
    content = b"downloaded weights" * 1000
    path = tmp_path / "model.safetensors"
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    waits = []

    class Recording:
        evict_page_cache = False

        def wait(self, nbytes, should_stop=None):
            waits.append(nbytes)
            return True

        def holding(self):
            return False

    worker.verify_throttle = lambda: Recording()

    assert await worker._verify_checksum(_download(path, digest)) is True

    assert sum(waits) == len(content)


@pytest.mark.asyncio
async def test_a_corrupt_download_is_still_rejected_and_seeds_nothing(worker, tmp_path, mock_db):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"downloaded weights")

    assert await worker._verify_checksum(_download(path, "f" * 64)) is False

    assert model_hash_cache_repo.get(str(path)) is None


@pytest.mark.asyncio
async def test_a_download_with_no_checksum_is_not_read_at_all(worker, tmp_path, mock_db, monkeypatch):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"downloaded weights")
    monkeypatch.setattr(file_digest_module, "read_chunks", lambda *args, **kwargs: pytest.fail("read"))

    assert await worker._verify_checksum(_download(path, None)) is True


def test_the_governor_hands_out_a_cap_only_throttle_for_verification():
    governor = IndexingGovernor(throttle=ReadThrottle())
    governor.set_level("low")
    low = governor.verification_throttle()
    governor.set_level("max")
    maximum = governor.verification_throttle()

    assert low is not governor.throttle and low is not governor.verification_throttle()
    assert low.holding() is False
    assert (low._bytes_per_second, low._pauses_for_generation) == (96 * 1024 * 1024, False)
    assert (maximum._bytes_per_second, maximum._pauses_for_generation) == (None, False)
