import contextlib

import httpx
import pytest

from src.features.prompt_database.embedding import OllamaEmbeddingProvider
from src.features.remote_execution.transport import WorkerTransport
from src.features.remote_execution.worker.model_depot import ModelDepot
from src.platform.filesystem.s3_driver import S3FileStorageDriver
from src.platform.http.tls import client_ssl_context
from src.platform.worker_protocol import ContentDigest
from src.platform.worker_protocol.model_fetch import ModelFetchRequestV1


@pytest.fixture
def constructed(monkeypatch):
    seen = []

    class Stop(Exception):
        pass

    def recording(*args, **kwargs):
        seen.append(kwargs)
        raise Stop()

    monkeypatch.setattr(httpx, "AsyncClient", recording)
    monkeypatch.setattr(httpx, "Client", recording)
    return seen


def _shared(seen):
    return bool(seen) and all(kwargs.get("verify") is client_ssl_context() for kwargs in seen)


@pytest.mark.asyncio
async def test_ollama_embedding_provider_uses_shared_context(constructed):
    provider = OllamaEmbeddingProvider()
    with contextlib.suppress(Exception):
        await provider.embed(["a"])
    with contextlib.suppress(Exception):
        await provider.is_available()

    assert len(constructed) == 2
    assert _shared(constructed)


def test_worker_transport_client_uses_shared_context(constructed):
    transport = WorkerTransport("http://worker.invalid", "token")
    with contextlib.suppress(Exception):
        transport._client(timeout=httpx.Timeout(5.0))

    assert _shared(constructed)


@pytest.mark.asyncio
async def test_model_depot_fetch_uses_shared_context(constructed, tmp_path):
    depot = ModelDepot(depot_dir=tmp_path)
    request = ModelFetchRequestV1(
        relative_path="fetched/model.safetensors",
        expected_digest=ContentDigest(algorithm="sha256", hex="0" * 64),
        expected_size=1,
        url="https://example.invalid/model.safetensors",
    )
    transfer = depot.start_fetch(request)
    with contextlib.suppress(Exception):
        await depot.run_fetch(transfer.id, request)

    assert _shared(constructed)


def test_s3_driver_default_client_uses_shared_context(constructed):
    with contextlib.suppress(Exception):
        S3FileStorageDriver(bucket="b", access_key_id="k", secret_access_key="s")

    assert _shared(constructed)
