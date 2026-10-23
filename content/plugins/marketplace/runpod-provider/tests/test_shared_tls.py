import asyncio

import httpx
import pytest

from backend import provisioning
from backend.catalog_client import fetch_catalog
from backend.client import RunPodClient
from backend.provisioning import RunPodProvisioningManager
from src.plugin_api.providers import client_ssl_context

from .test_catalog_client import CATALOG_PAYLOAD


class FakeHttp:
    instances = []

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.gets = 0
        self.closed = False
        FakeHttp.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        self.closed = True

    async def get(self, url, headers=None):
        self.gets += 1
        status = 200 if self.gets >= FakeHttp.answer_after else 503
        return httpx.Response(status)


@pytest.fixture
def fake_http(monkeypatch):
    FakeHttp.instances = []
    FakeHttp.answer_after = 3
    monkeypatch.setattr(provisioning.httpx, "AsyncClient", FakeHttp)
    return FakeHttp


async def _noop_report(progress):
    return None


async def _no_sleep(seconds):
    return None


def _manager(**kwargs):
    return RunPodProvisioningManager(None, None, None, sleep=kwargs.pop("sleep", _no_sleep), **kwargs)


def test_rest_client_uses_shared_tls_context(monkeypatch):
    seen = []
    real = httpx.AsyncClient

    def recording(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", recording)
    RunPodClient(api_key="k", transport=httpx.MockTransport(lambda request: httpx.Response(200)))

    assert seen[0]["verify"] is client_ssl_context()


@pytest.mark.asyncio
async def test_catalog_client_uses_shared_tls_context(monkeypatch):
    seen = []
    real = httpx.AsyncClient

    def recording(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", recording)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=CATALOG_PAYLOAD))
    await fetch_catalog("k", transport=transport)

    assert seen[0]["verify"] is client_ssl_context()


@pytest.mark.asyncio
async def test_readiness_loop_reuses_one_client_with_shared_context(fake_http):
    manager = _manager(handshake_attempts=5)

    assert await manager._wait_for_worker("https://p", "tok", _noop_report) is True

    assert len(fake_http.instances) == 1
    assert fake_http.instances[0].gets == 3
    assert fake_http.instances[0].kwargs["verify"] is client_ssl_context()
    assert fake_http.instances[0].closed


@pytest.mark.asyncio
async def test_readiness_client_is_closed_when_the_loop_times_out(fake_http):
    fake_http.answer_after = 100
    manager = _manager(handshake_attempts=4)

    assert await manager._wait_for_worker("https://p", "tok", _noop_report) is False

    assert len(fake_http.instances) == 1
    assert fake_http.instances[0].gets == 4
    assert fake_http.instances[0].closed


@pytest.mark.asyncio
async def test_readiness_client_is_closed_when_the_loop_is_cancelled(fake_http):
    fake_http.answer_after = 100
    started = asyncio.Event()

    async def blocking_sleep(seconds):
        started.set()
        await asyncio.Event().wait()

    manager = _manager(handshake_attempts=4, sleep=blocking_sleep)
    task = asyncio.create_task(manager._wait_for_worker("https://p", "tok", _noop_report))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(fake_http.instances) == 1
    assert fake_http.instances[0].closed


@pytest.mark.asyncio
async def test_injected_probe_never_builds_a_client(fake_http):
    calls = []

    async def probe(base_url, token):
        calls.append((base_url, token))
        return True

    manager = _manager(readiness_probe=probe)

    assert await manager._wait_for_worker("https://p", "tok", _noop_report) is True
    assert calls == [("https://p", "tok")]
    assert fake_http.instances == []
