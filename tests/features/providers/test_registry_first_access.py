import asyncio

import pytest

import src.features.providers.registry as module
from src.features.providers.registry import ProviderRegistry


class _Provider:
    def matches_download_url(self, url):
        return True

    async def initialize(self, settings):
        return True


@pytest.fixture
def fresh_module(monkeypatch):
    async def discover(self):
        self._providers["acme"] = _Provider()
        return []

    monkeypatch.setattr(module, "_provider_registry", None)
    monkeypatch.setattr(module, "_discovery_done", False)
    monkeypatch.setattr(ProviderRegistry, "discover_providers", discover)
    monkeypatch.setattr(ProviderRegistry, "_get_provider_settings", lambda self, provider_id: {})
    monkeypatch.setattr(module, "PluginRepository", lambda: None)


@pytest.mark.asyncio
async def test_sync_accessor_inside_a_running_loop_starts_discovery(fresh_module):
    registry = module.get_provider_registry()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert registry.get_provider("acme") is not None


@pytest.mark.asyncio
async def test_ensure_after_sync_first_access_waits_for_discovery(fresh_module):
    module.get_provider_registry()

    registry = await module.ensure_providers_discovered()

    assert registry.get_provider("acme") is not None


@pytest.mark.asyncio
async def test_concurrent_ensure_calls_discover_once(fresh_module, monkeypatch):
    calls = []

    async def discover(self):
        calls.append(1)
        await asyncio.sleep(0)
        self._providers["acme"] = _Provider()
        return []

    monkeypatch.setattr(ProviderRegistry, "discover_providers", discover)

    first, second = await asyncio.gather(
        module.ensure_providers_discovered(), module.ensure_providers_discovered()
    )

    assert first is second
    assert first.get_provider("acme") is not None
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_ensure_after_reset_repopulates_the_registry(fresh_module):
    registry = await module.ensure_providers_discovered()
    module.reset_provider_registry()
    assert registry.get_provider("acme") is None

    again = await module.ensure_providers_discovered()

    assert again.get_provider("acme") is not None


@pytest.mark.asyncio
async def test_reset_mid_flight_hands_concurrent_awaiters_the_new_populated_registry(
    fresh_module, monkeypatch
):
    started = asyncio.Event()
    calls = []

    async def discover(self):
        calls.append(1)
        if len(calls) == 1:
            started.set()
            await asyncio.Event().wait()
        self._providers["acme"] = _Provider()
        return []

    monkeypatch.setattr(ProviderRegistry, "discover_providers", discover)

    waiter = asyncio.ensure_future(module.ensure_providers_discovered())
    await started.wait()
    module.reset_provider_registry()

    registry = await asyncio.wait_for(waiter, timeout=2)

    assert registry.get_provider("acme") is not None
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_a_failed_discovery_surfaces_once_and_the_next_ensure_retries(fresh_module, monkeypatch):
    calls = []

    async def discover(self):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("hook exploded")
        self._providers["acme"] = _Provider()
        return []

    monkeypatch.setattr(ProviderRegistry, "discover_providers", discover)

    with pytest.raises(RuntimeError, match="hook exploded"):
        await module.ensure_providers_discovered()

    registry = await module.ensure_providers_discovered()

    assert registry.get_provider("acme") is not None
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_cancelling_the_awaiter_still_raises_cancelled(fresh_module, monkeypatch):
    started = asyncio.Event()

    async def discover(self):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(ProviderRegistry, "discover_providers", discover)

    waiter = asyncio.ensure_future(module.ensure_providers_discovered())
    await started.wait()
    waiter.cancel()

    with pytest.raises(asyncio.CancelledError):
        await waiter
    module.reset_provider_registry()
