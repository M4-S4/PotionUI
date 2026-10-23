"""`RemoteModelsController`'s mapping of a `WorkerTransportError` onto the
`APIResponse` error code the frontend section branches on - `worker_unreachable`
for a genuine connect failure or a real worker-produced protocol error,
`worker_not_running` for a gateway answering on a stopped/still-starting
worker's behalf (see `WorkerUnreachableError.reason` in `transport.py`).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.features.remote_execution import routes as routes_module
from src.features.remote_execution.routes import RemoteModelsController
from src.features.remote_execution.transport import WorkerProtocolError, WorkerUnreachableError


class StubTransport:
    def __init__(self, exc: Exception):
        self._exc = exc

    async def list_transfers(self):
        raise self._exc


async def _transfers_response(exc: Exception):
    controller = RemoteModelsController(container=object())
    with patch.object(RemoteModelsController, "_transport", return_value=StubTransport(exc)):
        return await controller.transfers("backend-1")


@pytest.mark.asyncio
async def test_a_not_running_worker_maps_to_worker_not_running():
    exc = WorkerUnreachableError(
        "worker is not running (gateway answered HTTP 404 for /v1/models/transfers)", reason="not_running",
    )

    response = await _transfers_response(exc)

    assert response.success is False
    assert response.error == "worker_not_running"
    assert response.message == "The worker is stopped or still starting"


@pytest.mark.asyncio
async def test_a_connect_failure_stays_worker_unreachable():
    exc = WorkerUnreachableError("could not reach worker at http://fake-worker: connection refused")

    response = await _transfers_response(exc)

    assert response.success is False
    assert response.error == "worker_unreachable"
    assert response.message == str(exc)


@pytest.mark.asyncio
async def test_a_real_protocol_error_stays_worker_unreachable():
    exc = WorkerProtocolError("transfer listing failed: HTTP 500 boom")

    response = await _transfers_response(exc)

    assert response.success is False
    assert response.error == "worker_unreachable"
    assert response.message == str(exc)


@pytest.mark.asyncio
async def test_the_push_route_hands_the_digest_service_to_the_bundle_builder():
    digests = object()
    container = SimpleNamespace(
        model_repository="repository", model_locator="locator", model_index_manager=SimpleNamespace(digests=digests),
    )
    captured = {}

    async def push_models(model_ids, **kwargs):
        captured.update(kwargs, model_ids=model_ids)
        return [{"model_id": "m1"}]

    controller = RemoteModelsController(container=container)
    with patch.object(RemoteModelsController, "_transport", return_value="transport"), patch.object(
        routes_module.ops, "push_models", push_models
    ):
        response = await controller.push("backend-1", SimpleNamespace(model_ids=["m1"]))

    assert response.success is True
    assert captured["digests"] is digests
    assert (captured["model_ids"], captured["model_repository"], captured["model_locator"], captured["transport"]) == (
        ["m1"], "repository", "locator", "transport",
    )
