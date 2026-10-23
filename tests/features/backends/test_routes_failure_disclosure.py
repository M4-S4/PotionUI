from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException

from src.features.backends.backend_config import BackendConfigStore
from src.features.backends.model_listing import ModelListingNotSupported
from src.features.backends.routes import BackendController
from src.platform.security.failure_text import GENERIC_FAILURE
from src.platform.security.user import AccountType, User
from src.platform.settings.settings import Settings

LEAK = "secret connection string at /srv/app/db.sqlite"


def _user(account_type):
    user = Mock(spec=User)
    user.account_type = account_type
    return user


@pytest.fixture
def controller():
    store = Mock(spec=BackendConfigStore)
    registry = Mock()
    registry.backend_config_store = store
    return BackendController(Mock(spec=Settings), registry)


async def _refusal(call):
    with pytest.raises(HTTPException) as caught:
        await call
    return caught.value


@pytest.mark.asyncio
async def test_generic_failure_on_a_user_route_hides_the_exception(controller):
    controller.backend_config_store.get_backends.side_effect = RuntimeError(LEAK)

    refusal = await _refusal(controller.list_backends())

    assert refusal.status_code == 400
    assert refusal.detail["error"] == "backend_list_failed"
    assert "secret" not in refusal.detail["message"]
    assert GENERIC_FAILURE in refusal.detail["message"]


@pytest.mark.asyncio
async def test_admin_only_route_shows_the_scrubbed_exception(controller):
    controller.backend_config_store.get_backend.return_value = Mock()
    controller.backend_config_store.remove_backend.side_effect = RuntimeError(LEAK)

    refusal = await _refusal(controller.delete_backend("native-1"))

    assert refusal.status_code == 400
    assert refusal.detail["error"] == "backend_delete_failed"
    assert "secret connection string" in refusal.detail["message"]
    assert "/srv/app" not in refusal.detail["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize("account_type,shows_detail", [(AccountType.ADMIN, True), (AccountType.USER, False)])
async def test_optimization_route_detail_depends_on_the_caller(controller, account_type, shows_detail):
    user = _user(account_type)
    with patch.object(controller, "_require_admin"), \
            patch.object(controller, "_require_local_backend", side_effect=RuntimeError(LEAK)):
        refusal = await _refusal(controller.get_backend_optimizations("native-1", user))

    assert refusal.detail["error"] == "backend_optimizations_failed"
    assert ("secret connection string" in refusal.detail["message"]) is shows_detail
    assert "/srv/app" not in refusal.detail["message"]


@pytest.mark.asyncio
async def test_value_error_keeps_its_message_and_status(controller):
    controller.backend_config_store.set_default_backend.side_effect = ValueError("Backend 'x' not found")

    refusal = await _refusal(controller.set_default_backend("x"))

    assert refusal.status_code == 404
    assert refusal.detail["error"] == "backend_not_found"
    assert refusal.detail["message"] == "Backend 'x' not found"


@pytest.mark.asyncio
@pytest.mark.parametrize("call,probe,key", [
    ("list_backend_health", "health_check", "health"),
    ("get_backend_health", "health_check", "health"),
    ("get_backend_system_info", "get_system_info", None),
])
async def test_probe_failure_is_scrubbed_inside_the_200_body(controller, call, probe, key):
    config = Mock(id="native-1", enabled=True, engine="native")
    config.name = "Local"
    controller.backend_config_store.get_backends.return_value = [config]
    controller.backend_config_store.get_backend.return_value = config
    instance = Mock()
    setattr(instance, probe, AsyncMock(side_effect=RuntimeError(LEAK)))
    controller.backend_registry.get_backend.return_value = instance

    method = getattr(controller, call)
    response = await (method() if call == "list_backend_health" else method("native-1"))

    payload = response.data[0] if call == "list_backend_health" else response.data
    body = payload[key] if key else payload
    assert "/srv/app" not in body["error"]
    assert "<server path>" in body["error"]
    assert "secret connection string" in body["error"]


@pytest.mark.asyncio
async def test_unsupported_model_listing_answers_400_with_its_code(controller):
    backend = Mock(engine="comfyui")
    backend.name = "Remote"
    controller.backend_registry.get_backend.return_value = backend
    unsupported = ModelListingNotSupported("This backend cannot list models")

    with patch("src.features.models.backend_indexer.backend_model_indexer.index_backend",
               AsyncMock(side_effect=unsupported)):
        refusal = await _refusal(controller.index_backend_models("remote-1"))

    assert refusal.status_code == 400
    assert refusal.detail["error"] == "model_listing_not_supported"
    assert refusal.detail["message"] == "This backend cannot list models"
