import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from unittest.mock import MagicMock, Mock

from src.features.models.indexing_coordinator import ModelIndexingCoordinator
from src.features.models.routes import ModelController, build_router
from src.platform.security.current_user import get_current_active_user, get_current_admin_user
from src.platform.security.user import AccountType, User
from tests.features.models.fakes import OpenPlugins, ProgressScanner


@pytest.fixture
def app():
    coordinator = ModelIndexingCoordinator(
        model_repository=MagicMock(), plugin_registry=OpenPlugins(), scanner=ProgressScanner(), spawn=lambda target: None
    )
    collaborators = Mock()
    collaborators.indexing = coordinator
    user = Mock(spec=User)
    user.id = "admin-1"
    user.account_type = AccountType.ADMIN
    container = Mock()
    container.model_controller = ModelController(collaborators, Mock(), Mock())
    fastapi_app = FastAPI()
    fastapi_app.include_router(build_router(container))
    fastapi_app.dependency_overrides[get_current_admin_user] = lambda: user
    fastapi_app.state.coordinator = coordinator
    return fastapi_app


@pytest.mark.asyncio
async def test_the_level_can_be_changed_and_read_back(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        changed = await client.put("/api/models/indexing/settings", json={"level": "normal"})
        unchanged = await client.put("/api/models/indexing/settings", json={})

    assert changed.status_code == 200
    assert changed.json()["data"] == {"level": "normal"}
    assert unchanged.json()["data"] == {"level": "normal"}
    assert app.state.coordinator.scanner.governor.level() == "normal"


@pytest.mark.asyncio
async def test_an_unknown_level_is_rejected_and_changes_nothing(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.put("/api/models/indexing/settings", json={"level": "turbo"})

    assert response.status_code == 422
    assert app.state.coordinator.scanner.governor.level() == "low"


@pytest.mark.asyncio
async def test_a_regular_user_is_refused_by_the_real_admin_check_and_nothing_changes(app):
    regular = Mock(spec=User)
    regular.id = "user-1"
    regular.account_type = AccountType.USER
    app.dependency_overrides.clear()
    app.dependency_overrides[get_current_active_user] = lambda: regular

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.put("/api/models/indexing/settings", json={"level": "max"})

    assert response.status_code == 403
    assert app.state.coordinator.scanner.governor.level() == "low"
