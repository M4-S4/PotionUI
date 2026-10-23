from unittest.mock import AsyncMock, Mock

import pytest

from src.features.automation.dto import CreateAutomationRequest, HttpRequestProbeRequest
from src.features.automation.http_request import HttpRequestError
from src.features.automation.nodes import actions
from src.features.automation.routes import AutomationController
from src.platform.plugins.automation_nodes import NodeTypeRegistry
from src.platform.security.user import AccountType, User


@pytest.fixture
def manager():
    manager = Mock()
    manager.create = AsyncMock()
    manager.test_http_request = AsyncMock(return_value={"status": 200})
    manager.repository = Mock()
    return manager


@pytest.fixture
def controller(manager):
    registry = NodeTypeRegistry()
    actions.register(registry)
    return AutomationController(manager, registry=registry)


@pytest.fixture
def admin():
    return Mock(spec=User, id="admin-1", account_type=AccountType.ADMIN)


@pytest.fixture
def member():
    return Mock(spec=User, id="member-1", account_type=AccountType.USER)


def http_graph():
    return {"nodes": [{"id": "n1", "type": "action.http_request", "config": {"url": "https://a.example.com"}}], "edges": []}


@pytest.mark.asyncio
async def test_member_cannot_author_a_graph_with_an_http_request_node(controller, manager, member):
    result = await controller.create_automation(CreateAutomationRequest(name="x", graph=http_graph()), member)
    assert result.success is False
    assert result.error == "admin_required"
    manager.create.assert_not_called()


def test_test_endpoint_sits_behind_the_admin_gated_router(controller):
    from types import SimpleNamespace

    from src.features.automation.routes import build_router
    from src.platform.security.current_user import get_current_admin_user

    router = build_router(SimpleNamespace(automation_controller=controller))
    route = next(r for r in router.routes if r.path == "/api/automations/http-request/test")
    assert "POST" in route.methods
    assert any(dep.dependency is get_current_admin_user for dep in router.dependencies)


@pytest.mark.asyncio
async def test_admin_test_endpoint_returns_the_output(controller, admin):
    result = await controller.test_http_request(HttpRequestProbeRequest(config={"url": "https://a.example.com"}), admin)
    assert result.success is True
    assert result.data == {"status": 200}


@pytest.mark.asyncio
async def test_blocked_target_comes_back_as_a_readable_error(controller, manager, admin):
    manager.test_http_request.side_effect = HttpRequestError("Requests to 127.0.0.1 are blocked.")
    result = await controller.test_http_request(HttpRequestProbeRequest(config={"url": "http://127.0.0.1"}), admin)
    assert result.success is False
    assert result.error == "http_request_failed"
    assert "blocked" in result.message


def test_node_catalog_marks_it_admin_only(controller, member):
    from src.features.automation.routes import _serialize_node_type

    spec = controller.registry.get("action.http_request")
    assert _serialize_node_type(spec)["requires_admin"] is True
