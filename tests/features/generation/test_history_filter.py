from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.features.generation.history_filter import HistoryFilter, parse_tag_ids
from src.features.generation.history_facade import GenerationHistoryFacade
from src.features.generation.routes import GenerationController, build_router
from src.features.generation.run_report_recorder import RunReportRecorder
from src.platform.http.base_controller import APIResponse
from src.platform.security.current_user import get_current_active_user
from src.platform.security.user import AccountType, User


def _user():
    return User(
        id="user-1",
        username="user-1",
        email="user-1@example.com",
        password_hash="$2b$12$x",
        account_type=AccountType.USER,
        created_at=datetime.utcnow(),
        last_login=None,
    )


@pytest.fixture
def facade():
    facade = Mock(spec=GenerationHistoryFacade)
    facade.get_history_async = AsyncMock(
        return_value={"generations": [], "total": 0, "limit": 50, "offset": 0, "filters": {}}
    )
    return facade


@pytest.fixture
def client(facade):
    controller = GenerationController(Mock(), facade, Mock(), Mock(spec=RunReportRecorder))
    app = FastAPI()
    app.include_router(build_router(SimpleNamespace(_generation_controller=controller)))
    app.dependency_overrides[get_current_active_user] = lambda: _user()
    return TestClient(app)


def test_tag_ids_are_split_on_commas_and_trimmed():
    assert parse_tag_ids(" a, b ,,c") == ["a", "b", "c"]
    assert parse_tag_ids("") is None
    assert parse_tag_ids(None) is None


def test_overrides_replace_fields_on_a_copy():
    base = HistoryFilter(user_id="u1", status="completed")

    resolved = HistoryFilter.resolve(base, {"status": "failed", "search": "cat"})

    assert (resolved.user_id, resolved.status, resolved.search) == ("u1", "failed", "cat")
    assert base.status == "completed"


def test_an_unknown_filter_name_is_rejected():
    with pytest.raises(TypeError):
        HistoryFilter.resolve(None, {"statuss": "failed"})


def test_every_history_query_parameter_reaches_the_filter(client, facade):
    response = client.get(
        "/api/generations/history",
        params={
            "limit": 10,
            "offset": 5,
            "status": "completed",
            "created_from": "2024-01-01",
            "created_to": "2024-01-31",
            "completed_from": "2024-02-01",
            "completed_to": "2024-02-28",
            "tag_ids": "t1, t2",
            "include_tags": "false",
            "media_type": "IMAGE",
            "search": "fox",
            "mode": "txt2img",
            "preset_id": "p1",
            "model_name": "flux",
            "min_rating": 3,
            "favorites_only": "true",
            "collection_id": "c1",
            "include_descendants": "false",
            "unsorted": "true",
            "used_phrasebook_value_id": "pv1",
            "sort_by": "created_at",
            "sort_dir": "asc",
            "system_tag": "portrait",
            "semantic_query": "red fox",
            "grid_id": "g1",
        },
    )

    assert response.status_code == 200
    call = facade.get_history_async.call_args
    assert call.args[0] == HistoryFilter(
        user_id="user-1",
        status="completed",
        created_from="2024-01-01",
        created_to="2024-01-31",
        completed_from="2024-02-01",
        completed_to="2024-02-28",
        tag_ids=["t1", "t2"],
        media_type="IMAGE",
        search="fox",
        mode="txt2img",
        preset_id="p1",
        model_name="flux",
        min_rating=3,
        favorites_only=True,
        collection_id="c1",
        include_descendants=False,
        unsorted=True,
        used_phrasebook_value_id="pv1",
        system_tag="portrait",
        semantic_query="red fox",
        grid_id="g1",
    )
    assert call.kwargs == {
        "limit": 10,
        "offset": 5,
        "include_tags": False,
        "sort_by": "created_at",
        "sort_dir": "asc",
        "group_grids": True,
    }


def test_the_history_route_defaults_match_the_documented_contract(client, facade):
    client.get("/api/generations/history")

    call = facade.get_history_async.call_args
    assert call.args[0] == HistoryFilter(user_id="user-1")
    assert call.kwargs["limit"] == 50
    assert call.kwargs["include_tags"] is True


def test_a_caller_cannot_choose_another_users_history(client, facade):
    client.get("/api/generations/history", params={"user_id": "someone-else"})

    assert facade.get_history_async.call_args.args[0].user_id == "user-1"


def test_the_facade_keeps_the_keyword_form_for_tools_and_plugins():
    query = Mock()
    facade = GenerationHistoryFacade.__new__(GenerationHistoryFacade)
    facade._query = query

    facade.get_history(user_id="u1", search="cat", limit=7)

    query.get_history.assert_called_once_with(
        None, limit=7, offset=0, include_tags=True, sort_by=None, sort_dir=None, group_grids=False,
        user_id="u1", search="cat",
    )
