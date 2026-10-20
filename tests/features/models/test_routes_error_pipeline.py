from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from src.features.downloads.exceptions import DownloadNotFoundException
from src.features.models import (
    ModelDownloadException,
    ModelIndexingException,
    ModelNotFoundException,
    routes,
)
from src.features.models.exceptions import (
    InvalidModelTypeException,
    ModelTypeConflictException,
    ModelTypeNotAssignableException,
)
from src.features.models.search_filter import InvalidModelSearch
from src.features.models.routes import ModelController
from src.platform.security.user import AccountType

LEAKY = "connection to /srv/secrets/db.sqlite refused"


def _user(account_type):
    return Mock(id="u1", account_type=account_type)


@pytest.fixture
def controller():
    return ModelController(Mock(), Mock(), Mock())


async def test_an_unexpected_failure_hides_the_exception_text_from_a_non_admin(controller, monkeypatch):
    monkeypatch.setattr(routes.operations, "list_models", AsyncMock(side_effect=RuntimeError(LEAKY)))

    response = await controller.list_models(_user(AccountType.USER))

    assert response.success is False
    assert response.error == "list_models_failed"
    assert "secrets" not in response.message
    assert response.message.startswith("Failed to list models")


async def test_an_unexpected_failure_shows_an_admin_the_scrubbed_text(controller, monkeypatch):
    monkeypatch.setattr(routes.operations, "list_models", AsyncMock(side_effect=RuntimeError(LEAKY)))

    response = await controller.list_models(_user(AccountType.ADMIN))

    assert "refused" in response.message
    assert "/srv/secrets" not in response.message


async def test_one_exception_class_keeps_a_different_code_per_endpoint(controller, monkeypatch):
    monkeypatch.setattr(routes.operations, "start_indexing", Mock(side_effect=ModelIndexingException("busy")))
    monkeypatch.setattr(routes.operations, "delete_model", Mock(side_effect=ModelIndexingException("in use")))

    started = await controller.index_models()
    deleted = await controller.delete_model("m1")

    assert (started.error, started.message) == ("indexing_blocked", "busy")
    assert (deleted.error, deleted.message) == ("delete_blocked", "in use")


async def test_a_type_conflict_is_a_409_with_the_domain_message(controller, monkeypatch):
    monkeypatch.setattr(
        routes.operations, "set_model_type", Mock(side_effect=ModelTypeConflictException("taken", "m2"))
    )
    request = Mock(model_type="lora")

    with pytest.raises(HTTPException) as caught:
        await controller.set_model_type("m1", request, _user(AccountType.ADMIN))

    assert caught.value.status_code == 409
    assert caught.value.detail["error"] == "model_type_conflict"
    assert caught.value.detail["message"] == "taken"


def _ops(name, error, asynchronous=False):
    def arrange(controller, monkeypatch):
        fake = AsyncMock(side_effect=error) if asynchronous else Mock(side_effect=error)
        monkeypatch.setattr(routes.operations, name, fake)
    return arrange


def _search(controller, monkeypatch):
    monkeypatch.setattr(routes, "parse_model_search", Mock(side_effect=InvalidModelSearch("bad search")))


def _favorite(controller, monkeypatch):
    controller.user_model_meta_repository.set_favorite.side_effect = ValueError("not allowed")


def _library_name(controller, monkeypatch):
    controller.user_model_meta_repository.set_custom_name.side_effect = ValueError("too long")


def _status(error):
    def arrange(controller, monkeypatch):
        controller.download_queue.get_download = Mock(side_effect=error)
    return arrange


ADMIN = AccountType.ADMIN
CASES = [
    ("search", _search, lambda c: c.list_models(_user(ADMIN)), 422, "invalid_model_search", "bad search"),
    ("type-missing", _ops("set_model_type", ModelNotFoundException("gone")),
     lambda c: c.set_model_type("m", Mock(model_type="x"), _user(ADMIN)), 404, "model_not_found", "gone"),
    ("type-invalid", _ops("set_model_type", InvalidModelTypeException("no")),
     lambda c: c.set_model_type("m", Mock(model_type="x"), _user(ADMIN)), 422, "model_type_not_assignable", "no"),
    ("type-conflict", _ops("set_model_type", ModelTypeConflictException("taken", "m2")),
     lambda c: c.set_model_type("m", Mock(model_type="x"), _user(ADMIN)), 409, "model_type_conflict", "taken"),
    ("reset-missing", _ops("reset_model_type", ModelNotFoundException("gone")),
     lambda c: c.reset_model_type("m"), 404, "model_not_found", "gone"),
    ("reset-unassignable", _ops("reset_model_type", ModelTypeNotAssignableException("no")),
     lambda c: c.reset_model_type("m"), 422, "model_type_not_assignable", "no"),
    ("reset-conflict", _ops("reset_model_type", ModelTypeConflictException("taken")),
     lambda c: c.reset_model_type("m"), 409, "model_type_conflict", "taken"),
    ("by-id-missing", _ops("get_model_by_id", ModelNotFoundException("gone"), True),
     lambda c: c.get_model_by_id("m", _user(ADMIN)), None, "model_not_found", "gone"),
    ("download-missing", _status(DownloadNotFoundException("no such download")),
     lambda c: c.get_recommendation_download("d"), None, "download_not_found", "no such download"),
    ("download-leak", _status(RuntimeError(LEAKY)),
     lambda c: c.get_recommendation_download("d"), None, "download_status_failed", None),
    ("reorder", _ops("reorder_model_previews", ModelIndexingException("stale order")),
     lambda c: c.reorder_model_previews("m", Mock(ordered_ids=["a"])), None, "invalid_reorder", "stale order"),
    ("favorite", _favorite,
     lambda c: c.set_model_favorite("m", Mock(is_favorite=True), _user(ADMIN)), None, "set_favorite_failed", "not allowed"),
    ("library-name", _library_name,
     lambda c: c.set_model_library_name("m", SimpleNamespace(name="n"), _user(ADMIN)), None, "set_library_name_failed", "too long"),
    ("download-blocked", _ops("start_download_and_index", ModelDownloadException("blocked")),
     lambda c: c.download_and_index_model(Mock(), Mock(model_type=None)), None, "download_blocked", "blocked"),
]


@pytest.mark.parametrize("arrange,call,status,code,message", [case[1:] for case in CASES], ids=[case[0] for case in CASES])
async def test_each_refusal_keeps_its_status_and_code(controller, monkeypatch, arrange, call, status, code, message):
    arrange(controller, monkeypatch)

    if status is None:
        response = await call(controller)
        assert (response.success, response.error) == (False, code)
        assert LEAKY not in (response.message or "")
        if message is not None:
            assert response.message == message
        return

    with pytest.raises(HTTPException) as caught:
        await call(controller)
    assert caught.value.status_code == status
    assert caught.value.detail["error"] == code
    assert caught.value.detail["message"] == message


async def test_a_blocked_download_schedules_no_background_task(controller, monkeypatch):
    monkeypatch.setattr(routes.operations, "start_download_and_index", Mock(side_effect=ModelDownloadException("x")))
    tasks = Mock()

    response = await controller.download_and_index_model(tasks, Mock(model_type=None))

    assert response.error == "download_blocked"
    tasks.add_task.assert_not_called()
