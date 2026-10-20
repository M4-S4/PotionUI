import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.bootstrap.errors import register_error_handlers
from src.features.plans.errors import LimitExceeded
from src.platform.http.base_controller import BaseController
from src.platform.http.error_pipeline import Fail, guarded, raw_detail
from src.platform.security.failure_text import failure_detail
from src.platform.security.user import AccountType


class Missing(Exception):
    pass


class User:
    def __init__(self, admin):
        self.account_type = AccountType.ADMIN if admin else AccountType.USER


TABLE = {Missing: Fail("missing", "Gone", status=404), KeyError: Fail("key_error")}
FALLBACK = Fail("failed", "Failed", detail=failure_detail)


class Stub(BaseController):
    @guarded(TABLE, FALLBACK, admin="user")
    async def run(self, raised, user):
        raise raised

    @guarded(TABLE, FALLBACK, admin="user")
    def run_sync(self, raised, user):
        raise raised

    @guarded(TABLE)
    async def no_fallback(self, raised):
        raise raised

    @guarded({}, Fail("failed", "Failed", detail=raw_detail))
    async def raw(self, raised):
        raise raised

    @guarded(TABLE, FALLBACK, admin="user")
    async def ok(self, user):
        return {"ok": True}


def refusal():
    return LimitExceeded([{
        "kind": "storage_bytes", "code": "storage_quota_exceeded", "label": "Storage", "format": "bytes",
        "used": 1, "limit": 1, "percent": 100.0, "resets_at": None, "message": "Full.",
    }], "upload", "")


def client():
    app = FastAPI()
    register_error_handlers(app)
    stub = Stub()

    @app.get("/soft")
    async def soft():
        return await stub.run(KeyError("k"), User(False))

    @app.get("/hard")
    async def hard():
        return await stub.run(Missing("x"), User(False))

    @app.get("/leak/{role}")
    async def leak(role: str):
        return await stub.run(RuntimeError("db password=hunter2"), User(role == "admin"))

    @app.get("/sync")
    def sync():
        return stub.run_sync(RuntimeError("db password=hunter2"), User(True))

    @app.get("/limit")
    async def limit():
        return await stub.run(refusal(), User(True))

    @app.get("/ok")
    async def ok():
        return await stub.ok(User(False))

    return TestClient(app, raise_server_exceptions=False)


def test_a_table_entry_without_status_is_a_200_soft_envelope():
    response = client().get("/soft")

    assert response.status_code == 200
    assert response.json() == {"success": False, "data": None, "message": "'k'", "error": "key_error"}


def test_a_table_entry_with_status_raises_that_status_code_and_message():
    response = client().get("/hard")

    assert response.status_code == 404
    assert response.json()["detail"]["error"] == "missing"
    assert response.json()["detail"]["message"] == "Gone"


def test_the_fallback_hides_raw_text_from_non_admins():
    body = client().get("/leak/user").json()

    assert body["error"] == "failed"
    assert "db password=hunter2" not in body["message"]


def test_the_fallback_shows_raw_text_to_admins():
    assert "db password=hunter2" in client().get("/leak/admin").json()["message"]


def test_a_sync_handler_binds_admin_through_the_signature():
    response = client().get("/sync")

    assert response.status_code == 200
    assert "db password=hunter2" in response.json()["message"]


def test_a_successful_call_returns_its_value():
    assert client().get("/ok").json() == {"ok": True}


async def test_without_a_fallback_an_unmapped_exception_propagates():
    with pytest.raises(RuntimeError):
        await Stub().no_fallback(RuntimeError("x"))


async def test_a_raw_detail_fallback_is_the_exception_text():
    response = await Stub().raw(RuntimeError("boom"))

    assert response.message == "Failed: boom"


def test_a_pass_through_bypasses_the_table_and_the_global_handler_answers_403():
    response = client().get("/limit")

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "storage_quota_exceeded"
