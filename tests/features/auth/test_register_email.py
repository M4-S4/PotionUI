from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.bootstrap.errors import register_error_handlers
from src.features.auth.routes import build_router
from src.platform.security import Auth, LoginHandoffStore, current_user
from src.platform.security.user import AccountType, User


@pytest.fixture
def auth():
    manager = Mock(spec=Auth)
    manager.register.side_effect = lambda *, username, email, password, **kwargs: (
        User(id="u1", username=username, email=email, password_hash="h", account_type=AccountType.ADMIN),
        "a.b.c",
    )
    return manager


@pytest.fixture
def client(auth):
    previous = current_user._auth
    current_user.set_auth(auth)
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(
        build_router(SimpleNamespace(auth=auth, login_handoff=LoginHandoffStore(), content_policy_resolver=None))
    )
    with TestClient(app) as test_client:
        yield test_client
    current_user._auth = previous


def _body(email):
    return {"username": "owner", "email": email, "password": "longenough1"}


@pytest.mark.parametrize("email", ["admin@potionui.local", "me@localhost", "x@example.test"])
def test_register_accepts_local_emails(client, auth, email):
    response = client.post("/api/auth/register", json=_body(email))
    assert response.status_code == 200
    assert auth.register.call_args.kwargs["email"] == email


def test_register_rejects_malformed_email_with_field_location(client):
    response = client.post("/api/auth/register", json=_body("not-an-email"))
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert errors[0]["loc"] == ["body", "email"]
    assert "@-sign" in errors[0]["msg"]
