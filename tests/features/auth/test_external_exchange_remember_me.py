import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jose import jwt

from src.features.auth.routes import build_router
from src.platform.plugins import PluginRegistry
from src.platform.plugins.hooks import HookContext
from src.platform.security import Auth, LoginHandoffStore
from src.platform.security import current_user
from src.platform.security.config import AuthConfig
from src.platform.security.token import TokenCodec
from src.platform.security.user import AccountType, User

SECRET = "exchange-remember-me-secret"
DAY = 24 * 3600


@pytest.fixture
def user():
    return User(
        id="user-1", username="alice", email="alice@example.com",
        password_hash="h", account_type=AccountType.USER,
    )


@pytest.fixture
def auth(user):
    config = Mock(spec=AuthConfig)
    config.secret_key = SECRET
    config.algorithm = "HS256"
    config.access_token_expire_minutes = 24 * 60
    config.remember_me_token_expire_days = 30
    users = Mock()
    users.get_by_id.return_value = user
    registry = Mock(spec=PluginRegistry)
    registry.execute_hook.return_value = (HookContext(hook_name="t", plugin_id="t", data={}), [])
    return Auth(
        user_repository=users,
        password_hasher=Mock(),
        token_codec=TokenCodec(config),
        auth_config=config,
        plugin_registry=registry,
        instance_claim=Mock(),
        claim_tokens=Mock(),
        settings=Mock(),
    )


@pytest.fixture
def handoff():
    return LoginHandoffStore()


@pytest.fixture
def client(auth, handoff):
    previous = current_user._auth
    current_user.set_auth(auth)
    app = FastAPI()
    app.include_router(build_router(SimpleNamespace(auth=auth, login_handoff=handoff, content_policy_resolver=None)))
    with TestClient(app) as test_client:
        yield test_client
    current_user._auth = previous


def _lifetime(token):
    claims = jwt.decode(token, SECRET, algorithms=["HS256"])
    return claims["exp"] - claims["iat"]


def _redeem(client, auth, user, handoff, **payload):
    code = handoff.issue(auth.issue_session(user))
    return client.post("/api/auth/external/exchange", json={"code": code, **payload})


def test_remember_me_returns_a_thirty_day_session(client, auth, user, handoff):
    response = _redeem(client, auth, user, handoff, remember_me=True)

    token = response.json()["access_token"]
    assert response.status_code == 200
    assert _lifetime(token) == 30 * DAY
    assert jwt.decode(token, SECRET, algorithms=["HS256"])["user_id"] == "user-1"


def test_remember_me_sets_the_media_cookie_from_the_long_token(client, auth, user, handoff):
    response = _redeem(client, auth, user, handoff, remember_me=True)

    cookie = response.headers["set-cookie"]
    max_age = int(cookie.split("Max-Age=")[1].split(";")[0])
    assert response.json()["access_token"] in cookie
    assert 30 * DAY - 60 <= max_age <= 30 * DAY


def test_without_remember_me_the_default_session_is_kept(client, auth, user, handoff):
    response = _redeem(client, auth, user, handoff)

    assert _lifetime(response.json()["access_token"]) == DAY


def test_remember_me_false_keeps_the_default_session(client, auth, user, handoff):
    response = _redeem(client, auth, user, handoff, remember_me=False)

    assert _lifetime(response.json()["access_token"]) == DAY


def test_remember_me_with_an_unknown_code_is_still_rejected(client):
    response = client.post("/api/auth/external/exchange", json={"code": "nope", "remember_me": True})

    assert response.status_code == 401


def test_remember_me_does_not_fire_a_second_login(client, auth, user, handoff):
    code = handoff.issue(auth.issue_session(user))
    auth.users.update_last_login.reset_mock()
    auth.plugins.execute_hook.reset_mock()

    client.post("/api/auth/external/exchange", json={"code": code, "remember_me": True})

    auth.users.update_last_login.assert_not_called()
    auth.plugins.execute_hook.assert_not_called()
