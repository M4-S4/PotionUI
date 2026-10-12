import time
from unittest.mock import Mock

import pytest
from jose import jwt

from src.platform.security.config import AuthConfig
from src.platform.security.session_renewal import renewed_token
from src.platform.security.token import TokenCodec

SECRET = "session-renewal-test-secret"
HOUR = 3600


@pytest.fixture
def codec():
    config = Mock(spec=AuthConfig)
    config.secret_key = SECRET
    config.algorithm = "HS256"
    config.access_token_expire_minutes = 60
    return TokenCodec(config)


def _token(issued_ago, lifetime, extra=None, secret=SECRET):
    now = int(time.time())
    claims = {"sub": "alice", "user_id": "user-1", "iat": now - issued_ago, "exp": now - issued_ago + lifetime}
    claims.update(extra or {})
    return jwt.encode(claims, secret, algorithm="HS256")


def _claims(token):
    return jwt.decode(token, SECRET, algorithms=["HS256"])


def test_a_fresh_token_is_not_renewed(codec):
    assert renewed_token(codec, _token(60, 24 * HOUR)) is None


def test_a_token_just_before_half_life_is_not_renewed(codec):
    assert renewed_token(codec, _token(11 * HOUR, 24 * HOUR)) is None


def test_a_token_past_half_life_is_renewed_with_the_same_lifetime(codec):
    fresh = renewed_token(codec, _token(13 * HOUR, 24 * HOUR))

    claims = _claims(fresh)
    assert claims["exp"] - claims["iat"] == 24 * HOUR
    assert abs(claims["iat"] - time.time()) < 5


def test_the_renewed_token_keeps_the_identity_claims(codec):
    fresh = renewed_token(codec, _token(13 * HOUR, 24 * HOUR, extra={"role": "x"}))

    claims = _claims(fresh)
    assert claims["sub"] == "alice"
    assert claims["user_id"] == "user-1"
    assert claims["role"] == "x"


def test_a_remember_me_lifetime_is_preserved(codec):
    thirty_days = 30 * 24 * HOUR

    fresh = renewed_token(codec, _token(16 * 24 * HOUR, thirty_days))

    claims = _claims(fresh)
    assert claims["exp"] - claims["iat"] == thirty_days


def test_the_clock_can_be_injected(codec):
    token = _token(60, 24 * HOUR)

    assert renewed_token(codec, token, now=time.time() + 13 * HOUR) is not None


def test_a_token_without_iat_is_never_renewed(codec):
    legacy = jwt.encode(
        {"sub": "alice", "user_id": "user-1", "exp": int(time.time()) + 60},
        SECRET,
        algorithm="HS256",
    )

    assert renewed_token(codec, legacy) is None


def test_a_token_without_exp_is_never_renewed(codec):
    token = jwt.encode(
        {"sub": "alice", "user_id": "user-1", "iat": int(time.time()) - 13 * HOUR},
        SECRET,
        algorithm="HS256",
    )

    assert renewed_token(codec, token) is None


def test_an_expired_token_is_not_renewed(codec):
    assert renewed_token(codec, _token(25 * HOUR, 24 * HOUR)) is None


def test_a_token_signed_with_another_secret_is_not_renewed(codec):
    assert renewed_token(codec, _token(13 * HOUR, 24 * HOUR, secret="someone-else")) is None


def test_garbage_is_not_renewed(codec):
    assert renewed_token(codec, "not-a-jwt") is None


def test_a_non_positive_lifetime_is_not_renewed(codec):
    assert renewed_token(codec, _token(0, 0)) is None
