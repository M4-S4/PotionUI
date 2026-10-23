import pytest
from fastapi import HTTPException

from src.platform.http.base_controller import BaseController
from src.platform.http.errors import (
    AccessDeniedError,
    ConflictError,
    DomainError,
    NotFoundError,
    domain_failure,
)


def test_typed_errors_keep_value_error_semantics():
    for error in (NotFoundError("x_missing", "gone"), AccessDeniedError("x_denied", "no"), ConflictError("x_dup", "twice")):
        assert isinstance(error, ValueError)
        assert isinstance(error, DomainError)


def test_status_codes_and_payload():
    assert NotFoundError("a", "m").status_code == 404
    assert AccessDeniedError("a", "m").status_code == 403
    assert ConflictError("a", "m").status_code == 409
    assert DomainError("a", "m", 422, problems=[1]).payload() == {
        "success": False, "error": "a", "message": "m", "problems": [1],
    }


def test_domain_failure_ignores_other_exceptions():
    assert domain_failure(ValueError("plain")) is None
    failure = domain_failure(NotFoundError("a", "m"))
    assert (failure.code, failure.message, failure.status_code) == ("a", "m", 404)


class _Controller(BaseController):
    pass


@pytest.mark.asyncio
async def test_run_maps_a_domain_error_to_its_status_and_body():
    def refuse():
        raise DomainError("rule_cap", "Too many", 409, cap=3)

    with pytest.raises(HTTPException) as caught:
        await _Controller().run("op_failed", refuse)

    assert caught.value.status_code == 409
    assert caught.value.detail == {"success": False, "error": "rule_cap", "message": "Too many", "cap": 3}


@pytest.mark.asyncio
async def test_run_soft_keeps_the_envelope_for_a_domain_error():
    def refuse():
        raise NotFoundError("x_missing", "gone")

    response = await _Controller().run("op_failed", refuse, soft=True)

    assert (response.success, response.error, response.message) == (False, "x_missing", "gone")


@pytest.mark.asyncio
async def test_run_soft_hides_unexpected_text_from_regular_users_and_shows_admins():
    def boom():
        raise RuntimeError("secret at /srv/data")

    regular = await _Controller().run("op_failed", boom, soft=True, message="Failed to op")
    admin = await _Controller().run("op_failed", boom, soft=True, message="Failed to op", admin=True)

    assert "secret" not in regular.message
    assert regular.message.startswith("Failed to op")
    assert "secret at <server path>" in admin.message


@pytest.mark.asyncio
async def test_run_hard_sanitises_unexpected_errors_and_runs_off_the_loop():
    import threading

    seen = []

    def boom():
        seen.append(threading.current_thread() is threading.main_thread())
        raise RuntimeError("secret")

    with pytest.raises(HTTPException) as caught:
        await _Controller().run("op_failed", boom)

    assert caught.value.status_code == 500
    assert "secret" not in str(caught.value.detail)
    assert seen == [False]
