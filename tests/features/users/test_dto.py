import pytest
from pydantic import ValidationError

from src.features.users.dto import UserCreate, UserUpdate


def test_create_accepts_local_domain():
    assert UserCreate(username="ada", email="ada@potionui.local", password="x").email == "ada@potionui.local"


def test_create_rejects_malformed():
    with pytest.raises(ValidationError):
        UserCreate(username="ada", email="ada@", password="x")


def test_update_accepts_local_domain_and_none():
    assert UserUpdate(email="me@localhost").email == "me@localhost"
    assert UserUpdate().email is None


def test_update_rejects_malformed():
    with pytest.raises(ValidationError):
        UserUpdate(email="not-an-email")
