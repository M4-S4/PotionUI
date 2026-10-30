import importlib.util
from pathlib import Path

import pytest

from src.features.users.repository import user_repo
from src.platform.security.password import PasswordHasher

SCRIPT = Path(__file__).resolve().parent.parent.parent / "scripts" / "reset_password.py"


@pytest.fixture
def reset(mock_db, monkeypatch):
    spec = importlib.util.spec_from_file_location("reset_password_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("POTIONUI_NEW_PASSWORD", raising=False)
    monkeypatch.delenv("POTIONUI_NEW_PASSWORD_REPEAT", raising=False)
    monkeypatch.delenv("POTIONUI_RESET_USERNAME", raising=False)

    def run(*args, env=None):
        for key, value in (env or {}).items():
            monkeypatch.setenv(key, value)
        monkeypatch.setattr("sys.argv", ["reset_password.py", *args])
        return module.main()

    return run


def _owner_and_member():
    owner, became_owner = user_repo.create_claiming_instance(
        username="owner", email="owner@example.com", password_hash=PasswordHasher().hash("old-owner")
    )
    assert became_owner
    member, _ = user_repo.create_claiming_instance(
        username="member", email="member@example.com", password_hash=PasswordHasher().hash("old-member")
    )
    return owner, member


def _verifies(user_id, password):
    return PasswordHasher().verify(password, user_repo.get_by_id(user_id).password_hash)


def test_an_empty_username_resets_the_owner_and_names_it(reset, capsys):
    owner, member = _owner_and_member()

    assert reset("", env={"POTIONUI_NEW_PASSWORD": "fresh-pw", "POTIONUI_NEW_PASSWORD_REPEAT": "fresh-pw"}) == 0

    assert _verifies(owner.id, "fresh-pw")
    assert _verifies(member.id, "old-member")
    assert "Sign in as: owner" in capsys.readouterr().out


def test_no_arguments_reset_the_owner(reset):
    owner, _ = _owner_and_member()

    assert reset(env={"POTIONUI_NEW_PASSWORD": "fresh-pw"}) == 0

    assert _verifies(owner.id, "fresh-pw")


def test_a_named_user_is_reset(reset):
    owner, member = _owner_and_member()

    assert reset("member", env={"POTIONUI_NEW_PASSWORD": "fresh-pw"}) == 0

    assert _verifies(member.id, "fresh-pw")
    assert _verifies(owner.id, "old-owner")


def test_the_username_can_come_from_the_environment(reset):
    owner, member = _owner_and_member()

    assert reset(env={"POTIONUI_RESET_USERNAME": "member", "POTIONUI_NEW_PASSWORD": "fresh-pw"}) == 0

    assert _verifies(member.id, "fresh-pw")
    assert _verifies(owner.id, "old-owner")


def test_a_mismatched_repeat_changes_nothing(reset, capsys):
    owner, _ = _owner_and_member()

    assert reset("", env={"POTIONUI_NEW_PASSWORD": "fresh-pw", "POTIONUI_NEW_PASSWORD_REPEAT": "other"}) == 1

    assert _verifies(owner.id, "old-owner")
    assert "do not match" in capsys.readouterr().err


def test_an_empty_password_changes_nothing(reset):
    owner, _ = _owner_and_member()

    assert reset("", env={"POTIONUI_NEW_PASSWORD": ""}) == 1

    assert _verifies(owner.id, "old-owner")


def test_an_unclaimed_install_has_no_owner_to_reset(reset, capsys):
    assert reset("", env={"POTIONUI_NEW_PASSWORD": "fresh-pw"}) == 1

    assert "no owner account" in capsys.readouterr().err


def test_an_unknown_username_is_reported(reset, capsys):
    _owner_and_member()

    assert reset("nobody", env={"POTIONUI_NEW_PASSWORD": "fresh-pw"}) == 1

    assert "No user named 'nobody'" in capsys.readouterr().err
