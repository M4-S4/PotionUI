import sqlite3

import pytest

from tests.fixtures.seed import seed_user


def _row(db, user_id):
    with db.get_cursor() as cursor:
        cursor.execute(
            "SELECT id, username, email, password_hash, account_type FROM users WHERE id = ?", (user_id,)
        )
        return tuple(cursor.fetchone())


def test_defaults_derive_username_and_email_from_the_id(mock_db):
    assert seed_user(mock_db, "alice") == "alice"

    assert _row(mock_db, "alice") == ("alice", "alice", "alice@example.test", "x", "USER")


def test_explicit_fields_are_stored(mock_db):
    seed_user(mock_db, "root", username="admin", email="a@b.test", password_hash="h", account_type="ADMIN")

    assert _row(mock_db, "root") == ("root", "admin", "a@b.test", "h", "ADMIN")


def test_seeding_the_same_id_twice_fails(mock_db):
    seed_user(mock_db, "dup")

    with pytest.raises(sqlite3.IntegrityError):
        seed_user(mock_db, "dup")
