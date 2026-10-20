import pytest

import tests.conftest as ct
from src.platform.security.secrets import (
    ENVELOPE_PREFIX,
    SecretCipher,
    configure_secret_cipher,
    generate_key,
)
from tests.platform.database.test_migration_060_tab_drafts import _load_migration

PLAINTEXT = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"


@pytest.fixture
def env():
    database = ct.TestDatabase()
    cipher = SecretCipher([generate_key()])
    configure_secret_cipher(cipher)
    _load_migration("001_baseline", database).up()
    yield database, cipher, _load_migration("061_encrypt_s3_secret_key", database)
    configure_secret_cipher(None)
    database.close()


def _set(database, value):
    with database.get_connection() as conn:
        conn.execute("UPDATE settings SET value = ? WHERE key = 's3_secret_key'", (value,))
        conn.commit()


def _raw(database):
    with database.get_connection() as conn:
        return conn.execute("SELECT value FROM settings WHERE key = 's3_secret_key'").fetchone()[0]


def test_encrypts_plaintext_row(env):
    database, cipher, migration = env
    _set(database, PLAINTEXT)
    migration.up()
    raw = _raw(database)
    assert raw.startswith(ENVELOPE_PREFIX)
    assert cipher.decrypt(raw, context="test") == PLAINTEXT


def test_leaves_encrypted_row_untouched(env):
    database, cipher, migration = env
    encrypted = cipher.encrypt(PLAINTEXT)
    _set(database, encrypted)
    migration.up()
    assert _raw(database) == encrypted


def test_leaves_empty_row_empty_and_is_idempotent(env):
    database, cipher, migration = env
    migration.up()
    assert _raw(database) == ""
    _set(database, PLAINTEXT)
    migration.up()
    once = _raw(database)
    migration.up()
    assert _raw(database) == once
