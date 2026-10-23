import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

import tests.conftest as ct
from src.platform.security.secrets import (
    ENV_KEY,
    ENV_KEY_FILE,
    SecretCipher,
    configure_secret_cipher,
    generate_key,
    resolve_secret_keys,
)
from src.platform.settings.repository import SettingRepository
from tests.platform.database._support import load_migration

PLAINTEXT = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "rotate_secret_key.py"


class _Empty:
    def iter_encrypted_settings(self):
        return []

    def iter_encrypted_configs(self):
        return []

    def iter_encrypted_api_keys(self):
        return []


def _load_script():
    spec = importlib.util.spec_from_file_location("rotate_secret_key_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def env(monkeypatch, tmp_path):
    database = ct.TestDatabase()
    old_key = generate_key()
    monkeypatch.setenv(ENV_KEY, old_key.decode("ascii"))
    monkeypatch.delenv(ENV_KEY_FILE, raising=False)
    configure_secret_cipher(None)
    with patch("src.platform.database.database.db", database):
        load_migration("001_baseline", database).up()
        with database.get_connection() as conn:
            conn.execute(
                "UPDATE settings SET value = ? WHERE key = 's3_secret_key'",
                (SecretCipher([old_key]).encrypt(PLAINTEXT),),
            )
            conn.commit()
        yield database, old_key, tmp_path / "secret.key"
    configure_secret_cipher(None)
    database.close()


def _raw(database):
    with database.get_connection() as conn:
        return conn.execute("SELECT value FROM settings WHERE key = 's3_secret_key'").fetchone()[0]


def test_rotate_reencrypts_the_s3_secret_under_the_new_key(env, monkeypatch):
    database, old_key, key_file = env
    script = _load_script()
    before = _raw(database)
    assert script.cmd_rotate(_Empty(), _Empty(), _Empty(), SettingRepository(), key_file, False) == 0
    after = _raw(database)
    assert after != before
    monkeypatch.delenv(ENV_KEY)
    monkeypatch.setenv(ENV_KEY_FILE, str(key_file))
    keys = resolve_secret_keys(allow_generate=False)
    assert keys[0] != old_key
    assert SecretCipher([keys[0]]).decrypt(after, context="test") == PLAINTEXT


def test_status_reports_an_unreadable_s3_secret(env, monkeypatch, capsys):
    database, old_key, key_file = env
    script = _load_script()
    assert script.cmd_status(_Empty(), _Empty(), _Empty(), SettingRepository()) == 0
    monkeypatch.setenv(ENV_KEY, generate_key().decode("ascii"))
    assert script.cmd_status(_Empty(), _Empty(), _Empty(), SettingRepository()) == 1
    assert "settings:s3_secret_key" in capsys.readouterr().out
