from unittest.mock import patch

import pytest

from src.features.llm.records import LLMConfiguration
from src.features.llm.repository import LLMConfigurationRepository, LLMRepository
from src.platform.security.secrets import SecretCipher, configure_secret_cipher, generate_key

from tests.features.plugins.test_credential_encryption import fresh_file_database
from tests.platform.database._support import load_migration


def _record(**overrides) -> LLMConfiguration:
    base = dict(
        id="cfg-1", name="n", type="ollama", enabled=True, base_url="http://localhost",
        api_key=None, model="m", system_message="s",
    )
    base.update(overrides)
    return LLMConfiguration(**base)


@pytest.fixture
def db(tmp_path):
    database = fresh_file_database(tmp_path / "db.sqlite")
    load_migration("065_llm_show_thinking", database).up()
    load_migration("069_llm_profile_overrides", database).up()
    configure_secret_cipher(SecretCipher([generate_key()]))
    with patch("src.platform.database.database.db", database):
        yield database
    configure_secret_cipher(None)
    database.close()


def _stored(database, config_id="cfg-1"):
    with database.get_cursor() as cursor:
        cursor.execute("SELECT profile_overrides FROM llm_configurations WHERE id = ?", (config_id,))
        return cursor.fetchone()["profile_overrides"]


def test_a_new_config_has_no_overrides(db):
    repo = LLMConfigurationRepository()
    repo.create(_record())

    assert repo.get_by_id("cfg-1").profile_overrides is None
    assert _stored(db) is None


def test_overrides_are_stored_as_compact_sorted_json_and_read_back(db):
    repo = LLMConfigurationRepository()
    repo.create(_record())

    assert repo.set_profile_overrides("cfg-1", {"replay": "preserve", "context_window": 65536}) is True

    assert _stored(db) == '{"context_window":65536,"replay":"preserve"}'
    assert repo.get_by_id("cfg-1").profile_overrides == {"context_window": 65536, "replay": "preserve"}
    assert repo.get_all()["cfg-1"].profile_overrides == {"context_window": 65536, "replay": "preserve"}


def test_empty_overrides_clear_the_column(db):
    repo = LLMConfigurationRepository()
    repo.create(_record())
    repo.set_profile_overrides("cfg-1", {"replay": "never"})

    repo.set_profile_overrides("cfg-1", {})

    assert _stored(db) is None
    repo.set_profile_overrides("cfg-1", {"replay": "never"})
    repo.set_profile_overrides("cfg-1", None)
    assert _stored(db) is None


def test_an_unknown_config_reports_failure(db):
    assert LLMConfigurationRepository().set_profile_overrides("missing", {"replay": "never"}) is False


def test_editing_the_configuration_keeps_its_overrides(db):
    repo = LLMConfigurationRepository()
    repo.create(_record())
    repo.set_profile_overrides("cfg-1", {"replay": "preserve"})

    assert repo.update("cfg-1", _record(name="renamed", model="other")) is True

    loaded = repo.get_by_id("cfg-1")
    assert (loaded.name, loaded.model) == ("renamed", "other")
    assert loaded.profile_overrides == {"replay": "preserve"}


def test_the_high_level_repository_exposes_overrides_on_the_config(db):
    LLMConfigurationRepository().create(_record())
    repository = LLMRepository()

    assert repository.get_configuration("cfg-1").profile_overrides is None
    assert repository.set_profile_overrides("cfg-1", {"vision": True}) is True

    assert repository.get_configuration("cfg-1").profile_overrides == {"vision": True}


def test_editing_through_the_high_level_repository_keeps_overrides(db):
    repository = LLMRepository()
    LLMConfigurationRepository().create(_record())
    repository.set_profile_overrides("cfg-1", {"vision": True})

    config = repository.get_configuration("cfg-1")
    assert repository.update_configuration("cfg-1", config.model_copy(update={"name": "renamed"})) is True

    reloaded = repository.get_configuration("cfg-1")
    assert reloaded.name == "renamed"
    assert reloaded.profile_overrides == {"vision": True}


def test_malformed_stored_json_reads_as_no_overrides(db):
    LLMConfigurationRepository().create(_record())
    with db.get_cursor() as cursor:
        cursor.execute("UPDATE llm_configurations SET profile_overrides = '{not json' WHERE id = 'cfg-1'")

    assert LLMConfigurationRepository().get_by_id("cfg-1").profile_overrides is None
