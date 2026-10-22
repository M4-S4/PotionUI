from unittest.mock import patch

import pytest

from src.features.llm.dto import LLMConfigRequest
from src.features.llm.mappers import config_to_response
from src.features.llm.records import LLMConfiguration
from src.features.llm.repository import LLMConfig, LLMConfigurationRepository, LLMRepository
from src.platform.security.secrets import SecretCipher, configure_secret_cipher, generate_key

from tests.features.plugins.test_credential_encryption import fresh_file_database
from tests.platform.database._support import load_migration


def _request(**overrides) -> LLMConfigRequest:
    base = dict(
        name="n", type="openai", enabled=True, base_url="http://localhost",
        model="m", system_message="s",
    )
    base.update(overrides)
    return LLMConfigRequest(**base)


def _record(**overrides) -> LLMConfiguration:
    base = dict(
        id="cfg-1", name="n", type="openai", enabled=True, base_url="http://localhost",
        api_key=None, model="m", system_message="s",
    )
    base.update(overrides)
    return LLMConfiguration(**base)


@pytest.fixture
def db(tmp_path):
    database = fresh_file_database(tmp_path / "db.sqlite")
    load_migration("065_llm_show_thinking", database).up()
    configure_secret_cipher(SecretCipher([generate_key()]))
    with patch("src.platform.database.database.db", database):
        yield database
    configure_secret_cipher(None)
    database.close()


def test_a_request_hides_thinking_unless_asked():
    assert _request().show_thinking is False


def test_a_request_can_show_thinking():
    assert _request(show_thinking=True).show_thinking is True


def test_the_request_reaches_the_config_model():
    assert LLMConfig(**_request(id="cfg-1", show_thinking=True).model_dump()).show_thinking is True


def test_the_response_reports_the_flag():
    config = LLMConfig(**_request(id="cfg-1", show_thinking=True).model_dump())
    assert config_to_response(config, is_default=False).show_thinking is True


def test_a_created_config_round_trips_the_flag(db):
    repo = LLMConfigurationRepository()
    assert repo.create(_record(show_thinking=True)) is True
    assert repo.get_by_id("cfg-1").show_thinking is True
    assert repo.get_all()["cfg-1"].show_thinking is True


def test_a_config_created_without_the_flag_hides_thinking(db):
    repo = LLMConfigurationRepository()
    repo.create(_record())
    assert repo.get_by_id("cfg-1").show_thinking is False


def test_an_update_turns_the_flag_on_and_off(db):
    repo = LLMConfigurationRepository()
    repo.create(_record())

    assert repo.update("cfg-1", _record(show_thinking=True)) is True
    assert repo.get_by_id("cfg-1").show_thinking is True

    assert repo.update("cfg-1", _record(show_thinking=False)) is True
    assert repo.get_by_id("cfg-1").show_thinking is False


def test_the_high_level_repository_exposes_the_flag(db):
    LLMConfigurationRepository().create(_record(show_thinking=True))
    assert LLMRepository().get_configuration("cfg-1").show_thinking is True
