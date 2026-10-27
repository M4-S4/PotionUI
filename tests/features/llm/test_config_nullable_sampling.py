from unittest.mock import patch

import pydantic
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
        name="n", type="ollama", enabled=True, base_url="http://localhost", model="m", system_message="s",
    )
    base.update(overrides)
    return LLMConfigRequest(**base)


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


def test_a_request_without_sampling_values_leaves_them_to_the_model():
    request = _request()

    assert request.temperature is None and request.max_tokens is None


def test_a_request_can_clear_the_sampling_values_with_null():
    request = _request(temperature=None, max_tokens=None)

    assert request.temperature is None and request.max_tokens is None


def test_a_request_keeps_the_values_it_is_given():
    request = _request(temperature=0.0, max_tokens=512)

    assert (request.temperature, request.max_tokens) == (0.0, 512)


def test_the_timeout_stays_a_number():
    assert _request().timeout == 30
    with pytest.raises(pydantic.ValidationError):
        _request(timeout=None)


def test_the_unset_values_reach_the_config_model_and_the_response():
    config = LLMConfig(**_request(id="cfg-1").model_dump())

    response = config_to_response(config, is_default=False)

    assert config.temperature is None and config.max_tokens is None
    assert response.temperature is None and response.max_tokens is None
    assert response.model_dump()["timeout"] == 30


def test_unset_values_round_trip_through_the_database(db):
    repo = LLMConfigurationRepository()

    assert repo.create(_record(temperature=None, max_tokens=None)) is True

    loaded = repo.get_by_id("cfg-1")
    assert loaded.temperature is None and loaded.max_tokens is None


def test_values_can_be_set_then_cleared_by_an_update(db):
    repo = LLMConfigurationRepository()
    repo.create(_record(temperature=0.4, max_tokens=800))
    assert (repo.get_by_id("cfg-1").temperature, repo.get_by_id("cfg-1").max_tokens) == (0.4, 800)

    assert repo.update("cfg-1", _record(temperature=None, max_tokens=None)) is True

    loaded = repo.get_by_id("cfg-1")
    assert loaded.temperature is None and loaded.max_tokens is None


def test_the_high_level_repository_serves_unset_values(db):
    LLMConfigurationRepository().create(_record(temperature=None, max_tokens=None))

    config = LLMRepository().get_configuration("cfg-1")

    assert config.temperature is None and config.max_tokens is None
