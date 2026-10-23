import pytest
from pydantic import ValidationError

from src.features.llm.dto import LLMConfigRequest


def _ollama_request(**provider_options) -> LLMConfigRequest:
    return LLMConfigRequest(
        name="n",
        type="ollama",
        enabled=True,
        base_url="http://localhost:11434",
        model="llama3",
        system_message="s",
        provider_options=provider_options,
    )


def test_think_omitted_is_stored_as_false_and_other_options_survive():
    req = _ollama_request(keep_alive="5m")
    assert req.provider_options == {"keep_alive": "5m", "think": False}


def test_think_none_is_stored_as_false():
    req = _ollama_request(think=None)
    assert req.provider_options["think"] is False


def test_no_provider_options_at_all_still_stores_think_false():
    req = LLMConfigRequest(
        name="n", type="ollama", enabled=True, base_url="http://localhost:11434", model="llama3", system_message="s",
    )
    assert req.provider_options == {"think": False}


def test_a_non_ollama_type_keeps_its_options_untouched():
    req = LLMConfigRequest(
        name="n", type="openai", enabled=True, base_url="", model="m", system_message="s",
    )
    assert req.provider_options is None


def test_the_callers_dict_is_not_mutated():
    options = {"keep_alive": "5m"}
    LLMConfigRequest(
        name="n", type="ollama", enabled=True, base_url="http://localhost:11434", model="llama3",
        system_message="s", provider_options=options,
    )
    assert options == {"keep_alive": "5m"}


@pytest.mark.parametrize("value", [True, False])
def test_think_accepts_a_real_bool(value):
    req = _ollama_request(think=value)
    assert req.provider_options["think"] is value


@pytest.mark.parametrize("value", ["low", "medium", "high"])
def test_think_accepts_named_levels(value):
    req = _ollama_request(think=value)
    assert req.provider_options["think"] == value


@pytest.mark.parametrize("value", [1, 0, "true", "false", "LOW", "off", "on", 1.0, [], {}])
def test_think_rejects_malformed_or_truthiness_coerced_values(value):
    with pytest.raises(ValidationError, match="think"):
        _ollama_request(think=value)


def test_think_key_is_unvalidated_for_a_non_ollama_type():
    """provider_options.think is Ollama-specific; a native or other config
    leaves it as an unvalidated free-form key (meaningless there, so it is
    not this validator's business)."""
    req = LLMConfigRequest(
        name="n",
        type="native",
        enabled=True,
        base_url="",
        model="m",
        system_message="s",
        provider_options={"think": "not-a-real-value"},
    )
    assert req.provider_options["think"] == "not-a-real-value"
