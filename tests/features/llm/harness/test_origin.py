from src.features.llm.harness.messages import Origin, credential_fingerprint
from src.features.llm.repository import LLMConfig


def config(**fields):
    values = {
        "id": "c1",
        "name": "n",
        "type": "openai",
        "enabled": True,
        "base_url": "https://api.openai.com/v1",
        "api_key": "key-1",
        "model": "gpt-5.6",
        "system_message": "",
    }
    values.update(fields)
    return LLMConfig(**values)


def test_an_origin_is_stamped_from_the_adapter_name_and_the_config():
    origin = Origin.for_config("openai-responses", config())

    assert origin == Origin(
        adapter="openai-responses",
        provider="openai",
        model="gpt-5.6",
        credential=credential_fingerprint("https://api.openai.com/v1", "key-1"),
    )


def test_each_config_field_that_defines_the_origin_changes_it():
    base = Origin.for_config("openai-responses", config())

    assert Origin.for_config("openai-chat", config()) != base
    assert Origin.for_config("openai-responses", config(type="openai-compatible")) != base
    assert Origin.for_config("openai-responses", config(model="gpt-5.5")) != base
    assert Origin.for_config("openai-responses", config(api_key="key-2")) != base
    assert Origin.for_config("openai-responses", config(base_url="https://eu.api.openai.com/v1")) != base


def test_fields_outside_the_origin_do_not_change_it():
    base = Origin.for_config("openai-responses", config())

    assert Origin.for_config("openai-responses", config(name="other", temperature=0.1, show_thinking=True)) == base


def test_a_config_without_a_key_still_has_a_credential_fingerprint():
    origin = Origin.for_config("ollama-chat", config(type="ollama", api_key=None, base_url="http://localhost:11434"))

    assert origin.credential == credential_fingerprint("http://localhost:11434", None)
