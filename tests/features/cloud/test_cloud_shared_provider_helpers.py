import dataclasses
from typing import ClassVar, Optional

import pytest
from pydantic import Field, ValidationError

from src.features.cloud.contracts import CloudBackendConfig, CloudProvider, load_catalog_file, valid_specs
from src.features.cloud.http import CloudHttp
from src.features.cloud.inline import decode_inline, encode_media
from src.features.cloud.testing.fake import fake_specs


class AcmeConfig(CloudBackendConfig):
    driver: str = Field(default="cloud.acme")

    engine_label: ClassVar[Optional[str]] = "Acme"
    api_key_source: ClassVar[str] = "Created at acme.example."


class AcmeProvider(CloudProvider):
    key: ClassVar[str] = "acme"
    label: ClassVar[str] = "Acme"
    config_class: ClassVar[type] = AcmeConfig
    default_base_url: ClassVar[str] = "https://api.acme.example/v1/"

    async def discover(self):
        return []

    async def submit(self, request):
        raise NotImplementedError


def acme(**fields):
    return AcmeConfig(id="a", name="b", **{"api_key": " sk-1 ", **fields})


@pytest.mark.parametrize("key", [None, "", "   ", 7])
def test_a_missing_api_key_names_the_engine(key):
    data = {"id": "a", "name": "b"}
    if key is not None:
        data["api_key"] = key
    with pytest.raises(ValidationError, match="Add your Acme API key."):
        AcmeConfig(**data)


def test_the_key_is_trimmed_and_the_help_text_names_where_to_get_it():
    assert acme().api_key == "sk-1"
    field = next(item for item in AcmeConfig.engine_fields() if item["name"] == "api_key")
    assert field["secret"] is True and field["description"].startswith("Created at acme.example. ")


@pytest.mark.parametrize("address", ["proxy.example", "ftp://proxy.example", "https://a b.example"])
def test_a_base_url_must_be_a_web_address(address):
    with pytest.raises(ValidationError):
        acme(base_url=address)


def test_the_base_url_falls_back_to_the_provider_default_and_the_key_goes_in_the_default_header():
    assert acme().base_url == "" and acme(base_url=" https://proxy.example/v1/ ").base_url == "https://proxy.example/v1"
    assert AcmeProvider.api_base_url(acme()) == "https://api.acme.example/v1"
    assert AcmeProvider.api_base_url(acme(base_url="https://proxy.example/v1")) == "https://proxy.example/v1"
    assert AcmeProvider.auth_headers(acme()) == {"Authorization": "Bearer sk-1"}


def test_the_catalog_file_supplies_the_suggested_models(tmp_path, monkeypatch):
    catalog = tmp_path / "cloud_models.yml"
    catalog.write_text("suggested: [a, 1, b]\n", encoding="utf-8")
    provider = AcmeProvider(acme(), CloudHttp("http://127.0.0.1:1"))

    assert provider.suggested_model_ids() == ()
    monkeypatch.setattr(AcmeProvider, "catalog_file", catalog)
    assert provider.suggested_model_ids() == ("a", "b")


@pytest.mark.parametrize("text", ["", "- a\n- b\n", "key: [unclosed\n"])
def test_an_unreadable_catalog_file_is_empty(tmp_path, text):
    path = tmp_path / "bad.yml"
    path.write_text(text, encoding="utf-8")

    assert load_catalog_file(path) == {}
    assert load_catalog_file(tmp_path / "missing.yml") == {}


def test_valid_specs_drops_none_and_invalid_specs_and_logs_the_rest(caplog):
    good, other = fake_specs()[:2]
    broken = dataclasses.replace(other, label="")

    with caplog.at_level("WARNING"):
        kept = valid_specs([good, None, broken], "ACME")

    assert kept == [good]
    assert f"[ACME] skipped {broken.provider_model_id}" in caplog.text


def test_inline_media_round_trips_and_bad_data_decodes_to_nothing(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"\x00\x01payload")

    assert decode_inline(encode_media(path)) == b"\x00\x01payload"
    assert decode_inline("") == b"" and decode_inline(None) == b""
    assert decode_inline("a") == b""
