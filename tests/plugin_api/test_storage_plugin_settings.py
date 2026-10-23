from types import SimpleNamespace

import pytest

from src.features.providers.registry import ProviderRegistry
from src.plugin_api import plugin_settings
from src.platform.plugins.runtime_registries import get_global_plugin_registry


class Repo:
    def __init__(self, **stored):
        self.stored = stored

    def get_plugin_settings(self, plugin_id):
        return [
            SimpleNamespace(setting_key=key, setting_value=value)
            for key, value in self.stored.items()
        ]


def test_shipped_runpod_manifest_yields_typed_defaults_and_keeps_an_emptied_cuda_list():
    defaults = plugin_settings("runpod-provider", Repo())

    assert defaults["volume_size_gb"] == 100
    assert defaults["gpu_type_id"] == "NVIDIA GeForce RTX 4090"
    assert defaults["worker_image"].startswith("ghcr.io/")
    assert defaults["allowed_cuda_versions"] == "13.0"
    assert defaults["api_key"] is None
    assert plugin_settings("runpod-provider", Repo(allowed_cuda_versions=""))["allowed_cuda_versions"] == ""
    assert plugin_settings("runpod-provider", Repo(volume_size_gb="5"))["volume_size_gb"] == 10


def test_shipped_oidc_manifest_defaults_apply_to_blank_strings():
    values = plugin_settings("oidc-auth", Repo(scopes="", label=""))

    assert values["scopes"] == "openid profile email"
    assert values["label"] == "SSO"
    assert "account_creation_notice" not in values


@pytest.mark.parametrize("provider_id", ["civitai", "huggingface"])
def test_shipped_provider_manifests_mark_api_key_secret(provider_id):
    registry = ProviderRegistry(get_global_plugin_registry())
    registry._providers[provider_id] = object()

    assert registry.get_provider_secret_keys(provider_id) == ["api_key"]
    schema = registry.get_provider_settings_schema(provider_id)
    assert schema["properties"]["api_key"]["format"] == "password"
    assert schema["properties"]["rate_limit_delay"]["maximum"] == 10.0
