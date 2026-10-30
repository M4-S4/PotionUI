from pathlib import Path

import pytest
import yaml

from src.features.fields.model import Model
from src.features.models.routes import ModelController

PRESETS = Path("content/presets/marketplace")
MATTING_TABS = [
    "ImageTools/modes/remove_background/tabs/generation.yml",
    "ImageTools/modes/prepare/tabs/generation.yml",
    "Pixal3D/modes/views2mesh/tabs/generation.yml",
    "Pixal3D/modes/img2mesh/tabs/generation.yml",
    "TRELLIS2/modes/img2mesh/tabs/generation.yml",
]
BIREFNET_GENERAL = {
    "name": "BiRefNet-general.safetensors",
    "link": "https://huggingface.co/ZhengPeng7/BiRefNet/resolve/main/model.safetensors",
    "size": "444 MB",
    "sha256": "9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154",
}


class _NoProviders:
    def is_provider_initialized(self, provider):
        return False


class _NoInstalledFiles:
    def existing_filenames(self, candidates):
        return set()


def _fields_named(node, name):
    if isinstance(node, dict):
        if node.get("name") == name and node.get("type") == "model":
            yield node
        for value in node.values():
            yield from _fields_named(value, name)
    elif isinstance(node, list):
        for item in node:
            yield from _fields_named(item, name)


def _matting_fields(tab):
    document = yaml.safe_load((PRESETS / tab).read_text(encoding="utf-8"))
    return list(_fields_named(document, "matting_model"))


@pytest.mark.parametrize("tab", MATTING_TABS)
def test_the_matting_picker_offers_birefnet_general(tab):
    fields = _matting_fields(tab)

    assert len(fields) == 1
    assert fields[0]["configuration"]["model_type"] == "detection_segm"
    assert fields[0]["configuration"]["recommendations"] == [BIREFNET_GENERAL]


@pytest.mark.parametrize("tab", MATTING_TABS)
def test_the_recommendation_downloads_under_its_own_name(tab):
    recommendation = _matting_fields(tab)[0]["configuration"]["recommendations"][0]

    assert ModelController._recommendation_filename(recommendation["name"]) == recommendation["name"]


@pytest.mark.parametrize("tab", MATTING_TABS)
def test_the_serialized_recommendation_keeps_its_link_and_checksum(tab, monkeypatch):
    monkeypatch.setattr("src.features.providers.registry.get_provider_registry", lambda: _NoProviders())
    monkeypatch.setattr("src.features.fields.model.current_form_lookups", lambda: _NoInstalledFiles())
    recommendations = _matting_fields(tab)[0]["configuration"]["recommendations"]

    serialized = Model._serialize_recommendations(recommendations)

    assert serialized == [{**BIREFNET_GENERAL, "installed": False}]
    assert len(serialized[0]["sha256"]) == 64
