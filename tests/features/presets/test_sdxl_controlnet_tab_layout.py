from __future__ import annotations
from tests.fixtures.form_lookup import all_field_names

from unittest.mock import Mock

import pytest

from src.features.presets import PresetTemplateLoader
from src.features.presets.form_serializer import PresetFormSerializer
from src.platform.templating.processor import TemplateProcessor

EXPECTED_LOOP_FIELD_NAMES = {
    f"controlnet_{i}_{suffix}"
    for i in (1, 2, 3)
    for suffix in ("model", "type", "image", "start", "end", "scale")
}


@pytest.fixture(scope="module")
def loader():
    loader = PresetTemplateLoader(["content/presets/marketplace"])
    loader.load_presets()
    return loader


@pytest.fixture(scope="module")
def serializer(loader):
    return PresetFormSerializer(loader, TemplateProcessor(settings=Mock()))


@pytest.fixture(scope="module")
def controlnet_tab(loader, serializer):
    matches = [p for p in loader.presets if str(p.path).replace("\\", "/").endswith("presets/marketplace/SDXL")]
    assert len(matches) == 1, f"SDXL matched {len(matches)} presets"
    preset = matches[0]
    schema = serializer.process_form_fields(preset.modes["txt2img"].forms[0], preset.id)
    tabs = schema["properties"]["tabs"]["children"]
    titles = [t.get("title") for t in tabs]
    assert "ControlNet" in titles, f"SDXL txt2img tabs are {titles}"
    return tabs[titles.index("ControlNet")]


def _section(controlnet_tab, title):
    return next(c for c in controlnet_tab["children"] if c.get("title") == title)


def test_controlnet_tab_top_level_is_the_general_and_units_sections(controlnet_tab):
    top_level = controlnet_tab["children"]
    assert [c.get("type") for c in top_level] == ["section", "section"]
    assert [c.get("title") for c in top_level] == ["General", "ControlNet Units"]


def test_general_section_holds_the_toggles(controlnet_tab):
    general = _section(controlnet_tab, "General")
    assert all_field_names(general) == {"enable_controlnet", "controlnet_auto_preprocess"}


def test_controlnet_units_section_still_expands_to_three_accordions(controlnet_tab):
    units = _section(controlnet_tab, "ControlNet Units")
    accordions = units["children"]
    assert [a.get("type") for a in accordions] == ["accordion", "accordion", "accordion"]
    assert [a.get("title") for a in accordions] == ["ControlNet 1", "ControlNet 2", "ControlNet 3"]
    assert all_field_names(units) == EXPECTED_LOOP_FIELD_NAMES
