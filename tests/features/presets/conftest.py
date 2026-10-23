import pytest

from src.features.presets import PresetTemplateLoader

H3_PRESET_ID = "01KX47H3MINIMAX000000000VA"


@pytest.fixture(scope="module")
def h3_template():
    loader = PresetTemplateLoader(["content/presets"])
    loader.load_presets()
    template = next((p for p in loader.presets if p.id == H3_PRESET_ID), None)
    assert template is not None, (
        f"MiniMax-H3 preset {H3_PRESET_ID} not loaded from content/presets "
        f"(loaded: {sorted(p.id for p in loader.presets)})"
    )
    return template
