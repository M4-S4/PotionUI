from pathlib import Path

import pytest

from src.features.presets import PresetTemplateLoader


@pytest.fixture(scope="module")
def presets():
    loader = PresetTemplateLoader(["content/presets"])
    loader.load_presets()
    return {Path(p.path).as_posix().rsplit("/", 1)[-1]: p for p in loader.presets}


def _copy(preset, mode):
    return ((preset.vars.get("prompt") or {}).get("modes") or {}).get(mode) or {}


def _optional_modes(preset):
    names = []
    for entry in preset.vars.get("optional_prompt_modes") or []:
        names.append(entry if isinstance(entry, str) else entry["mode"])
    return names


def test_qwen_edit_text_says_the_prompt_is_placement_and_style(presets):
    copy = _copy(presets["QwenImage-2.1"], "text")

    assert copy["label"] == "Placement and style"
    assert copy["placeholder"] == "in white capitals at the top"
    assert "text" in _optional_modes(presets["QwenImage-2.1"])


def test_h3_talking_video_says_the_prompt_is_the_scene(presets):
    copy = _copy(presets["MiniMax-H3"], "talk")

    assert copy["label"] == "Scene"
    assert copy["placeholder"] == "a newsroom, soft window light"
    assert "talk" in _optional_modes(presets["MiniMax-H3"])


@pytest.mark.parametrize("preset_name", ["QwenImage-2.1", "MiniMax-H3", "LTX-2.5"])
def test_every_optional_prompt_mode_labels_its_prompt(presets, preset_name):
    preset = presets[preset_name]

    for mode in _optional_modes(preset):
        copy = _copy(preset, mode)
        assert copy.get("label"), f"{preset_name}/{mode} has no prompt_label"
        assert copy.get("placeholder"), f"{preset_name}/{mode} has no prompt_placeholder"
        assert len(copy["label"]) <= 32, f"{preset_name}/{mode} label is too long"
