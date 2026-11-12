from pathlib import Path

import pytest
import yaml

from .conftest import H3_12GB_ID, H3_FAST_12GB_ID

PLUGIN = Path(__file__).resolve().parents[1]
SHARED = PLUGIN.parents[2] / "presets" / "_shared"
MEANINGFUL = ("speed_profile", "resolution", "frames", "steps", "source_video")


def _path(text, preset):
    return Path(text.replace("{{ paths.preset }}", str(preset)).replace("{{ paths._shared }}", str(SHARED)))


def _walk(items, preset, found):
    for item in items or []:
        if not isinstance(item, dict):
            continue
        if "name" in item:
            found[item["name"]] = item
        children = item.get("children")
        if isinstance(children, str):
            _walk(yaml.safe_load(_path(children, preset).read_text(encoding="utf-8")).get("fields"), preset, found)
        elif isinstance(children, list):
            _walk(children, preset, found)
    return found


def real_fields(folder, mode):
    preset = PLUGIN / "presets" / folder
    form = yaml.safe_load((preset / "modes" / mode / "form.yml").read_text(encoding="utf-8"))
    return _walk(form["fields"], preset, {})


CASES = [
    ("use_h3_continue", H3_12GB_ID, "MiniMax-H3-12GB", "extend"),
    ("use_h3_fast_12gb_continue", H3_FAST_12GB_ID, "MiniMax-H3-Fast-12GB", "video"),
    ("use_h3_fast_12gb_continue", H3_FAST_12GB_ID, "MiniMax-H3-Fast-12GB", "extend"),
]


@pytest.mark.parametrize(("use", "preset_id", "folder", "mode"), CASES)
def test_the_fields_the_tests_give_a_12_gb_preset_are_those_of_its_real_form(core, use, preset_id, folder, mode):
    getattr(core, use)()
    fixture = core.fields[(preset_id, mode)]
    real = real_fields(folder, mode)

    for name, spec in fixture.items():
        assert name in real, (folder, mode, name)
        field = real[name]
        assert spec["type"] == field["type"], (folder, mode, name)
        assert spec["default"] == field.get("default"), (folder, mode, name)
        if spec["type"] == "select":
            assert [option["value"] for option in spec["options"]] == [option["value"] for option in field["configuration"]["options"]], (folder, mode, name)
    assert [name for name in MEANINGFUL if name in real] == [name for name in MEANINGFUL if name in fixture], (folder, mode)
