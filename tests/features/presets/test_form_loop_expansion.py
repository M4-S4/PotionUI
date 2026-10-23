from pathlib import Path

import pytest

from src.features.forms.binding import bind_form
from src.features.presets.loader import PresetTemplateLoader

PRESET_ID = "01AAAAAAAAAAAAAAAAAAAAAAAAA"

PRESET_YML = (
    "schema: 1\n"
    f'id: "{PRESET_ID}"\n'
    'name: "Test"\n'
    'version: "1.0.0"\n'
    'category: "image"\n'
    'engine: "native"\n'
    "modes:\n"
    "  - txt2img\n"
)

SLOT_LOOP = (
    "fields:\n"
    "  - type: section\n"
    "    label: Slots\n"
    "    children:\n"
    "      - type: \"@loop\"\n"
    "        configuration:\n"
    "          count: {count}\n"
    "          template:\n"
    "            type: accordion\n"
    "            label: \"Slot {{{{ loop.index }}}}\"\n"
    "            children:\n"
    "              - name: \"slot_{{{{ loop.index }}}}_model\"\n"
    "                type: string\n"
    "                default: \"\"\n"
)


def _load(tmp_path: Path, form_yml: str):
    preset_dir = tmp_path / "presets" / "Foo" / "std"
    mode_dir = preset_dir / "modes" / "txt2img"
    mode_dir.mkdir(parents=True)
    (preset_dir / "preset.yml").write_text(PRESET_YML)
    (mode_dir / "pipeline.yml").write_text("pipeline: []\n")
    (mode_dir / "form.yml").write_text(form_yml)
    loader = PresetTemplateLoader([str(tmp_path / "presets")])
    loader.load_presets()
    return loader


def _names(fields) -> list:
    names = []
    for field in fields:
        if field.name:
            names.append(field.name)
        if isinstance(field.children, list):
            names.extend(_names(field.children))
    return names


def test_loop_is_expanded_into_concrete_fields_at_load(tmp_path):
    loader = _load(tmp_path, SLOT_LOOP.format(count=3))
    assert loader.load_errors == {}
    section = loader.presets[0].modes["txt2img"].forms[0].fields[0]
    assert [c.type for c in section.children] == ["accordion"] * 3
    assert [c.label for c in section.children] == ["Slot 1", "Slot 2", "Slot 3"]
    assert _names(section.children) == ["slot_1_model", "slot_2_model", "slot_3_model"]


def test_loop_generated_fields_are_bound_not_stripped(tmp_path):
    loader = _load(tmp_path, SLOT_LOOP.format(count=2))
    raw = {"slot_1_model": "a.safetensors", "slot_2_model": "b.safetensors"}

    bound = bind_form(loader.presets[0], "txt2img", None, raw, "user_1")

    assert bound.values["slot_1_model"] == "a.safetensors"
    assert bound.values["slot_2_model"] == "b.safetensors"
    assert bound.stripped == []


def test_loop_generated_fields_apply_their_own_default_when_absent(tmp_path):
    loader = _load(tmp_path, SLOT_LOOP.format(count=2))

    bound = bind_form(loader.presets[0], "txt2img", None, {}, "user_1")

    assert bound.values["slot_1_model"] == ""
    assert bound.values["slot_2_model"] == ""


def test_loop_over_items_with_when_skips_filtered_items(tmp_path):
    form = (
        "fields:\n"
        "  - type: \"@loop\"\n"
        "    configuration:\n"
        "      items: [a, b, c]\n"
        "      when: \"{{ item != 'b' }}\"\n"
        "      template:\n"
        "        name: \"opt_{{ item }}\"\n"
        "        type: string\n"
    )
    loader = _load(tmp_path, form)
    assert loader.load_errors == {}
    assert _names(loader.presets[0].modes["txt2img"].forms[0].fields) == ["opt_a", "opt_c"]


def test_loop_over_dict_items_unpacks_key_and_value(tmp_path):
    form = (
        "fields:\n"
        "  - type: \"@loop\"\n"
        "    configuration:\n"
        "      items: {x: 1, y: 2}\n"
        "      as: \"key,value\"\n"
        "      template:\n"
        "        name: \"{{ key }}_{{ value }}\"\n"
        "        type: string\n"
    )
    loader = _load(tmp_path, form)
    assert loader.load_errors == {}
    assert _names(loader.presets[0].modes["txt2img"].forms[0].fields) == ["x_1", "y_2"]


def test_loop_beyond_the_item_cap_is_a_load_error(tmp_path):
    loader = _load(tmp_path, SLOT_LOOP.format(count=10_001))
    assert any("cap" in message for errors in loader.load_errors.values() for message in errors)


def test_loop_without_configuration_is_a_load_error(tmp_path):
    loader = _load(tmp_path, "fields:\n  - type: \"@loop\"\n")
    assert loader.presets == []
    assert any("requires configuration" in m for errors in loader.load_errors.values() for m in errors)
