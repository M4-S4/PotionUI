from pathlib import Path

import yaml

from src.features.presets.linter import PresetLinter
from src.features.presets.loader import PresetTemplateLoader

PRESET_ID = "copy-preset"


def _write_preset(root: Path, form: dict, vars_block: dict = None) -> Path:
    preset_dir = root / PRESET_ID
    manifest = {
        "schema": 1, "id": PRESET_ID, "name": PRESET_ID, "version": "1.0.0",
        "category": "image", "engine": "native", "modes": ["text"],
    }
    if vars_block is not None:
        manifest["vars"] = vars_block
    mode_dir = preset_dir / "modes" / "text"
    mode_dir.mkdir(parents=True)
    (preset_dir / "preset.yml").write_text(yaml.dump(manifest), encoding="utf-8")
    (mode_dir / "pipeline.yml").write_text("pipeline: []\n", encoding="utf-8")
    (mode_dir / "form.yml").write_text(yaml.dump({"fields": [], **form}), encoding="utf-8")
    return preset_dir / "preset.yml"


def _load(tmp_path: Path, form: dict, vars_block: dict = None):
    root = tmp_path / "presets"
    _write_preset(root, form, vars_block)
    loader = PresetTemplateLoader([str(root)])
    loader.load_presets()
    preset = next((p for p in loader.presets if p.id == PRESET_ID), None)
    return preset, loader


def _lint(tmp_path: Path, form: dict, vars_block: dict = None):
    root = tmp_path / "presets"
    _write_preset(root, form, vars_block)
    return PresetLinter([str(root)]).lint()


def test_a_core_mode_label_and_placeholder_reach_the_presets_prompt_vars(tmp_path):
    preset, loader = _load(
        tmp_path, {"prompt_label": "Placement and style", "prompt_placeholder": "in white capitals at the top"},
    )

    assert loader.load_errors == {}
    assert preset.vars["prompt"]["modes"]["text"] == {
        "label": "Placement and style",
        "placeholder": "in white capitals at the top",
    }


def test_the_label_keeps_a_preset_declared_segment_template_override(tmp_path):
    preset, _ = _load(
        tmp_path,
        {"prompt_label": "Scene"},
        vars_block={"prompt": {"modes": {"text": {"segment_templates": []}}}},
    )

    assert preset.vars["prompt"]["modes"]["text"] == {"segment_templates": [], "label": "Scene"}


def test_a_core_mode_prompt_requirement_in_its_form_joins_the_presets_lists(tmp_path):
    preset, _ = _load(tmp_path, {"prompt": "optional", "prompt_label": "Scene"})

    assert preset.vars["optional_prompt_modes"] == ["text"]


def test_a_mode_with_no_copy_leaves_the_presets_vars_untouched(tmp_path):
    preset, _ = _load(tmp_path, {}, vars_block={"fps": 24})

    assert preset.vars == {"fps": 24}


def test_blank_copy_is_ignored(tmp_path):
    preset, _ = _load(tmp_path, {"prompt_label": "   "}, vars_block={"fps": 24})

    assert preset.vars == {"fps": 24}


def test_the_linter_rejects_a_bad_prompt_requirement(tmp_path):
    issues = _lint(tmp_path, {"prompt": "sometimes"})

    assert any(i.level == "error" and "prompt must be required, optional or none" in i.message for i in issues)


def test_the_linter_rejects_a_non_text_label(tmp_path):
    issues = _lint(tmp_path, {"prompt": "optional", "prompt_label": 3})

    assert any(i.level == "error" and "prompt_label" in i.message for i in issues)


def test_the_linter_flags_a_label_on_a_mode_without_a_prompt(tmp_path):
    issues = _lint(tmp_path, {"prompt": "none", "prompt_label": "Scene"})

    assert any(i.level == "warning" and "prompt_label is set but this mode has no prompt" in i.message for i in issues)


def test_the_linter_flags_a_label_on_a_mode_the_preset_lists_as_promptless(tmp_path):
    issues = _lint(tmp_path, {"prompt_placeholder": "a scene"}, vars_block={"promptless_modes": ["text"]})

    assert any(i.level == "warning" and "prompt_placeholder is set" in i.message for i in issues)


def test_the_linter_accepts_a_label_on_an_optional_prompt_mode(tmp_path):
    issues = _lint(tmp_path, {"prompt": "optional", "prompt_label": "Scene", "prompt_placeholder": "a newsroom"})

    assert [i for i in issues if i.level == "error" or "prompt_" in i.message] == []
