from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.features.presets.linter import PresetLinter
from src.features.presets.loader import PresetTemplateLoader
from src.features.presets.schema import validate_form_file

TARGET_ID = "target-preset"


def _write_target(root: Path, vars_block: dict = None, llm_block: dict = None) -> None:
    preset_dir = root / TARGET_ID
    manifest = {
        "schema": 1, "id": TARGET_ID, "name": TARGET_ID, "version": "1.0.0",
        "category": "image", "engine": "native", "modes": ["txt2img"],
    }
    if vars_block is not None:
        manifest["vars"] = vars_block
    if llm_block is not None:
        manifest["llm"] = llm_block
    mode_dir = preset_dir / "modes" / "txt2img"
    mode_dir.mkdir(parents=True)
    (preset_dir / "preset.yml").write_text(yaml.dump(manifest), encoding="utf-8")
    (mode_dir / "pipeline.yml").write_text("pipeline: []\n", encoding="utf-8")
    (mode_dir / "form.yml").write_text("fields: []\n", encoding="utf-8")


def _write_mode(plugin_dir: Path, name: str, form: dict) -> None:
    mode_dir = plugin_dir / "contributed" / "modes" / name
    mode_dir.mkdir(parents=True)
    (mode_dir / "pipeline.yml").write_text("pipeline: []\n", encoding="utf-8")
    (mode_dir / "form.yml").write_text(yaml.dump({"fields": [], **form}), encoding="utf-8")


def _load(tmp_path: Path, forms: dict, **target):
    core_root = tmp_path / "presets"
    _write_target(core_root, **target)
    plugin_dir = tmp_path / "plugin"
    for name, form in forms.items():
        _write_mode(plugin_dir, name, form)
    manifest = SimpleNamespace(
        id="some-plugin", plugin_dir=plugin_dir,
        preset_modes=[{"target": TARGET_ID, "modes_root": "contributed"}],
    )
    loader = PresetTemplateLoader([str(core_root)], plugin_registry=SimpleNamespace(get_enabled_plugins=lambda: [manifest]))
    loader.load_presets()
    return next(p for p in loader.presets if p.id == TARGET_ID), loader


def test_an_optional_prompt_mode_joins_the_targets_optional_prompt_modes(tmp_path):
    target, loader = _load(
        tmp_path, {"extra": {"prompt": "optional"}}, vars_block={"optional_prompt_modes": ["existing"]},
    )

    assert loader.load_errors == {}
    assert target.vars["optional_prompt_modes"] == ["existing", "extra"]
    assert not target.vars.get("promptless_modes")


def test_a_prompt_none_mode_joins_the_targets_promptless_modes(tmp_path):
    target, _ = _load(tmp_path, {"extra": {"prompt": "none"}})

    assert target.vars["promptless_modes"] == ["extra"]
    assert not target.vars.get("optional_prompt_modes")


def test_a_required_prompt_mode_leaves_the_target_vars_alone(tmp_path):
    target, _ = _load(tmp_path, {"extra": {"prompt": "required"}}, vars_block={"fps": 24})

    assert target.vars == {"fps": 24}


def test_an_llm_guide_becomes_the_modes_guide_and_keeps_the_others(tmp_path):
    target, _ = _load(
        tmp_path, {"extra": {"llm_guide": "Write one plain sentence.\n"}},
        llm_block={"guide": "base", "modes": {"txt2img": {"guide": "core guide"}}},
    )

    assert target.llm["guide"] == "base"
    assert target.llm["modes"] == {
        "txt2img": {"guide": "core guide"},
        "extra": {"guide": "Write one plain sentence."},
    }


def test_segment_templates_replace_the_flat_list_for_that_mode_only(tmp_path):
    flat = [{"name": "Flat", "segments": [{"name": "s", "content": ""}]}]
    target, _ = _load(
        tmp_path, {"extra": {"segment_templates": []}},
        vars_block={"prompt": {"segment_join": "paragraph", "segment_templates": flat, "modes": {"other": {"segment_templates": flat}}}},
    )

    prompt = target.vars["prompt"]
    assert prompt["segment_templates"] == flat
    assert prompt["segment_join"] == "paragraph"
    assert prompt["modes"] == {"other": {"segment_templates": flat}, "extra": {"segment_templates": []}}


def test_a_mode_declaring_nothing_leaves_vars_and_llm_untouched(tmp_path):
    target, _ = _load(tmp_path, {"extra": {}}, vars_block={"fps": 24}, llm_block={"guide": "base"})

    assert target.vars == {"fps": 24}
    assert target.llm["guide"] == "base"
    assert "modes" not in target.llm
    assert "extra" in target.modes


def test_a_rejected_contribution_does_not_touch_the_targets_vars(tmp_path):
    target, loader = _load(tmp_path, {"txt2img": {"prompt": "optional", "llm_guide": "x"}}, vars_block={"fps": 24})

    assert loader.load_errors
    assert target.vars == {"fps": 24}
    assert target.llm is None


def test_an_unknown_prompt_value_is_a_schema_error():
    validated, errors = validate_form_file({"fields": [], "prompt": "sometimes"}, prefix="form.yml")

    assert validated is None
    assert errors


@pytest.mark.parametrize("form", [{"prompt": "optional"}, {"llm_guide": "g"}, {"segment_templates": []}])
def test_the_form_schema_accepts_each_prompting_key(form):
    validated, errors = validate_form_file({"fields": [], **form}, prefix="form.yml")

    assert errors == []
    assert validated is not None


def test_the_linter_accepts_a_contributed_mode_that_declares_prompting(tmp_path):
    core_root = tmp_path / "presets"
    _write_target(core_root)
    plugin_dir = tmp_path / "plugin"
    _write_mode(plugin_dir, "extra", {"prompt": "optional", "llm_guide": "g", "segment_templates": []})
    manifest = SimpleNamespace(
        id="some-plugin", plugin_dir=plugin_dir,
        preset_modes=[{"target": TARGET_ID, "modes_root": "contributed"}],
    )

    issues = PresetLinter([str(core_root)], plugin_manifests=[manifest]).lint()

    assert [i for i in issues if i.level == "error"] == []
