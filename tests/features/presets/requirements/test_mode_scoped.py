from pathlib import Path
from unittest.mock import MagicMock, patch

import asyncio

import pytest

from src.features.presets import operations
from src.features.presets.collaborators import PresetCollaborators
from src.features.presets.linter import PresetLinter
from src.features.presets.loader import PresetTemplateLoader
from src.features.presets.requirements.builtin import register_builtin_requirement_checkers
from src.features.presets.requirements.evaluator import RequirementsCache, summarize
from src.features.presets.requirements.contracts import RequirementResult
from src.features.presets.templates import FormTemplate, ModeTemplate, PresetTemplate
from src.platform.plugins.requirement_checkers import requirement_checker_registry

MISSING_PACKAGE = "definitely-not-a-real-package-xyz"


def setup_module(module):
    if not requirement_checker_registry.all():
        register_builtin_requirement_checkers(requirement_checker_registry)


def _write_preset(tmp_path: Path, form_yaml: str, variant_form_yaml: str = "") -> Path:
    preset_dir = tmp_path / "presets" / "Foo" / "std"
    preset_dir.mkdir(parents=True)
    (preset_dir / "preset.yml").write_text(
        (
            'schema: 1\n'
            'id: "01MODEREQAAAAAAAAAAAAAAAAAA"\n'
            'name: "Test"\n'
            'version: "1.0.0"\n'
            'category: "image"\n'
            'engine: "native"\n'
            'requirements:\n'
            '  - type: platform\n'
            '    os: [linux, windows, darwin]\n'
            'modes:\n'
            '  - txt2img\n'
            '  - colorize\n'
        )
    )
    (preset_dir / "tests.yml").write_text("schema: 1\ncases: []\n")
    for mode, form in (("txt2img", "fields: []\n"), ("colorize", form_yaml)):
        mode_dir = preset_dir / "modes" / mode
        mode_dir.mkdir(parents=True)
        (mode_dir / "pipeline.yml").write_text("pipeline: []\n")
        (mode_dir / "form.yml").write_text(form)
    if variant_form_yaml:
        variant_dir = preset_dir / "modes" / "colorize" / "variants" / "alt"
        variant_dir.mkdir(parents=True)
        (variant_dir / "form.yml").write_text(variant_form_yaml)
    return preset_dir


COLORIZE_FORM = (
    "fields: []\n"
    "requirements:\n"
    f"  - type: python_package\n    name: {MISSING_PACKAGE}\n    hint: get the file\n"
)


def _preset(requirements, modes=("txt2img", "colorize")) -> PresetTemplate:
    return PresetTemplate(
        id="preset-1", name="Preset", version="1.0.0", path="/tmp/preset", engine="native",
        modes={m: ModeTemplate(forms=[FormTemplate(name="default", fields=[])], pipes=[]) for m in modes},
        requirements=requirements,
    )


def _collaborators(preset, cache):
    defaults = dict(
        preset_loader=MagicMock(), preset_processor=MagicMock(), template_processor=MagicMock(),
        file_repo=MagicMock(find_preset_by_id=MagicMock(return_value=preset)),
        db_repo=MagicMock(), user_repo=MagicMock(), group_repo=MagicMock(),
        pipeline_builder=MagicMock(), pipe_catalog=MagicMock(), plugins=MagicMock(),
        settings=MagicMock(), backend_registry=None, model_index=None, gpu_monitor=None,
        requirements_cache=cache,
    )
    with patch("src.features.presets.collaborators.PresetFormSerializer"):
        return PresetCollaborators(**defaults)


SCOPED_PRESET_REQUIREMENTS = [
    {"type": "platform", "os": ["linux", "windows", "darwin"]},
    {"type": "python_package", "name": MISSING_PACKAGE, "mode": "colorize"},
]


class TestLoaderMergesModeRequirements:
    def test_mode_entries_are_appended_with_their_mode(self, tmp_path):
        _write_preset(tmp_path, COLORIZE_FORM)
        loader = PresetTemplateLoader([str(tmp_path)])
        loader.load_presets()

        assert loader.load_errors == {}
        requirements = loader.presets[0].requirements
        assert requirements[0] == {"type": "platform", "os": ["linux", "windows", "darwin"], "optional": False}
        assert requirements[1] == {
            "type": "python_package", "name": MISSING_PACKAGE, "hint": "get the file", "optional": False,
            "mode": "colorize",
        }
        assert len(requirements) == 2

    def test_a_variant_form_does_not_contribute(self, tmp_path):
        _write_preset(
            tmp_path, "fields: []\n",
            variant_form_yaml=f"fields: []\nrequirements:\n  - type: python_package\n    name: {MISSING_PACKAGE}\n",
        )
        loader = PresetTemplateLoader([str(tmp_path)])
        loader.load_presets()

        assert [spec.get("mode") for spec in loader.presets[0].requirements] == [None]


class TestSummaryIsPresetWide:
    def test_summarize_ignores_scoped_entries(self):
        specs = [{"type": "a"}, {"type": "b", "mode": "colorize"}]
        results = [RequirementResult(status="ok", detail=""), RequirementResult(status="missing", detail="")]

        assert summarize(specs, results) == {"ok": 1, "missing": 0, "unknown": 0, "optional_missing": 0}

    @pytest.mark.asyncio
    async def test_admin_summary_excludes_scoped_miss_but_item_carries_mode(self):
        preset = _preset(SCOPED_PRESET_REQUIREMENTS)
        collaborators = _collaborators(preset, RequirementsCache())

        data = await operations.get_preset_requirements(collaborators, preset.id)

        assert data["summary"]["missing"] == 0
        assert data["summary"]["ok"] == 1
        scoped = next(item for item in data["results"] if item["mode"] == "colorize")
        assert scoped["status"] == "missing"
        assert scoped["name"] == MISSING_PACKAGE
        assert next(item for item in data["results"] if item["type"] == "platform")["mode"] is None

    def test_list_style_peek_ignores_scoped_miss(self):
        from src.features.presets.operations.query import _peek_requirements_summary

        preset = _preset(SCOPED_PRESET_REQUIREMENTS)
        cache = RequirementsCache()
        collaborators = _collaborators(preset, cache)
        asyncio.run(operations.get_preset_requirements(collaborators, preset.id))

        assert _peek_requirements_summary(collaborators, preset.id)["missing"] == 0


class TestPerModeMissingView:
    @pytest.mark.asyncio
    async def test_cold_cache_is_unknown_for_scoped_mode_only(self):
        preset = _preset(SCOPED_PRESET_REQUIREMENTS)
        collaborators = _collaborators(preset, RequirementsCache())

        modes = {m["name"]: m for m in operations.get_available_modes(collaborators, preset.id)["modes"]}

        assert modes["colorize"]["requirements_missing"] is None
        assert modes["txt2img"]["requirements_missing"] == []

    @pytest.mark.asyncio
    async def test_refresh_populates_missing_names_for_the_scoped_mode_only(self):
        preset = _preset(SCOPED_PRESET_REQUIREMENTS)
        collaborators = _collaborators(preset, RequirementsCache())

        await operations.refresh_mode_requirements(collaborators, preset.id)
        modes = {m["name"]: m for m in operations.get_available_modes(collaborators, preset.id)["modes"]}

        assert modes["colorize"]["requirements_missing"] == [MISSING_PACKAGE]
        assert modes["txt2img"]["requirements_missing"] == []

    @pytest.mark.asyncio
    async def test_refresh_skips_evaluation_without_scoped_entries(self):
        preset = _preset([SCOPED_PRESET_REQUIREMENTS[0]])
        cache = RequirementsCache()
        collaborators = _collaborators(preset, cache)

        await operations.refresh_mode_requirements(collaborators, preset.id)

        assert cache.peek_host_summary(requirement_checker_registry, preset) is None

    @pytest.mark.asyncio
    async def test_optional_scoped_miss_is_not_reported(self):
        preset = _preset([{"type": "python_package", "name": MISSING_PACKAGE, "mode": "colorize", "optional": True}])
        collaborators = _collaborators(preset, RequirementsCache())

        await operations.refresh_mode_requirements(collaborators, preset.id)
        modes = {m["name"]: m for m in operations.get_available_modes(collaborators, preset.id)["modes"]}

        assert modes["colorize"]["requirements_missing"] == []


class TestLintModeRequirements:
    def _lint(self, tmp_path, form_yaml, variant_form_yaml=""):
        _write_preset(tmp_path, form_yaml, variant_form_yaml)
        return PresetLinter([str(tmp_path)]).lint()

    def test_valid_mode_entry_has_no_requirements_issue(self, tmp_path):
        issues = self._lint(tmp_path, COLORIZE_FORM)

        assert not any("requirements" in i.message for i in issues if i.level == "error")

    def test_unknown_type_in_mode_form_is_error(self, tmp_path):
        issues = self._lint(tmp_path, "fields: []\nrequirements:\n  - type: nope\n")

        assert any(
            i.level == "error" and "modes/colorize/form.yml requirements[0]" in i.message and "unknown type" in i.message
            for i in issues
        )

    def test_bad_arguments_in_mode_form_are_error(self, tmp_path):
        issues = self._lint(tmp_path, "fields: []\nrequirements:\n  - type: model\n")

        assert any(
            i.level == "error" and "modes/colorize/form.yml requirements[0] (model)" in i.message for i in issues
        )

    def test_variant_requirements_are_error(self, tmp_path):
        issues = self._lint(
            tmp_path, "fields: []\n",
            variant_form_yaml=f"fields: []\nrequirements:\n  - type: python_package\n    name: {MISSING_PACKAGE}\n",
        )

        assert any(i.level == "error" and "only read from modes/colorize/form.yml" in i.message for i in issues)
