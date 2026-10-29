import importlib.util
import sys
from pathlib import Path

import pytest

from src.features.presets.linter import PresetLinter
from tests.features.presets.test_linter import _write_preset


def _form(fields_yaml):
    return "name: custom\nfields:\n" + fields_yaml


def _lint(tmp_path, preset_id, fields_yaml):
    preset_dir = _write_preset(tmp_path, "presets/native/Foo/std", preset_id, ["txt2img"])
    form_file = preset_dir / "modes" / "txt2img" / "form.yml"
    form_file.write_text(_form(fields_yaml))
    return PresetLinter([str(tmp_path)]).lint()


def _errors(issues):
    return [issue for issue in issues if issue.level == "error"]


def _field_problems(issues):
    return [
        issue for issue in issues
        if issue.level == "error" or "FieldConfigSpec" in issue.message
    ]


GOOD_FORM = """  - name: fit
    type: segmented
    label: Fit
    default: pad
    configuration:
      options:
        - { label: Pad, value: pad }
        - { label: Cover, value: cover }
      fit: true
      tooltip: How a clip meets the frame
  - name: angle
    type: segmented
    label: Angle
    default: 90
    configuration:
      options: [90, 180, 270]
  - name: where
    type: anchor
    label: Anchor
    default: top_left
    configuration:
      tooltip: Where the image sits
  - name: bare_anchor
    type: anchor
    label: Bare anchor
  - name: fill
    type: color
    label: Fill
    default: transparent
    configuration:
      allow_transparent: true
  - name: key
    type: color
    label: Key
    default: "#00B140"
"""


class TestGoodConfigsPass:
    def test_a_preset_using_each_type_has_no_errors_or_config_warnings(self, tmp_path):
        issues = _lint(tmp_path, "01TOOLFIELDSGOODAAAAAAAAAA", GOOD_FORM)

        assert _field_problems(issues) == []

    def test_each_type_alone_is_clean(self, tmp_path):
        snippets = {
            "01TOOLFIELDSSEGONLYAAAAAA": "  - { name: s, type: segmented, label: S, configuration: { options: [a, b] } }\n",
            "01TOOLFIELDSANCHONLYAAAAA": "  - { name: a, type: anchor, label: A, default: bottom_right }\n",
            "01TOOLFIELDSCOLORONLYAAAA": "  - { name: c, type: color, label: C, default: '#fff' }\n",
        }

        for preset_id, snippet in snippets.items():
            assert _field_problems(_lint(tmp_path / preset_id, preset_id, snippet)) == [], preset_id


class TestBadConfigsFail:
    @pytest.mark.parametrize(
        "case,fields,fragment",
        [
            (
                "segmented without options",
                "  - { name: s, type: segmented, label: S }\n",
                "needs a non-empty `configuration.options` list",
            ),
            (
                "segmented with empty options",
                "  - { name: s, type: segmented, label: S, configuration: { options: [] } }\n",
                "needs a non-empty `configuration.options` list",
            ),
            (
                "segmented option without a value",
                "  - { name: s, type: segmented, label: S, configuration: { options: [{ label: Pad }] } }\n",
                "needs a `value`",
            ),
            (
                "segmented duplicate values",
                "  - { name: s, type: segmented, label: S, configuration: { options: [a, a] } }\n",
                "must be unique",
            ),
            (
                "segmented default outside options",
                "  - { name: s, type: segmented, label: S, default: z, configuration: { options: [a, b] } }\n",
                "is not one of the declared options",
            ),
            (
                "segmented option value is a list",
                "  - { name: s, type: segmented, label: S, configuration: { options: [{ value: [1] }] } }\n",
                "must be a string or a number",
            ),
            (
                "anchor default not one of the nine",
                "  - { name: a, type: anchor, label: A, default: middle }\n",
                "must be one of",
            ),
            (
                "anchor hyphenated default",
                "  - { name: a, type: anchor, label: A, default: top-left }\n",
                "must be one of",
            ),
            (
                "color default not hex",
                "  - { name: c, type: color, label: C, default: red }\n",
                "colour",
            ),
            (
                "color default transparent without allow_transparent",
                "  - { name: c, type: color, label: C, default: transparent }\n",
                "allow_transparent",
            ),
            (
                "color allow_transparent not a bool",
                "  - { name: c, type: color, label: C, configuration: { allow_transparent: 'yes' } }\n",
                "must be a boolean",
            ),
            (
                "color default is a number",
                "  - { name: c, type: color, label: C, default: 12 }\n",
                "must be a #RRGGBB string",
            ),
        ],
    )
    def test_bad_config_is_a_lint_error(self, tmp_path, case, fields, fragment):
        issues = _lint(tmp_path, "01TOOLFIELDSBADAAAAAAAAAAA", fields)

        assert any(fragment in issue.message for issue in _errors(issues)), (case, [i.message for i in issues])

    def test_unknown_config_key_on_a_new_type_is_a_warning(self, tmp_path):
        issues = _lint(
            tmp_path,
            "01TOOLFIELDSUNKNOWNAAAAAAA",
            "  - { name: a, type: anchor, label: A, configuration: { cells: 9 } }\n",
        )

        assert any(
            issue.level == "warning" and "type 'anchor'" in issue.message and "['cells']" in issue.message
            for issue in issues
        )

    def test_the_error_names_the_field_and_type(self, tmp_path):
        issues = _lint(tmp_path, "01TOOLFIELDSNAMEDAAAAAAAAA", "  - { name: where, type: anchor, label: A, default: nope }\n")

        assert any("'where' (anchor)" in issue.message for issue in _errors(issues))


def _load_cli():
    path = Path(__file__).resolve().parents[3] / "scripts" / "preset_lint.py"
    spec = importlib.util.spec_from_file_location("preset_lint_cli_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPresetLintCli:
    def test_cli_exits_clean_for_a_preset_using_each_type(self, tmp_path, monkeypatch, capsys):
        preset_dir = _write_preset(tmp_path, "presets/native/Foo/std", "01TOOLFIELDSCLIGOODAAAAAA", ["txt2img"])
        (preset_dir / "modes" / "txt2img" / "form.yml").write_text(_form(GOOD_FORM))
        monkeypatch.setattr(sys, "argv", ["preset_lint.py", str(tmp_path)])

        code = _load_cli().main()

        assert code == 0
        assert "No issues found." in capsys.readouterr().out

    def test_cli_exits_nonzero_for_a_bad_config(self, tmp_path, monkeypatch, capsys):
        preset_dir = _write_preset(tmp_path, "presets/native/Foo/std", "01TOOLFIELDSCLIBADAAAAAAA", ["txt2img"])
        (preset_dir / "modes" / "txt2img" / "form.yml").write_text(
            _form("  - { name: a, type: anchor, label: A, default: nowhere }\n")
        )
        monkeypatch.setattr(sys, "argv", ["preset_lint.py", str(tmp_path)])

        code = _load_cli().main()

        assert code == 1
        assert "'a' (anchor)" in capsys.readouterr().out
