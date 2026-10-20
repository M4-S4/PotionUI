from pathlib import Path

import pytest

from src.features.presets.children_refs import ChildrenVarsError, parse_children_ref
from src.features.presets.linter import PresetLinter
from src.features.presets.loader import PresetTemplateLoader

FRAGMENT = """
fields:
  - name: steps
    type: number
    label: "{{ vars.family }} steps"
    default: "{{ vars.steps }}"
    configuration: {min: 1, max: 100}
"""


@pytest.fixture
def loader(monkeypatch):
    monkeypatch.setattr("src.features.presets.loader._known_field_types", lambda: {"section", "number"})
    return PresetTemplateLoader([])


def _section(tmp_path: Path, include, fragment: str = FRAGMENT):
    (tmp_path / "frag.yml").write_text(fragment, encoding="utf-8")
    return {"name": "box", "type": "section", "children": include}


def _build(loader, tmp_path, include, fragment: str = FRAGMENT):
    field = _section(tmp_path, include, fragment)
    return loader._build_field_template(field, tmp_path, "form.yml")


def _include(**vars):
    return {"file": "{{ paths.preset }}/frag.yml", "vars": vars}


def test_vars_are_substituted_with_types_kept(loader, tmp_path):
    built = _build(loader, tmp_path, _include(family="flux3", steps=50))
    child = built.children[0]
    assert child.default == 50
    assert child.label == "flux3 steps"


def test_string_include_without_vars_still_works(loader, tmp_path):
    fragment = "fields:\n  - {name: steps, type: number, default: 4}\n"
    built = _build(loader, tmp_path, "{{ paths.preset }}/frag.yml", fragment)
    assert built.children[0].default == 4


def test_missing_var_is_an_error(loader, tmp_path):
    with pytest.raises(ChildrenVarsError, match="missing vars.*steps"):
        _build(loader, tmp_path, _include(family="flux3"))


def test_unknown_var_is_an_error(loader, tmp_path):
    with pytest.raises(ChildrenVarsError, match="unknown vars.*extra"):
        _build(loader, tmp_path, _include(family="flux3", steps=1, extra=2))


def test_include_without_vars_leaves_fragment_untouched(loader, tmp_path):
    fragment = "fields:\n  - {name: steps, type: number, label: \"{{ vars.family }}\"}\n"
    built = _build(loader, tmp_path, "{{ paths.preset }}/frag.yml", fragment)
    assert built.children[0].label == "{{ vars.family }}"


@pytest.mark.parametrize("value", [[1, 2], {"a": 1}, None])
def test_non_scalar_var_is_an_error(value):
    with pytest.raises(ChildrenVarsError, match="scalar"):
        parse_children_ref({"file": "x.yml", "vars": {"steps": value}})


def test_malformed_include_shapes_are_errors():
    with pytest.raises(ChildrenVarsError, match="unknown keys"):
        parse_children_ref({"file": "x.yml", "var": {}})
    with pytest.raises(ChildrenVarsError, match="file"):
        parse_children_ref({"vars": {}})
    with pytest.raises(ChildrenVarsError, match="mapping"):
        parse_children_ref({"file": "x.yml", "vars": [1]})


def test_schema_rejects_non_scalar_var(loader, tmp_path):
    from src.features.presets.schema import validate_field_list

    _, errors = validate_field_list(
        [{"name": "box", "type": "section", "children": {"file": "x.yml", "vars": {"steps": [1]}}}],
        prefix="form.yml",
    )
    assert errors


def test_nested_include_in_a_vars_fragment_is_an_error(loader, tmp_path):
    fragment = """
fields:
  - name: inner
    type: section
    label: "{{ vars.family }}"
    children: "{{ paths.preset }}/other.yml"
"""
    with pytest.raises(ChildrenVarsError, match="cannot include another"):
        _build(loader, tmp_path, _include(family="flux3"), fragment)


def test_linter_reads_fragment_with_vars(tmp_path):
    (tmp_path / "frag.yml").write_text(FRAGMENT, encoding="utf-8")
    linter = PresetLinter([])
    data = linter._read_fragment(_include(family="z", steps=7), tmp_path)
    assert data["fields"][0]["default"] == 7
    assert linter._resolve_children_path(_include(family="z", steps=7), tmp_path) == Path(str(tmp_path) + "/frag.yml")


def _lint_vars_preset(tmp_path: Path, include: str, fragment: str | None):
    from tests.features.presets.test_shared_tab_fragments import _write_preset

    shared = tmp_path / "shared"
    (shared / "x").mkdir(parents=True)
    if fragment is not None:
        (shared / "x" / "tab.yml").write_text(fragment, encoding="utf-8")
    preset_dir = _write_preset(tmp_path / "presets", "unused")
    (preset_dir / "modes" / "txt2img" / "form.yml").write_text(
        "fields:\n"
        "  - name: tabs\n"
        "    type: tabs\n"
        "    children:\n"
        "      - name: t\n"
        "        type: tab\n"
        "        label: T\n"
        f"        children: {include}\n",
        encoding="utf-8",
    )
    return PresetLinter([str(tmp_path / "presets")], shared_path=shared).lint()


def test_lint_flags_missing_fragment_in_vars_include(tmp_path):
    issues = _lint_vars_preset(
        tmp_path, '{file: "{{ paths._shared }}/x/missing.yml", vars: {a: 1}}', None
    )
    assert [i for i in issues if i.level == "error" and "children fragment not found" in i.message]


def test_lint_accepts_vars_include_and_flags_bad_vars(tmp_path):
    fragment = "fields:\n  - {name: steps, type: number, default: \"{{ vars.steps }}\"}\n"
    include = '{file: "{{ paths._shared }}/x/tab.yml", vars: {steps: 4}}'
    issues = _lint_vars_preset(tmp_path, include, fragment)
    assert [i for i in issues if i.level == "error" and "children" in i.message] == []

    bad = '{file: "{{ paths._shared }}/x/tab.yml", vars: {steps: 4, extra: 1}}'
    issues = _lint_vars_preset(tmp_path / "bad", bad, fragment)
    assert [i for i in issues if i.level == "error" and "unknown vars" in i.message]


def _serializer():
    from unittest.mock import Mock

    from src.features.presets.form_serializer import PresetFormSerializer
    from src.platform.templating import TemplateProcessor

    return PresetFormSerializer(Mock(), TemplateProcessor(settings=None))


def _serializer_fields(tmp_path: Path, **vars):
    from src.features.presets.templates import FieldTemplate

    (tmp_path / "frag.yml").write_text(FRAGMENT, encoding="utf-8")
    include = {"file": str(tmp_path / "frag.yml"), "vars": vars}
    return _serializer()._resolve_external_children([FieldTemplate(name="box", type="section", children=include)], {})


def test_serializer_substitutes_vars(tmp_path):
    resolved = _serializer_fields(tmp_path, family="flux3", steps=50)
    child = resolved[0].children[0]
    assert child.default == 50
    assert child.label == "flux3 steps"


def test_serializer_surfaces_missing_var(tmp_path):
    with pytest.raises(ChildrenVarsError, match="missing vars"):
        _serializer_fields(tmp_path, family="flux3")
