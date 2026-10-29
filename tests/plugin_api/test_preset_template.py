from pathlib import Path
from types import SimpleNamespace

import pytest

from src.features.presets.file_repository import FilePresetRepository
from src.features.presets.loader import PresetTemplateLoader
from src.features.presets.repository import DatabasePresetRepository
from src.features.user_groups.repository import UserGroupRepository
from src.platform.plugins import runtime_registries
from src.plugin_api import presets as api
from src.plugin_api.presets import PresetInstallState

SOLO = "01TESTPRESETTEMPLATESOLO00"
GROUPED = "01TESTPRESETTEMPLATEGROUP0"
IDLE = "01TESTPRESETTEMPLATEIDLE00"

FORM = """name: "custom"
fields:
  - name: "source_image"
    type: "image"
    label: "Source"
"""


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _preset(root, folder, preset_id):
    preset = root / folder
    _write(preset / "preset.yml", (
        f'schema: 1\nid: "{preset_id}"\nname: "{folder}"\nversion: "1.0.0"\ncategory: "image"\n'
        'engine: "native"\nmodes:\n  - txt2img\n'
    ))
    _write(preset / "tests.yml", "schema: 1\ncases: []\n")
    _write(preset / "modes" / "txt2img" / "form.yml", FORM)
    _write(preset / "modes" / "txt2img" / "pipeline.yml", "pipeline: []\n")


@pytest.fixture
def world(mock_db, tmp_path, monkeypatch):
    with mock_db.get_cursor() as cursor:
        cursor.execute(
            "INSERT INTO users (id, username, email, password_hash, account_type) VALUES ('member', 'member', 'member@example.test', 'x', 'USER')"
        )
    root = tmp_path / "presets"
    shared = tmp_path / "shared"
    for folder, preset_id in (("Solo", SOLO), ("Grouped", GROUPED), ("Idle", IDLE)):
        _preset(root, folder, preset_id)
    loader = PresetTemplateLoader([str(root)], shared_path=shared)
    presets = DatabasePresetRepository()
    groups = UserGroupRepository()
    for preset_id in (SOLO, GROUPED, IDLE):
        presets.install_preset(preset_id)
    presets.assign_preset_to_user(SOLO, "member")
    groups.assign_preset_to_group(groups.create_group("studio").id, GROUPED)
    container = SimpleNamespace(
        file_preset_repository=FilePresetRepository(loader),
        preset_collaborators=SimpleNamespace(db_repo=presets, group_repo=groups),
    )
    monkeypatch.setattr(runtime_registries, "_container", container)
    return SimpleNamespace(loader=loader, root=root, shared=shared)


def test_a_loaded_preset_is_handed_over_as_its_loader_built_it(world):
    template = api.preset_template(SOLO)

    assert world.loader.load_errors == {}
    assert template is world.loader.get(SOLO)
    assert Path(template.path).resolve() == (world.root / "Solo").resolve()
    assert template.modes["txt2img"].forms[0].fields[0].name == "source_image"
    assert api.preset_template("01TESTPRESETTEMPLATENONE00") is None


def test_a_path_a_preset_names_has_its_variables_filled(world):
    preset_root = Path(api.preset_template(SOLO).path)

    assert api.resolve_preset_path(SOLO, "{{ paths.preset }}/options.yml") == preset_root / "options.yml"
    assert api.resolve_preset_path(SOLO, "{{paths._shared}}/lists/names.yml") == world.shared / "lists" / "names.yml"
    assert api.resolve_preset_path("01TESTPRESETTEMPLATENONE00", "{{ paths.preset }}/options.yml") is None


def test_a_preset_is_assigned_through_a_user_or_a_group(world):
    assert api.preset_install_state(SOLO) == PresetInstallState(installed=True, assigned=True)
    assert api.preset_install_state(GROUPED) == PresetInstallState(installed=True, assigned=True)
    assert api.preset_install_state(IDLE) == PresetInstallState(installed=True, assigned=False)
    assert api.preset_install_state("01TESTPRESETTEMPLATENONE00") == PresetInstallState(installed=False, assigned=False)


def test_the_template_reads_are_part_of_the_presets_api():
    assert {"PresetInstallState", "PresetTemplate", "preset_install_state", "preset_template", "resolve_preset_path"} <= set(api.__all__)
