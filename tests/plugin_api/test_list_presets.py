from types import SimpleNamespace

import pytest

from src.features.presets.file_repository import FilePresetRepository
from src.features.presets.loader import PresetTemplateLoader
from src.features.presets.repository import DatabasePresetRepository
from src.features.user_groups.repository import UserGroupRepository
from src.platform.database.rows import now_utc
from src.platform.plugins import runtime_registries
from src.platform.security.user import AccountType, User
from src.plugin_api import presets as api
from src.plugin_api.presets import PresetDescription, describe_preset, list_presets

DIRECT = "01TESTLISTPRESETDIRECT0000"
GROUP = "01TESTLISTPRESETGROUP00000"
OTHER = "01TESTLISTPRESETOTHER00000"
UNINSTALLED = "01TESTLISTPRESETUNINSTALL0"
NO_FILES = "01TESTLISTPRESETNOFILES000"

FORM = """name: "custom"
fields:
  - name: "source_image"
    type: "image"
    label: "Source"
    required: true
"""


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _preset(root, folder, preset_id, name, category):
    preset = root / folder
    _write(preset / "preset.yml", (
        f'schema: 1\nid: "{preset_id}"\nname: "{name}"\nversion: "1.0.0"\ncategory: "{category}"\n'
        'engine: "native"\nrequires:\n  min_vram_gb: 8\nmodes:\n  - txt2img\n'
    ))
    _write(preset / "tests.yml", "schema: 1\ncases: []\n")
    _write(preset / "modes" / "txt2img" / "form.yml", FORM)
    _write(preset / "modes" / "txt2img" / "pipeline.yml", "pipeline: []\n")


def _user(user_id, account_type=AccountType.USER):
    return User(
        id=user_id, username=user_id, email=f"{user_id}@example.test", password_hash="$2b$12$x",
        account_type=account_type, created_at=now_utc(), last_login=None,
    )


class World:
    def __init__(self, db, root):
        for user_id in ("member", "stranger", "admin"):
            with db.get_cursor() as cursor:
                cursor.execute(
                    "INSERT INTO users (id, username, email, password_hash, account_type) VALUES (?, ?, ?, 'x', 'USER')",
                    (user_id, user_id, f"{user_id}@example.test"),
                )
        self.member = _user("member")
        self.stranger = _user("stranger")
        self.admin = _user("admin", AccountType.ADMIN)
        _preset(root, "Direct", DIRECT, "Direct", "image")
        _preset(root, "Group", GROUP, "Group", "video")
        _preset(root, "Other", OTHER, "Other", "audio")
        _preset(root, "Uninstalled", UNINSTALLED, "Uninstalled", "image")
        self.loader = PresetTemplateLoader([str(root)])
        self.files = FilePresetRepository(self.loader)
        self.presets = DatabasePresetRepository()
        groups = UserGroupRepository()
        for preset_id in (DIRECT, GROUP, OTHER, NO_FILES):
            self.presets.install_preset(preset_id)
        self.presets.assign_preset_to_user(DIRECT, "member")
        self.presets.assign_preset_to_user(NO_FILES, "member")
        studio = groups.create_group("studio")
        groups.add_user_to_group(studio.id, "member")
        groups.assign_preset_to_group(studio.id, GROUP)
        self.container = SimpleNamespace(
            file_preset_repository=self.files,
            preset_collaborators=SimpleNamespace(db_repo=self.presets),
        )

    def route_order(self, preset_ids):
        return [entry["id"] for entry in self.files.list_all_presets() if entry["id"] in preset_ids]


@pytest.fixture
def world(mock_db, tmp_path, monkeypatch):
    built = World(mock_db, tmp_path / "presets")
    monkeypatch.setattr(runtime_registries, "_container", built.container)
    return built


def test_a_user_gets_the_presets_assigned_directly_and_through_a_group(world):
    listed = list_presets(world.member)

    assert world.loader.load_errors == {}
    assert [preset.id for preset in listed] == world.route_order({DIRECT, GROUP})


def test_a_user_without_assignments_gets_nothing(world):
    assert list_presets(world.stranger) == []


def test_an_admin_gets_every_installed_preset_and_no_other(world):
    assert [preset.id for preset in list_presets(world.admin)] == world.route_order({DIRECT, GROUP, OTHER})


def test_an_installed_preset_whose_files_are_missing_is_left_out(world):
    assert NO_FILES not in {preset.id for preset in list_presets(world.member)}
    assert NO_FILES not in {preset.id for preset in list_presets(world.admin)}


def test_each_entry_is_the_full_description(world):
    listed = list_presets(world.admin)

    assert all(isinstance(preset, PresetDescription) for preset in listed)
    assert listed == [describe_preset(preset.id) for preset in listed]
    group = next(preset for preset in listed if preset.id == GROUP)
    assert (group.name, group.engine, group.category, group.requires) == ("Group", "native", "video", {"min_vram_gb": 8})
    assert group.modes["txt2img"].forms["custom"]["source_image"].required is True


def test_listing_needs_a_user(world):
    with pytest.raises(PermissionError):
        list_presets(None)


def test_list_presets_is_importable_from_the_package_root():
    import src.plugin_api as root

    assert root.list_presets is api.list_presets
    assert "list_presets" in root.__all__
    assert "list_presets" in api.__all__
