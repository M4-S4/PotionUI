"""Coverage for the shared path-containment primitive and the Director
document helpers built on it (shared between video_director/music_director)."""

import ntpath
from pathlib import Path

import pytest

from src.platform.util import path_resolution

from src.platform.util.path_resolution import (
    apply_preset_mode_overlay,
    is_within,
    resolve_media_ref,
    resolve_within,
)


class TestResolveWithin:
    def test_a_relative_path_inside_root_resolves(self, tmp_path):
        (tmp_path / "inputs").mkdir()
        (tmp_path / "inputs" / "a.png").write_bytes(b"x")

        resolved = resolve_within(tmp_path, "inputs/a.png")

        assert resolved == (tmp_path / "inputs" / "a.png").resolve()

    def test_a_traversal_escape_is_rejected(self, tmp_path):
        assert resolve_within(tmp_path, "../../etc/passwd") is None

    def test_an_absolute_path_outside_root_is_rejected(self, tmp_path):
        assert resolve_within(tmp_path, "/etc/passwd") is None

    def test_an_absolute_path_inside_root_is_accepted(self, tmp_path):
        target = tmp_path / "a.png"
        target.write_bytes(b"x")

        assert resolve_within(tmp_path, str(target)) == target.resolve()

    def test_must_exist_rejects_an_absent_but_contained_path(self, tmp_path):
        assert resolve_within(tmp_path, "missing.png", must_exist=True) is None

    def test_must_exist_false_does_not_require_existence(self, tmp_path):
        assert resolve_within(tmp_path, "missing.png") == (tmp_path / "missing.png").resolve()


class TestApplyPresetModeOverlay:
    def test_no_overrides_returns_a_copy_of_capabilities(self):
        caps = {"modes": {"director": {"audio": True}}}
        result = apply_preset_mode_overlay(caps, "director")
        assert result == caps
        assert result is not caps

    def test_an_unmentioned_mode_falls_through_unchanged(self):
        caps = {"modes": {"director": {"audio": True}}}
        caps["preset_mode_overrides"] = {"refs": {"modes": {"director": {"audio": False}}}}

        result = apply_preset_mode_overlay(caps, "video")

        assert result["modes"]["director"]["audio"] is True
        assert "preset_mode_overrides" not in result

    def test_a_matching_override_shallow_merges_the_named_mode(self):
        caps = {
            "modes": {"director": {"audio": True, "max_segments": 4}},
            "preset_mode_overrides": {"refs": {"modes": {"director": {"audio": False}}}},
        }

        result = apply_preset_mode_overlay(caps, "refs")

        assert result["modes"]["director"] == {"audio": False, "max_segments": 4}


class TestResolveMediaRef:
    def test_a_contained_existing_path_resolves(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"x")
        errors: list = []

        result = resolve_media_ref({"relative_path": "a.png"}, tmp_path, "ctx", errors)

        assert result["path"] == str((tmp_path / "a.png").resolve())
        assert errors == []

    def test_an_escaping_path_is_rejected_with_a_clear_error(self, tmp_path):
        errors: list = []

        result = resolve_media_ref({"path": "../../etc/passwd"}, tmp_path, "ctx", errors)

        assert result is None
        assert "escapes" in errors[0]

    def test_a_missing_path_or_relative_path_is_rejected(self, tmp_path):
        errors: list = []

        result = resolve_media_ref({}, tmp_path, "ctx", errors)

        assert result is None
        assert "requires a 'path'" in errors[0]


class TestIsWithin:
    def test_a_descendant_is_within(self, tmp_path):
        assert is_within(tmp_path, tmp_path / "a" / "b")

    def test_the_base_itself_is_within(self, tmp_path):
        assert is_within(tmp_path, tmp_path)

    def test_a_sibling_sharing_a_name_prefix_is_not_within(self, tmp_path):
        assert not is_within(tmp_path / "plugin", tmp_path / "plugin-evil" / "x")

    def test_dotdot_segments_are_collapsed(self, tmp_path):
        assert not is_within(tmp_path / "root", tmp_path / "root" / ".." / "other")

    def test_a_relative_target_against_an_absolute_base_is_not_within(self, tmp_path):
        assert not is_within(tmp_path, Path("relative/dir"))

    def test_accepts_plain_strings(self, tmp_path):
        assert is_within(str(tmp_path), str(tmp_path / "child"))


class TestIsWithinWindows:
    @pytest.fixture(autouse=True)
    def _windows_paths(self, monkeypatch):
        monkeypatch.setattr(path_resolution, "_path", ntpath)

    def test_containment_ignores_case(self):
        assert is_within("C:\\Base", "c:\\base\\x")

    def test_a_target_on_another_drive_is_not_within(self):
        assert not is_within("C:\\a", "D:\\a")

    def test_a_sibling_sharing_a_prefix_is_not_within(self):
        assert not is_within("C:\\plugin", "C:\\plugin-evil\\x")

    def test_forward_slashes_are_normalised(self):
        assert is_within("C:\\Base", "C:/base/sub/x")
