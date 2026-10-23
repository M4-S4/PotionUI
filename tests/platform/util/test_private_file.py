import os
import stat

import pytest

from src.platform.util import private_file
from src.platform.util.private_file import write_private_file

posix_only = pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")


def test_writes_text_and_creates_parent_directories(tmp_path):
    target = tmp_path / "nested" / "dir" / "secret"

    write_private_file(target, "value")

    assert target.read_text(encoding="utf-8") == "value"


def test_replaces_an_existing_file(tmp_path):
    target = tmp_path / "secret"
    target.write_text("old", encoding="utf-8")

    write_private_file(target, "new")

    assert target.read_text(encoding="utf-8") == "new"


@posix_only
def test_final_file_is_0600_even_under_a_permissive_umask(tmp_path):
    previous = os.umask(0)
    try:
        write_private_file(tmp_path / "secret", "value")
    finally:
        os.umask(previous)

    assert stat.S_IMODE((tmp_path / "secret").stat().st_mode) == 0o600


def test_works_without_fchmod(tmp_path, monkeypatch):
    monkeypatch.delattr(os, "fchmod", raising=False)

    write_private_file(tmp_path / "secret", "value")

    assert (tmp_path / "secret").read_text(encoding="utf-8") == "value"


def test_failed_replace_leaves_no_temp_file_and_keeps_the_original(tmp_path, monkeypatch):
    target = tmp_path / "secret"
    target.write_text("old", encoding="utf-8")

    def _boom(*_args, **_kwargs):
        raise OSError("disk gone")

    monkeypatch.setattr(private_file.os, "replace", _boom)

    with pytest.raises(OSError):
        write_private_file(target, "new")

    assert target.read_text(encoding="utf-8") == "old"
    assert [entry.name for entry in tmp_path.iterdir()] == ["secret"]
