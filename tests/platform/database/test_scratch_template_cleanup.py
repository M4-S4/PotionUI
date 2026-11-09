import os
import subprocess
import sys
from pathlib import Path

from src.platform.database.testing import _remove_template_directory

REPO_ROOT = Path(__file__).resolve().parents[3]
PREFIX = "potionui-test-db-template-"
HOLDER = (
    "import sys\n"
    "from src.platform.database.testing import scratch_database\n"
    "scratch_database(sys.argv[1])\n"
    "from src.platform.database.testing import template_db_path\n"
    "print(template_db_path(), flush=True)\n"
)


def _environment(tmp_path):
    temp_root = tmp_path / "temp"
    temp_root.mkdir(exist_ok=True)
    environment = dict(os.environ)
    environment.pop("PYTEST_CURRENT_TEST", None)
    environment.pop("PYTEST_XDIST_WORKER", None)
    for name in ("TMPDIR", "TEMP", "TMP"):
        environment[name] = str(temp_root)
    environment["POTIONUI_DB_PATH"] = str(tmp_path / "unused" / "db.sqlite")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return temp_root, environment


def test_the_scratch_database_template_is_removed_when_its_process_ends(tmp_path):
    temp_root, environment = _environment(tmp_path)

    result = subprocess.run(
        [sys.executable, "-B", "-c", HOLDER, str(tmp_path / "scratch.sqlite")],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    template = Path(result.stdout.strip())
    assert template.parent.parent == temp_root
    assert (tmp_path / "scratch.sqlite").is_file()
    assert not template.parent.exists()
    assert [path.name for path in temp_root.iterdir() if path.name.startswith(PREFIX)] == []


def test_another_process_does_not_remove_the_scratch_database_template(tmp_path):
    directory = tmp_path / f"{PREFIX}kept"
    directory.mkdir()

    _remove_template_directory(directory, os.getpid() + 1)

    assert directory.is_dir()
