import os
import stat
import tempfile
from pathlib import Path

PRIVATE_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR


def _unlink_quietly(name: str) -> None:
    try:
        os.unlink(name)
    except OSError:
        pass


def write_private_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, PRIVATE_FILE_MODE)
        else:
            os.chmod(tmp_name, PRIVATE_FILE_MODE)
        handle = os.fdopen(fd, "w", encoding="utf-8")
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        _unlink_quietly(tmp_name)
        raise
    try:
        with handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        _unlink_quietly(tmp_name)
        raise
    try:
        os.chmod(path, PRIVATE_FILE_MODE)
    except OSError:
        pass
