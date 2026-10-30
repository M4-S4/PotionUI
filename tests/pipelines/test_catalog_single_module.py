import sys
from pathlib import Path

import pytest

from src.pipelines.catalog import PipeCatalog

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_PIPES = REPO_ROOT / "src" / "pipelines" / "pipes"


@pytest.mark.parametrize("name", ["output_skipper", "media_tool"])
def test_core_pipe_loaded_through_catalog_is_the_importable_module_object(name):
    catalog = PipeCatalog(str(CORE_PIPES), str(REPO_ROOT / "pipes" / "custom"))

    pipe_class = catalog.get_pipe(name)

    module_name = f"src.pipelines.pipes.{name}.main"
    assert pipe_class is not None
    assert pipe_class.__module__ == module_name
    assert sys.modules[module_name] is sys.modules[pipe_class.__module__]
    assert getattr(sys.modules[module_name], pipe_class.__name__) is pipe_class
    assert f"pipes.{name}" not in sys.modules


def test_core_pipe_path_given_relative_still_uses_the_package_module(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    catalog = PipeCatalog("src/pipelines/pipes", "pipes/custom")

    pipe_class = catalog.get_pipe("output_skipper")

    assert pipe_class.__module__ == "src.pipelines.pipes.output_skipper.main"


def test_pipes_outside_the_core_package_keep_file_loading(tmp_path):
    core = tmp_path / "core"
    (core / "solo").mkdir(parents=True)
    (core / "solo" / "main.py").write_text(
        "from src.pipelines.contracts import BasePipe\n"
        "class SoloPipe(BasePipe):\n"
        "    name = 'solo'\n",
        encoding="utf-8",
    )
    catalog = PipeCatalog(str(core), str(tmp_path / "custom"))

    pipe_class = catalog.get_pipe("solo")

    assert pipe_class is not None
    assert pipe_class.__module__ == "pipes.solo"
