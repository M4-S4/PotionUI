import logging

from src.pipelines.catalog import PipeCatalog

PIPE_SOURCE = "\n".join([
    "from typing import Any, Dict, List",
    "from src.pipelines.contracts import BasePipe, PipeConfigSpec, PipeInput, PipeInputSpec, PipeOutput, PipeOutputSpec",
    "",
    "class {cls}(BasePipe):",
    "    name = '{name}'",
    "    description = 'fixture'",
    "",
    "    def process(self, pipe_input: PipeInput, generation_outputs: callable) -> PipeOutput:",
    "        return PipeOutput(output={{}})",
    "",
    "    @classmethod",
    "    def get_default_config(cls) -> Dict[str, Any]:",
    "        return {{}}",
    "",
    "    @classmethod",
    "    def inputs(cls) -> List[PipeInputSpec]:",
    "        return []",
    "",
    "    @classmethod",
    "    def outputs(cls) -> List[PipeOutputSpec]:",
    "        return []",
    "",
    "    @classmethod",
    "    def configuration(cls) -> List[PipeConfigSpec]:",
    "        return []",
    "",
])


def _write_pipe(root, relative, cls, name):
    target = root.joinpath(*relative.split("/"))
    target.mkdir(parents=True, exist_ok=True)
    (target / "main.py").write_text(PIPE_SOURCE.format(cls=cls, name=name), encoding="utf-8")


def _catalog(tmp_path):
    core = tmp_path / "core"
    custom = tmp_path / "custom"
    core.mkdir()
    custom.mkdir()
    return PipeCatalog(str(core), str(custom)), core, custom


def test_discover_pipes_loads_every_location(tmp_path):
    catalog, core, _ = _catalog(tmp_path)
    _write_pipe(core, "alpha", "AlphaPipe", "alpha")
    _write_pipe(core, "beta/variant", "BetaVariantPipe", "generator")

    catalog.discover_pipes()

    assert set(catalog.pipes) == {"alpha", "beta/variant"}
    assert catalog.get_pipe_source("alpha") == "core"


def test_main_pipe_is_reachable_by_directory_key_and_by_class_name(tmp_path):
    catalog, _, custom = _catalog(tmp_path)
    _write_pipe(custom, "folder_name", "RenamedPipe", "declared_name")

    catalog.discover_pipes()

    assert catalog.get_pipe("folder_name") is catalog.get_pipe("declared_name")
    assert catalog.get_pipe("declared_name") is not None
    assert catalog.get_pipe_source("declared_name") == "custom"


def test_variants_register_only_under_dir_and_variant(tmp_path):
    catalog, core, _ = _catalog(tmp_path)
    _write_pipe(core, "family/one", "OnePipe", "generator")

    catalog.discover_pipes()

    assert "generator" not in catalog.pipes
    assert "family/one" in catalog.pipes


def test_import_error_is_logged_at_discover_time_and_does_not_abort(tmp_path, caplog):
    catalog, core, _ = _catalog(tmp_path)
    _write_pipe(core, "good", "GoodPipe", "good")
    broken = core / "broken"
    broken.mkdir()
    (broken / "main.py").write_text("raise RuntimeError('boom at import')\n", encoding="utf-8")

    with caplog.at_level(logging.ERROR):
        catalog.discover_pipes()

    assert "good" in catalog.pipes
    assert "broken" not in catalog.pipes
    assert any("boom at import" in r.getMessage() or "broken" in r.getMessage() for r in caplog.records)


def test_get_available_pipes_discovers_once_and_returns_every_pipe(tmp_path):
    catalog, core, _ = _catalog(tmp_path)
    _write_pipe(core, "alpha", "AlphaPipe", "alpha")

    first = catalog.get_available_pipes()
    _write_pipe(core, "late", "LatePipe", "late")
    second = catalog.get_available_pipes()

    assert len(first) == 1
    assert len(second) == 1


def test_rediscovery_picks_up_a_new_pipe(tmp_path):
    catalog, core, _ = _catalog(tmp_path)
    _write_pipe(core, "alpha", "AlphaPipe", "alpha")
    catalog.discover_pipes()
    _write_pipe(core, "late", "LatePipe", "late")

    catalog.discover_pipes()

    assert {"alpha", "late"} <= set(catalog.pipes)
