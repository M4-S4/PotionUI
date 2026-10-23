import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "build-plugins.mjs"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not on PATH")

RECORDING_ESBUILD = """\
const { appendFileSync } = require('fs');
module.exports = {
	build: async (options) => {
		appendFileSync(process.env.BUILD_PLUGINS_TEST_LOG, JSON.stringify({ ...options, cwd: process.cwd() }) + '\\n');
	}
};
"""
FAKE_ESBUILD_SVELTE = "module.exports = () => ({ name: 'svelte' });\n"
FAKE_JS_YAML = "module.exports = { load: (text) => JSON.parse(text) };\n"


def _fake_toolchain(repo: Path) -> None:
    plugins = repo / "content" / "plugins"
    (plugins / "node_modules").mkdir(parents=True)
    (plugins / "package.json").write_text('{"name": "plugins", "private": true}\n')
    for name, source in (
        ("esbuild", RECORDING_ESBUILD),
        ("esbuild-svelte", FAKE_ESBUILD_SVELTE),
        ("js-yaml", FAKE_JS_YAML),
    ):
        module = plugins / "node_modules" / name
        module.mkdir()
        (module / "package.json").write_text(json.dumps({"name": name, "main": "index.js"}))
        (module / "index.js").write_text(source)


def _plugin(folder: Path, plugin_id: str) -> None:
    src = folder / "frontend" / "src"
    src.mkdir(parents=True)
    manifest = {"id": plugin_id, "pages": [{"component": "Page.js"}], "sidebar_widgets": [{"component": "Widget.js"}]}
    (folder / "manifest.yml").write_text(json.dumps(manifest))
    (src / "index.js").write_text("import Page from './Page.svelte';\nexport { Page };\n")
    (src / "Page.svelte").write_text("<div></div>\n")
    (src / "Widget.svelte").write_text("<div></div>\n")


def _link(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        if sys.platform != "win32":
            raise
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, repo / "scripts" / "build-plugins.mjs")
    _fake_toolchain(repo)
    local = repo / "content" / "plugins" / "local"
    local.mkdir()
    _plugin(local / "plain-plugin", "plain-plugin")
    elsewhere = tmp_path / "elsewhere" / "linked-plugin"
    _plugin(elsewhere, "linked-plugin")
    _link(local / "linked-plugin", elsewhere)
    return repo


def _run(repo: Path, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    log = repo / "esbuild-calls.jsonl"
    log.unlink(missing_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "PLUGIN_BUILD"}
    env["BUILD_PLUGINS_TEST_LOG"] = str(log)
    return subprocess.run(
        [NODE, str(repo / "scripts" / "build-plugins.mjs"), *args],
        cwd=cwd or repo, env=env, capture_output=True, text=True,
    )


def _entries(options: dict) -> list:
    entry_points = options["entryPoints"]
    return list(entry_points.values()) if isinstance(entry_points, dict) else entry_points


def _esbuild_calls(repo: Path, *args: str, cwd: Path | None = None) -> dict:
    result = _run(repo, *args, cwd=cwd)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = {}
    for line in (repo / "esbuild-calls.jsonl").read_text(encoding="utf-8").splitlines():
        options = json.loads(line)
        plugin_id = Path(_entries(options)[0]).parts[-4]
        calls.setdefault(plugin_id, []).append(options)
    return calls


def test_only_a_linked_plugin_folder_resolves_from_the_path_as_given(repo):
    calls = _esbuild_calls(repo)

    assert len(calls["plain-plugin"]) == 2 and len(calls["linked-plugin"]) == 2
    assert [options["preserveSymlinks"] for options in calls["plain-plugin"]] == [False, False]
    assert [options["preserveSymlinks"] for options in calls["linked-plugin"]] == [True, True]


def test_a_linked_plugin_is_handed_to_esbuild_by_its_linked_path(repo):
    for options in _esbuild_calls(repo)["linked-plugin"]:
        for entry in _entries(options):
            assert Path(entry).parent.parent.parent == repo / "content" / "plugins" / "local" / "linked-plugin"


def test_release_and_debug_modes_differ_in_minify_and_sourcemap_only(repo):
    release = _esbuild_calls(repo)["plain-plugin"]
    debug = _esbuild_calls(repo, "--debug")["plain-plugin"]

    assert [(o["minify"], o["sourcemap"], o["legalComments"]) for o in release] == [(True, False, "eof")] * 2
    assert [(o["minify"], o["sourcemap"], o["legalComments"]) for o in debug] == [(False, True, "eof")] * 2
    for release_options, debug_options in zip(release, debug):
        unchanged = {k: v for k, v in release_options.items() if k not in ("minify", "sourcemap")}
        assert unchanged == {k: v for k, v in debug_options.items() if k not in ("minify", "sourcemap")}


def test_a_dangling_plugin_link_is_skipped_with_a_warning_and_the_rest_still_build(repo, tmp_path):
    gone = tmp_path / "gone" / "dead-plugin"
    gone.mkdir(parents=True)
    dangling = repo / "content" / "plugins" / "local" / "dead-plugin"
    _link(dangling, gone)
    gone.rmdir()
    if dangling.exists():
        pytest.skip("platform keeps the link resolvable after its target is removed")

    result = _run(repo)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "dead-plugin" in result.stderr
    ids = {Path(_entries(json.loads(line))[0]).parts[-4] for line in (repo / "esbuild-calls.jsonl").read_text(encoding="utf-8").splitlines()}
    assert ids == {"plain-plugin", "linked-plugin"}


def test_component_entries_are_built_from_any_manifest_section_and_split_into_shared_chunks(repo):
    plugin = repo / "content" / "plugins" / "local" / "plain-plugin"
    manifest = json.loads((plugin / "manifest.yml").read_text())
    manifest["organize"] = {"facts": [{"component": "Fact.js"}], "actions": [{"component": "Action.svelte"}]}
    (plugin / "manifest.yml").write_text(json.dumps(manifest))
    for name in ("Fact", "Action"):
        (plugin / "frontend" / "src" / f"{name}.svelte").write_text("<div></div>\n")

    page_call, component_call = _esbuild_calls(repo, "plain-plugin")["plain-plugin"]

    assert component_call["splitting"] is True
    assert component_call["chunkNames"] == "chunk-[hash]"
    assert {Path(entry).name for entry in component_call["entryPoints"]} == {"Widget.svelte", "Fact.svelte", "Action.svelte"}


def _component_only_plugin(folder: Path, plugin_id: str) -> None:
    src = folder / "frontend" / "src"
    src.mkdir(parents=True)
    (folder / "manifest.yml").write_text(json.dumps({"id": plugin_id, "sidebar_widgets": [{"component": "Widget.js"}]}))
    (src / "Widget.svelte").write_text("<div></div>\n")


@pytest.mark.parametrize("plugin_id", ["plain-plugin", "component-only"])
def test_stale_chunks_are_removed_from_dist_and_other_files_are_kept(repo, plugin_id):
    local = repo / "content" / "plugins" / "local"
    if plugin_id == "component-only":
        _component_only_plugin(local / plugin_id, plugin_id)
    dist = local / plugin_id / "frontend" / "dist"
    dist.mkdir(parents=True)
    for name in ("chunk-ABC123.js", "chunk-ABC123.js.map", "Widget.js", "chunky.js", "notchunk-X.js"):
        (dist / name).write_text("x")

    result = _run(repo, plugin_id)

    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(p.name for p in dist.iterdir()) == ["Widget.js", "chunky.js", "notchunk-X.js"]


def test_a_page_is_split_into_chunks_under_its_own_folder(repo):
    plugin = repo / "content" / "plugins" / "local" / "plain-plugin"

    page_call = _esbuild_calls(repo, "plain-plugin")["plain-plugin"][0]

    assert page_call["entryPoints"] == {"Page": str(plugin / "frontend" / "src" / "index.js")}
    assert (page_call["splitting"], page_call["entryNames"], page_call["chunkNames"]) == (True, "[name]", "chunks/[name]-[hash]")
    assert Path(page_call["outdir"]) == plugin / "frontend" / "dist"
    assert "outfile" not in page_call


def test_a_page_build_replaces_its_chunks_folder_and_keeps_the_component_chunks_rule(repo):
    dist = repo / "content" / "plugins" / "local" / "plain-plugin" / "frontend" / "dist"
    (dist / "chunks").mkdir(parents=True)
    (dist / "chunks" / "engine-OLD.js").write_text("x")
    (dist / "Page.js").write_text("x")

    result = _run(repo, "plain-plugin")

    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(p.name for p in dist.iterdir()) == ["Page.js"]


@pytest.mark.parametrize("has_frontend", [True, False])
def test_every_build_runs_from_the_frontend_folder_whatever_the_working_directory(repo, tmp_path, has_frontend):
    if has_frontend:
        (repo / "frontend").mkdir()
    expected = (repo / "frontend" if has_frontend else repo).resolve()

    for start in (repo, tmp_path):
        calls = [o for options in _esbuild_calls(repo, cwd=start).values() for o in options]
        assert calls and {Path(o["cwd"]).resolve() for o in calls} == {expected}
