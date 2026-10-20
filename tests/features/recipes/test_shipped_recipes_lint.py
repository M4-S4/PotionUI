"""Every recipe the repository ships passes the lint `scripts/recipe_lint.py`
runs in the release gate: the core marketplace tree and the `recipes:` root of
every marketplace plugin. A shipped `smoke.timeout_seconds` out of range is a
lint error, so it fails here."""

from pathlib import Path

import pytest

import yaml

from scripts import recipe_lint
from src.features.recipes.catalog import plugin_recipe_roots
from src.features.recipes.schema import parse_recipe
from src.platform.plugins.loader import PluginLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_RECIPES = REPO_ROOT / "content" / "recipes" / "marketplace"
PLUGINS = REPO_ROOT / "content" / "plugins" / "marketplace"


@pytest.fixture(scope="module")
def manifests():
    return PluginLoader(str(PLUGINS), str(REPO_ROOT / "content" / "plugins" / "_none")).discover_plugins()


def _plugin_recipe_files(manifests):
    return sorted(f for root in plugin_recipe_roots(manifests) for f in Path(root.path).glob("*.yml"))


def test_the_shipped_trees_are_not_empty(manifests):
    assert sorted(CORE_RECIPES.glob("*.yml"))
    assert _plugin_recipe_files(manifests)


def test_every_shipped_recipe_passes_the_recipe_lint(manifests):
    preset_index = recipe_lint.load_preset_index(plugin_manifests=manifests)
    extra_kinds = recipe_lint.plugin_step_kinds(manifests)
    files = sorted(CORE_RECIPES.glob("*.yml")) + _plugin_recipe_files(manifests)

    failures = {
        str(path.relative_to(REPO_ROOT)): issues
        for path in files
        if (issues := recipe_lint.lint_recipe_file(path, preset_index, extra_kinds))
    }

    assert failures == {}


def test_the_ltx25_starter_smoke_waits_ten_minutes():
    path = CORE_RECIPES / "ltx25-starter.yml"
    recipe = parse_recipe(yaml.safe_load(path.read_text(encoding="utf-8")), source_path=str(path))

    assert recipe.smoke.timeout_seconds == 600
