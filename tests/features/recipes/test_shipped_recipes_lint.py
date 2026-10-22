"""Every recipe the repository ships passes the lint `scripts/recipe_lint.py`
runs in the release gate: the core marketplace tree and the `recipes:` root of
every marketplace plugin. A shipped `smoke.timeout_seconds` out of range is a
lint error, so it fails here.

The default of 300 s is too short for a video model on a 12 GB card, so a
recipe whose smoke generates video declares `smoke.timeout_seconds`, or is
named with its reason in `VIDEO_SMOKES_ON_THE_DEFAULT`. The recipes whose smoke
needs more than the default are pinned in `LONG_SMOKES`."""

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

LONG_SMOKES = {
    "ltx25-starter": 600,
    "vp-wan22-14b-i2v": 600,
    "vp-wan22-14b-lightning-12gb": 600,
    "wan22-starter": 600,
}

VIDEO_SMOKES_ON_THE_DEFAULT = {
    "ltx2-starter": "smoke duration on 12 GB unmeasured; LTX-2.3 22B ran out of GPU memory there (docs/user/hardware-requirements.md)",
    "minimax-h3-fast-starter": "smoke duration on 12 GB unmeasured; the plugin's 12 GB preset needs 6 min 11 s for a 5 s clip (docs/user/hardware-requirements.md)",
    "minimax-h3-starter": "smoke duration on 12 GB unmeasured; the plugin's 12 GB preset needs 9 min 48 s for a 5 s clip (docs/user/hardware-requirements.md)",
    "vp-wan21-t2v-1-3b-8gb": "smoke duration unmeasured; a 5 s clip takes 5 min 40 s to 5 min 55 s on the 8 GB tier (docs/user/hardware-requirements.md)",
    "vp-wan22-ti2v-5b-8gb": "smoke duration unmeasured; a 5 s clip takes 4 min 38 s to 4 min 51 s on the 8 GB tier (docs/user/hardware-requirements.md)",
}


@pytest.fixture(scope="module")
def manifests():
    return PluginLoader(str(PLUGINS), str(REPO_ROOT / "content" / "plugins" / "_none")).discover_plugins()


@pytest.fixture(scope="module")
def preset_index(manifests):
    return recipe_lint.load_preset_index(plugin_manifests=manifests)


def _plugin_recipe_files(manifests):
    return sorted(f for root in plugin_recipe_roots(manifests) for f in Path(root.path).glob("*.yml"))


@pytest.fixture(scope="module")
def smokes(manifests):
    found = {}
    for path in sorted(CORE_RECIPES.glob("*.yml")) + _plugin_recipe_files(manifests):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data.get("smoke"):
            recipe = parse_recipe(data, source_path=str(path))
            found[recipe.id] = (recipe.smoke, data["smoke"].get("timeout_seconds") is not None)
    return found


def _is_video(smoke, preset_index):
    return smoke.mode == "video" or preset_index[smoke.preset_id].category == "video"


def _video_smokes_on_the_default(smokes, preset_index):
    return {
        recipe_id
        for recipe_id, (smoke, declared) in smokes.items()
        if _is_video(smoke, preset_index) and not declared
    }


def test_the_shipped_trees_are_not_empty(manifests):
    assert sorted(CORE_RECIPES.glob("*.yml"))
    assert _plugin_recipe_files(manifests)


def test_every_shipped_recipe_passes_the_recipe_lint(manifests, preset_index):
    extra_kinds = recipe_lint.plugin_step_kinds(manifests)
    files = sorted(CORE_RECIPES.glob("*.yml")) + _plugin_recipe_files(manifests)

    failures = {
        str(path.relative_to(REPO_ROOT)): issues
        for path in files
        if (issues := recipe_lint.lint_recipe_file(path, preset_index, extra_kinds))
    }

    assert failures == {}


@pytest.mark.parametrize(("recipe_id", "minimum"), sorted(LONG_SMOKES.items()))
def test_a_smoke_known_to_run_past_five_minutes_declares_enough_time(smokes, recipe_id, minimum):
    smoke, _ = smokes[recipe_id]

    assert smoke.timeout_seconds >= minimum


def test_every_video_smoke_declares_its_timeout_or_is_listed_as_keeping_the_default(smokes, preset_index):
    undeclared = _video_smokes_on_the_default(smokes, preset_index)

    assert sorted(undeclared - VIDEO_SMOKES_ON_THE_DEFAULT.keys()) == []


def test_the_list_of_video_smokes_keeping_the_default_names_only_ones_that_still_do(smokes, preset_index):
    undeclared = _video_smokes_on_the_default(smokes, preset_index)

    assert sorted(VIDEO_SMOKES_ON_THE_DEFAULT.keys() - undeclared) == []
    assert all(VIDEO_SMOKES_ON_THE_DEFAULT.values())


def test_the_smokes_known_to_run_past_five_minutes_are_seen_as_video_smokes(smokes, preset_index):
    assert all(_is_video(smokes[recipe_id][0], preset_index) for recipe_id in LONG_SMOKES)
