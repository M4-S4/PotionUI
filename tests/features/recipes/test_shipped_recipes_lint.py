"""Every recipe the repository ships passes the lint `scripts/recipe_lint.py`
runs in the release gate: the core marketplace tree and the `recipes:` root of
every marketplace plugin. A shipped `smoke.timeout_seconds` out of range is a
lint error, so it fails here.

The default of 300 s is too short for a video model on a 12 GB card, so a
recipe whose smoke generates video declares `smoke.timeout_seconds`, is named
with its cold smoke time in `VIDEO_SMOKES_MEASURED_WITHIN_THE_DEFAULT` when one
and a half times that time fits the default, or waits for a measure in
`VIDEO_SMOKES_AWAITING_A_MEASURE` with its reason. The recipes whose smoke
needs more than the default are pinned in `LONG_SMOKES`.

A smoke that runs on a Video Director document (every preset mode that shows
the Director with a `t2v` sub-mode) takes its clip length from that document's
duration, the shortest the preset allows, so a `frames` or `duration` field in
its `smoke.form` would never be read: such a field is a lint failure here."""

from pathlib import Path

import pytest

import yaml

from scripts import recipe_lint
from src.features.recipes.catalog import plugin_recipe_roots
from src.features.recipes.executors._fixture_form import fixture_video_director_capabilities
from src.features.recipes.schema import DEFAULT_SMOKE_TIMEOUT_SECONDS, parse_recipe
from src.platform.plugins.loader import PluginLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_RECIPES = REPO_ROOT / "content" / "recipes" / "marketplace"
PLUGINS = REPO_ROOT / "content" / "plugins" / "marketplace"

LONG_SMOKES = {
    "ltx25-starter": 600,
    "minimax-h3-fast-starter": 400,
    "minimax-h3-starter": 400,
    "vp-wan22-14b-i2v": 600,
    "vp-wan22-14b-lightning-12gb": 600,
    "wan22-starter": 600,
}

VIDEO_SMOKES_AWAITING_A_MEASURE = {
    "ltx2-starter": "smoke duration on 12 GB unmeasured; LTX-2.3 22B ran out of GPU memory there (docs/user/hardware-requirements.md)",
}

VIDEO_SMOKES_MEASURED_WITHIN_THE_DEFAULT = {
    "vp-wan21-t2v-1-3b-8gb": 64,
    "vp-wan22-ti2v-5b-8gb": 111,
}

VIDEO_SMOKES_ON_THE_DEFAULT = {**VIDEO_SMOKES_AWAITING_A_MEASURE, **VIDEO_SMOKES_MEASURED_WITHIN_THE_DEFAULT}


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


def _runs_on_a_director_document(smoke, preset_index):
    return fixture_video_director_capabilities(preset_index[smoke.preset_id], smoke.mode) is not None


CLIP_LENGTH_FIELDS = {"frames", "duration", "num_frames"}


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
    assert all(VIDEO_SMOKES_AWAITING_A_MEASURE.values())


@pytest.mark.parametrize(("recipe_id", "seconds"), sorted(VIDEO_SMOKES_MEASURED_WITHIN_THE_DEFAULT.items()))
def test_a_measured_smoke_keeps_the_default_only_with_half_its_time_to_spare(smokes, recipe_id, seconds):
    smoke, _ = smokes[recipe_id]

    assert smoke.timeout_seconds == DEFAULT_SMOKE_TIMEOUT_SECONDS
    assert seconds * 1.5 <= DEFAULT_SMOKE_TIMEOUT_SECONDS


def test_the_smokes_known_to_run_past_five_minutes_are_seen_as_video_smokes(smokes, preset_index):
    assert all(_is_video(smokes[recipe_id][0], preset_index) for recipe_id in LONG_SMOKES)


def test_a_smoke_on_a_director_document_sets_no_clip_length_in_its_form(smokes, preset_index):
    director_smokes = {
        recipe_id for recipe_id, (smoke, _) in smokes.items() if _runs_on_a_director_document(smoke, preset_index)
    }
    assert {"minimax-h3-starter", "minimax-h3-fast-starter"} <= director_smokes

    dead_fields = {
        recipe_id: sorted(CLIP_LENGTH_FIELDS & set((smokes[recipe_id][0].form or {}).keys()))
        for recipe_id in director_smokes
    }

    assert {recipe_id: fields for recipe_id, fields in dead_fields.items() if fields} == {}
