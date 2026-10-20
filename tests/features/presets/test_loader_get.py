import yaml

from src.features.presets.loader import PresetTemplateLoader
from tests.features.presets.test_reload_rescan import _write_preset


def _manifest_name(preset_dir, name):
    path = preset_dir / "preset.yml"
    data = yaml.safe_load(path.read_text())
    data["name"] = name
    path.write_text(yaml.dump(data))


def test_get_loads_lazily_and_returns_the_template(tmp_path):
    _write_preset(tmp_path, "alpha", ["txt2img"])
    loader = PresetTemplateLoader([str(tmp_path)])

    found = loader.get("alpha")

    assert found is not None
    assert found.id == "alpha"
    assert loader.presets_by_id["alpha"] is found


def test_get_returns_none_for_unknown_and_empty_ids(tmp_path):
    _write_preset(tmp_path, "alpha", ["txt2img"])
    loader = PresetTemplateLoader([str(tmp_path)])

    assert loader.get("missing") is None
    assert loader.get("") is None
    assert loader.get(None) is None


def test_clear_cache_empties_the_index_and_next_get_reloads(tmp_path):
    _write_preset(tmp_path, "alpha", ["txt2img"])
    loader = PresetTemplateLoader([str(tmp_path)])
    assert loader.get("alpha") is not None

    loader.clear_cache()

    assert loader.presets_by_id == {}
    assert loader.get("alpha") is not None


def test_a_duplicate_id_resolves_to_the_first_loaded_preset(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_dir = _write_preset(first_root, "dup", ["txt2img"])
    second_dir = _write_preset(second_root, "dup", ["txt2img"])
    _manifest_name(first_dir, "First")
    _manifest_name(second_dir, "Second")
    loader = PresetTemplateLoader([str(first_root), str(second_root)])

    loader.load_presets()
    expected = next(p for p in loader.presets if p.id == "dup")

    assert loader.get("dup") is expected
    assert loader.all() is loader.presets


def test_load_preset_by_id_delegates_to_get(tmp_path):
    _write_preset(tmp_path, "alpha", ["txt2img"])
    loader = PresetTemplateLoader([str(tmp_path)])

    assert loader.load_preset_by_id("alpha") is loader.get("alpha")


def _callback_lookups(loader, preset_id, seen):
    def callback(presets):
        seen.append((loader.get(preset_id), [p.id for p in loader.all()]))

    return callback


def _run_in_thread(fn):
    import threading

    thread = threading.Thread(target=fn, daemon=True)
    thread.start()
    thread.join(timeout=5)
    return not thread.is_alive()


def test_callback_can_look_up_presets_during_the_lazy_load(tmp_path):
    _write_preset(tmp_path, "alpha", ["txt2img"])
    loader = PresetTemplateLoader([str(tmp_path)])
    seen = []
    loader.on_presets_changed = _callback_lookups(loader, "alpha", seen)

    finished = _run_in_thread(lambda: loader.get("alpha"))

    assert finished
    assert len(seen) == 1
    found, ids = seen[0]
    assert found is not None and found.id == "alpha"
    assert ids == ["alpha"]


def test_callback_can_look_up_presets_during_reload(tmp_path):
    _write_preset(tmp_path, "alpha", ["txt2img"])
    loader = PresetTemplateLoader([str(tmp_path)])
    loader.load_presets()
    loader.clear_cache()
    seen = []
    loader.on_presets_changed = _callback_lookups(loader, "alpha", seen)

    finished = _run_in_thread(loader.reload)

    assert finished
    assert len(seen) == 1
    assert seen[0][0] is not None
